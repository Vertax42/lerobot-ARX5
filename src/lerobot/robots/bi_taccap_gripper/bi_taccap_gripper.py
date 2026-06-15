#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Bimanual TacCap-Gripper handheld data-collection device for LeRobot.

Two independent TacCap-Gripper units driven as one robot. Like the single
``taccap_gripper`` this is a passive recording device — ``send_action()`` is a
no-op (the jaw motors stay disabled; we read encoders only). Pose comes from a
per-side Pico4 Ultra tracker (``Pico4TrackerReader``); tactile + wrist cameras go
through the standard ``cameras`` framework.

Implemented with the *reimplement-with-prefixes* pattern (cf. ``bi_elite_cs66_rt``):
per-side hardware handles live in dicts keyed ``"left"``/``"right"`` and every
observation/action key is ``left_``/``right_`` prefixed. The per-side reading logic
is the same as the single ``TaccapGripper``.

Observation features (per side ``{s}`` in left/right):
    {s}_tcp.x/y/z, {s}_tcp.r1..r6   -- Pico4 tracker → EE 6D pose (if enable_tracker)
    {s}_gripper.pos                 -- normalised jaw [0=closed, 1=open]
    {s}_imu.accel/gyro/mag.{x,y,z}  -- optional
    {s}_wrist                       -- wrist UVC frame (if enable_wrist_camera)
    {s}_tactile_0 / {s}_tactile_1   -- tactile frames (from `cameras`)
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import get_logger

from ..robot import Robot
from ..taccap_gripper.taccap_gripper import resolve_wrist_camera_path
from .config_bi_taccap_gripper import BiTaccapGripperConfig

_SIDES = ("left", "right")

# ---- TacCap-Gripper SDK -----------------------------------------------------
try:
    from xense.taccap import (
        LeaderGripper,
        find_one,
        scan_grippers,
    )

    TACCAP_SDK_AVAILABLE = True
except ImportError:
    TACCAP_SDK_AVAILABLE = False

# ---- Pico4 tracker reader ---------------------------------------------------
try:
    from lerobot.teleoperators.pico4.tracker import Pico4TrackerReader

    PICO4_TRACKER_AVAILABLE = True
except ImportError:
    PICO4_TRACKER_AVAILABLE = False


class BiTaccapGripper(Robot):
    """Bimanual TacCap-Gripper handheld data-collection device.

    Operated manually — no action is applied to either jaw. Emits ``left_``/
    ``right_`` prefixed pose + jaw + tactile + wrist for downstream learning.

    6D rotation convention (matches ``vive_tracker`` / single ``taccap_gripper``):
        r1..r3 = first column of the rotation matrix
        r4..r6 = second column of the rotation matrix
    """

    config_class = BiTaccapGripperConfig
    name = "bi_taccap_gripper"

    def __init__(self, config: BiTaccapGripperConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(f"BiTaccapGripper-{config.id or 'default'}")

        any_gripper = any(getattr(config, f"{s}_enable_gripper") for s in _SIDES)
        any_tracker = any(getattr(config, f"{s}_enable_tracker") for s in _SIDES)
        if any_gripper and not TACCAP_SDK_AVAILABLE:
            raise ImportError(
                "xense.taccap SDK not available. Build it from the vendored "
                "submodule third_party/taccap-gripper (run setup_env.sh --install)."
            )
        if any_tracker and not PICO4_TRACKER_AVAILABLE:
            raise ImportError(
                "Pico4TrackerReader not available. Ensure "
                "src/lerobot/teleoperators/pico4/tracker.py is importable."
            )

        # Per-side hardware handles, populated on connect.
        self._gripper: dict[str, Any] = {s: None for s in _SIDES}  # LeaderGripper
        self._endpoints: dict[str, Any] = {s: None for s in _SIDES}  # GripperEndpoints
        self._tracker: dict[str, Pico4TrackerReader | None] = {s: None for s in _SIDES}

        # Tactile cameras (wrist cameras are added per side in connect()).
        self.cameras = make_cameras_from_configs(config.cameras)

        self._is_connected = False

    # ------------------------------------------------------------------ schema

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {}

        for side in _SIDES:
            if getattr(self.config, f"{side}_enable_tracker"):
                for k in ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6"):
                    features[f"{side}_tcp.{k}"] = float
            if getattr(self.config, f"{side}_enable_gripper"):
                features[f"{side}_gripper.pos"] = float
            if getattr(self.config, f"{side}_enable_imu"):
                for axis in ("x", "y", "z"):
                    features[f"{side}_imu.accel.{axis}"] = float
                    features[f"{side}_imu.gyro.{axis}"] = float
                    features[f"{side}_imu.mag.{axis}"] = float

        # Tactile cameras (keys are already left_/right_ prefixed in config.cameras).
        for cam_name, cam_cfg in self.config.cameras.items():
            features[cam_name] = (cam_cfg.height, cam_cfg.width, 3)

        # Wrist cameras (wired per side from config, not from `cameras`).
        for side in _SIDES:
            if getattr(self.config, f"{side}_enable_wrist_camera"):
                features[f"{side}_wrist"] = (
                    getattr(self.config, f"{side}_wrist_camera_height"),
                    getattr(self.config, f"{side}_wrist_camera_width"),
                    3,
                )

        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """The 'demonstration' action the rig emits (pose + jaw per side, no cameras)."""
        features: dict[str, type] = {}
        for side in _SIDES:
            if getattr(self.config, f"{side}_enable_tracker"):
                for k in ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6"):
                    features[f"{side}_tcp.{k}"] = float
            if getattr(self.config, f"{side}_enable_gripper"):
                features[f"{side}_gripper.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Factory calibration; the only per-unit step is the gripper encoder
        zero, which lives in firmware (set once via the SDK's calibrate.py)."""
        return self.is_connected

    # ------------------------------------------------------------------ lifecycle

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.logger.info("Connecting BiTacCap-Gripper...")

        for side in _SIDES:
            # 1. Gripper SDK (MCU transport only; cameras come from the LeRobot
            #    camera framework, so open_cameras stays False).
            if getattr(self.config, f"{side}_enable_gripper"):
                endpoints = self._discover_gripper(side)
                self._endpoints[side] = endpoints
                self.logger.info(
                    f"  [{side}] TacCap-Gripper: side={endpoints.side} "
                    f"fw_sn={endpoints.firmware_sn!r} mcu={endpoints.mcu_serial!r}"
                )
                self._gripper[side] = LeaderGripper(endpoints.mcu_device)
                self.logger.info(
                    f"  [{side}] ✅ LeaderGripper attached (MCU-only, read-only)"
                )

            # 2. Wrist camera (V4L2 path from config; usable even if gripper off).
            if getattr(self.config, f"{side}_enable_wrist_camera"):
                self._attach_wrist_camera(side)

            # 3. Pico4 tracker.
            if getattr(self.config, f"{side}_enable_tracker"):
                tracker = Pico4TrackerReader(
                    tracker_sn=getattr(self.config, f"{side}_tracker_sn"),
                    tracker_to_ee_pos=getattr(self.config, f"{side}_tracker_to_ee_pos"),
                    tracker_to_ee_quat=getattr(self.config, f"{side}_tracker_to_ee_quat"),
                    device_wait_timeout=self.config.tracker_wait_timeout,
                    logger_name=f"{self.config.id or 'bi'}-{side}",
                )
                init_pose = (
                    np.asarray(
                        getattr(self.config, f"{side}_init_tcp_pose"), dtype=np.float64
                    )
                    if getattr(self.config, f"{side}_enable_init_pose_alignment")
                    else None
                )
                tracker.connect(current_tcp_pose_quat=init_pose)
                self._tracker[side] = tracker
                if init_pose is not None:
                    self.logger.info(
                        f"  [{side}] ✅ Pico4 tracker connected with UMI alignment"
                    )
                else:
                    self.logger.info(
                        f"  [{side}] ✅ Pico4 tracker connected (raw xrt frame)"
                    )

        # 4. Cameras (tactile + the wrist cameras added above).
        for cam_name, cam in self.cameras.items():
            self.logger.info(f"  Connecting camera {cam_name}...")
            cam.connect()
        if self.cameras:
            self.logger.info(f"  ✅ {len(self.cameras)} camera(s) connected")

        self._is_connected = True
        self.logger.info(f"✅ {self} connected.")

    def _attach_wrist_camera(self, side: str) -> None:
        """Wire the ``{side}`` wrist UVC camera into ``self.cameras`` under key
        ``{side}_wrist``. ``{side}_wrist_camera_index_or_path`` (explicit override)
        wins; otherwise ``{side}_wrist_camera_serial`` is resolved via
        ``/dev/v4l/by-id``."""
        wrist_path = getattr(self.config, f"{side}_wrist_camera_index_or_path")
        if not wrist_path:
            serial = getattr(self.config, f"{side}_wrist_camera_serial")
            if not serial:
                raise RuntimeError(
                    f"{side}_enable_wrist_camera=True but neither "
                    f"{side}_wrist_camera_serial nor {side}_wrist_camera_index_or_path is set."
                )
            wrist_path = resolve_wrist_camera_path(serial)
            self.logger.info(f"  [{side}] Resolved wrist serial {serial!r} -> {wrist_path}")

        cfg = OpenCVCameraConfig(
            index_or_path=wrist_path,
            width=getattr(self.config, f"{side}_wrist_camera_width"),
            height=getattr(self.config, f"{side}_wrist_camera_height"),
            fps=getattr(self.config, f"{side}_wrist_camera_fps"),
        )
        self.cameras.update(make_cameras_from_configs({f"{side}_wrist": cfg}))
        self.logger.info(
            f"  [{side}] ✅ Wrist camera wired at {wrist_path!r} "
            f"({getattr(self.config, f'{side}_wrist_camera_width')}x"
            f"{getattr(self.config, f'{side}_wrist_camera_height')} @ "
            f"{getattr(self.config, f'{side}_wrist_camera_fps')}fps)"
        )

    def _discover_gripper(self, side: str):
        """Locate exactly one TacCap-Gripper for ``side``, optionally filtered by
        ``{side}_firmware_sn`` (the stable identity — the MCU serial is the CH343
        chip serial which can change on chip swap)."""
        firmware_sn = getattr(self.config, f"{side}_firmware_sn")
        if firmware_sn is None:
            return find_one()
        all_eps = list(scan_grippers())
        candidates = [eps for eps in all_eps if eps.firmware_sn == firmware_sn]
        if not candidates:
            seen = [eps.firmware_sn for eps in all_eps]
            raise RuntimeError(
                f"No TacCap-Gripper with {side}_firmware_sn={firmware_sn!r} found. "
                f"Visible firmware SNs: {seen!r}."
            )
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple TacCap-Grippers match {side}_firmware_sn={firmware_sn!r}. "
                "Firmware SNs are supposed to be unique — check your firmware burning."
            )
        return candidates[0]

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        self.logger.info(f"Disconnecting {self}...")

        for cam_name, cam in self.cameras.items():
            try:
                if cam.is_connected:
                    cam.disconnect()
            except Exception as e:  # pragma: no cover — best-effort teardown
                self.logger.error(f"  Camera {cam_name} disconnect error: {e}")

        for side in _SIDES:
            tracker = self._tracker[side]
            if tracker is not None:
                try:
                    tracker.disconnect()
                except Exception as e:  # pragma: no cover
                    self.logger.error(f"  [{side}] Pico4 tracker disconnect error: {e}")
                self._tracker[side] = None

            gripper = self._gripper[side]
            if gripper is not None:
                try:
                    if getattr(gripper, "is_streaming", False):
                        gripper.stop_streaming()
                except Exception as e:  # pragma: no cover
                    self.logger.warn(f"  [{side}] stop_streaming raised: {e}")
                # LeaderGripper has no explicit close; transport released on GC.
                self._gripper[side] = None
            self._endpoints[side] = None

        self._is_connected = False
        self.logger.info(f"✅ {self} disconnected.")

    def calibrate(self) -> None:
        """Encoder zero is set out-of-band per unit via the SDK's calibrate.py."""
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------ data

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        obs: dict[str, Any] = {}

        for side in _SIDES:
            if getattr(self.config, f"{side}_enable_tracker") and self._tracker[side] is not None:
                for k, v in self._tracker[side].get_action().items():
                    obs[f"{side}_{k}"] = v

            if getattr(self.config, f"{side}_enable_gripper") and self._gripper[side] is not None:
                obs[f"{side}_gripper.pos"] = self._read_gripper_normalized(side)

            if getattr(self.config, f"{side}_enable_imu") and self._gripper[side] is not None:
                try:
                    imu = self._gripper[side].imu.read_once()
                    accel, gyro, mag = imu.accel_mps2, imu.gyro_radps, imu.mag_uT
                    for i, axis in enumerate(("x", "y", "z")):
                        obs[f"{side}_imu.accel.{axis}"] = float(accel[i])
                        obs[f"{side}_imu.gyro.{axis}"] = float(gyro[i])
                        obs[f"{side}_imu.mag.{axis}"] = float(mag[i])
                except Exception as e:
                    self.logger.warn(f"  [{side}] IMU read failed: {e}")

        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()

        return obs

    def send_action(self, action: dict[str, Any] | None = None) -> dict[str, Any]:
        """No-op: passive demonstration device. The operators drive the jaws
        mechanically — we never command either motor."""
        return action or {}

    def get_action(self) -> dict[str, Any]:
        """Return the prefixed pose + gripper dict the rig emits as a teleoperator
        (used when recording demonstrations)."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        action: dict[str, Any] = {}
        for side in _SIDES:
            if getattr(self.config, f"{side}_enable_tracker") and self._tracker[side] is not None:
                for k, v in self._tracker[side].get_action().items():
                    action[f"{side}_{k}"] = v
            if getattr(self.config, f"{side}_enable_gripper") and self._gripper[side] is not None:
                action[f"{side}_gripper.pos"] = self._read_gripper_normalized(side)
        return action

    # ------------------------------------------------------------------ helpers

    def _read_gripper_normalized(self, side: str) -> float:
        """Read cooked encoder position (rad ≥ 0 post-``set_zero``) and normalise
        to [0, 1] via ``clip(rad / {side}_gripper_open_rad, 0, 1)``."""
        try:
            sample = self._gripper[side].encoder.read_once()
            rad = float(sample.position_rad)
        except Exception as e:
            self.logger.warn(f"  [{side}] Encoder read failed: {e}")
            return 0.0
        opened = getattr(self.config, f"{side}_gripper_open_rad")
        if opened <= 0.0:  # guarded in config.__post_init__ but be defensive
            return 0.0
        return float(np.clip(rad / opened, 0.0, 1.0))

    def get_endpoints(self):
        """Per-side hardware discovery info populated on connect."""
        return dict(self._endpoints)
