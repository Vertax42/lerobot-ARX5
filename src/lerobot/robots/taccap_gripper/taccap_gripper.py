#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
TacCap-Gripper handheld data-collection device for LeRobot.

This is a passive recording device — ``send_action()`` is a no-op. The
gripper motor is **not** enabled; we read the encoder only. Pose comes
from a Pico4 Ultra independent tracker physically mounted on top
(``Pico4TrackerReader``). Tactile and wrist cameras are configured via
the standard ``cameras`` framework.

Observation features:
    tcp.x, tcp.y, tcp.z              -- Pico4 tracker → EE position (m)
    tcp.r1..tcp.r6                   -- 6D rotation representation
    gripper.pos                      -- normalised jaw [0=closed, 1=open]
    imu.accel.{x,y,z} (optional)     -- m/s²
    imu.gyro.{x,y,z}  (optional)     -- rad/s
    imu.mag.{x,y,z}   (optional)     -- µT
    <camera_name>                    -- one (H, W, 3) entry per config camera
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
from .config_taccap_gripper import TaccapGripperConfig

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


class TaccapGripper(Robot):
    """TacCap-Gripper handheld data-collection device.

    The device is operated manually — there is no action signal applied
    to the gripper (motor stays disabled). It records pose + jaw state
    + tactile + wrist for downstream policy learning.

    6D rotation convention (matches ``xense_flare`` and ``vive_tracker``):
        r1..r3 = first column of the rotation matrix
        r4..r6 = second column of the rotation matrix
    """

    config_class = TaccapGripperConfig
    name = "taccap_gripper"

    def __init__(self, config: TaccapGripperConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(f"TaccapGripper-{config.id or 'default'}")

        if config.enable_gripper and not TACCAP_SDK_AVAILABLE:
            raise ImportError(
                "xense.taccap SDK not available. Install the taccap-gripper "
                "PyPI package (source at /home/ubuntu/TacCap-Gripper)."
            )
        if config.enable_tracker and not PICO4_TRACKER_AVAILABLE:
            raise ImportError(
                "Pico4TrackerReader not available. Ensure "
                "src/lerobot/teleoperators/pico4/tracker.py is importable."
            )

        # Hardware handles, populated on connect.
        self._gripper: Any = None  # xense.taccap.LeaderGripper
        self._endpoints: Any = None  # xense.taccap.GripperEndpoints
        self._tracker: Pico4TrackerReader | None = None

        self.cameras = make_cameras_from_configs(config.cameras)

        self._is_connected = False

    # ------------------------------------------------------------------ schema

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {}

        if self.config.enable_tracker:
            features["tcp.x"] = float
            features["tcp.y"] = float
            features["tcp.z"] = float
            features["tcp.r1"] = float
            features["tcp.r2"] = float
            features["tcp.r3"] = float
            features["tcp.r4"] = float
            features["tcp.r5"] = float
            features["tcp.r6"] = float

        if self.config.enable_gripper:
            features["gripper.pos"] = float

        if self.config.enable_imu:
            for axis in ("x", "y", "z"):
                features[f"imu.accel.{axis}"] = float
                features[f"imu.gyro.{axis}"] = float
                features[f"imu.mag.{axis}"] = float

        for cam_name, cam_cfg in self.config.cameras.items():
            features[cam_name] = (cam_cfg.height, cam_cfg.width, 3)

        if self.config.enable_wrist_camera:
            features["wrist_cam"] = (
                self.config.wrist_camera_height,
                self.config.wrist_camera_width,
                3,
            )

        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """The 'demonstration' action this device emits when used as a teleop.

        Matches ``xense_flare``: pose (tcp.x/y/z, tcp.r1-r6) + gripper.pos.
        No camera data — that lives in observation only.
        """
        features: dict[str, type] = {}
        if self.config.enable_tracker:
            features["tcp.x"] = float
            features["tcp.y"] = float
            features["tcp.z"] = float
            features["tcp.r1"] = float
            features["tcp.r2"] = float
            features["tcp.r3"] = float
            features["tcp.r4"] = float
            features["tcp.r5"] = float
            features["tcp.r6"] = float
        if self.config.enable_gripper:
            features["gripper.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """The TacCap-Gripper uses factory calibration; we only need the
        gripper open/closed endpoints, which live in the config."""
        return self.is_connected

    # ------------------------------------------------------------------ lifecycle

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.logger.info("Connecting TacCap-Gripper...")

        # 1. Gripper SDK (MCU transport only — we do NOT call .tactile_*.start()
        #    or .wrist_camera.start(); those streams come through the cameras
        #    framework instead).
        if self.config.enable_gripper:
            self._endpoints = self._discover_gripper()
            self.logger.info(
                f"  TacCap-Gripper: side={self._endpoints.side} "
                f"mcu={self._endpoints.mcu_serial} "
                f"fw_sn={getattr(self._endpoints, 'firmware_sn', '?')!r}"
            )
            self.logger.info(
                f"  Tactile serials: left={self._endpoints.tactile_left_serial!r} "
                f"right={self._endpoints.tactile_right_serial!r}"
            )
            self.logger.info(f"  Wrist video path: {self._endpoints.wrist_video!r}")
            self._gripper = LeaderGripper.open()
            self.logger.info("  ✅ LeaderGripper attached (read-only — motor stays disabled)")

        # Auto-wire the wrist camera using the V4L2 path the SDK reports.
        # We discover endpoints independently if the gripper itself is
        # disabled — wrist camera is still part of the same hardware unit.
        if self.config.enable_wrist_camera:
            self._attach_wrist_camera()

        # 2. Pico4 tracker.
        if self.config.enable_tracker:
            self._tracker = Pico4TrackerReader(
                tracker_sn=self.config.tracker_sn,
                tracker_to_ee_pos=self.config.tracker_to_ee_pos,
                tracker_to_ee_quat=self.config.tracker_to_ee_quat,
                device_wait_timeout=self.config.tracker_wait_timeout,
                logger_name=self.config.id or "robot",
            )
            self._tracker.connect()
            self.logger.info("  ✅ Pico4 tracker connected")

        # 3. Cameras (tactile + wrist).
        for cam_name, cam in self.cameras.items():
            self.logger.info(f"  Connecting camera {cam_name}...")
            cam.connect()
        if self.cameras:
            self.logger.info(f"  ✅ {len(self.cameras)} camera(s) connected")

        self._is_connected = True
        self.logger.info(f"✅ {self} connected.")

    def _attach_wrist_camera(self) -> None:
        """Build an ``OpenCVCameraConfig`` pointed at ``endpoints.wrist_video``
        and add it to ``self.cameras`` under key ``wrist_cam``.

        Discovers endpoints on demand if ``enable_gripper`` was False, so
        the wrist camera can be used standalone."""
        if self._endpoints is None:
            self._endpoints = self._discover_gripper()
            self.logger.info(
                f"  TacCap-Gripper endpoints (wrist-only): "
                f"side={self._endpoints.side} mcu={self._endpoints.mcu_serial}"
            )

        wrist_path = self._endpoints.wrist_video
        if not wrist_path:
            raise RuntimeError(
                "GripperEndpoints.wrist_video is empty — the SDK could not "
                "resolve a V4L2 path for the wrist camera on this unit."
            )

        cfg = OpenCVCameraConfig(
            index_or_path=wrist_path,
            width=self.config.wrist_camera_width,
            height=self.config.wrist_camera_height,
            fps=self.config.wrist_camera_fps,
        )
        wrist_dict = make_cameras_from_configs({"wrist_cam": cfg})
        self.cameras.update(wrist_dict)
        self.logger.info(
            f"  ✅ Wrist camera auto-wired at {wrist_path!r} "
            f"({self.config.wrist_camera_width}x{self.config.wrist_camera_height} "
            f"@ {self.config.wrist_camera_fps}fps)"
        )

    def _discover_gripper(self):
        """Locate exactly one TacCap-Gripper, optionally filtered by MCU serial."""
        if self.config.mcu_serial is None:
            return find_one()
        candidates = [eps for eps in scan_grippers() if eps.mcu_serial == self.config.mcu_serial]
        if not candidates:
            seen = [eps.mcu_serial for eps in scan_grippers()]
            raise RuntimeError(
                f"No TacCap-Gripper with mcu_serial={self.config.mcu_serial!r} found. "
                f"Visible MCUs: {seen!r}."
            )
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple TacCap-Grippers match mcu_serial={self.config.mcu_serial!r}. "
                "Serials are supposed to be unique — check your hardware."
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

        if self._tracker is not None:
            try:
                self._tracker.disconnect()
            except Exception as e:  # pragma: no cover
                self.logger.error(f"  Pico4 tracker disconnect error: {e}")
            self._tracker = None

        if self._gripper is not None:
            try:
                if getattr(self._gripper, "is_streaming", False):
                    self._gripper.stop_streaming()
            except Exception as e:  # pragma: no cover
                self.logger.warn(f"  stop_streaming raised: {e}")
            # LeaderGripper has no explicit close; transport is released on GC.
            self._gripper = None

        self._endpoints = None
        self._is_connected = False
        self.logger.info(f"✅ {self} disconnected.")

    def calibrate(self) -> None:
        """Calibration here means measuring the jaw open/closed encoder
        endpoints, which is an out-of-band operator workflow. See
        ``calibrate_gripper_range.py`` in this package."""
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------ data

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        obs: dict[str, Any] = {}

        if self.config.enable_tracker and self._tracker is not None:
            obs.update(self._tracker.get_action())

        if self.config.enable_gripper and self._gripper is not None:
            obs["gripper.pos"] = self._read_gripper_normalized()

        if self.config.enable_imu and self._gripper is not None:
            try:
                imu = self._gripper.imu.read_once()
                accel = imu.accel_mps2
                gyro = imu.gyro_radps
                mag = imu.mag_uT
                obs["imu.accel.x"] = float(accel[0])
                obs["imu.accel.y"] = float(accel[1])
                obs["imu.accel.z"] = float(accel[2])
                obs["imu.gyro.x"] = float(gyro[0])
                obs["imu.gyro.y"] = float(gyro[1])
                obs["imu.gyro.z"] = float(gyro[2])
                obs["imu.mag.x"] = float(mag[0])
                obs["imu.mag.y"] = float(mag[1])
                obs["imu.mag.z"] = float(mag[2])
            except Exception as e:
                self.logger.warn(f"IMU read failed: {e}")

        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()

        return obs

    def send_action(self, action: dict[str, Any] | None = None) -> dict[str, Any]:
        """No-op: this is a passive demonstration device. We never command
        the jaw motor — the operator drives the gripper mechanically."""
        return action or {}

    def get_action(self) -> dict[str, Any]:
        """Return the same pose + gripper dict that the device emits as a
        teleoperator (used when recording demonstrations)."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        action: dict[str, Any] = {}
        if self.config.enable_tracker and self._tracker is not None:
            action.update(self._tracker.get_action())
        if self.config.enable_gripper and self._gripper is not None:
            action["gripper.pos"] = self._read_gripper_normalized()
        return action

    # ------------------------------------------------------------------ helpers

    def _read_gripper_normalized(self) -> float:
        """Read encoder position (radians) and normalise to [0, 1]."""
        try:
            sample = self._gripper.encoder.read_once()
            rad = float(sample.position_rad)
        except Exception as e:
            self.logger.warn(f"Encoder read failed: {e}")
            return 0.0
        closed = self.config.gripper_closed_rad
        opened = self.config.gripper_open_rad
        denom = opened - closed
        if denom == 0.0:  # guarded in config.__post_init__ but be defensive
            return 0.0
        normalized = (rad - closed) / denom
        return float(np.clip(normalized, 0.0, 1.0))

    def get_endpoints(self):
        """Hardware discovery info populated on connect (None otherwise)."""
        return self._endpoints
