#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Bimanual Elite CS66 robot integration for LeRobot.

Two Elite CS66 controllers, each driven exactly like the single-arm
``EliteCS66RT`` (RTSI state stream + EliteDriver reverse-socket servoj, an
optional background Cartesian servo loop, rotvec-continuity handling and
min-jerk reset). Per-arm state is kept in ``{"left": ..., "right": ...}`` dicts
so the single-arm logic is reused per side rather than duplicated line-by-line.

Action / observation keys are ``left_``/``right_`` prefixed:
    left_tcp.x/y/z + left_tcp.r1..r6   (+ optional left_joint_*),  left_gripper.pos
    right_tcp.x/y/z + right_tcp.r1..r6  (+ optional right_joint_*), right_gripper.pos
Grippers are per-arm serial (USB) devices addressed by board SN (no IP/MAC).
Cameras (head + per-arm wrist + optional tactiles) live at the bimanual level;
tactile images come from separate XenseTactileCamera devices (already namespaced
left_tactile_* / right_tactile_*), not the gripper.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import (
    BiEliteCS66RTConfig,
    BiEliteCS66RTControlMode,
)
from lerobot.robots.elite_cs66_rt import elite_cs66_rt as _elite_mod
from lerobot.robots.elite_cs66_rt.elite_cs66_rt import (
    _import_elite_sdk,
    _quaternion_to_rotvec,
    _rotvec_continuity_shift,
    _rotvec_to_quaternion,
    _slerp_quaternion_wxyz,
)
from lerobot.robots.grippers import SerialGripper
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import (
    get_logger,
    quaternion_to_euler,
    quaternion_to_rotation_6d,
    rotation_6d_to_quaternion,
)
from lerobot.utils.rotation import Rotation

# Bare (unprefixed) per-arm schema; shared with the single-arm driver.
TCP_POSITION_KEYS = ("tcp.x", "tcp.y", "tcp.z")
TCP_ROTATION_6D_KEYS = ("tcp.r1", "tcp.r2", "tcp.r3", "tcp.r4", "tcp.r5", "tcp.r6")
JOINT_POSITION_KEYS = tuple(f"joint_{i}.pos" for i in range(1, 7))
JOINT_VELOCITY_KEYS = tuple(f"joint_{i}.vel" for i in range(1, 7))
JOINT_EFFORT_KEYS = tuple(f"joint_{i}.effort" for i in range(1, 7))

_SIDES = ("left", "right")

# Single-arm RTSI/recipe resource directory, reused as the on-disk fallback when
# the SDK package doesn't ship the recipes. No need to duplicate recipe files.
_ELITE_RESOURCE_DIR = Path(_elite_mod.__file__).resolve().parent / "resource"


class BiEliteCS66RT(Robot):
    """Two Elite CS66 arms using elite_cs_sdk external control.

    Cartesian mode: action/observation features are ``{side}_tcp.x/y/z`` plus
    ``{side}_tcp.r1..r6``. Joint mode: ``{side}_joint_1.pos .. {side}_joint_6.pos``
    streamed with ``writeServoj(..., cartesian=False)``.
    """

    config_class = BiEliteCS66RTConfig
    name = "bi_elite_cs66_rt"

    # Same required RTSI output fields as the single-arm driver (SDK helpers
    # silently return zero vectors when these are absent).
    _REQUIRED_RTSI_OUTPUT_FIELDS = _elite_mod.EliteCS66RT._REQUIRED_RTSI_OUTPUT_FIELDS

    def __init__(self, config: BiEliteCS66RTConfig):
        super().__init__(config)
        self.config = config
        logger_suffix = config.id if config.id is not None else hex(id(self))
        self.logger = get_logger(f"BiEliteCS66RT.{logger_suffix}")

        self._cs = None  # shared elite_cs_sdk module (imported in connect())
        self._is_connected = False

        # Per-arm SDK handles + servo state.
        self._driver: dict[str, Any] = {s: None for s in _SIDES}
        self._dashboard: dict[str, Any] = {s: None for s in _SIDES}
        self._rtsi: dict[str, Any] = {s: None for s in _SIDES}

        self._gripper: dict[str, SerialGripper | None] = {
            "left": SerialGripper(config.left_gripper) if config.left_gripper is not None else None,
            "right": SerialGripper(config.right_gripper) if config.right_gripper is not None else None,
        }

        self._last_tcp_command: dict[str, np.ndarray | None] = {s: None for s in _SIDES}
        self._target_tcp_command: dict[str, np.ndarray | None] = {s: None for s in _SIDES}
        self._servo_thread: dict[str, threading.Thread | None] = {s: None for s in _SIDES}
        self._servo_stop_event: dict[str, threading.Event] = {s: threading.Event() for s in _SIDES}
        self._servo_lock: dict[str, threading.Lock] = {s: threading.Lock() for s in _SIDES}
        self._servo_error: dict[str, BaseException | None] = {s: None for s in _SIDES}
        self._last_action_time: dict[str, float] = {s: 0.0 for s in _SIDES}
        self._start_tcp_pose: dict[str, np.ndarray | None] = {s: None for s in _SIDES}
        self._reset_start_tcp_pose: dict[str, np.ndarray | None] = {s: None for s in _SIDES}
        self._reset_target_tcp_pose: dict[str, np.ndarray | None] = {s: None for s in _SIDES}
        self._reset_start_time: dict[str, float] = {s: 0.0 for s in _SIDES}
        self._reset_end_time: dict[str, float] = {s: 0.0 for s in _SIDES}
        self._reset_moving: dict[str, bool] = {s: False for s in _SIDES}
        self._external_command_received: dict[str, bool] = {s: False for s in _SIDES}

        # Prefixed key tuples (built once).
        self._tcp_pos_keys = {s: tuple(f"{s}_{k}" for k in TCP_POSITION_KEYS) for s in _SIDES}
        self._tcp_rot_keys = {s: tuple(f"{s}_{k}" for k in TCP_ROTATION_6D_KEYS) for s in _SIDES}
        self._joint_pos_keys = {s: tuple(f"{s}_{k}" for k in JOINT_POSITION_KEYS) for s in _SIDES}
        self._joint_vel_keys = {s: tuple(f"{s}_{k}" for k in JOINT_VELOCITY_KEYS) for s in _SIDES}
        self._joint_effort_keys = {s: tuple(f"{s}_{k}" for k in JOINT_EFFORT_KEYS) for s in _SIDES}
        self._gripper_key = {s: f"{s}_gripper.pos" for s in _SIDES}

        # Per-arm world←base rotation R = Rz(γ)·Rz(β)·Rx(α): tilt α about base-X
        # and zrot β about Z come from the teach pendant (fix the gravity vector
        # only); world_yaw γ aligns each arm's heading into ONE shared gravity-
        # aligned world frame (x=facing, y=left, z=up). Used at the get_observation
        # / send_action boundaries; all internal servo state stays in base frame.
        self._R_wb: dict[str, np.ndarray] = {
            side: self._resolve_world_rotation(config, side) for side in _SIDES
        }

        self.cameras = make_cameras_from_configs(config.cameras)

    @staticmethod
    def _resolve_world_rotation(config: BiEliteCS66RTConfig, side: str) -> np.ndarray:
        """world<-base rotation for one arm.

        Uses the explicit ``{side}_world_rotation`` matrix from config when set
        (re-orthonormalized defensively), else builds it from the tilt/zrot/yaw
        angles. The explicit path is needed when the mounting isn't a clean
        Rz·Rx (e.g. the left arm tilts about base-Y, not base-X).
        """
        override = getattr(config, f"{side}_world_rotation")
        if override is not None:
            R = np.asarray(override, dtype=np.float64)
            if R.shape != (3, 3):
                raise ValueError(f"{side}_world_rotation must be 3x3, got {R.shape}")
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
            return R
        return BiEliteCS66RT._mount_rotation(
            getattr(config, f"{side}_mount_tilt_deg"),
            getattr(config, f"{side}_mount_zrot_deg"),
            getattr(config, f"{side}_mount_world_yaw_deg"),
        )

    @staticmethod
    def _mount_rotation(tilt_deg: float, zrot_deg: float, world_yaw_deg: float) -> np.ndarray:
        """world←base rotation matrix for one arm: R = Rz(world_yaw)·Rz(zrot)·Rx(tilt).

        Built from axis-angle rotvecs (this repo's ``Rotation`` has no
        ``from_euler``): Rx(tilt) about base-X, Rz(zrot) about Z (teach-pendant
        mounting), then Rz(world_yaw) about world-Z to align headings across arms.
        """
        rx = Rotation.from_rotvec([np.deg2rad(tilt_deg), 0.0, 0.0]).as_matrix()
        rz = Rotation.from_rotvec([0.0, 0.0, np.deg2rad(zrot_deg)]).as_matrix()
        ryaw = Rotation.from_rotvec([0.0, 0.0, np.deg2rad(world_yaw_deg)]).as_matrix()
        return ryaw @ rz @ rx

    def _base_pose6_to_world(self, side: str, pose6: np.ndarray) -> np.ndarray:
        """Lift a base-frame ``[x,y,z,rx,ry,rz]`` (rotvec) pose into world frame."""
        R_wb = self._R_wb[side]
        pose6 = np.asarray(pose6, dtype=np.float64)
        pos = R_wb @ pose6[:3]
        rot = R_wb @ Rotation.from_rotvec(pose6[3:6]).as_matrix()
        rotvec = Rotation.from_matrix(rot).as_rotvec()
        return np.concatenate([pos, rotvec])

    def _world_pose6_to_base(self, side: str, pose6: np.ndarray) -> np.ndarray:
        """Map a world-frame ``[x,y,z,rx,ry,rz]`` (rotvec) pose back to base frame."""
        R_bw = self._R_wb[side].T
        pose6 = np.asarray(pose6, dtype=np.float64)
        pos = R_bw @ pose6[:3]
        rot = R_bw @ Rotation.from_rotvec(pose6[3:6]).as_matrix()
        rotvec = Rotation.from_matrix(rot).as_rotvec()
        return np.concatenate([pos, rotvec])

    # =========================================================================
    # Per-arm config accessors
    # =========================================================================

    def _arm_ip(self, side: str) -> str:
        return getattr(self.config, f"{side}_robot_ip")

    def _arm_local_ip(self, side: str) -> str:
        return getattr(self.config, f"{side}_local_ip")

    def _arm_start_pose(self, side: str) -> list[float]:
        return list(getattr(self.config, f"{side}_start_position_rad"))

    def _arm_home_pose(self, side: str) -> list[float]:
        return list(getattr(self.config, f"{side}_home_position_rad"))

    # =========================================================================
    # Feature descriptors
    # =========================================================================

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = {}

        for side in _SIDES:
            if self.config.observe_tcp:
                features.update(dict.fromkeys(self._tcp_pos_keys[side] + self._tcp_rot_keys[side], float))
            if self.config.observe_joints:
                features.update(dict.fromkeys(self._joint_pos_keys[side], float))
                features.update(dict.fromkeys(self._joint_vel_keys[side], float))
                features.update(dict.fromkeys(self._joint_effort_keys[side], float))

            if self._gripper[side] is not None:
                features[self._gripper_key[side]] = float

        # Tactile sensors are XenseTactileCamera entries in self.cameras, so they
        # are covered by the camera loop below (same as head / wrist cams).
        for cam_name in self.cameras:
            features[cam_name] = (
                self.config.cameras[cam_name].height,
                self.config.cameras[cam_name].width,
                3,
            )
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for side in _SIDES:
            if self.config.control_mode == BiEliteCS66RTControlMode.JOINT_SERVO:
                features.update(dict.fromkeys(self._joint_pos_keys[side], float))
            else:
                features.update(dict.fromkeys(self._tcp_pos_keys[side] + self._tcp_rot_keys[side], float))
            if self._gripper[side] is not None:
                features[self._gripper_key[side]] = float
        return features

    # =========================================================================
    # Connection state
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and all(self._driver[s] is not None for s in _SIDES)
            and all(self._rtsi[s] is not None for s in _SIDES)
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        # Elite CS66 is factory calibrated; no runtime calibration step.
        return True

    def calibrate(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

    def configure(self) -> None:
        pass

    # =========================================================================
    # Recipe / driver config helpers (shared across both arms)
    # =========================================================================

    def _resolve_sdk_resource(self, filename: str) -> str:
        assert self._cs is not None
        module_file = getattr(self._cs, "__file__", None)
        if not module_file:
            raise RuntimeError("Cannot resolve elite_cs_sdk package path.")
        path = Path(module_file).resolve().parent / filename
        if not path.exists():
            raise FileNotFoundError(f"Elite SDK resource not found: {path}")
        return str(path)

    @staticmethod
    def _read_recipe_fields(path: str) -> list[str]:
        lines = Path(path).read_text().splitlines()
        return [s for s in (line.strip() for line in lines) if s and not s.startswith("#")]

    def _validate_output_recipe(self, path: str) -> None:
        fields = set(self._read_recipe_fields(path))
        missing = [f for f in self._REQUIRED_RTSI_OUTPUT_FIELDS if f not in fields]
        if missing:
            raise ValueError(
                f"RTSI output recipe at {path} is missing required field(s): {missing}. "
                "These are read by SDK helpers (getActualTCPPose, getActualJointPositions, "
                "getActualJointVelocity, getActualJointTorques) which silently return zero "
                "vectors when the field is absent — leaving the robot believing it's at the "
                "world origin and risking a MoveJ into the floor on the next reset."
            )

    def _resolve_recipe(self, configured: str | Path | None, filename: str) -> str:
        if configured is not None:
            path = Path(configured).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Configured RTSI recipe not found: {path}")
            return str(path)

        from contextlib import suppress

        sdk_path = None
        with suppress(FileNotFoundError):
            sdk_path = self._resolve_sdk_resource(filename)
        if sdk_path:
            return sdk_path

        module_recipe = _ELITE_RESOURCE_DIR / filename
        if module_recipe.exists():
            return str(module_recipe)
        raise FileNotFoundError(
            f"Could not find {filename}. Set rtsi_output_recipe/rtsi_input_recipe in BiEliteCS66RTConfig."
        )

    def _make_driver_config(self, side: str):
        assert self._cs is not None
        cfg = self._cs.EliteDriverConfig()
        cfg.robot_ip = self._arm_ip(side)
        cfg.local_ip = self._arm_local_ip(side)
        cfg.servoj_time = self.config.servoj_time
        cfg.servoj_lookahead_time = self.config.servoj_lookahead_time
        cfg.servoj_gain = self.config.servoj_gain
        cfg.headless_mode = True
        # Two EliteDriver instances on one host can't share the local reverse /
        # trajectory / script-command TCP server ports — the 2nd arm would hit
        # "Address already in use". Offset one arm's ports; the SDK substitutes
        # these into the pushed external_control.script (REVERSE/TRAJECTORY/
        # SCRIPT_COMMAND port placeholders) so the controller connects back to
        # the matching ports.
        offset = getattr(self.config, f"{side}_driver_port_offset")
        cfg.reverse_port += offset
        cfg.script_sender_port += offset
        cfg.trajectory_port += offset
        cfg.script_command_port += offset
        if self.config.script_file_path is not None:
            cfg.script_file_path = str(Path(self.config.script_file_path).expanduser())
        else:
            cfg.script_file_path = self._resolve_sdk_resource("external_control.script")
        return cfg

    # =========================================================================
    # Connect / disconnect
    # =========================================================================

    def connect(self, calibrate: bool = False, go_to_start: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected, do not run connect() twice.")

        self._cs = _import_elite_sdk()
        # RT scheduling is best-effort (needs CAP_SYS_NICE / rtprio). Disable via
        # servo_fifo_scheduling=False — two FIFO-99 servo threads + the GIL can
        # priority-invert and stall one arm's feeding (see config docstring).
        if self.config.servo_fifo_scheduling:
            try:
                self._cs.setCurrentThreadFiFoScheduling(self._cs.getThreadFiFoMaxPriority())
            except Exception as exc:
                self.logger.warn(f"Failed to enable FIFO scheduling for Bi Elite CS66 control thread: {exc}")

        try:
            # --- Bring up both controllers (+ their grippers) in parallel ---
            self.logger.info(
                f"Connecting both arms in parallel: "
                f"left={self._arm_ip('left')}, right={self._arm_ip('right')}"
            )
            with ThreadPoolExecutor(max_workers=2) as ex:
                futs = {side: ex.submit(self._connect_arm, side) for side in _SIDES}
                for side in _SIDES:
                    futs[side].result()

            # --- Connect bimanual cameras in parallel ---
            if self.cameras:
                self.logger.info(
                    f"Connecting {len(self.cameras)} camera(s): {', '.join(self.cameras.keys())}..."
                )
                with ThreadPoolExecutor(max_workers=len(self.cameras)) as ex:
                    cam_futs = [ex.submit(cam.connect) for cam in self.cameras.values()]
                    for f in cam_futs:
                        f.result()
        except BaseException:
            self._cleanup_after_failed_connect()
            raise

        self._is_connected = True

        # --- Bring each arm to its start_position AND immediately hand off to
        #     its servo loop, per-arm, in parallel ---
        # The reverse socket the controller opened in _connect_arm has an
        # effectively-infinite recv timeout UNTIL the first command; the first
        # MoveJ command arms the move_j_timeout_ms recv budget. From then on any
        # feeding gap > that budget drops the connection. The dangerous gap is
        # the MoveJ -> servo-loop handoff. Doing MoveJ + seed + servo-loop start
        # inside ONE per-arm worker keeps the faster-finishing arm from sitting
        # idle while the slower arm finishes (the previous "wait for both, then
        # seed/start sequentially" structure starved whichever arm finished
        # first -> intermittent "socket timed out ... reverse_socket" RST).
        def _bring_arm_online(side: str) -> None:
            if go_to_start:
                self._move_j_blocking(
                    side, self._arm_start_pose(side), self.config.start_move_duration_s
                )
            if self.config.control_mode == BiEliteCS66RTControlMode.CARTESIAN_SERVO:
                current_tcp = np.asarray(self._rtsi[side].getActualTCPPose(), dtype=np.float64)
                self._last_tcp_command[side] = current_tcp.copy()
                self._target_tcp_command[side] = current_tcp.copy()
                self._start_tcp_pose[side] = current_tcp.copy()
                self._last_action_time[side] = time.monotonic()
                if self.config.use_background_servo_loop:
                    self._start_servo_loop(side)

        try:
            if go_to_start:
                self.logger.info(
                    "Bi Elite CS66 moving both arms to start_position over "
                    f"{self.config.start_move_duration_s:.1f}s..."
                )
            with ThreadPoolExecutor(max_workers=2) as ex:
                online_futs = {side: ex.submit(_bring_arm_online, side) for side in _SIDES}
                for side in _SIDES:
                    online_futs[side].result()
        except BaseException:
            self._is_connected = False
            self._cleanup_after_failed_connect()
            raise

        self.logger.info("BiEliteCS66RT connected and ready.")

    def _connect_arm(self, side: str) -> None:
        """Bring up one Elite controller: RTSI + dashboard + EliteDriver handshake (+ gripper)."""
        ip = self._arm_ip(side)

        output_recipe = self._resolve_recipe(self.config.rtsi_output_recipe, "output_recipe.txt")
        self._validate_output_recipe(output_recipe)
        input_recipe = self._resolve_recipe(self.config.rtsi_input_recipe, "input_recipe.txt")

        rtsi = self._cs.RtsiIOInterface(output_recipe, input_recipe, self.config.rtsi_frequency)
        if not rtsi.connect(ip):
            raise ConnectionError(f"Failed to connect Elite RTSI server ({side}) at {ip}:30004")
        self._rtsi[side] = rtsi

        dashboard = self._cs.DashboardClientInterface()
        if not dashboard.connect(ip):
            raise ConnectionError(f"Failed to connect Elite dashboard ({side}) at {ip}")
        self._dashboard[side] = dashboard

        if not dashboard.powerOn():
            raise RuntimeError(f"Elite CS66 ({side}) powerOn() failed.")
        if not dashboard.brakeRelease():
            raise RuntimeError(f"Elite CS66 ({side}) brakeRelease() failed.")

        driver_config = self._make_driver_config(side)
        driver_construct_time = time.monotonic()
        driver = self._cs.EliteDriver(driver_config)
        self._driver[side] = driver

        if self.config.external_control_settle_s > 0:
            time.sleep(self.config.external_control_settle_s)

        if not driver.isRobotConnected() and not driver.sendExternalControlScript():
            raise RuntimeError(f"Failed to send Elite external control script ({side}).")

        deadline = time.monotonic() + self.config.connect_timeout_s
        while not driver.isRobotConnected():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for Elite external control script connection ({side})."
                )
            time.sleep(0.01)

        remaining = self.config.external_control_settle_s - (time.monotonic() - driver_construct_time)
        if remaining > 0:
            time.sleep(remaining)

        gripper = self._gripper[side]
        if gripper is not None:
            self.logger.info(f"{side} arm: connecting gripper ({type(gripper).__name__})...")
            gripper.connect()

    def _cleanup_after_failed_connect(self) -> None:
        for side in _SIDES:
            # A servo loop may already be running if the other arm's bring-up
            # failed after this one handed off; stop it before tearing down.
            try:
                self._stop_servo_loop(side)
            except Exception:
                pass
            try:
                if self._driver[side] is not None:
                    self._driver[side].stopControl(1000)
            except Exception:
                pass
            try:
                if self._dashboard[side] is not None:
                    self._dashboard[side].disconnect()
            except Exception:
                pass
            try:
                if self._rtsi[side] is not None:
                    self._rtsi[side].disconnect()
            except Exception:
                pass
            gripper = self._gripper[side]
            if gripper is not None:
                try:
                    if getattr(gripper, "_is_connected", False):
                        gripper.disconnect()
                except Exception:
                    pass
            self._driver[side] = None
            self._dashboard[side] = None
            self._rtsi[side] = None

        for cam in self.cameras.values():
            try:
                if cam.is_connected:
                    cam.disconnect()
            except Exception:
                pass
        self._is_connected = False

    def disconnect(self) -> None:
        # Idempotent: quiet no-op if nothing was ever brought up.
        any_handle = any(
            self._driver[s] is not None or self._rtsi[s] is not None or self._dashboard[s] is not None
            for s in _SIDES
        )
        if not self._is_connected and not any_handle:
            self.logger.warn(f"{self} is not connected, skipping disconnect.")
            return

        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()

        for side in _SIDES:
            self._stop_servo_loop(side)

        # Smooth return to home for both arms in parallel before teardown.
        with ThreadPoolExecutor(max_workers=2) as ex:
            home_futs = {side: ex.submit(self._return_home_arm, side) for side in _SIDES}
            for side in _SIDES:
                home_futs[side].result()

        for side in _SIDES:
            driver = self._driver[side]
            if driver is not None:
                try:
                    driver.writeIdle(self.config.command_timeout_ms)
                    driver.stopControl(1000)
                finally:
                    self._driver[side] = None

            dashboard = self._dashboard[side]
            if dashboard is not None:
                try:
                    dashboard.disconnect()
                finally:
                    self._dashboard[side] = None

            rtsi = self._rtsi[side]
            if rtsi is not None:
                try:
                    rtsi.disconnect()
                finally:
                    self._rtsi[side] = None

            gripper = self._gripper[side]
            if gripper is not None:
                try:
                    if getattr(gripper, "_is_connected", False):
                        gripper.disconnect()
                except Exception as exc:
                    self.logger.warn(f"{side} gripper disconnect failed: {exc}")

        self._is_connected = False

    def _return_home_arm(self, side: str) -> None:
        """Blocking MoveJ to home for one arm, then re-kill its servo loop."""
        if self._driver[side] is None or self._rtsi[side] is None:
            return
        try:
            self.logger.info(
                f"{side} arm: returning to home_position over "
                f"{self.config.home_move_duration_s:.1f}s..."
            )
            self._move_j_blocking(side, self._arm_home_pose(side), self.config.home_move_duration_s)
        except Exception as exc:
            self.logger.warn(
                f"{side} arm: return-to-home failed; proceeding with shutdown anyway: {exc}"
            )
        # _move_j_blocking may have restarted the servo loop in its finally block.
        self._stop_servo_loop(side)

    # =========================================================================
    # Servo loop (per arm)
    # =========================================================================

    def _start_servo_loop(self, side: str) -> None:
        if self._servo_thread[side] is not None and self._servo_thread[side].is_alive():
            return
        self._servo_error[side] = None
        self._servo_stop_event[side].clear()
        thread = threading.Thread(
            target=self._servo_loop,
            args=(side,),
            name=f"BiEliteCS66RTServoLoop-{side}-{self.config.id or hex(id(self))}",
            daemon=True,
        )
        self._servo_thread[side] = thread
        thread.start()

    def _stop_servo_loop(self, side: str) -> None:
        self._servo_stop_event[side].set()
        thread = self._servo_thread[side]
        if thread is not None:
            thread.join(timeout=2.0)
            self._servo_thread[side] = None

    def _servo_loop(self, side: str) -> None:
        driver = self._driver[side]
        assert driver is not None
        assert self._cs is not None

        if self.config.servo_fifo_scheduling:
            try:
                self._cs.setCurrentThreadFiFoScheduling(self._cs.getThreadFiFoMaxPriority())
            except Exception as exc:
                self.logger.warn(f"Failed to enable FIFO scheduling for {side} servo loop: {exc}")

        lock = self._servo_lock[side]
        stop_event = self._servo_stop_event[side]
        next_tick = time.monotonic()
        consecutive_failures = 0
        max_consecutive_failures = self.config.servo_failure_tolerance_ticks
        while not stop_event.is_set():
            try:
                now = time.monotonic()
                with lock:
                    target, reset_active = self._get_servo_target_locked(side, now)
                    last_action_time = self._last_action_time[side]

                if target is None:
                    driver.writeIdle(self.config.command_timeout_ms)
                elif not reset_active and not self._external_command_received[side]:
                    driver.writeIdle(self.config.command_timeout_ms)
                else:
                    if not reset_active and now - last_action_time > self.config.command_stale_timeout_s:
                        driver.writeIdle(self.config.command_timeout_ms)
                    else:
                        ok = driver.writeServoj(target.tolist(), self.config.command_timeout_ms, True)
                        if not ok:
                            consecutive_failures += 1
                            if consecutive_failures > max_consecutive_failures:
                                raise RuntimeError(
                                    f"Elite writeServoj(cartesian=True) failed "
                                    f"{consecutive_failures} ticks in a row ({side})."
                                )
                        else:
                            consecutive_failures = 0
                            with lock:
                                self._last_tcp_command[side] = target

                next_tick += self.config.servoj_time
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()
            except BaseException as exc:
                self._servo_error[side] = exc
                stop_event.set()
                break

    def _get_servo_target_locked(self, side: str, now: float) -> tuple[np.ndarray | None, bool]:
        if self._reset_moving[side]:
            if self._reset_start_tcp_pose[side] is None or self._reset_target_tcp_pose[side] is None:
                self._reset_moving[side] = False
            elif now >= self._reset_end_time[side]:
                target = self._reset_target_tcp_pose[side].copy()
                self._target_tcp_command[side] = target.copy()
                self._last_action_time[side] = now
                self._last_tcp_command[side] = target.copy()
                self._reset_moving[side] = False
                return target, True
            else:
                duration = max(
                    self._reset_end_time[side] - self._reset_start_time[side], self.config.servoj_time
                )
                alpha = self._min_jerk((now - self._reset_start_time[side]) / duration)
                target = self._interpolate_tcp_pose(
                    self._reset_start_tcp_pose[side], self._reset_target_tcp_pose[side], alpha
                )
                self._last_action_time[side] = now
                return target, True

        target = (
            None if self._target_tcp_command[side] is None else self._target_tcp_command[side].copy()
        )
        return target, False

    def _is_reset_moving_locked(self, side: str, now: float) -> bool:
        if not self._reset_moving[side]:
            return False
        if now < self._reset_end_time[side]:
            return True
        if self._reset_target_tcp_pose[side] is not None:
            self._target_tcp_command[side] = self._reset_target_tcp_pose[side].copy()
            self._last_tcp_command[side] = self._reset_target_tcp_pose[side].copy()
            self._last_action_time[side] = now
        self._reset_moving[side] = False
        return False

    def _raise_servo_error_if_any(self, side: str) -> None:
        if self._servo_error[side] is not None:
            raise RuntimeError(
                f"Bi Elite CS66 background servo loop failed ({side}): {self._servo_error[side]}"
            ) from self._servo_error[side]

    # =========================================================================
    # MoveJ trajectory primitive (per arm)
    # =========================================================================

    def _move_j_blocking(self, side: str, target_joints: list[float], duration_s: float) -> None:
        driver = self._driver[side]
        assert driver is not None
        if len(target_joints) != 6:
            raise ValueError(f"_move_j_blocking expects 6 joint angles, got {len(target_joints)}")

        servo_was_running = (
            self._servo_thread[side] is not None and self._servo_thread[side].is_alive()
        )
        if servo_was_running:
            self._stop_servo_loop(side)

        done_event = threading.Event()
        result_box: dict[str, Any] = {}

        def _on_done(result):
            result_box["result"] = result
            done_event.set()

        driver.setTrajectoryResultCallback(_on_done)
        timeout_ms = self.config.move_j_timeout_ms

        try:
            if not driver.writeTrajectoryControlAction(
                self._cs.TrajectoryControlAction.START, 1, timeout_ms
            ):
                raise RuntimeError("writeTrajectoryControlAction(START) failed")
            if not driver.writeTrajectoryPoint(list(target_joints), float(duration_s), 0.0, False):
                raise RuntimeError("writeTrajectoryPoint failed")

            deadline = time.monotonic() + duration_s + 5.0
            while not done_event.is_set():
                if not driver.writeTrajectoryControlAction(
                    self._cs.TrajectoryControlAction.NOOP, 0, timeout_ms
                ):
                    raise RuntimeError("writeTrajectoryControlAction(NOOP) failed")
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"MoveJ ({side}) to {target_joints} did not complete within "
                        f"{duration_s + 5.0:.1f} s"
                    )
                time.sleep(0.02)

            result = result_box.get("result")
            if result is not None and result != self._cs.TrajectoryMotionResult.SUCCESS:
                raise RuntimeError(f"MoveJ ({side}) finished with non-success result: {result}")
        finally:
            try:
                driver.writeIdle(timeout_ms)
            except Exception:
                pass
            if servo_was_running:
                try:
                    current_tcp = np.asarray(self._rtsi[side].getActualTCPPose(), dtype=np.float64)
                    with self._servo_lock[side]:
                        self._last_tcp_command[side] = current_tcp.copy()
                        self._target_tcp_command[side] = current_tcp.copy()
                        self._last_action_time[side] = time.monotonic()
                        self._external_command_received[side] = False
                except Exception:
                    pass
                self._start_servo_loop(side)

    # =========================================================================
    # Interpolation helpers (translated from the single-arm driver)
    # =========================================================================

    @staticmethod
    def _min_jerk(alpha: float) -> float:
        alpha = min(max(alpha, 0.0), 1.0)
        return alpha * alpha * alpha * (10.0 + alpha * (-15.0 + 6.0 * alpha))

    @staticmethod
    def _interpolate_tcp_pose(start: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
        pose = np.asarray(start, dtype=np.float64).copy()
        target = np.asarray(target, dtype=np.float64)
        pose[:3] = start[:3] + alpha * (target[:3] - start[:3])

        start_quat = _rotvec_to_quaternion(start[3:6])
        target_quat = _rotvec_to_quaternion(target[3:6])
        interp_principal = _quaternion_to_rotvec(
            _slerp_quaternion_wxyz(start_quat, target_quat, alpha)
        )
        pose[3:6] = _rotvec_continuity_shift(interp_principal, start[3:6])
        return pose

    # =========================================================================
    # Observation
    # =========================================================================

    def _tcp_rotvec_to_feature_values(self, side: str, tcp_pose: np.ndarray) -> dict[str, float]:
        pos_keys = self._tcp_pos_keys[side]
        rot_keys = self._tcp_rot_keys[side]
        values = {
            pos_keys[0]: float(tcp_pose[0]),
            pos_keys[1]: float(tcp_pose[1]),
            pos_keys[2]: float(tcp_pose[2]),
        }
        quat = _rotvec_to_quaternion(tcp_pose[3:6])
        r6d = quaternion_to_rotation_6d(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        values.update({key: float(value) for key, value in zip(rot_keys, r6d, strict=True)})
        return values

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs: dict[str, Any] = {}

        for side in _SIDES:
            rtsi = self._rtsi[side]
            assert rtsi is not None

            if self.config.observe_tcp:
                # RTSI reports the TCP pose in the (tilted) base frame; lift it
                # into the gravity-aligned world frame before publishing.
                tcp_pose = np.asarray(rtsi.getActualTCPPose(), dtype=np.float64)
                tcp_world = self._base_pose6_to_world(side, tcp_pose)
                obs.update(self._tcp_rotvec_to_feature_values(side, tcp_world))
            if self.config.observe_joints:
                joints = rtsi.getActualJointPositions()
                obs.update(
                    {k: float(v) for k, v in zip(self._joint_pos_keys[side], joints, strict=True)}
                )
                joint_vel = rtsi.getActualJointVelocity()
                obs.update(
                    {k: float(v) for k, v in zip(self._joint_vel_keys[side], joint_vel, strict=True)}
                )
                joint_effort = rtsi.getActualJointTorques()
                obs.update(
                    {k: float(v) for k, v in zip(self._joint_effort_keys[side], joint_effort, strict=True)}
                )

            gripper = self._gripper[side]
            if gripper is not None:
                obs[self._gripper_key[side]] = gripper.get_gripper_position()

        # Tactile images come from XenseTactileCamera entries in self.cameras.
        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()
        return obs

    # =========================================================================
    # Action
    # =========================================================================

    def _cartesian_action_to_tcp_pose(self, side: str, action: dict[str, Any]) -> np.ndarray:
        with self._servo_lock[side]:
            last_tcp = (
                None if self._last_tcp_command[side] is None else self._last_tcp_command[side].copy()
            )

        if last_tcp is not None:
            last_base = last_tcp
        else:
            assert self._rtsi[side] is not None
            last_base = np.asarray(self._rtsi[side].getActualTCPPose(), dtype=np.float64)

        # The incoming action is in the world frame; merge it against the last
        # commanded pose expressed in world so partial (position-only) actions
        # keep the same per-axis semantics as the single-arm driver, then map the
        # merged target back into base for the servo loop / SDK IK.
        target_world = self._base_pose6_to_world(side, last_base)

        pos_keys = self._tcp_pos_keys[side]
        rot_keys = self._tcp_rot_keys[side]
        for i, key in enumerate(pos_keys):
            if key in action:
                target_world[i] = float(action[key])

        if any(key in action for key in rot_keys):
            if not all(key in action for key in rot_keys):
                raise ValueError(
                    f"Incomplete rotation-6D action ({side}). Expected {rot_keys[0]} through "
                    f"{rot_keys[-1]} together."
                )
            r6d = np.array([float(action[key]) for key in rot_keys], dtype=np.float64)
            target_world[3:6] = _quaternion_to_rotvec(rotation_6d_to_quaternion(r6d))

        target = self._world_pose6_to_base(side, target_world)
        # Re-express the base-frame target rotvec on the same ±2π·axis branch as
        # our own last-commanded base rotvec (continuous by construction, NOT
        # RTSI's reported pose) so the SDK IK seed stays smooth. See single-arm
        # driver for why we anchor on the commanded rotvec.
        target[3:6] = _rotvec_continuity_shift(target[3:6], last_base[3:6])

        return target

    def _trace_send_action(self, side: str, action: dict[str, Any], target_tcp: np.ndarray) -> None:
        if not self.config.trace_servoj:
            return
        try:
            current = np.asarray(self._rtsi[side].getActualTCPPose(), dtype=np.float64)
        except Exception:
            return
        last = (
            self._last_tcp_command[side].copy()
            if self._last_tcp_command[side] is not None
            else current.copy()
        )

        d_lin_vs_last = float(np.linalg.norm(target_tcp[:3] - last[:3]))
        tgt_rot = Rotation.from_rotvec(target_tcp[3:6])
        last_rot = Rotation.from_rotvec(last[3:6])
        d_ang_vs_last = float(np.linalg.norm((tgt_rot * last_rot.inv()).as_rotvec()))

        msg = (
            f"[{side}] send_action tgt=({target_tcp[0]:+.4f},{target_tcp[1]:+.4f},{target_tcp[2]:+.4f},"
            f"rv=[{target_tcp[3]:+.3f},{target_tcp[4]:+.3f},{target_tcp[5]:+.3f}]) "
            f"d_lin(vs_last={d_lin_vs_last*1000:.2f}mm) "
            f"d_ang(vs_last={np.rad2deg(d_ang_vs_last):.2f}deg)"
        )
        self.logger.debug(msg)

        if (
            self.config.trace_translation_threshold > 0
            and d_lin_vs_last > self.config.trace_translation_threshold
        ) or (
            self.config.trace_rotation_threshold > 0
            and d_ang_vs_last > self.config.trace_rotation_threshold
        ):
            self.logger.warn(f"LARGE STEP {msg}")

    def _trace_send_action_joint(
        self, side: str, target_joints: list[float], current_joints: list[float]
    ) -> None:
        if not self.config.trace_servoj:
            return
        deltas = [t - c for t, c in zip(target_joints, current_joints, strict=True)]
        max_abs_delta = max((abs(d) for d in deltas), default=0.0)
        msg = (
            f"[{side}] joint send_action "
            f"tgt=[{','.join(f'{j:+.3f}' for j in target_joints)}] "
            f"delta=[{','.join(f'{d:+.3f}' for d in deltas)}] max_abs={max_abs_delta:.3f}rad"
        )
        self.logger.debug(msg)
        if self.config.trace_joint_threshold > 0 and max_abs_delta > self.config.trace_joint_threshold:
            self.logger.warn(f"LARGE JOINT STEP {msg}")

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        sent: dict[str, Any] = {}
        for side in _SIDES:
            self._send_arm_action(side, action, sent)
        return sent or action

    def _send_arm_action(self, side: str, action: dict[str, Any], sent: dict[str, Any]) -> None:
        driver = self._driver[side]
        assert driver is not None
        self._raise_servo_error_if_any(side)

        gripper = self._gripper[side]
        gripper_key = self._gripper_key[side]

        if self.config.control_mode == BiEliteCS66RTControlMode.CARTESIAN_SERVO:
            if self.config.use_background_servo_loop:
                with self._servo_lock[side]:
                    reset_moving = self._is_reset_moving_locked(side, time.monotonic())
                if reset_moving:
                    if gripper is not None and gripper_key in action:
                        gripper.set_gripper_position(float(action[gripper_key]))
                        sent[gripper_key] = float(action[gripper_key])
                    return

            target_tcp = self._cartesian_action_to_tcp_pose(side, action)
            self._trace_send_action(side, action, target_tcp)
            if self.config.use_background_servo_loop:
                with self._servo_lock[side]:
                    self._target_tcp_command[side] = target_tcp.copy()
                    self._last_action_time[side] = time.monotonic()
                    self._external_command_received[side] = True
            else:
                ok = driver.writeServoj(target_tcp.tolist(), self.config.command_timeout_ms, True)
                if not ok:
                    raise RuntimeError(f"Elite writeServoj(cartesian=True) failed ({side}).")
                self._last_tcp_command[side] = target_tcp
                self._external_command_received[side] = True
            # Report the sent pose back in the world frame so callers (display /
            # replay) stay consistent with get_observation. The dataset action is
            # recorded from the teleop/policy action, not this return value.
            sent.update(
                self._tcp_rotvec_to_feature_values(
                    side, self._base_pose6_to_world(side, target_tcp)
                )
            )
        else:
            joint_keys = self._joint_pos_keys[side]
            if not all(key in action for key in joint_keys):
                missing = [key for key in joint_keys if key not in action]
                raise ValueError(f"Missing joint servo action keys ({side}): {missing}")
            target_joints = [float(action[key]) for key in joint_keys]
            assert self._rtsi[side] is not None
            current_joints = list(self._rtsi[side].getActualJointPositions())
            self._trace_send_action_joint(side, target_joints, current_joints)
            ok = driver.writeServoj(target_joints, self.config.command_timeout_ms, False)
            if not ok:
                raise RuntimeError(f"Elite writeServoj(cartesian=False) failed ({side}).")
            sent.update(dict(zip(joint_keys, target_joints, strict=True)))

        if gripper is not None and gripper_key in action:
            gripper.set_gripper_position(float(action[gripper_key]))
            sent[gripper_key] = float(action[gripper_key])

    # =========================================================================
    # Reset
    # =========================================================================

    def reset_to_initial_position(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = {side: ex.submit(self._reset_arm, side) for side in _SIDES}
            for side in _SIDES:
                futs[side].result()

    def _reset_arm(self, side: str) -> None:
        if self.config.control_mode != BiEliteCS66RTControlMode.CARTESIAN_SERVO:
            self._move_j_blocking(side, self._arm_start_pose(side), self.config.reset_duration_s)
            return

        if self._start_tcp_pose[side] is None:
            return

        if self.config.use_background_servo_loop:
            assert self._rtsi[side] is not None
            now = time.monotonic()
            with self._servo_lock[side]:
                if self._is_reset_moving_locked(side, now):
                    return
                if self._last_tcp_command[side] is not None:
                    self._reset_start_tcp_pose[side] = self._last_tcp_command[side].copy()
                else:
                    self._reset_start_tcp_pose[side] = np.asarray(
                        self._rtsi[side].getActualTCPPose(), dtype=np.float64
                    )
                target_pose = self._start_tcp_pose[side].copy()
                target_principal = _quaternion_to_rotvec(_rotvec_to_quaternion(target_pose[3:6]))
                target_pose[3:6] = _rotvec_continuity_shift(
                    target_principal, self._reset_start_tcp_pose[side][3:6]
                )
                self._reset_target_tcp_pose[side] = target_pose
                self._reset_start_time[side] = now
                self._reset_end_time[side] = now + self.config.reset_duration_s
                self._reset_moving[side] = True
                self._last_action_time[side] = now
            return

        driver = self._driver[side]
        assert driver is not None
        assert self._rtsi[side] is not None
        if self._last_tcp_command[side] is not None:
            start_pose = self._last_tcp_command[side].copy()
        else:
            start_pose = np.asarray(self._rtsi[side].getActualTCPPose(), dtype=np.float64)
        target_pose = self._start_tcp_pose[side].copy()
        target_principal = _quaternion_to_rotvec(_rotvec_to_quaternion(target_pose[3:6]))
        target_pose[3:6] = _rotvec_continuity_shift(target_principal, start_pose[3:6])
        start_time = time.monotonic()
        duration = max(self.config.reset_duration_s, self.config.servoj_time)

        while True:
            now = time.monotonic()
            alpha = (now - start_time) / duration
            if alpha >= 1.0:
                pose = target_pose
            else:
                pose = self._interpolate_tcp_pose(start_pose, target_pose, self._min_jerk(alpha))
            ok = driver.writeServoj(pose.tolist(), self.config.command_timeout_ms, True)
            if not ok:
                raise RuntimeError(f"Elite writeServoj(cartesian=True) failed during reset ({side}).")
            self._last_tcp_command[side] = pose
            if alpha >= 1.0:
                break
            time.sleep(self.config.servoj_time)

    # =========================================================================
    # RT status + pose getters
    # =========================================================================

    @property
    def rt_running(self) -> bool:
        return all(
            self._servo_thread[s] is not None and self._servo_thread[s].is_alive() for s in _SIDES
        )

    @property
    def rt_moving(self) -> bool:
        moving = False
        for side in _SIDES:
            with self._servo_lock[side]:
                moving = moving or self._is_reset_moving_locked(side, time.monotonic())
        return moving

    def _arm_tcp_pose_quat(self, side: str) -> np.ndarray:
        rtsi = self._rtsi[side]
        assert rtsi is not None
        # Return the pose in the gravity-aligned world frame, consistent with
        # get_observation (RTSI reports it in the tilted base frame).
        tcp_pose = self._base_pose6_to_world(
            side, np.asarray(rtsi.getActualTCPPose(), dtype=np.float64)
        )
        quat = _rotvec_to_quaternion(tcp_pose[3:6])
        gripper = self._gripper[side]
        gripper_pos = gripper.get_gripper_position() if gripper is not None else 0.0
        return np.array(
            [tcp_pose[0], tcp_pose[1], tcp_pose[2], quat[0], quat[1], quat[2], quat[3], gripper_pos],
            dtype=np.float64,
        )

    def _arm_tcp_pose_euler(self, side: str, tcp_pose: np.ndarray | None = None) -> np.ndarray:
        if tcp_pose is None:
            rtsi = self._rtsi[side]
            assert rtsi is not None
            tcp_pose = np.asarray(rtsi.getActualTCPPose(), dtype=np.float64)
        # ``tcp_pose`` (whether from RTSI or a passed-in _last_tcp_command) is in
        # the tilted base frame; lift it into world for a consistent report.
        tcp_pose = self._base_pose6_to_world(side, np.asarray(tcp_pose, dtype=np.float64))
        quat = _rotvec_to_quaternion(tcp_pose[3:6])
        euler = quaternion_to_euler(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        gripper = self._gripper[side]
        gripper_pos = gripper.get_gripper_position() if gripper is not None else 0.0
        return np.array(
            [tcp_pose[0], tcp_pose[1], tcp_pose[2], euler[0], euler[1], euler[2], gripper_pos],
            dtype=np.float64,
        )

    def get_current_tcp_pose_quat(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._arm_tcp_pose_quat("left"), self._arm_tcp_pose_quat("right")

    def get_current_tcp_pose_euler(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._arm_tcp_pose_euler("left"), self._arm_tcp_pose_euler("right")

    def get_commanded_tcp_pose_euler(self) -> tuple[np.ndarray, np.ndarray]:
        """Last commanded TCP pose (Euler + gripper) per arm.

        Prefer this over ``get_current_tcp_pose_euler`` when re-seeding a teleop
        accumulator: ``_last_tcp_command`` is continuous with our servoj stream,
        whereas RTSI's rotvec can be in a different ±2π branch near singularities.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        poses = []
        for side in _SIDES:
            last = self._last_tcp_command[side]
            if last is None:
                poses.append(self._arm_tcp_pose_euler(side))
            else:
                poses.append(self._arm_tcp_pose_euler(side, np.asarray(last, dtype=np.float64)))
        return poses[0], poses[1]
