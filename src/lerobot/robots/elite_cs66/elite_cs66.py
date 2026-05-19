#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Elite CS66 robot integration for LeRobot."""

import importlib
import threading
import time
from contextlib import suppress
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.elite_cs66.config_elite_cs66 import (
    EliteCS66Config,
    EliteCS66ControlMode,
)
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import (
    euler_to_quaternion,
    get_logger,
    quaternion_to_rotation_6d,
    rotation_6d_to_quaternion,
)
from lerobot.utils.rotation import Rotation

TCP_POSITION_KEYS = ("tcp.x", "tcp.y", "tcp.z")
TCP_ROTATION_6D_KEYS = ("tcp.r1", "tcp.r2", "tcp.r3", "tcp.r4", "tcp.r5", "tcp.r6")
JOINT_POSITION_KEYS = tuple(f"joint_{i}.pos" for i in range(1, 7))
JOINT_VELOCITY_KEYS = tuple(f"joint_{i}.vel" for i in range(1, 7))
JOINT_EFFORT_KEYS = tuple(f"joint_{i}.effort" for i in range(1, 7))


def _import_elite_sdk():
    try:
        return importlib.import_module("elite_cs_sdk")
    except ImportError as exc:
        raise ImportError(
            "elite_cs_sdk is not installed in this environment. Install/build the Elite CS SDK "
            "inside the active LeRobot environment before connecting an Elite CS66 robot."
        ) from exc


def _rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = Rotation.from_rotvec(rotvec).as_quat()
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def _quaternion_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion [qw, qx, qy, qz], got shape {quat.shape}")
    return Rotation.from_quat(np.array([quat[1], quat[2], quat[3], quat[0]])).as_rotvec()


def _slerp_quaternion_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        quat = q0 + alpha * (q1 - q0)
        return quat / np.linalg.norm(quat)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    scale_0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    scale_1 = sin_theta / sin_theta_0
    return scale_0 * q0 + scale_1 * q1


class EliteCS66(Robot):
    """Single Elite CS66 arm using elite_cs_sdk external control.

    Cartesian mode:
        action/observation features are tcp.x/y/z plus tcp.r1..tcp.r6.
        Elite's native [rx, ry, rz] rotation vector is kept as an internal SDK
        detail and converted inside send_action()/get_observation().

    Joint mode:
        action features are joint_1.pos ... joint_6.pos and are streamed with
        writeServoj(..., cartesian=False).
    """

    config_class = EliteCS66Config
    name = "elite_cs66"

    def __init__(self, config: EliteCS66Config):
        super().__init__(config)
        self.config = config
        logger_suffix = config.id if config.id is not None else hex(id(self))
        self.logger = get_logger(f"EliteCS66.{logger_suffix}")

        self._cs = None
        self._dashboard = None
        self._driver = None
        self._rtsi = None
        self._is_connected = False
        self._gripper_position = float(config.initial_gripper_position)
        self._last_tcp_command: np.ndarray | None = None
        self._target_tcp_command: np.ndarray | None = None
        self._servo_thread: threading.Thread | None = None
        self._servo_stop_event = threading.Event()
        self._servo_lock = threading.Lock()
        self._servo_error: BaseException | None = None
        self._last_action_time = 0.0
        self._start_tcp_pose: np.ndarray | None = None
        self._reset_start_tcp_pose: np.ndarray | None = None
        self._reset_target_tcp_pose: np.ndarray | None = None
        self._reset_start_time = 0.0
        self._reset_end_time = 0.0
        self._reset_moving = False

        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = {}

        if self.config.use_joint_observation:
            features.update(dict.fromkeys(JOINT_POSITION_KEYS, float))
            features.update(dict.fromkeys(JOINT_VELOCITY_KEYS, float))
            features.update(dict.fromkeys(JOINT_EFFORT_KEYS, float))
        else:
            features.update(dict.fromkeys(TCP_POSITION_KEYS + TCP_ROTATION_6D_KEYS, float))

        if self.config.use_gripper:
            features["gripper.pos"] = float

        for cam_name in self.cameras:
            features[cam_name] = (self.config.cameras[cam_name].height, self.config.cameras[cam_name].width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.config.control_mode == EliteCS66ControlMode.JOINT_SERVO:
            features = dict.fromkeys(JOINT_POSITION_KEYS, float)
        else:
            features = dict.fromkeys(TCP_POSITION_KEYS + TCP_ROTATION_6D_KEYS, float)

        if self.config.use_gripper:
            features["gripper.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self._driver is not None
            and self._rtsi is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        return self.is_connected

    def calibrate(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

    def configure(self) -> None:
        pass

    def _resolve_sdk_resource(self, filename: str) -> str:
        assert self._cs is not None
        module_file = getattr(self._cs, "__file__", None)
        if not module_file:
            raise RuntimeError("Cannot resolve elite_cs_sdk package path.")
        path = Path(module_file).resolve().parent / filename
        if not path.exists():
            raise FileNotFoundError(f"Elite SDK resource not found: {path}")
        return str(path)

    def _resolve_recipe(self, configured: str | Path | None, filename: str) -> str:
        if configured is not None:
            path = Path(configured).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Configured RTSI recipe not found: {path}")
            return str(path)

        sdk_path = None
        with suppress(FileNotFoundError):
            sdk_path = self._resolve_sdk_resource(filename)
        if sdk_path:
            return sdk_path

        module_recipe = Path(__file__).resolve().parent / "resource" / filename
        if module_recipe.exists():
            return str(module_recipe)
        raise FileNotFoundError(
            f"Could not find {filename}. Set rtsi_output_recipe/rtsi_input_recipe in EliteCS66Config."
        )

    def _make_driver_config(self):
        assert self._cs is not None
        cfg = self._cs.EliteDriverConfig()
        cfg.robot_ip = self.config.robot_ip
        cfg.local_ip = self.config.local_ip
        cfg.servoj_time = self.config.servoj_time
        cfg.servoj_lookahead_time = self.config.servoj_lookahead_time
        cfg.servoj_gain = self.config.servoj_gain
        cfg.headless_mode = self.config.headless_mode
        if self.config.script_file_path is not None:
            cfg.script_file_path = str(Path(self.config.script_file_path).expanduser())
        else:
            cfg.script_file_path = self._resolve_sdk_resource("external_control.script")
        return cfg

    def connect(self, calibrate: bool = False, go_to_start: bool = False) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected, do not run connect() twice.")

        self._cs = _import_elite_sdk()
        if self.config.enable_realtime_scheduling:
            try:
                self._cs.setCurrentThreadFiFoScheduling(self._cs.getThreadFiFoMaxPriority())
            except Exception as exc:
                self.logger.warn(f"Failed to enable FIFO scheduling for Elite CS66 control thread: {exc}")

        output_recipe = self._resolve_recipe(self.config.rtsi_output_recipe, "output_recipe.txt")
        input_recipe = self._resolve_recipe(self.config.rtsi_input_recipe, "input_recipe.txt")

        self._rtsi = self._cs.RtsiIOInterface(output_recipe, input_recipe, self.config.rtsi_frequency)
        if not self._rtsi.connect(self.config.robot_ip):
            self._rtsi = None
            raise ConnectionError(f"Failed to connect Elite RTSI server at {self.config.robot_ip}:30004")

        self._dashboard = self._cs.DashboardClientInterface()
        if not self._dashboard.connect(self.config.robot_ip):
            self._cleanup_after_failed_connect()
            raise ConnectionError(f"Failed to connect Elite dashboard at {self.config.robot_ip}")

        if self.config.power_on_on_connect and not self._dashboard.powerOn():
            self._cleanup_after_failed_connect()
            raise RuntimeError("Elite CS66 powerOn() failed.")

        if self.config.brake_release_on_connect and not self._dashboard.brakeRelease():
            self._cleanup_after_failed_connect()
            raise RuntimeError("Elite CS66 brakeRelease() failed.")

        driver_config = self._make_driver_config()
        self._driver = self._cs.EliteDriver(driver_config)

        if self.config.start_external_control_on_connect:
            if driver_config.headless_mode:
                if not self._driver.isRobotConnected() and not self._driver.sendExternalControlScript():
                    self._cleanup_after_failed_connect()
                    raise RuntimeError("Failed to send Elite external control script.")
            elif self.config.play_program_on_connect and not self._dashboard.playProgram():
                self._cleanup_after_failed_connect()
                raise RuntimeError("Failed to play Elite external control program.")

            deadline = time.monotonic() + self.config.connect_timeout_s
            while not self._driver.isRobotConnected():
                if time.monotonic() > deadline:
                    self._cleanup_after_failed_connect()
                    raise TimeoutError("Timed out waiting for Elite external control script connection.")
                time.sleep(0.01)

        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        if self.config.control_mode == EliteCS66ControlMode.CARTESIAN_SERVO:
            current_tcp = np.asarray(self._rtsi.getActualTCPPose(), dtype=np.float64)
            self._last_tcp_command = current_tcp.copy()
            self._target_tcp_command = current_tcp.copy()
            self._start_tcp_pose = current_tcp.copy()
            self._last_action_time = time.monotonic()
            if self.config.use_background_servo_loop:
                self._start_servo_loop()

    def _cleanup_after_failed_connect(self) -> None:
        try:
            if self._driver is not None:
                self._driver.stopControl(1000)
        except Exception:
            pass
        try:
            if self._dashboard is not None:
                self._dashboard.disconnect()
        except Exception:
            pass
        try:
            if self._rtsi is not None:
                self._rtsi.disconnect()
        except Exception:
            pass
        self._driver = None
        self._dashboard = None
        self._rtsi = None
        self._is_connected = False

    def _start_servo_loop(self) -> None:
        if self._servo_thread is not None and self._servo_thread.is_alive():
            return
        self._servo_error = None
        self._servo_stop_event.clear()
        self._servo_thread = threading.Thread(
            target=self._servo_loop,
            name=f"EliteCS66ServoLoop-{self.config.id or hex(id(self))}",
            daemon=True,
        )
        self._servo_thread.start()

    def _stop_servo_loop(self) -> None:
        self._servo_stop_event.set()
        if self._servo_thread is not None:
            self._servo_thread.join(timeout=2.0)
            self._servo_thread = None

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
        pose[3:6] = _quaternion_to_rotvec(_slerp_quaternion_wxyz(start_quat, target_quat, alpha))
        return pose

    def _get_servo_target_locked(self, now: float) -> tuple[np.ndarray | None, bool]:
        if self._reset_moving:
            if self._reset_start_tcp_pose is None or self._reset_target_tcp_pose is None:
                self._reset_moving = False
            elif now >= self._reset_end_time:
                target = self._reset_target_tcp_pose.copy()
                self._target_tcp_command = target.copy()
                self._last_action_time = now
                self._last_tcp_command = target.copy()
                self._reset_moving = False
                return target, True
            else:
                duration = max(self._reset_end_time - self._reset_start_time, self.config.servoj_time)
                alpha = self._min_jerk((now - self._reset_start_time) / duration)
                target = self._interpolate_tcp_pose(
                    self._reset_start_tcp_pose,
                    self._reset_target_tcp_pose,
                    alpha,
                )
                self._last_action_time = now
                return target, True

        target = None if self._target_tcp_command is None else self._target_tcp_command.copy()
        return target, False

    def _servo_loop(self) -> None:
        assert self._driver is not None
        assert self._cs is not None

        if self.config.enable_realtime_scheduling:
            try:
                self._cs.setCurrentThreadFiFoScheduling(self._cs.getThreadFiFoMaxPriority())
            except Exception as exc:
                self.logger.warn(f"Failed to enable FIFO scheduling for Elite CS66 servo loop: {exc}")

        next_tick = time.perf_counter()
        while not self._servo_stop_event.is_set():
            try:
                now = time.monotonic()
                with self._servo_lock:
                    target, reset_active = self._get_servo_target_locked(now)
                    last_action_time = self._last_action_time

                if target is None:
                    self._driver.writeIdle(self.config.command_timeout_ms)
                else:
                    if not reset_active and now - last_action_time > self.config.command_stale_timeout_s:
                        self._driver.writeIdle(self.config.command_timeout_ms)
                    else:
                        ok = self._driver.writeServoj(target.tolist(), self.config.command_timeout_ms, True)
                        if not ok:
                            raise RuntimeError("Elite writeServoj(cartesian=True) failed in servo loop.")
                        with self._servo_lock:
                            self._last_tcp_command = target

                next_tick += self.config.servoj_time
                sleep_s = next_tick - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.perf_counter()
            except BaseException as exc:
                self._servo_error = exc
                self._servo_stop_event.set()
                break

    def _raise_servo_error_if_any(self) -> None:
        if self._servo_error is not None:
            raise RuntimeError(f"Elite CS66 background servo loop failed: {self._servo_error}") from self._servo_error

    def _is_reset_moving_locked(self, now: float) -> bool:
        if not self._reset_moving:
            return False
        if now < self._reset_end_time:
            return True
        if self._reset_target_tcp_pose is not None:
            self._target_tcp_command = self._reset_target_tcp_pose.copy()
            self._last_tcp_command = self._reset_target_tcp_pose.copy()
            self._last_action_time = now
        self._reset_moving = False
        return False

    def _tcp_rotvec_to_feature_values(self, tcp_pose: np.ndarray) -> dict[str, float]:
        values = {
            "tcp.x": float(tcp_pose[0]),
            "tcp.y": float(tcp_pose[1]),
            "tcp.z": float(tcp_pose[2]),
        }
        quat = _rotvec_to_quaternion(tcp_pose[3:6])
        r6d = quaternion_to_rotation_6d(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        values.update({key: float(value) for key, value in zip(TCP_ROTATION_6D_KEYS, r6d, strict=True)})
        return values

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        assert self._rtsi is not None
        obs: dict[str, Any] = {}

        if self.config.use_joint_observation:
            joints = self._rtsi.getActualJointPositions()
            obs.update({key: float(value) for key, value in zip(JOINT_POSITION_KEYS, joints, strict=True)})
            joint_vel = self._rtsi.getActualJointVelocity()
            obs.update({key: float(value) for key, value in zip(JOINT_VELOCITY_KEYS, joint_vel, strict=True)})
            joint_effort = self._rtsi.getActualJointTorques()
            obs.update({key: float(value) for key, value in zip(JOINT_EFFORT_KEYS, joint_effort, strict=True)})
        else:
            tcp_pose = np.asarray(self._rtsi.getActualTCPPose(), dtype=np.float64)
            obs.update(self._tcp_rotvec_to_feature_values(tcp_pose))

        if self.config.use_gripper:
            obs["gripper.pos"] = self._gripper_position

        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()
        return obs

    def _clip_gripper(self, value: float) -> float:
        return min(max(value, self.config.gripper_min_position), self.config.gripper_max_position)

    def _cartesian_action_to_tcp_pose(self, action: dict[str, Any]) -> np.ndarray:
        if self._last_tcp_command is not None:
            target = self._last_tcp_command.copy()
        else:
            assert self._rtsi is not None
            target = np.asarray(self._rtsi.getActualTCPPose(), dtype=np.float64)

        current = target.copy()
        if self._rtsi is not None:
            current = np.asarray(self._rtsi.getActualTCPPose(), dtype=np.float64)

        for i, key in enumerate(TCP_POSITION_KEYS):
            if key in action:
                target[i] = float(action[key])

        if any(key in action for key in TCP_ROTATION_6D_KEYS):
            if not all(key in action for key in TCP_ROTATION_6D_KEYS):
                raise ValueError("Incomplete rotation-6D action. Expected tcp.r1 through tcp.r6 together.")
            r6d = np.array([float(action[key]) for key in TCP_ROTATION_6D_KEYS], dtype=np.float64)
            target[3:6] = _quaternion_to_rotvec(rotation_6d_to_quaternion(r6d))
        elif all(key in action for key in ("roll", "pitch", "yaw")):
            quat = euler_to_quaternion(float(action["roll"]), float(action["pitch"]), float(action["yaw"]))
            target[3:6] = _quaternion_to_rotvec(quat)

        if self.config.max_relative_translation > 0:
            delta = target[:3] - current[:3]
            norm = float(np.linalg.norm(delta))
            if norm > self.config.max_relative_translation:
                target[:3] = current[:3] + delta / norm * self.config.max_relative_translation

        if self.config.max_relative_rotation > 0:
            delta = target[3:6] - current[3:6]
            norm = float(np.linalg.norm(delta))
            if norm > self.config.max_relative_rotation:
                target[3:6] = current[3:6] + delta / norm * self.config.max_relative_rotation

        return target

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        assert self._driver is not None
        self._raise_servo_error_if_any()

        sent: dict[str, Any] = {}

        if self.config.control_mode == EliteCS66ControlMode.CARTESIAN_SERVO:
            if self.config.use_background_servo_loop:
                with self._servo_lock:
                    reset_moving = self._is_reset_moving_locked(time.monotonic())
                if reset_moving:
                    if self.config.use_gripper and "gripper.pos" in action:
                        self._gripper_position = self._clip_gripper(float(action["gripper.pos"]))
                        sent["gripper.pos"] = self._gripper_position
                    return sent or action

            target_tcp = self._cartesian_action_to_tcp_pose(action)
            if self.config.use_background_servo_loop:
                with self._servo_lock:
                    self._target_tcp_command = target_tcp.copy()
                    self._last_action_time = time.monotonic()
            else:
                ok = self._driver.writeServoj(target_tcp.tolist(), self.config.command_timeout_ms, True)
                if not ok:
                    raise RuntimeError("Elite writeServoj(cartesian=True) failed.")
                self._last_tcp_command = target_tcp
            sent.update(self._tcp_rotvec_to_feature_values(target_tcp))
        else:
            if not all(key in action for key in JOINT_POSITION_KEYS):
                missing = [key for key in JOINT_POSITION_KEYS if key not in action]
                raise ValueError(f"Missing joint servo action keys: {missing}")
            target_joints = [float(action[key]) for key in JOINT_POSITION_KEYS]
            ok = self._driver.writeServoj(target_joints, self.config.command_timeout_ms, False)
            if not ok:
                raise RuntimeError("Elite writeServoj(cartesian=False) failed.")
            sent.update(dict(zip(JOINT_POSITION_KEYS, target_joints, strict=True)))

        if self.config.use_gripper and "gripper.pos" in action:
            self._gripper_position = self._clip_gripper(float(action["gripper.pos"]))
            sent["gripper.pos"] = self._gripper_position

        return sent

    def disconnect(self) -> None:
        if not self._is_connected and self._driver is None and self._rtsi is None and self._dashboard is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._servo_stop_event.set()
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()

        self._stop_servo_loop()

        if self._driver is not None:
            try:
                if self.config.stop_control_on_disconnect:
                    self._driver.writeIdle(self.config.command_timeout_ms)
                    self._driver.stopControl(1000)
            finally:
                self._driver = None

        if self._dashboard is not None:
            try:
                self._dashboard.disconnect()
            finally:
                self._dashboard = None

        if self._rtsi is not None:
            try:
                self._rtsi.disconnect()
            finally:
                self._rtsi = None

        self._is_connected = False

    @property
    def rt_running(self) -> bool:
        return self._servo_thread is not None and self._servo_thread.is_alive()

    @property
    def rt_moving(self) -> bool:
        with self._servo_lock:
            return self._is_reset_moving_locked(time.monotonic())

    def get_current_tcp_pose_quat(self) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        assert self._rtsi is not None
        tcp_pose = np.asarray(self._rtsi.getActualTCPPose(), dtype=np.float64)
        quat = _rotvec_to_quaternion(tcp_pose[3:6])
        gripper_pos = self._gripper_position if self.config.use_gripper else 0.0
        return np.array(
            [tcp_pose[0], tcp_pose[1], tcp_pose[2], quat[0], quat[1], quat[2], quat[3], gripper_pos],
            dtype=np.float64,
        )

    def get_current_tcp_pose_euler(self) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        assert self._rtsi is not None
        tcp_pose = np.asarray(self._rtsi.getActualTCPPose(), dtype=np.float64)
        quat = _rotvec_to_quaternion(tcp_pose[3:6])
        euler = self._quaternion_to_euler_wxyz(quat)
        gripper_pos = self._gripper_position if self.config.use_gripper else 0.0
        return np.array(
            [tcp_pose[0], tcp_pose[1], tcp_pose[2], euler[0], euler[1], euler[2], gripper_pos],
            dtype=np.float64,
        )

    def reset_to_initial_position(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._start_tcp_pose is None:
            return
        if self.config.control_mode != EliteCS66ControlMode.CARTESIAN_SERVO:
            raise RuntimeError("reset_to_initial_position() is only supported in Cartesian servo mode.")

        if self.config.use_background_servo_loop:
            assert self._rtsi is not None
            now = time.monotonic()
            with self._servo_lock:
                if self._is_reset_moving_locked(now):
                    return
                current_tcp = np.asarray(self._rtsi.getActualTCPPose(), dtype=np.float64)
                self._reset_start_tcp_pose = current_tcp.copy()
                self._reset_target_tcp_pose = self._start_tcp_pose.copy()
                self._reset_start_time = now
                self._reset_end_time = now + self.config.reset_duration_s
                self._reset_moving = True
                self._last_action_time = now
            return

        assert self._driver is not None
        ok = self._driver.writeServoj(
            self._start_tcp_pose.tolist(),
            self.config.command_timeout_ms,
            True,
        )
        if not ok:
            raise RuntimeError("Elite writeServoj(cartesian=True) failed during reset.")
        self._last_tcp_command = self._start_tcp_pose.copy()

    @staticmethod
    def _quaternion_to_euler_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat_wxyz, dtype=np.float64)
        if quat.shape != (4,):
            raise ValueError(f"Expected quaternion [qw, qx, qy, qz], got shape {quat.shape}")
        qw, qx, qy, qz = quat
        t0 = 2.0 * (qw * qx + qy * qz)
        t1 = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(t0, t1)
        t2 = 2.0 * (qw * qy - qz * qx)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        t3 = 2.0 * (qw * qz + qx * qy)
        t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(t3, t4)
        return np.array([roll, pitch, yaw], dtype=np.float64)
