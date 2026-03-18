#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Any

from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import euler_to_quaternion, quaternion_to_rotation_6d

TCP_POSITION_KEYS = ("tcp.x", "tcp.y", "tcp.z")
TCP_ROTATION_KEYS = ("tcp.r1", "tcp.r2", "tcp.r3", "tcp.r4", "tcp.r5", "tcp.r6")
TCP_KEYS = TCP_POSITION_KEYS + TCP_ROTATION_KEYS


class MockRobotControlMode(str, Enum):
    JOINT_CONTROL = "joint_control"
    CARTESIAN_CONTROL = "cartesian_control"


@RobotConfig.register_subclass("mock_robot")
@dataclass
class MockRobotConfig(RobotConfig):
    n_motors: int = 6
    control_mode: MockRobotControlMode = MockRobotControlMode.JOINT_CONTROL
    use_gripper: bool = True
    gripper_min_position: float = 0.0
    gripper_max_position: float = 1.0
    initial_joint_positions: list[float] | None = None
    initial_gripper_position: float = 0.0
    # [tcp.x, tcp.y, tcp.z, tcp.r1, ..., tcp.r6]
    initial_tcp_pose: list[float] | None = None
    camera_shapes: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    calibrated: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.n_motors < 1:
            raise ValueError(f"n_motors must be >= 1, got {self.n_motors}")
        if self.gripper_min_position >= self.gripper_max_position:
            raise ValueError(
                "gripper_min_position must be smaller than gripper_max_position, got "
                f"{self.gripper_min_position} >= {self.gripper_max_position}"
            )
        if self.initial_joint_positions is not None and len(self.initial_joint_positions) != self.n_motors:
            raise ValueError(
                "initial_joint_positions length must match n_motors, got "
                f"{len(self.initial_joint_positions)} != {self.n_motors}"
            )
        if self.initial_tcp_pose is not None and len(self.initial_tcp_pose) != len(TCP_KEYS):
            raise ValueError(
                "initial_tcp_pose must be [tcp.x, tcp.y, tcp.z, tcp.r1, ..., tcp.r6], got length "
                f"{len(self.initial_tcp_pose)}"
            )
        if not self.gripper_min_position <= self.initial_gripper_position <= self.gripper_max_position:
            raise ValueError(
                "initial_gripper_position must be in [gripper_min_position, gripper_max_position], got "
                f"{self.initial_gripper_position}"
            )


class MockRobot(Robot):
    """Mock robot for testing with configurable joint/cartesian control modes."""

    config_class = MockRobotConfig
    name = "mock_robot"

    def __init__(self, config: MockRobotConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._is_calibrated = config.calibrated

        self.motors = [f"joint_{i + 1}" for i in range(config.n_motors)]
        initial_joints = config.initial_joint_positions or [0.0] * config.n_motors
        self._joint_state = {f"{motor}.pos": float(v) for motor, v in zip(self.motors, initial_joints, strict=True)}

        if config.initial_tcp_pose is not None:
            self._tcp_state = {k: float(v) for k, v in zip(TCP_KEYS, config.initial_tcp_pose, strict=True)}
        else:
            # Identity orientation in 6D rotation representation.
            self._tcp_state = {
                "tcp.x": 0.0,
                "tcp.y": 0.0,
                "tcp.z": 0.0,
                "tcp.r1": 1.0,
                "tcp.r2": 0.0,
                "tcp.r3": 0.0,
                "tcp.r4": 0.0,
                "tcp.r5": 1.0,
                "tcp.r6": 0.0,
            }

        self._gripper_state = float(config.initial_gripper_position)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        if self.config.control_mode == MockRobotControlMode.JOINT_CONTROL:
            features: dict[str, type | tuple[int, int, int]] = {f"{motor}.pos": float for motor in self.motors}
        else:
            features = {k: float for k in TCP_KEYS}

        if self.config.use_gripper:
            features["gripper.pos"] = float

        for cam_name, shape in self.config.camera_shapes.items():
            features[f"observation.images.{cam_name}"] = shape
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.config.control_mode == MockRobotControlMode.JOINT_CONTROL:
            features = {f"{motor}.pos": float for motor in self.motors}
        else:
            features = {k: float for k in TCP_KEYS}

        if self.config.use_gripper:
            features["gripper.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected")

        self._is_connected = True
        if calibrate:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._is_calibrated = True

    def configure(self) -> None:
        pass

    def _clip_gripper(self, value: float) -> float:
        clipped = min(max(value, self.config.gripper_min_position), self.config.gripper_max_position)
        if clipped != value:
            warnings.warn(
                "Gripper action is out of range. Clipped "
                f"{value} -> {clipped} in [{self.config.gripper_min_position}, {self.config.gripper_max_position}]"
            )
        return clipped

    def _parse_float(self, value: Any, key: str) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            warnings.warn(f"Ignoring non-numeric action[{key}]={value!r}")
            return None

    def _extract_first_float(self, action: dict[str, Any], keys: tuple[str, ...]) -> float | None:
        for key in keys:
            if key in action:
                return self._parse_float(action[key], key)
        return None

    def _apply_positional_fallback(self, action: dict[str, Any], keys: list[str]) -> dict[str, float]:
        warnings.warn(
            "Action keys do not match mock_robot schema; falling back to positional mapping by action value order."
        )
        applied: dict[str, float] = {}
        for key, value in zip(keys, action.values(), strict=False):
            parsed = self._parse_float(value, key)
            if parsed is None:
                continue
            if key == "gripper.pos":
                parsed = self._clip_gripper(parsed)
                self._gripper_state = parsed
            elif key in self._joint_state:
                self._joint_state[key] = parsed
            else:
                self._tcp_state[key] = parsed
            applied[key] = parsed
        return applied

    def _apply_joint_action(self, action: dict[str, Any]) -> dict[str, float]:
        applied: dict[str, float] = {}
        recognized_any = False

        for idx, motor in enumerate(self.motors, start=1):
            key = f"{motor}.pos"
            value = self._extract_first_float(
                action,
                (
                    key,
                    f"motor_{idx}.pos",
                    motor,
                    f"joint_{idx}",
                    f"motor_{idx}",
                ),
            )
            if value is None:
                continue
            recognized_any = True
            self._joint_state[key] = value
            applied[key] = value

        if self.config.use_gripper:
            gripper = self._extract_first_float(action, ("gripper.pos", "gripper_pos", "gripper"))
            if gripper is not None:
                recognized_any = True
                self._gripper_state = self._clip_gripper(gripper)
                applied["gripper.pos"] = self._gripper_state

        if not recognized_any and action:
            fallback_keys = list(self._joint_state)
            if self.config.use_gripper:
                fallback_keys.append("gripper.pos")
            return self._apply_positional_fallback(action, fallback_keys)

        return applied

    def _parse_cartesian_from_euler(self, action: dict[str, Any]) -> dict[str, float]:
        euler_keys = ("roll", "pitch", "yaw")
        has_any_euler = any(k in action for k in euler_keys)
        if not has_any_euler:
            return {}
        if not all(k in action for k in euler_keys):
            warnings.warn("Incomplete Euler action provided. Expected roll/pitch/yaw together.")
            return {}

        roll = self._parse_float(action["roll"], "roll")
        pitch = self._parse_float(action["pitch"], "pitch")
        yaw = self._parse_float(action["yaw"], "yaw")
        if roll is None or pitch is None or yaw is None:
            return {}

        q = euler_to_quaternion(roll, pitch, yaw)
        r6d = quaternion_to_rotation_6d(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        return {k: float(v) for k, v in zip(TCP_ROTATION_KEYS, r6d, strict=True)}

    def _parse_cartesian_rotation(self, action: dict[str, Any]) -> dict[str, float]:
        has_any_r6 = any(k in action for k in TCP_ROTATION_KEYS)
        if has_any_r6:
            if not all(k in action for k in TCP_ROTATION_KEYS):
                warnings.warn("Incomplete TCP rotation action provided. Expected tcp.r1~tcp.r6 together.")
                return {}
            rot_values = {}
            for k in TCP_ROTATION_KEYS:
                v = self._parse_float(action[k], k)
                if v is None:
                    return {}
                rot_values[k] = v
            return rot_values
        return self._parse_cartesian_from_euler(action)

    def _apply_cartesian_action(self, action: dict[str, Any]) -> dict[str, float]:
        applied: dict[str, float] = {}
        recognized_any = False

        pos_aliases = {
            "tcp.x": ("tcp.x", "x"),
            "tcp.y": ("tcp.y", "y"),
            "tcp.z": ("tcp.z", "z"),
        }
        for key, aliases in pos_aliases.items():
            value = self._extract_first_float(action, aliases)
            if value is None:
                continue
            recognized_any = True
            self._tcp_state[key] = value
            applied[key] = value

        rot_values = self._parse_cartesian_rotation(action)
        if rot_values:
            recognized_any = True
            for k, v in rot_values.items():
                self._tcp_state[k] = v
            applied.update(rot_values)

        if self.config.use_gripper:
            gripper = self._extract_first_float(action, ("gripper.pos", "gripper_pos", "gripper"))
            if gripper is not None:
                recognized_any = True
                self._gripper_state = self._clip_gripper(gripper)
                applied["gripper.pos"] = self._gripper_state

        if not recognized_any and action:
            fallback_keys = list(TCP_KEYS)
            if self.config.use_gripper:
                fallback_keys.append("gripper.pos")
            return self._apply_positional_fallback(action, fallback_keys)

        return applied

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs: dict[str, Any] = {}
        if self.config.control_mode == MockRobotControlMode.JOINT_CONTROL:
            obs.update(self._joint_state)
        else:
            obs.update(self._tcp_state)

        if self.config.use_gripper:
            obs["gripper.pos"] = self._gripper_state

        for cam_name, shape in self.config.camera_shapes.items():
            import numpy as np

            obs[f"observation.images.{cam_name}"] = np.zeros(shape, dtype=np.uint8)
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.config.control_mode == MockRobotControlMode.JOINT_CONTROL:
            return self._apply_joint_action(action)
        return self._apply_cartesian_action(action)

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._is_connected = False
