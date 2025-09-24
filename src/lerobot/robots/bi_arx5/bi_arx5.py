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

import logging
import math
import time
from functools import cached_property
from typing import Any, Sequence

import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_bi_arx5 import BiARX5Config

# 导入ARX5接口 Stanford-Real-Robot
try:
    from .ARX5_SDK.python import arx5_interface as arx5
except ImportError as e:
    if "LogLevel" in str(e) and "already registered" in str(e):
        # LogLevel already registered, try to get the existing module
        import sys

        if "lerobot.robots.bi_arx5.ARX5_SDK.python.arx5_interface" in sys.modules:
            arx5 = sys.modules["lerobot.robots.bi_arx5.ARX5_SDK.python.arx5_interface"]
        else:
            raise e
    else:
        raise e


logger = logging.getLogger(__name__)


class BiARX5(Robot):
    """
    [Bimanual ARX5 Arms](https://github.com/ARXroboticsX/ARX_X5) designed by ARXROBOTICS
    """

    config_class = BiARX5Config
    name = "bi_arx5"

    def __init__(self, config: BiARX5Config):
        super().__init__(config)
        self.config = config

        # init left and right arm when connect
        self.left_arm = None
        self.right_arm = None
        self._is_connected = False

        # rpc timeout
        self.rpc_timeout: float = getattr(config, "rpc_timeout", 5.0)

        # init thread pool
        self._exec_left = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="left_arm"
        )
        self._exec_right = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="right_arm"
        )

        # Pre-compute action keys for faster lookup (performance optimization)
        self._left_joint_keys = [f"left_joint_{i+1}.pos" for i in range(6)]
        self._right_joint_keys = [f"right_joint_{i+1}.pos" for i in range(6)]

        # Pre-allocate JointState command buffers to avoid repeated allocation
        self._left_cmd_buffer = None
        self._right_cmd_buffer = None

        # use dict to store left and right arm configs
        self.robot_configs = {
            "left_config": arx5.RobotConfigFactory.get_instance().get_config(
                config.left_arm_model
            ),
            "right_config": arx5.RobotConfigFactory.get_instance().get_config(
                config.right_arm_model
            ),
        }

        self.controller_configs = {
            "left_config": arx5.ControllerConfigFactory.get_instance().get_config(
                "joint_controller", self.robot_configs["left_config"].joint_dof
            ),
            "right_config": arx5.ControllerConfigFactory.get_instance().get_config(
                "joint_controller", self.robot_configs["right_config"].joint_dof
            ),
        }

        # use multithreading by default
        if config.use_multithreading:
            self.controller_configs["left_config"].background_send_recv = True
            self.controller_configs["right_config"].background_send_recv = True
        else:
            self.controller_configs["left_config"].background_send_recv = False
            self.controller_configs["right_config"].background_send_recv = False

        self.cameras = make_cameras_from_configs(config.cameras)
        np.set_printoptions(precision=3, suppress=True)

    @property
    def _motors_ft(self) -> dict[str, type]:
        # ARX5 有 6个关节 + 1个夹爪
        joint_names = [f"joint_{i}" for i in range(1, 7)] + ["gripper"]
        return {f"left_{joint}.pos": float for joint in joint_names} | {
            f"right_{joint}.pos": float for joint in joint_names
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self.left_arm is not None
            and self.right_arm is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = False, go_to_home: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} already connected, do not run `robot.connect()` twice."
            )

        try:
            logger.info("正在创建左臂控制器...")
            self.left_arm = arx5.Arx5JointController(
                self.robot_configs["left_config"],
                self.controller_configs["left_config"],
                self.config.left_arm_port,
            )
            logger.info("✓ 左臂控制器创建成功")

            logger.info("正在创建右臂控制器...")
            self.right_arm = arx5.Arx5JointController(
                self.robot_configs["right_config"],
                self.controller_configs["right_config"],
                self.config.right_arm_port,
            )
            logger.info("✓ 右臂控制器创建成功")
        except Exception as e:
            logger.error(f"创建机器人控制器失败: {e}")
            # 清理已创建的实例
            self.left_arm = None
            self.right_arm = None
            raise e

        # set log lever
        self.set_log_level(self.config.log_level)

        # 如果需要校准，执行回零
        if go_to_home:
            self.reset_to_home()

        # 设置重力补偿模式：所有增益设为0，只保留重力补偿
        logger.info("Setting both arms to gravity compensation mode...")

        zero_gain = arx5.Gain(self.robot_configs["left_config"].joint_dof)
        zero_gain.kp()[:] = 0.0
        zero_gain.kd()[:] = 0.0
        zero_gain.gripper_kp = 0.0
        zero_gain.gripper_kd = 0.0
        self.left_arm.set_gain(zero_gain)
        self.right_arm.set_gain(zero_gain)

        logger.info("✓ Both arms are now in gravity compensation mode")
        # 连接摄像头
        for cam in self.cameras.values():
            cam.connect()

        # Initialize command buffers for optimized send_action
        self._left_cmd_buffer = arx5.JointState(
            self.robot_configs["left_config"].joint_dof
        )
        self._right_cmd_buffer = arx5.JointState(
            self.robot_configs["right_config"].joint_dof
        )

        self._is_connected = True
        logger.info("Dual-ARX5 connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.is_connected()

    def calibrate(self) -> None:
        """ARX 5 dont need to calib in runtime"""
        logger.info("ARX5 do not need to calib in runtime, skip...")
        return

    def configure(self) -> None:
        """配置ARX5双臂的控制增益"""
        if self.left_arm is None or self.right_arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            # 获取默认增益
            left_gain = self.left_arm.get_gain()
            right_gain = self.right_arm.get_gain()

            # 打印增益信息
            logger.info("Left arm gains:")
            logger.info(f"  kp: {left_gain.kp()}")
            logger.info(f"  kd: {left_gain.kd()}")
            logger.info(f"  gripper_kp: {left_gain.gripper_kp}")
            logger.info(f"  gripper_kd: {left_gain.gripper_kd}")

            logger.info("Right arm gains:")
            logger.info(f"  kp: {right_gain.kp()}")
            logger.info(f"  kd: {right_gain.kd()}")
            logger.info(f"  gripper_kp: {right_gain.gripper_kp}")
            logger.info(f"  gripper_kd: {right_gain.gripper_kd}")

            # 设置增益（保持默认值）
            self.left_arm.set_gain(left_gain)
            self.right_arm.set_gain(right_gain)

            logger.info(f"{self} configured with custom gains")

        except Exception as e:
            logger.error(f"Failed to configure {self}: {e}")
            raise e

    def setup_motors(self) -> None:
        """ARX5 motors use pre-configured IDs, no runtime setup needed"""
        logger.info(
            f"{self} ARX5 motors use pre-configured IDs, no runtime setup needed"
        )
        logger.info("Motor IDs are defined in the robot configuration:")
        logger.info("  - Joint motors: [1, 2, 4, 5, 6, 7]")
        logger.info("  - Gripper motor: 8")
        logger.info("Make sure your hardware matches these ID configurations")
        return

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Add "left_" prefix - get joint state from left arm
        left_joint_state = self.left_arm.get_joint_state()
        left_pos = (
            left_joint_state.pos().copy()
        )  # numpy array of joint positions (deep copy)

        # Create left arm observations with joint names matching _motors_ft
        for i in range(6):  # 6 joints
            obs_dict[f"left_joint_{i+1}.pos"] = float(left_pos[i])
        obs_dict["left_gripper.pos"] = float(left_joint_state.gripper_pos)

        # Add "right_" prefix - get joint state from right arm
        right_joint_state = self.right_arm.get_joint_state()
        right_pos = (
            right_joint_state.pos().copy()
        )  # numpy array of joint positions (deep copy)

        # Create right arm observations with joint names matching _motors_ft
        for i in range(6):  # 6 joints
            obs_dict[f"right_joint_{i+1}.pos"] = float(right_pos[i])
        obs_dict["right_gripper.pos"] = float(right_joint_state.gripper_pos)

        # Add camera observations
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Use pre-allocated JointState objects (avoid repeated allocation)
        left_cmd = self._left_cmd_buffer
        right_cmd = self._right_cmd_buffer

        # Batch extract using pre-computed keys for better performance
        # Left arm: use single get() call with fallback to avoid 'in' checks
        left_pos = left_cmd.pos()
        for i, key in enumerate(self._left_joint_keys):
            left_pos[i] = action.get(
                key, left_pos[i]
            )  # Keep previous value if key missing

        left_cmd.gripper_pos = action.get("left_gripper.pos", left_cmd.gripper_pos)

        # Right arm: same optimization
        right_pos = right_cmd.pos()
        for i, key in enumerate(self._right_joint_keys):
            right_pos[i] = action.get(
                key, right_pos[i]
            )  # Keep previous value if key missing

        right_cmd.gripper_pos = action.get("right_gripper.pos", right_cmd.gripper_pos)

        # Debug: Print commands before sending
        # print(
        #     f"Left arm command - pos: {left_cmd.pos()}, gripper: {left_cmd.gripper_pos}"
        # )
        # print(
        #     f"Right arm command - pos: {right_cmd.pos()}, gripper: {right_cmd.gripper_pos}"
        # )

        self.left_arm.set_joint_cmd(left_cmd)
        self.right_arm.set_joint_cmd(right_cmd)

        # Simply return the input action
        return action

    @staticmethod
    def _ease_in_out_quad(t: float) -> float:
        """Smooth easing function used for joint interpolation."""
        tt = t * 2.0
        if tt < 1.0:
            return (tt * tt) / 2.0
        tt -= 1.0
        return -(tt * (tt - 2.0) - 1.0) / 2.0

    def move_joint_trajectory(
        self,
        target_joint_poses: (
            dict[str, Sequence[float]] | Sequence[dict[str, Sequence[float]]]
        ),
        durations: float | Sequence[float],
        *,
        easing: str = "ease_in_out_quad",
        steps_per_segment: int | None = None,
    ) -> None:
        """Move both arms smoothly towards the provided joint targets.

        Args:
            target_joint_poses: A dictionary with "left" and "right" keys (each a
                sequence of 6 or 7 joint values including the gripper) or a
                sequence of such dictionaries to execute multiple segments.
            durations: Duration in seconds for the corresponding target poses.
            easing: Easing profile to apply ("ease_in_out_quad" or "linear").
            steps_per_segment: Optional fixed number of interpolation steps per
                segment. When omitted the controller's ``controller_dt`` is used
                to compute the number of steps from the duration.

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
            ValueError: If inputs are malformed.
        """

        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if isinstance(target_joint_poses, dict):
            trajectory = [target_joint_poses]
        else:
            trajectory = list(target_joint_poses)

        if isinstance(durations, (int, float)):
            segment_durations = [float(durations)]
        else:
            segment_durations = [float(d) for d in durations]

        if len(trajectory) != len(segment_durations):
            raise ValueError(
                "target_joint_poses and durations must have the same length"
            )

        # Determine controller timestep (fallback to 10 ms if unavailable)
        controller_dt = getattr(self.config, "interpolation_controller_dt", 0.01)

        # Fetch the current joint positions as starting state
        def _get_current_state() -> tuple[np.ndarray, np.ndarray]:
            left_state = self.left_arm.get_joint_state()
            right_state = self.right_arm.get_joint_state()
            left = np.concatenate(
                (left_state.pos().copy(), np.array([left_state.gripper_pos]))
            )
            right = np.concatenate(
                (right_state.pos().copy(), np.array([right_state.gripper_pos]))
            )
            return left, right

        current_left, current_right = _get_current_state()

        def _parse_target(
            segment: dict[str, Sequence[float]],
            default_left: np.ndarray,
            default_right: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            if not {"left", "right"}.issubset(segment):
                raise ValueError(
                    "Each segment must contain both 'left' and 'right' targets"
                )

            def _to_array(values: Sequence[float], default: np.ndarray) -> np.ndarray:
                arr = np.asarray(values, dtype=float)
                if arr.shape[0] not in (6, 7):
                    raise ValueError(
                        "Each arm target must provide 6 joint values (+ optional gripper)"
                    )
                if arr.shape[0] == 6:
                    arr = np.concatenate((arr, np.array([default[-1]])))
                return arr

            left_target = _to_array(segment["left"], default_left)
            right_target = _to_array(segment["right"], default_right)
            return left_target, right_target

        def _apply_easing(alpha: float) -> float:
            alpha = max(0.0, min(1.0, alpha))
            if easing == "ease_in_out_quad":
                return self._ease_in_out_quad(alpha)
            if easing == "linear":
                return alpha
            raise ValueError(f"Unsupported easing profile: {easing}")

        try:
            for segment, duration in zip(trajectory, segment_durations, strict=True):
                target_left, target_right = _parse_target(
                    segment, current_left, current_right
                )

                if duration <= 0:
                    action = {}
                    for i in range(6):
                        action[f"left_joint_{i+1}.pos"] = float(target_left[i])
                        action[f"right_joint_{i+1}.pos"] = float(target_right[i])
                    action["left_gripper.pos"] = float(target_left[6])
                    action["right_gripper.pos"] = float(target_right[6])
                    self.send_action(action)
                    current_left, current_right = target_left, target_right
                    continue

                steps = (
                    steps_per_segment
                    if steps_per_segment is not None
                    else max(1, int(math.ceil(duration / controller_dt)))
                )

                for step in range(1, steps + 1):
                    progress = step / steps
                    ratio = _apply_easing(progress)
                    interp_left = current_left + (target_left - current_left) * ratio
                    interp_right = (
                        current_right + (target_right - current_right) * ratio
                    )

                    action = {}
                    for i in range(6):
                        action[f"left_joint_{i+1}.pos"] = float(interp_left[i])
                        action[f"right_joint_{i+1}.pos"] = float(interp_right[i])
                    action["left_gripper.pos"] = float(interp_left[6])
                    action["right_gripper.pos"] = float(interp_right[6])

                    self.send_action(action)
                    time.sleep(duration / steps if steps_per_segment else controller_dt)

                current_left, current_right = target_left, target_right
        except KeyboardInterrupt:
            logger.warning(
                "Joint trajectory interrupted by user. Holding current pose."
            )

    def _disconnect_parallel(self):
        """Disconnect both arms in parallel (reset to home + set to damping)"""
        if self.left_arm is None or self.right_arm is None:
            logger.warning(
                "One or both arms are already None, skipping parallel disconnect"
            )
            return

        def disconnect_left_arm():
            try:
                logger.info("Disconnecting left arm...")

                self.left_arm.reset_to_home()
                self.left_arm.set_to_damping()
                logger.info("✓ Left arm disconnected successfully")
                return "left", None
            except Exception as e:
                logger.warning(f"Left arm disconnected failed: {e}")
                return "left", e

        def disconnect_right_arm():
            try:
                logger.info("Disconnecting right arm...")

                self.right_arm.reset_to_home()
                self.right_arm.set_to_damping()
                logger.info("✓ Right arm disconnected successfully")
                return "right", None
            except Exception as e:
                logger.warning(f"Right arm disconnected failed: {e}")
                return "right", e

        completed_tasks = []
        exceptions = []

        # Use configurable timeout (default: rpc_timeout, fallback: 5.0s)
        disconnect_timeout = getattr(
            self.config, "disconnect_timeout", self.rpc_timeout
        )

        try:
            fL = self._exec_left.submit(disconnect_left_arm)
            fR = self._exec_right.submit(disconnect_right_arm)

            for future in as_completed([fL, fR], timeout=disconnect_timeout):
                side, error = future.result()
                if error is None:
                    completed_tasks.append(side)
                else:
                    exceptions.append(f"{side.capitalize()} arm: {error}")
        except TimeoutError:
            # check if left and right arm are done
            if not fL.done():
                exceptions.append(
                    f"Left arm: timeout after {disconnect_timeout} seconds"
                )

            if not fR.done():
                exceptions.append(
                    f"Right arm: timeout after {disconnect_timeout} seconds"
                )

        if exceptions:
            logger.warning(f"Some disconnect tasks failed: {'; '.join(exceptions)}")
            logger.info(f"Successfully disconnected: {completed_tasks}")
        else:
            logger.info("Both arms disconnected and reset to home successfully")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # ARX5 SDK doesn't have explicit disconnect method
        # Set arms to damping mode for safety before destroying objects in parallel
        try:
            self._disconnect_parallel()
        except Exception as e:
            logger.warning(f"Failed to disconnect arms in parallel: {e}")

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        # Destroy arm objects - this triggers SDK cleanup
        self.left_arm = None
        self.right_arm = None

        # Shutdown thread pool executors
        try:
            logger.info("Shutting down thread pool executors...")
            self._exec_left.shutdown(wait=True)
            self._exec_right.shutdown(wait=True)
            logger.info("✓ Thread pool executors shut down successfully")
        except Exception as e:
            logger.warning(f"Failed to shutdown thread pool executors: {e}")

        self._is_connected = False

        logger.info(f"{self} disconnected.")

    def set_log_level(self, level: str):
        """Set robot log level

        Args:
            level: 日志级别字符串，支持: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL, OFF
        """
        # Convert string to LogLevel enum
        log_level_map = {
            "TRACE": arx5.LogLevel.TRACE,
            "DEBUG": arx5.LogLevel.DEBUG,
            "INFO": arx5.LogLevel.INFO,
            "WARNING": arx5.LogLevel.WARNING,
            "ERROR": arx5.LogLevel.ERROR,
            "CRITICAL": arx5.LogLevel.CRITICAL,
            "OFF": arx5.LogLevel.OFF,
        }

        if level.upper() not in log_level_map:
            raise ValueError(
                f"Invalid log level: {level}. Supported levels: {list(log_level_map.keys())}"
            )

        log_level = log_level_map[level.upper()]

        # Set log level for both arms if connected
        if self.left_arm is not None:
            self.left_arm.set_log_level(log_level)
        if self.right_arm is not None:
            self.right_arm.set_log_level(log_level)

    def _reset_to_home_parallel(self):
        """Reset both arms to home position in parallel"""
        if self.left_arm is None or self.right_arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        fL = self._exec_left.submit(self.left_arm.reset_to_home)
        fR = self._exec_right.submit(self.right_arm.reset_to_home)
        completed_tasks = []
        exceptions = []

        try:
            for future in as_completed([fL, fR], timeout=5.0):
                try:
                    future.result()  # get result
                    if future == fL:
                        completed_tasks.append("left")
                    else:
                        completed_tasks.append("right")
                except Exception as e:
                    if future == fL:
                        exceptions.append(f"Left arm: {e}")
                    else:
                        exceptions.append(f"Right arm: {e}")
        except TimeoutError:
            # check if left and right arm are done
            if not fL.done():
                exceptions.append("Left arm: timeout after 3 seconds")
            if not fR.done():
                exceptions.append("Right arm: timeout after 3 seconds")
        if exceptions:
            logger.warning(f"Some tasks failed: {'; '.join(exceptions)}")
            logger.info(f"Completed tasks: {completed_tasks}")
            raise Exception(f"Reset failed: {'; '.join(exceptions)}")

        logger.info("Both arms reset to home position parallel completed.")

    def reset_to_home(self):
        """Reset both arms to home position"""
        self._reset_to_home_parallel()

    def is_gravity_compensation_mode(self) -> bool:
        """Check if both arms are in gravity compensation mode"""
        if not self.is_connected:
            return False

        try:
            left_gain = self.left_arm.get_gain()
            right_gain = self.right_arm.get_gain()

            # 检查所有增益是否为0
            left_zero = (
                (left_gain.kp() == 0).all()
                and (left_gain.kd() == 0).all()
                and left_gain.gripper_kp == 0.0
                and left_gain.gripper_kd == 0.0
            )
            right_zero = (
                (right_gain.kp() == 0).all()
                and (right_gain.kd() == 0).all()
                and right_gain.gripper_kp == 0.0
                and right_gain.gripper_kd == 0.0
            )

            return left_zero and right_zero
        except Exception:
            return False

    def set_to_normal_position_control(self):
        """Switch from gravity compensation to normal position control mode"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Switching to normal position control mode...")

        # reset to default gain
        left_cfg = self.controller_configs["left_config"]
        right_cfg = self.controller_configs["right_config"]

        left_gain = self.left_arm.get_gain()
        left_gain.kp()[:] = left_cfg.default_kp
        left_gain.kd()[:] = left_cfg.default_kd
        left_gain.gripper_kp = left_cfg.default_gripper_kp
        left_gain.gripper_kd = left_cfg.default_gripper_kd
        self.left_arm.set_gain(left_gain)

        right_gain = self.right_arm.get_gain()
        right_gain.kp()[:] = right_cfg.default_kp
        right_gain.kd()[:] = right_cfg.default_kd
        right_gain.gripper_kp = right_cfg.default_gripper_kp
        right_gain.gripper_kd = right_cfg.default_gripper_kd
        self.right_arm.set_gain(right_gain)

        logger.info("✓ Both arms are now in normal position control mode")
