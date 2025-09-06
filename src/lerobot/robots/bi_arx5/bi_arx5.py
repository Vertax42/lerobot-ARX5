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
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError

from ..robot import Robot
from .config_bi_arx5 import BiARX5Config

# 导入ARX5接口 Stanford-Real-Robot
from .arx5_sdk.python import arx5_interface as arx5

# Native ARX5 python SDK
# from .arx_x5_python.bimanual import BimanualArm as arx5_bimanual

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

        # 初始化时不创建 arm 实例，在 connect 时创建
        self.left_arm = None
        self.right_arm = None
        self._is_connected = False

        # 预先配置arm配置，但不创建实例
        self.left_config = arx5.RobotConfigFactory.get_instance().get_config(
            config.left_arm_model
        )
        self.left_controller_config = (
            arx5.ControllerConfigFactory.get_instance().get_config(
                "joint_controller", self.left_config.joint_dof
            )
        )

        self.right_config = arx5.RobotConfigFactory.get_instance().get_config(
            config.right_arm_model
        )
        self.right_controller_config = (
            arx5.ControllerConfigFactory.get_instance().get_config(
                "joint_controller", self.right_config.joint_dof
            )
        )

        # use multithreading by default
        if config.use_multithreading:
            self.left_controller_config.background_send_recv = True
            self.right_controller_config.background_send_recv = True
        else:
            self.left_controller_config.background_send_recv = False
            self.right_controller_config.background_send_recv = False

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
        try:
            # try to get joint state
            self.left_arm.get_joint_state()
            self.right_arm.get_joint_state()

            return (
                self._is_connected
                and self.left_arm is not None
                and self.right_arm is not None
                and all(cam.is_connected for cam in self.cameras.values())
            )
        except Exception:
            return False

    def connect(self, calibrate: bool = True) -> None:
        # create arm instance
        self.left_arm = arx5.Arx5JointController(
            self.left_config, self.left_controller_config, self.config.left_arm_port
        )
        self.right_arm = arx5.Arx5JointController(
            self.right_config, self.right_controller_config, self.config.right_arm_port
        )

        # set log lever
        self.set_log_level(self.config.log_level)

        # 如果需要校准，执行回零
        if calibrate:
            self.reset_to_home()

        # 连接摄像头
        for cam in self.cameras.values():
            cam.connect()

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
        if self.left_arm is None or self.right_arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Add "left_" prefix - get joint state from left arm
        left_joint_state = self.left_arm.get_joint_state()
        left_pos = left_joint_state.pos().copy()  # numpy array of joint positions (deep copy)

        # Create left arm observations with joint names matching _motors_ft
        for i in range(6):  # 6 joints
            obs_dict[f"left_joint_{i+1}.pos"] = float(left_pos[i])
        obs_dict["left_gripper.pos"] = float(left_joint_state.gripper_pos)

        # Add "right_" prefix - get joint state from right arm
        right_joint_state = self.right_arm.get_joint_state()
        right_pos = right_joint_state.pos().copy()  # numpy array of joint positions (deep copy)

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
        if self.left_arm is None or self.right_arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Create JointState objects for both arms
        left_cmd = arx5.JointState(self.left_config.joint_dof)
        right_cmd = arx5.JointState(self.right_config.joint_dof)

        # Extract left arm joint positions
        left_positions = left_cmd.pos()
        for i in range(6):
            joint_key = f"left_joint_{i+1}.pos"
            if joint_key in action:
                left_positions[i] = action[joint_key]

        # Extract left arm gripper position
        if "left_gripper.pos" in action:
            left_cmd.gripper_pos = action["left_gripper.pos"]

        # Extract right arm joint positions
        right_positions = right_cmd.pos()
        for i in range(6):
            joint_key = f"right_joint_{i+1}.pos"
            if joint_key in action:
                right_positions[i] = action[joint_key]

        # Extract right arm gripper position
        if "right_gripper.pos" in action:
            right_cmd.gripper_pos = action["right_gripper.pos"]

        # Send commands to both arms
        self.left_arm.set_joint_cmd(left_cmd)
        self.right_arm.set_joint_cmd(right_cmd)

        # Return the commands that were actually sent
        sent_action = {}

        # Add left arm commands to return dict
        for i in range(6):
            sent_action[f"left_joint_{i+1}.pos"] = float(left_positions[i])
        sent_action["left_gripper.pos"] = float(left_cmd.gripper_pos)

        # Add right arm commands to return dict
        for i in range(6):
            sent_action[f"right_joint_{i+1}.pos"] = float(right_positions[i])
        sent_action["right_gripper.pos"] = float(right_cmd.gripper_pos)

        return sent_action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # ARX5 SDK doesn't have explicit disconnect method
        # Set arms to damping mode for safety before destroying objects
        try:
            if self.left_arm is not None:
                self.left_arm.set_to_damping()
            if self.right_arm is not None:
                self.right_arm.set_to_damping()
        except Exception as e:
            logger.warning(f"Failed to set arms to damping mode: {e}")

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        # Destroy arm objects - this triggers SDK cleanup
        self.left_arm = None
        self.right_arm = None
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

    def reset_to_home(self):
        """Reset both arms to home position"""
        if self.left_arm is None or self.right_arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            self.left_arm.reset_to_home()
            self.right_arm.reset_to_home()
            logger.info(f"{self} reset to home position")
        except Exception as e:
            logger.error(f"Failed to reset to home: {e}")
            raise e
