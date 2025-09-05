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

from ..robot import Robot
from .config_bi_arx5 import BiARX5Config

# 导入ARX5接口
from .arx5_sdk.python import arx5_interface as arx5

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

        # left arm config
        self.left_config = arx5.RobotConfigFactory.get_instance().get_config(
            config.left_arm_model
        )
        self.left_controller_config = (
            arx5.ControllerConfigFactory.get_instance().get_config(
                "joint_controller", self.left_config.joint_dof
            )
        )
        # right arm config
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

        self.left_arm = arx5.Arx5JointController(
            self.left_config, self.left_controller_config, config.left_arm_port
        )
        self.right_arm = arx5.Arx5JointController(
            self.right_config, self.right_controller_config, config.right_arm_port
        )

        self.cameras = make_cameras_from_configs(config.cameras)
        np.set_printoptions(precision=3, suppress=True)
        self.set_log_level(config.log_level)
        self.reset_to_home()

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
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
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Add "left_" prefix
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # Add "right_" prefix
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value
            for key, value in action.items()
            if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value
            for key, value in action.items()
            if key.startswith("right_")
        }

        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        # Add prefixes back
        prefixed_send_action_left = {
            f"left_{key}": value for key, value in send_action_left.items()
        }
        prefixed_send_action_right = {
            f"right_{key}": value for key, value in send_action_right.items()
        }

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

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

        # Set log level for both arms
        self.left_arm.set_log_level(log_level)
        self.right_arm.set_log_level(log_level)

    def reset_to_home(self):
        """Reset both arms to home position"""
        try:
            self.left_arm.reset_to_home()
            self.right_arm.reset_to_home()
        except Exception as e:
            logger.error(f"Failed to reset to home: {e}")
            raise e
