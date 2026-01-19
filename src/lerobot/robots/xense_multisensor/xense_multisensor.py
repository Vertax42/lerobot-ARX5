#!/usr/bin/env python

# Copyright 2025 The XenseRobotics Inc. team. All rights reserved.
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

import math
import os
import sys
import time
from collections.abc import Sequence
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.xense_multisensor.config_xense_multisensor import XenseMultisensorConfig
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import get_logger


class XenseMultisensor(Robot):
    """
    [Xense Multisensor]

    Xense Multisensor Robot with support for Joint and Cartesian control modes.
    """

    config_class = XenseMultisensorConfig
    name = "xense_multisensor"

    def __init__(self, config: XenseMultisensorConfig):
        super().__init__(config)
        self.config = config

        # Logger
        self.logger = get_logger("XenseMultisensor")

        # Connection state
        self._is_connected = False

        self.cameras = make_cameras_from_configs(config.cameras)
        np.set_printoptions(precision=3, suppress=True)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Return motor features based on control mode."""
        return {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        print(f"camera_features: {self._cameras_ft}")
        return {**self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Xense Multisensor is a pure observation device with no actions."""
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = False, go_to_start: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} already connected, do not run `robot.connect()` twice."
            )

        try:
            for cam in self.cameras.values():
                cam.connect()

            self._is_connected = True
            self.logger.info("✅ Xense Multisensor Robot connected.")
        except Exception as e:
            self.logger.error(f"Failed to connect Xense Multisensor Robot: {e}")
            raise e

    @property
    def is_calibrated(self) -> bool:
        """Xense Multisensor does not need to calibrate in runtime"""
        self.logger.info("Xense Multisensor does not need to calibrate in runtime, skip...")
        return True

    def calibrate(self) -> None:
        """Xense Multisensor does not need to calibrate in runtime"""
        self.logger.info("Xense Multisensor does not need to calibrate in runtime, skip...")
        return

    def configure(self) -> None:
        """Configure the robot"""
        self.logger.info("Xense Multisensor does not need to configure in runtime, skip...")
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        obs = {}
        for cam_key, cam in self.cameras.items():
            image = cam.async_read()
            obs[cam_key] = image
        return obs

    def send_action(self, action: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        No need to send action to Xense Multisensor, it is a pure observation device.
        """
        # No need to send action to Xense Multisensor, it is a pure observation device.
        return {}

    def disconnect(self) -> None:
        """Disconnect from the Xense Multisensor device."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        self.logger.info("Disconnecting from Xense Multisensor...")

        # Disconnect all cameras
        for cam_key, cam in self.cameras.items():
            try:
                if cam.is_connected:
                    cam.disconnect()
                    self.logger.info(f"  Disconnected camera: {cam_key}")
            except Exception as e:
                self.logger.error(f"  Error disconnecting camera {cam_key}: {e}")

        self._is_connected = False
        self.logger.info("✅ Xense Multisensor disconnected")