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

"""
TRLC Leader Teleoperator for LeRobot.

This teleoperator uses a 7-DOF Dynamixel-based leader arm (6 joints + gripper)
to provide joint position actions for teleoperation.

Action features (7D):
- joint_1.pos ~ joint_6.pos: Arm joint positions in radians
- gripper.pos: Gripper position normalized to [0=open, 1=closed]
"""

import logging
import time

import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus, OperatingMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .configuration_trlc_leader import TRLCLeaderConfig

logger = logging.getLogger(__name__)


class TRLCLeader(Teleoperator):
    """
    TRLC Leader arm teleoperator.

    Uses 7 Dynamixel xl330-m077 motors (6 arm joints + 1 gripper) to capture
    the operator's arm pose and gripper state for robot teleoperation.

    The gripper motor runs in current-position mode and its raw encoder reading
    is normalized to [0, 1] (0 = fully open, 1 = fully closed).
    """

    config_class = TRLCLeaderConfig
    name = "trlc_leader"

    def __init__(self, config: TRLCLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "joint_1": Motor(1, "xl330-m077", MotorNormMode.DEGREES),
                "joint_2": Motor(2, "xl330-m288", MotorNormMode.DEGREES),
                "joint_3": Motor(3, "xl330-m077", MotorNormMode.DEGREES),
                "joint_4": Motor(4, "xl330-m077", MotorNormMode.DEGREES),
                "joint_5": Motor(5, "xl330-m077", MotorNormMode.DEGREES),
                "joint_6": Motor(6, "xl330-m077", MotorNormMode.DEGREES),
                "gripper": Motor(7, "xl330-m077", MotorNormMode.DEGREES),
            },
        )
        # self.bus.default_baudrate = self.config.baudrate

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = False) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        self.configure()
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()

        # Set gripper to current-position mode and move to open position
        self.bus.write("Torque_Enable", "gripper", 0, normalize=False)
        self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value, normalize=False)
        self.bus.write("Current_Limit", "gripper", 100, normalize=False)
        self.bus.write("Torque_Enable", "gripper", 1, normalize=False)
        self.bus.write("Goal_Position", "gripper", self.config.gripper_open_pos, normalize=False)

    def setup_motors(self) -> None:
        for motor in self.bus.motors:
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        raw = self.bus.sync_read(normalize=False, data_name="Present_Position")

        action = {}
        for motor, val in raw.items():
            if motor == "gripper":
                gripper_range = self.config.gripper_open_pos - self.config.gripper_closed_pos
                action["gripper.pos"] = 1.0 - (val - self.config.gripper_closed_pos) / gripper_range
            else:
                # Convert raw encoder (0–4096) to radians in [-π, π]
                action[f"{motor}.pos"] = val / 4096 * 2 * np.pi - np.pi

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO: Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
