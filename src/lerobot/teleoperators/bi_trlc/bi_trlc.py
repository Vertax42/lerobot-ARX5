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
Bilateral TRLC Teleoperator for LeRobot.

This teleoperator manages two independent TRLC Leader arms (left + right),
each with 7-DOF Dynamixel motors (6 joints + gripper), to provide combined
joint position actions for bilateral teleoperation.

Action features (14D):
- left_joint_1.pos ~ left_joint_6.pos: Left arm joint positions in radians
- left_gripper.pos: Left gripper position normalized to [0=open, 1=closed]
- right_joint_1.pos ~ right_joint_6.pos: Right arm joint positions in radians
- right_gripper.pos: Right gripper position normalized to [0=open, 1=closed]
"""

import logging
import math
import time

import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus, OperatingMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .configuration_bi_trlc import BiTRLCConfig

logger = logging.getLogger(__name__)

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


class BiTRLC(Teleoperator):
    """
    Bilateral TRLC teleoperator (two arms: left + right).

    Manages two independent Dynamixel motor buses. Each arm has 7 motors
    (6 arm joints + 1 gripper). On connect(), both arms are calibrated by
    prompting the user to move each arm to its configured `start_joints` pose.

    Per-joint assembly offsets are computed so that:
        joint_angle = sign * (encoder / 4096 * 2π  -  offset)

    Each gripper runs in current-position mode; its raw encoder reading is
    normalized to [0, 1] (0 = fully open, 1 = fully closed).

    The combined 14-D action dict uses the prefixes ``left_`` and ``right_``
    for each arm's motors.
    """

    config_class = BiTRLCConfig
    name = "bi_trlc"

    def __init__(self, config: BiTRLCConfig):
        super().__init__(config)
        self.config = config

        self.left_bus = DynamixelMotorsBus(
            port=self.config.left_port,
            motors={
                "joint_1": Motor(1, "xl330-m288", MotorNormMode.DEGREES),
                "joint_2": Motor(2, "xl330-m288", MotorNormMode.DEGREES),
                "joint_3": Motor(3, "xl330-m288", MotorNormMode.DEGREES),
                "joint_4": Motor(4, "xl330-m288", MotorNormMode.DEGREES),
                "joint_5": Motor(5, "xl330-m288", MotorNormMode.DEGREES),
                "joint_6": Motor(6, "xl330-m288", MotorNormMode.DEGREES),
                "gripper": Motor(7, "xl330-m288", MotorNormMode.DEGREES),
            },
        )
        self.right_bus = DynamixelMotorsBus(
            port=self.config.right_port,
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

        self._left_joint_offsets: list[float] | None = None
        self._right_joint_offsets: list[float] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for motor in self.left_bus.motors:
            features[f"left_{motor}.pos"] = float
        for motor in self.right_bus.motors:
            features[f"right_{motor}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_bus.is_connected and self.right_bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return (
            self._left_joint_offsets is not None
            and self._right_joint_offsets is not None
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.left_bus.connect()
        self.right_bus.connect()
        self.configure()
        if calibrate:
            self.calibrate()
        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._left_joint_offsets = None
        self._right_joint_offsets = None
        self.left_bus.disconnect()
        self.right_bus.disconnect()
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------
    # Configuration & calibration
    # ------------------------------------------------------------------

    def configure(self) -> None:
        """Disable torque, configure motors, and set gripper mode for both arms."""
        for side, bus, cfg_open in [
            ("left", self.left_bus, self.config.left_gripper_open_pos),
            ("right", self.right_bus, self.config.right_gripper_open_pos),
        ]:
            bus.disable_torque()
            bus.configure_motors()

            # Set gripper to current-position mode and move to open position
            bus.write("Torque_Enable", "gripper", 0, normalize=False)
            bus.write(
                "Operating_Mode",
                "gripper",
                OperatingMode.CURRENT_POSITION.value,
                normalize=False,
            )
            bus.write("Current_Limit", "gripper", 100, normalize=False)
            bus.write("Torque_Enable", "gripper", 1, normalize=False)
            bus.write("Goal_Position", "gripper", cfg_open, normalize=False)
            logger.debug(f"{self} configured {side} arm.")

    def calibrate(self) -> None:
        """
        Prompt the user to move each arm to its `start_joints` pose, then
        compute per-joint assembly offsets for both arms:

            offset = raw_rad - sign * target_angle
        """
        self._left_joint_offsets = self._calibrate_arm(
            bus=self.left_bus,
            side="LEFT",
            target=self.config.left_start_joints,
            signs=self.config.left_joint_signs,
        )
        self._right_joint_offsets = self._calibrate_arm(
            bus=self.right_bus,
            side="RIGHT",
            target=self.config.right_start_joints,
            signs=self.config.right_joint_signs,
        )

    def _calibrate_arm(
        self,
        bus: DynamixelMotorsBus,
        side: str,
        target: list[float],
        signs: list[int],
    ) -> list[float]:
        """Calibrate one arm and return its per-joint offsets."""
        target_str = "  ".join(
            f"joint_{i + 1}: {math.degrees(v):+.1f}°" for i, v in enumerate(target)
        )
        print(f"\n[BiTRLC Calibration] {side} arm — move to start pose:")
        print(f"  {target_str}")
        time.sleep(2)

        # Average 10 readings to reduce noise
        readings = [
            bus.sync_read(normalize=False, data_name="Present_Position")
            for _ in range(10)
        ]

        offsets = []
        for i, motor in enumerate(ARM_JOINTS):
            raw_rad = np.mean([r[motor] for r in readings]) / 4096 * 2 * math.pi
            offsets.append(raw_rad - signs[i] * target[i])

        logger.info(
            f"{self} {side} arm calibrated: offsets={[f'{o:.4f}' for o in offsets]}"
        )
        return offsets

    def setup_motors(self) -> None:
        """Interactive per-motor ID assignment for both arms."""
        for side, bus in [("left", self.left_bus), ("right", self.right_bus)]:
            print(f"\n[BiTRLC] Setting up {side} arm motors:")
            for motor in bus.motors:
                input(
                    f"  Connect the controller board to the '{motor}' motor "
                    f"of the {side} arm only and press enter."
                )
                bus.setup_motor(motor)
                print(f"  '{motor}' ({side}) motor id set to {bus.motors[motor].id}")

    # ------------------------------------------------------------------
    # Teleoperation
    # ------------------------------------------------------------------

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        left_raw = self.left_bus.sync_read(
            normalize=False, data_name="Present_Position", num_retry=2
        )
        right_raw = self.right_bus.sync_read(
            normalize=False, data_name="Present_Position", num_retry=2
        )

        action: dict[str, float] = {}
        action.update(
            self._decode_arm(
                left_raw,
                "left",
                self.config.left_joint_signs,
                self._left_joint_offsets,
                self.config.left_gripper_open_pos,
                self.config.left_gripper_closed_pos,
            )
        )
        action.update(
            self._decode_arm(
                right_raw,
                "right",
                self.config.right_joint_signs,
                self._right_joint_offsets,
                self.config.right_gripper_open_pos,
                self.config.right_gripper_closed_pos,
            )
        )

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def _decode_arm(
        self,
        raw: dict[str, int],
        side: str,
        signs: list[int],
        offsets: list[float],
        gripper_open_pos: int,
        gripper_closed_pos: int,
    ) -> dict[str, float]:
        """Convert raw encoder readings for one arm into action entries."""
        result: dict[str, float] = {}
        gripper_range = gripper_open_pos - gripper_closed_pos

        for motor, val in raw.items():
            key = f"{side}_{motor}.pos"
            if motor == "gripper":
                result[key] = float(
                    np.clip(
                        1.0 - (val - gripper_closed_pos) / gripper_range,
                        0.0,
                        1.0,
                    )
                )
            else:
                j = ARM_JOINTS.index(motor)
                raw_rad = val / 4096 * 2 * math.pi
                result[key] = signs[j] * (raw_rad - offsets[j])

        return result

    def send_feedback(self, feedback: dict[str, float]) -> None:  # noqa: ARG002
        # TODO: Implement force feedback
        raise NotImplementedError
