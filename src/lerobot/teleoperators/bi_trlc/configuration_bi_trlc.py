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
Configuration for Bilateral TRLC Teleoperator.

The Bilateral TRLC uses two independent TRLC Leader arms (left + right),
each with 6 arm joints + 1 gripper motor (Dynamixel xl330-m077/xl330-m288).
"""

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_trlc")
@dataclass
class BiTRLCConfig(TeleoperatorConfig):
    """Configuration for the Bilateral TRLC teleoperator (two arms).

    Manages two independent TRLC Leader arms (left and right), each in a
    7-DOF configuration (6 arm joints + 1 gripper). Each arm's gripper uses
    current-position control mode and its raw encoder value is normalized to
    [0, 1].

    Calibration (runs automatically on every connect):
        On startup the user is prompted to move each arm to its `start_joints`.
        Per-joint assembly offsets are computed so that:
            joint_angle = sign * (encoder/4096*2π - offset)

        joint_signs: Direction multipliers ±1 per joint — set once based on
            how each motor is physically mounted (does not change between runs).
        start_joints: The known arm pose (radians) the user moves to at startup
            for calibration.  Default is all-zeros.

    Attributes:
        left_port: Serial port for the left arm Dynamixel motor bus
        right_port: Serial port for the right arm Dynamixel motor bus
        left_joint_signs: Direction signs (±1) for left arm joints 1-6
        right_joint_signs: Direction signs (±1) for right arm joints 1-6
        left_start_joints: Reference pose in radians for left arm calibration
        right_start_joints: Reference pose in radians for right arm calibration
        left_gripper_open_pos: Raw encoder value when left gripper is fully open
        left_gripper_closed_pos: Raw encoder value when left gripper is fully closed
        right_gripper_open_pos: Raw encoder value when right gripper is fully open
        right_gripper_closed_pos: Raw encoder value when right gripper is fully closed
    """

    id: str = "bi_trlc"

    left_port: str = "/dev/ttyTRLC_left"
    right_port: str = "/dev/ttyTRLC_right"

    left_joint_signs: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    right_joint_signs: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])

    left_start_joints: list[float] = field(default_factory=lambda: [0.0] * 6)
    right_start_joints: list[float] = field(default_factory=lambda: [0.0] * 6)

    left_gripper_open_pos: int = 2272
    left_gripper_closed_pos: int = 1647
    right_gripper_open_pos: int = 2282
    right_gripper_closed_pos: int = 1667
