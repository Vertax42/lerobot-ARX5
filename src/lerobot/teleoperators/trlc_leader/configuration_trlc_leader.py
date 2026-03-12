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
Configuration for TRLC Leader Teleoperator.

The TRLC Leader arm uses Dynamixel xl330-m077 motors for:
- 6 joint motors for arm control
- 1 gripper motor with current-position mode
"""

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("trlc_leader")
@dataclass
class TRLCLeaderConfig(TeleoperatorConfig):
    """Configuration for the TRLC Leader arm teleoperator.

    The TRLC Leader uses Dynamixel xl330-m077 motors in a 7-DOF configuration
    (6 arm joints + 1 gripper). The gripper uses current-position control mode
    and its raw encoder value is normalized to [0, 1].

    Calibration (runs automatically on every connect):
        On startup the user is prompted to move the arm to `start_joints`.
        The script reads the encoder values and computes per-joint offsets so
        that:  joint_angle = sign * (encoder/4096*2π - offset)

        joint_signs: Direction multipliers ±1 per joint — set once based on
            how each motor is physically mounted (does not change between runs).
        start_joints: The known arm pose (radians) the user moves to at startup
            for calibration.  Default is all-zeros.

    Attributes:
        port: Serial port for Dynamixel motor bus (e.g. "/dev/ttyUSB0")
        baudrate: Baud rate for the Dynamixel bus
        joint_signs: Direction signs (±1) for joints 1-6
        start_joints: Known reference pose in radians used for calibration
        gripper_open_pos: Raw encoder value when gripper is fully open
        gripper_closed_pos: Raw encoder value when gripper is fully closed
    """

    id: str = "trlc_leader"

    port: str = "/dev/ttyACM0"
    joint_signs: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    start_joints: list[float] = field(default_factory=lambda: [0.0] * 6)
    gripper_open_pos: int = 3287
    gripper_closed_pos: int = 2682
