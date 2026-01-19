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

from dataclasses import dataclass, field
from enum import Enum

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.xense import XenseCameraConfig, XenseOutputType
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("xense_multisensor")
@dataclass
class XenseMultisensorConfig(RobotConfig):
    """Configuration for Xense Multisensor robot."""

    # Camera configuration
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    def __post_init__(self):
        # Camera configuration based on tactile sensors setting
         self.cameras = {
                "head": RealSenseCameraConfig(
                    serial_number_or_name="230322271365", fps=60, width=640, height=480
                ),
                "left_wrist": RealSenseCameraConfig(
                    serial_number_or_name="230422271416", fps=60, width=640, height=480
                ),
                "right_wrist": RealSenseCameraConfig(
                    serial_number_or_name="230322274234", fps=60, width=640, height=480
                ),
                "right_tactile_0": XenseCameraConfig(
                    serial_number="OG000344",
                    fps=60,
                    output_types=[XenseOutputType.DIFFERENCE],
                    warmup_s=1.0,
                ),
                "left_tactile_0": XenseCameraConfig(
                    serial_number="OG000337",
                    fps=60,
                    output_types=[XenseOutputType.DIFFERENCE],
                    warmup_s=1.0,
                ),
            }
