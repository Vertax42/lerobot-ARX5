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
                "OS000097": XenseCameraConfig(
                    serial_number="OS000097",
                    fps=30,
                    output_types=[XenseOutputType.RECTIFY],
                    warmup_s=1.0,
                ),
                "OS000115": XenseCameraConfig(
                    serial_number="OS000115",
                    fps=30,
                    output_types=[XenseOutputType.RECTIFY],
                    warmup_s=1.0,
                    use_gpu=False,
                ),
                # "OG000456": XenseCameraConfig(
                #     serial_number="OG000456",
                #     fps=30,
                #     output_types=[XenseOutputType.RECTIFY],
                #     warmup_s=1.0,
                #     use_gpu=False,
                # ),
            }
