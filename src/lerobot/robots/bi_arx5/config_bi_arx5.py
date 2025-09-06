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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from ..config import RobotConfig


@RobotConfig.register_subclass("bi_arx5")
@dataclass
class BiARX5Config(RobotConfig):
    left_arm_model: str = "X5"
    left_arm_port: str = "can1"
    right_arm_model: str = "X5"
    right_arm_port: str = "can3"
    log_level: str = "DEBUG"
    use_multithreading: bool = True

    # Optional
    disable_torque_on_disconnect: bool = True
    max_relative_target: int | None = None
    use_degrees: bool = False

    # cameras
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "head": OpenCVCameraConfig(
                index_or_path="/dev/video0",
                fps=10,
                width=640,
                height=480,
            ),
            "left_wrist": OpenCVCameraConfig(
                index_or_path="/dev/video1",
                fps=30,
                width=640,
                height=480,
            ),
            "right_wrist": OpenCVCameraConfig(
                index_or_path="/dev/video2",
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
