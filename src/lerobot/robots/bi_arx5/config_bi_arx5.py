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
from lerobot.cameras.realsense import RealSenseCameraConfig
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
    rpc_timeout: float = 10.0
    controller_dt: float = 0.01  # 100Hz
    interpolation_controller_dt: float = 0.01
    inference_mode: bool = False
    default_preview_time: float = (
        0.015
        if inference_mode
        else 0.0  # For recording mode (0.0), for inference mode (0.015)
    )
    home_position: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    start_position: list[float] = field(
        default_factory=lambda: [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.002]
    )
    # cameras
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "head": RealSenseCameraConfig(
                "230322271365", fps=30, width=640, height=480
            ),
            "left_wrist": RealSenseCameraConfig(
                "230422271416", fps=30, width=640, height=480
            ),
            "right_wrist": RealSenseCameraConfig(
                "230322274234", fps=30, width=640, height=480
            ),
        }
    )
