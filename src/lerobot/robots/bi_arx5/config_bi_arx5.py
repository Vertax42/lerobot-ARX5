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
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.xense import XenseCameraConfig, XenseOutputType

# from lerobot.cameras.realsense import RealSenseCameraConfig
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
    # Preview time in seconds for action interpolation during inference
    # Higher values (0.03-0.05) provide smoother motion but more delay
    # Lower values (0.01-0.02) are more responsive but may cause jittering
    preview_time: float = 0.03  # Default 30ms for smooth inference
    gripper_open_readout: list[float] = field(default_factory=lambda: [-3.4, -3.4])
    home_position: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    start_position: list[float] = field(
        default_factory=lambda: [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.0]
    )
    # cameras
    # server_camera settings
    # cameras: dict[str, CameraConfig] = field(
    #     default_factory=lambda: {
    #         "head": OpenCVCameraConfig(
    #             index_or_path="/dev/video16", fps=60, width=640, height=480
    #         ),
    #         "left_wrist": OpenCVCameraConfig(
    #             index_or_path="/dev/video4", fps=60, width=640, height=480
    #         ),
    #         "right_wrist": OpenCVCameraConfig(
    #             index_or_path="/dev/video10", fps=60, width=640, height=480
    #         ),
    #     }
    # )
    # cameras: dict[str, CameraConfig] = field(
    #     default_factory=lambda: {
    #         "head": RealSenseCameraConfig(
    #             serial_number_or_name="230322271365", fps=30, width=640, height=480
    #         ),
    #         "left_wrist": RealSenseCameraConfig(
    #             serial_number_or_name="230422271416", fps=30, width=640, height=480
    #         ),
    #         "right_wrist": RealSenseCameraConfig(
    #             serial_number_or_name="230322274234", fps=30, width=640, height=480
    #         ),
    #         # "right_tactile_0": XenseCameraConfig(
    #         #     serial_number="OG000344",
    #         #     fps=30,  # Reduced from 60 to reduce loop overhead
    #         #     output_types=[XenseOutputType.DIFFERENCE],
    #         #     warmup_s=1.0,  # Increased warmup time for stable initialization
    #         # ),
    #         # "right_tactile_1": XenseCameraConfig(
    #         #     serial_number="OG000352",
    #         #     fps=30,  # Reduced from 60 to reduce loop overhead
    #         #     output_types=[XenseOutputType.DIFFERENCE],
    #         #     warmup_s=1.0,  # Increased warmup time for stable initialization
    #         # ),
    #     }
    # )
    # notebook_camera settings
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "head": OpenCVCameraConfig(
                index_or_path="/dev/video18", fps=30, width=640, height=480
            ),
            "left_wrist": OpenCVCameraConfig(
                index_or_path="/dev/video4", fps=30, width=640, height=480
            ),
            "right_wrist": OpenCVCameraConfig(
                index_or_path="/dev/video14", fps=30, width=640, height=480
            ),
            # "right_tactile_left": XenseCameraConfig(
            #     serial_number="OG000352",
            #     fps=30,  # Reduced from 60 to reduce loop overhead
            #     output_types=[XenseOutputType.DIFFERENCE],
            #     warmup_s=0.5,  # Increased warmup time for stable initialization
            # ),
            # "right_tactile_right": XenseCameraConfig(
            #     serial_number="OG000344",
            #     fps=30,  # Reduced from 60 to reduce loop overhead
            #     output_types=[XenseOutputType.DIFFERENCE],
            #     warmup_s=0.5,  # Increased warmup time for stable initialization
            # ),
            # "left_tactile_left": XenseCameraConfig(
            #     serial_number="OG000337",
            #     fps=30,  # Reduced from 60 to reduce loop overhead
            #     output_types=[XenseOutputType.DIFFERENCE],
            #     warmup_s=0.5,  # Increased warmup time for stable initialization
            # ),
            # "left_tactile_right": XenseCameraConfig(
            #     serial_number="OG000339",
            #     fps=30,  # Reduced from 60 to reduce loop overhead
            #     output_types=[XenseOutputType.DIFFERENCE],
            #     warmup_s=0.5,  # Increased warmup time for stable initialization
            # ),
        }
    )
