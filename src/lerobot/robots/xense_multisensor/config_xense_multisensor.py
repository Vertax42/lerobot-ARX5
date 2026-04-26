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

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.xense import XenseOutputType, XenseTactileCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.bi_flexiv_rizon4_rt.config_serial_gripper import SerialGripperConfig


@RobotConfig.register_subclass("xense_multisensor")
@dataclass
class XenseMultisensorConfig(RobotConfig):
    """Configuration for Xense Multisensor robot.

    This robot exposes the bi_flexiv_rizon4_rt camera suite plus two serial
    grippers, but does not connect to any robot arms.
    """

    left_use_gripper: bool = True
    left_gripper_sn: str = "000009"
    left_gripper_baudrate: int = 115200
    left_gripper_serial_timeout: float = 1.0
    left_gripper_min_pos: float = 0.0
    left_gripper_max_pos: float = 85.0
    left_gripper_v_max: float = 100.0
    left_gripper_f_max: float = 40.0
    left_gripper_init_open: bool = True

    right_use_gripper: bool = True
    right_gripper_sn: str = "000004"
    right_gripper_baudrate: int = 115200
    right_gripper_serial_timeout: float = 1.0
    right_gripper_min_pos: float = 0.0
    right_gripper_max_pos: float = 85.0
    right_gripper_v_max: float = 100.0
    right_gripper_f_max: float = 40.0
    right_gripper_init_open: bool = True

    left_gripper: SerialGripperConfig | None = field(default=None, init=False)
    right_gripper: SerialGripperConfig | None = field(default=None, init=False)

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "head": RealSenseCameraConfig(
                serial_number_or_name="135522074323",
                fps=30,
                width=640,
                height=480,
                warmup_s=1.0,
            ),
            "left_wrist": OpenCVCameraConfig(
                index_or_path="XC000003",
                fourcc="MJPG",
                width=640,
                height=480,
                fps=30,
                warmup_s=1.0,
            ),
            "right_wrist": OpenCVCameraConfig(
                index_or_path="XC000004",
                fourcc="MJPG",
                width=640,
                height=480,
                fps=30,
                warmup_s=1.0,
            ),
            "left_tactile_0": XenseTactileCameraConfig(
                serial_number="OG000867",
                fps=30,
                output_types=[XenseOutputType.RECTIFY],
                warmup_s=0.05,
            ),
            "left_tactile_1": XenseTactileCameraConfig(
                serial_number="OG000865",
                fps=30,
                output_types=[XenseOutputType.RECTIFY],
                warmup_s=0.05,
            ),
            "right_tactile_0": XenseTactileCameraConfig(
                serial_number="OG000142",
                fps=30,
                output_types=[XenseOutputType.RECTIFY],
                warmup_s=0.05,
            ),
            "right_tactile_1": XenseTactileCameraConfig(
                serial_number="OG000866",
                fps=30,
                output_types=[XenseOutputType.RECTIFY],
                warmup_s=0.05,
            ),
        }
    )

    def __post_init__(self):
        super().__post_init__()

        if self.left_use_gripper:
            self.left_gripper = SerialGripperConfig(
                sn=self.left_gripper_sn,
                baudrate=self.left_gripper_baudrate,
                serial_timeout=self.left_gripper_serial_timeout,
                gripper_min_pos=self.left_gripper_min_pos,
                gripper_max_pos=self.left_gripper_max_pos,
                gripper_v_max=self.left_gripper_v_max,
                gripper_f_max=self.left_gripper_f_max,
                init_open=self.left_gripper_init_open,
            )
        else:
            self.left_gripper = None

        if self.right_use_gripper:
            self.right_gripper = SerialGripperConfig(
                sn=self.right_gripper_sn,
                baudrate=self.right_gripper_baudrate,
                serial_timeout=self.right_gripper_serial_timeout,
                gripper_min_pos=self.right_gripper_min_pos,
                gripper_max_pos=self.right_gripper_max_pos,
                gripper_v_max=self.right_gripper_v_max,
                gripper_f_max=self.right_gripper_f_max,
                init_open=self.right_gripper_init_open,
            )
        else:
            self.right_gripper = None
