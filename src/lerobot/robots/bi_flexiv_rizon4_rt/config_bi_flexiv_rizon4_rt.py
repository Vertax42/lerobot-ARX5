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

"""Configuration for BiFlexivRizon4RT dual-arm robot (real-time via flexiv_rt)."""

from dataclasses import dataclass, field

import flexiv_rt

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.xense import XenseOutputType, XenseTactileCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.bi_flexiv_rizon4_rt.config_serial_gripper import SerialGripperConfig


@RobotConfig.register_subclass("bi_flexiv_rizon4_rt")
@dataclass
class BiFlexivRizon4RTConfig(RobotConfig):
    """Configuration for BiFlexivRizon4RT dual-arm robot with real-time control.

    Each arm has its own flexiv_rt.Robot and RT thread running at 1 kHz.
    Python-side send_action() (30-100 Hz) writes target poses to shared memory
    for each arm independently.

    Action/Observation keys are prefixed with "left_" or "right_":
        left_tcp.{x,y,z,r1-r6}, left_gripper.pos
        right_tcp.{x,y,z,r1-r6}, right_gripper.pos

    Attributes:
        left_robot_sn: Serial number of the left arm robot
        right_robot_sn: Serial number of the right arm robot
        bi_mount_type: Preset layout for robot/gripper/camera SNs and home/start poses ("forward" or "side")
        use_force: Enable force control axes (both arms)
        control_frequency: Python-side control loop frequency in Hz
        go_to_start: Move to start positions after connect
        left_start_position_degree: Left arm joint positions in degrees for start pose
        right_start_position_degree: Right arm joint positions in degrees for start pose
        start_vel_scale: Joint velocity scale for MoveJ (1-100)
        zero_ft_sensor_on_connect: Zero force-torque sensors on connect (both arms)
        stiffness_ratio: Multiplies nominal Cartesian stiffness K_x_nom
        damping_ratio: Cartesian damping ratio per axis (6D)
        force_control_frame: Reference frame for force control
        force_control_axis: Which axes to enable force control [x,y,z,rx,ry,rz]
        max_contact_wrench: Maximum contact wrench [fx,fy,fz,mx,my,mz]
        target_wrench: Default target wrench for force control
        ext_force_threshold: External TCP force threshold for collision detection [N]
        ext_torque_threshold: External joint torque threshold for collision detection [Nm]
    """

    # Robot identification
    left_robot_sn: str = "Rizon4s-063458"
    right_robot_sn: str = "Rizon4s-063670"
    bi_mount_type: str = "forward"
    # Force control
    use_force: bool = False

    # Python-side frequency (RT thread is always 1 kHz internally)
    control_frequency: float = 100.0  # Hz

    # Connection behavior
    go_to_start: bool = True

    # Camera configurations (external cameras)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    enable_tactile_sensors: bool = True

    # Cartesian impedance (shared for both arms)
    stiffness_ratio: float = 0.2
    damping_ratio: list[float] = field(default_factory=lambda: [0.7] * 6)

    # Force control settings (shared for both arms)
    force_control_frame: flexiv_rt.CoordType = flexiv_rt.CoordType.WORLD
    force_control_axis: list[bool] = field(
        default_factory=lambda: [False, False, False, False, False, False]
    )
    max_contact_wrench: list[float] = field(
        default_factory=lambda: [30.0, 30.0, 30.0, 5.0, 5.0, 5.0]
    )
    target_wrench: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    # Collision detection thresholds
    ext_force_threshold: float = 10.0  # N
    ext_torque_threshold: float = 5.0  # Nm

    # Start position parameters (left arm)
    left_start_position_degree: list[float] = field(
        default_factory=lambda: [88.79, 74.96, 22.75, 112.75, -0.39, 86.74, 1.24]
    )
    # left_TCP : x, y ,z, r, p, y = [955, 150, -110, 70, -170, 50]

    # Start position parameters (right arm)
    right_start_position_degree: list[float] = field(
        default_factory=lambda: [-24.41, 71.36, -4.67, 118.53, 3.91, 96.15, 3.60]
    )

    # right_TCP : x, y ,z, r, p, y = [955, -150, -110, -70, 170, -50]
    start_vel_scale: int = 50

    # Home position parameters (left arm) - used on disconnect
    left_home_position_degree: list[float] = field(
        default_factory=lambda: [88.79, 74.96, 22.75, 112.75, -0.39, 86.74, 1.24]
    )
    # Home position parameters (right arm) - used on disconnect
    right_home_position_degree: list[float] = field(
        default_factory=lambda: [-24.41, 71.36, -4.67, 118.53, 3.91, 96.15, 3.60]
    )
    home_vel_scale: int = 30

    # FT sensor zeroing
    zero_ft_sensor_on_connect: bool = True

    # Logging
    log_level: str = "INFO"

    # flexiv_rt.Robot connection
    connect_retries: int = 3
    retry_interval_sec: float = 1.0

    # CPU affinity for RT threads (2 = first user-available core; -1 = no binding)
    # Scheduler docs: core 0 reserved for system, core 1 reserved for Scheduler itself.
    # Binding left/right to separate cores eliminates inter-thread RT jitter.
    left_rt_cpu_affinity: int = 2
    right_rt_cpu_affinity: int = 3

    # ========== Left gripper settings ==========
    left_use_gripper: bool = True
    left_gripper_sn: str = "000001"
    left_gripper_baudrate: int = 115200
    left_gripper_serial_timeout: float = 1.0
    # -- shared motion parameters --
    left_gripper_min_pos: float = 0.0  # mm — fully closed
    left_gripper_max_pos: float = 85.0  # mm — fully open
    left_gripper_v_max: float = 100.0  # mm/s
    left_gripper_f_max: float = 40.0  # N
    left_gripper_init_open: bool = True

    # ========== Right gripper settings ==========
    right_use_gripper: bool = True
    right_gripper_sn: str = "000002"
    right_gripper_baudrate: int = 115200
    right_gripper_serial_timeout: float = 1.0
    # -- shared motion parameters --
    right_gripper_min_pos: float = 0.0  # mm — fully closed
    right_gripper_max_pos: float = 85.0  # mm — fully open
    right_gripper_v_max: float = 100.0  # mm/s
    right_gripper_f_max: float = 40.0  # N
    right_gripper_init_open: bool = True

    # Auto-created in __post_init__ (do not set directly)
    left_gripper: SerialGripperConfig | None = field(default=None, init=False)
    right_gripper: SerialGripperConfig | None = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

        # ── Apply preset positions and device identifiers based on mounting type ──
        _PRESETS = {
            "forward": {
                "left_sn": "Rizon4s-063458",
                "right_sn": "Rizon4s-063670",
                "left_gripper_sn": "000001",
                "right_gripper_sn": "000002",
                "left_start": [88.79, 74.96, 22.75, 112.75, -0.39, 86.74, 1.24],
                "right_start": [-24.41, 71.36, -4.67, 118.53, 3.91, 96.15, 3.60],
                "left_home": [88.79, 74.96, 22.75, 112.75, -0.39, 86.74, 1.24],
                "right_home": [-24.41, 71.36, -4.67, 118.53, 3.91, 96.15, 3.60],
                "head_camera_sn": "337322070722",
                "left_wrist_camera_sn": "XC000001",
                "right_wrist_camera_sn": "XC000002",
                "left_tactile_camera_sn_0": "OG000863",
                "left_tactile_camera_sn_1": "OG000864",
                "right_tactile_camera_sn_0": "OG000861",
                "right_tactile_camera_sn_1": "OG000862",
            },
            "side": {
                "left_sn": "Rizon4-063423",
                "right_sn": "Rizon4-062855",
                "left_gripper_sn": "000003",
                "right_gripper_sn": "000004",
                "left_start": [-18.95, 80.45, -80.35, -89.37, -12.83, -17.05, -9.80],
                "right_start": [12.67, -85.31, 85.44, 102.25, 5.88, 25.36, 0.0],
                "left_home": [-18.95, 80.45, -80.35, -89.37, -12.83, -17.05, -9.80],
                "right_home": [12.67, -85.31, 85.44, 102.25, 5.88, 25.36, 0.0],
                "head_camera_sn": "135522074323",
                "left_wrist_camera_sn": "XC000003",
                "right_wrist_camera_sn": "XC000004",
                "left_tactile_camera_sn_0": "OG000867",
                "left_tactile_camera_sn_1": "OG000865",
                "right_tactile_camera_sn_0": "OG000142",
                "right_tactile_camera_sn_1": "OG000866",
            },
        }
        if self.bi_mount_type not in _PRESETS:
            raise ValueError(
                f"Unknown mounting type {self.bi_mount_type!r}, expected one of {list(_PRESETS)}"
            )

        preset = _PRESETS[self.bi_mount_type]
        self.left_robot_sn = preset["left_sn"]
        self.right_robot_sn = preset["right_sn"]
        self.left_gripper_sn = preset["left_gripper_sn"]
        self.right_gripper_sn = preset["right_gripper_sn"]
        self.left_start_position_degree = preset["left_start"]
        self.right_start_position_degree = preset["right_start"]
        self.left_home_position_degree = preset["left_home"]
        self.right_home_position_degree = preset["right_home"]

        # Validate Cartesian/force parameters
        if len(self.force_control_axis) != 6:
            raise ValueError(
                f"force_control_axis must have 6 elements, got {len(self.force_control_axis)}"
            )
        if len(self.max_contact_wrench) != 6:
            raise ValueError(
                f"max_contact_wrench must have 6 elements, got {len(self.max_contact_wrench)}"
            )
        if len(self.target_wrench) != 6:
            raise ValueError(
                f"target_wrench must have 6 elements, got {len(self.target_wrench)}"
            )
        if len(self.damping_ratio) != 6:
            raise ValueError(
                f"damping_ratio must have 6 elements, got {len(self.damping_ratio)}"
            )

        # Validate start positions
        if len(self.left_start_position_degree) != 7:
            raise ValueError(
                f"left_start_position_degree must have 7 elements, got {len(self.left_start_position_degree)}"
            )
        if len(self.right_start_position_degree) != 7:
            raise ValueError(
                f"right_start_position_degree must have 7 elements, got {len(self.right_start_position_degree)}"
            )
        if not 1 <= self.start_vel_scale <= 100:
            raise ValueError(
                f"start_vel_scale must be between 1 and 100, got {self.start_vel_scale}"
            )
        if len(self.left_home_position_degree) != 7:
            raise ValueError(
                f"left_home_position_degree must have 7 elements, got {len(self.left_home_position_degree)}"
            )
        if len(self.right_home_position_degree) != 7:
            raise ValueError(
                f"right_home_position_degree must have 7 elements, got {len(self.right_home_position_degree)}"
            )
        if not 1 <= self.home_vel_scale <= 100:
            raise ValueError(
                f"home_vel_scale must be between 1 and 100, got {self.home_vel_scale}"
            )

        # Create left gripper config
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

        # Create right gripper config
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

        # Camera configuration based on tactile sensors setting
        self.cameras = {
            "head": RealSenseCameraConfig(
                serial_number_or_name=preset["head_camera_sn"],
                fps=30,
                width=640,
                height=480,
                warmup_s=1.0 if self.enable_tactile_sensors else 0.05,
            ),
            "left_wrist": OpenCVCameraConfig(
                index_or_path=preset["left_wrist_camera_sn"],
                fourcc="MJPG",
                width=640,
                height=480,
                fps=30,
                warmup_s=1.0,
            ),
            "right_wrist": OpenCVCameraConfig(
                index_or_path=preset["right_wrist_camera_sn"],
                fourcc="MJPG",
                width=640,
                height=480,
                fps=30,
                warmup_s=1.0,
            ),
        }
        if self.enable_tactile_sensors:
            self.cameras.update(
                {
                    "left_tactile_0": XenseTactileCameraConfig(
                        serial_number=preset["left_tactile_camera_sn_0"],
                        fps=30,
                        output_types=[XenseOutputType.RECTIFY],
                        warmup_s=0.05,
                    ),
                    "left_tactile_1": XenseTactileCameraConfig(
                        serial_number=preset["left_tactile_camera_sn_1"],
                        fps=30,
                        output_types=[XenseOutputType.RECTIFY],
                        warmup_s=0.05,
                    ),
                    "right_tactile_0": XenseTactileCameraConfig(
                        serial_number=preset["right_tactile_camera_sn_0"],
                        fps=30,
                        output_types=[XenseOutputType.RECTIFY],
                        warmup_s=0.05,
                    ),
                    "right_tactile_1": XenseTactileCameraConfig(
                        serial_number=preset["right_tactile_camera_sn_1"],
                        fps=30,
                        output_types=[XenseOutputType.RECTIFY],
                        warmup_s=0.05,
                    ),
                }
            )
        pass
