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
from typing import Union

import flexiv_rt

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.flexiv_rizon4.config_flare_gripper import FlareGripperConfig, SensorOutputType
from lerobot.robots.flexiv_rizon4.config_xense_gripper import GripperConfig


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
    left_robot_sn: str = "Rizon4-063423"
    right_robot_sn: str = "Rizon4-063424"

    # Force control
    use_force: bool = False

    # Python-side frequency (RT thread is always 1 kHz internally)
    control_frequency: float = 100.0  # Hz

    # Connection behavior
    go_to_start: bool = True

    # Camera configurations (external cameras)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

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
        default_factory=lambda: [-1.70, 4.48, 1.54, 136.22, 0.12, 41.74, -0.18]
    )
    # Start position parameters (right arm)
    right_start_position_degree: list[float] = field(
        default_factory=lambda: [-1.70, 4.48, 1.54, 136.22, 0.12, 41.74, -0.18]
    )
    start_vel_scale: int = 30

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
    left_gripper_type: str = "flare_gripper"  # Options: "flare_gripper", "xense_gripper"
    left_gripper_mac_addr: str = "e2b26adbb104"
    left_gripper_cam_size: tuple[int, int] = (640, 480)
    left_gripper_rectify_size: tuple[int, int] = (400, 700)
    left_gripper_sensor_output_type: SensorOutputType = SensorOutputType.RECTIFY
    left_gripper_sensor_keys: dict[str, str] = field(
        default_factory=lambda: {
            "OG000657": "left_right_tactile",
            "OG000450": "left_left_tactile",
        }
    )
    left_gripper_min_pos: float = 0.0
    left_gripper_max_pos: float = 85.0
    left_gripper_v_max: float = 80.0  # mm/s
    left_gripper_f_max: float = 20.0  # N
    left_gripper_init_open: bool = True

    # ========== Right gripper settings ==========
    right_use_gripper: bool = True
    right_gripper_type: str = "flare_gripper"  # Options: "flare_gripper", "xense_gripper"
    right_gripper_mac_addr: str = "e2b26adbb105"
    right_gripper_cam_size: tuple[int, int] = (640, 480)
    right_gripper_rectify_size: tuple[int, int] = (400, 700)
    right_gripper_sensor_output_type: SensorOutputType = SensorOutputType.RECTIFY
    right_gripper_sensor_keys: dict[str, str] = field(
        default_factory=lambda: {
            "OG000658": "right_right_tactile",
            "OG000451": "right_left_tactile",
        }
    )
    right_gripper_min_pos: float = 0.0
    right_gripper_max_pos: float = 85.0
    right_gripper_v_max: float = 80.0  # mm/s
    right_gripper_f_max: float = 20.0  # N
    right_gripper_init_open: bool = True

    # Auto-created in __post_init__ (do not set directly)
    left_gripper: Union[GripperConfig, FlareGripperConfig] | None = field(default=None, init=False)
    right_gripper: Union[GripperConfig, FlareGripperConfig] | None = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

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

        # Create left gripper config
        if self.left_use_gripper and self.left_gripper_type == "flare_gripper":
            self.left_gripper = FlareGripperConfig(
                mac_addr=self.left_gripper_mac_addr,
                cam_size=self.left_gripper_cam_size,
                rectify_size=self.left_gripper_rectify_size,
                sensor_output_type=self.left_gripper_sensor_output_type,
                sensor_keys=self.left_gripper_sensor_keys,
                gripper_max_pos=self.left_gripper_max_pos,
                gripper_v_max=self.left_gripper_v_max,
                gripper_f_max=self.left_gripper_f_max,
                init_open=self.left_gripper_init_open,
            )
        elif self.left_use_gripper and self.left_gripper_type == "xense_gripper":
            self.left_gripper = GripperConfig(
                mac_addr=self.left_gripper_mac_addr,
                rectify_size=self.left_gripper_rectify_size,
                sensor_output_type=self.left_gripper_sensor_output_type,
                sensor_keys=self.left_gripper_sensor_keys,
                gripper_min_pos=self.left_gripper_min_pos,
                gripper_max_pos=self.left_gripper_max_pos,
                gripper_v_max=self.left_gripper_v_max,
                gripper_f_max=self.left_gripper_f_max,
                init_open=self.left_gripper_init_open,
            )
        else:
            self.left_gripper = None

        # Create right gripper config
        if self.right_use_gripper and self.right_gripper_type == "flare_gripper":
            self.right_gripper = FlareGripperConfig(
                mac_addr=self.right_gripper_mac_addr,
                cam_size=self.right_gripper_cam_size,
                rectify_size=self.right_gripper_rectify_size,
                sensor_output_type=self.right_gripper_sensor_output_type,
                sensor_keys=self.right_gripper_sensor_keys,
                gripper_max_pos=self.right_gripper_max_pos,
                gripper_v_max=self.right_gripper_v_max,
                gripper_f_max=self.right_gripper_f_max,
                init_open=self.right_gripper_init_open,
            )
        elif self.right_use_gripper and self.right_gripper_type == "xense_gripper":
            self.right_gripper = GripperConfig(
                mac_addr=self.right_gripper_mac_addr,
                rectify_size=self.right_gripper_rectify_size,
                sensor_output_type=self.right_gripper_sensor_output_type,
                sensor_keys=self.right_gripper_sensor_keys,
                gripper_min_pos=self.right_gripper_min_pos,
                gripper_max_pos=self.right_gripper_max_pos,
                gripper_v_max=self.right_gripper_v_max,
                gripper_f_max=self.right_gripper_f_max,
                init_open=self.right_gripper_init_open,
            )
        else:
            self.right_gripper = None
