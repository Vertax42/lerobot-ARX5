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

"""Configuration for Flexiv Rizon4 RT robot (real-time via flexiv_rt)."""

from dataclasses import dataclass, field

import flexiv_rt

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.flexiv_rizon4.config_flare_gripper import FlareGripperConfig, SensorOutputType


@RobotConfig.register_subclass("flexiv_rizon4_rt")
@dataclass
class FlexivRizon4RTConfig(RobotConfig):
    """Configuration for Flexiv Rizon4 robot with real-time control via flexiv_rt.

    This driver uses the RT backend (flexiv_rt) instead of the NRT backend (flexivrdk).
    The C++ RT thread runs at 1 kHz via rdk::Scheduler with SCHED_FIFO priority.
    Python send_action() (30-100 Hz) writes to shared memory, which the RT thread
    reads every 1 ms for deterministic streaming.

    Architecture:
        LeRobot send_action(30-100 Hz) -> set_target_pose() -> SHM -> C++ RT(1 kHz)
            -> StreamCartesianMotionForce -> Robot

    Attributes:
        robot_sn: Serial number of the robot (e.g., "Rizon4-063423")
        use_force: Enable force control axes (when True, action includes target wrench)
        use_joint_observation: Include joint states in observation (even in Cartesian mode)
        control_frequency: Python-side control loop frequency in Hz (RT thread always 1 kHz)

        force_control_frame: Reference frame for force control (CoordType.WORLD or TCP)
        force_control_axis: Which axes to enable force control [x, y, z, rx, ry, rz]
        max_contact_wrench: Maximum contact wrench [fx, fy, fz, mx, my, mz] in N and Nm
        target_wrench: Default target wrench for force control

        stiffness_ratio: Multiplies nominal Cartesian stiffness K_x_nom (1.0 = nominal)
        damping_ratio: Cartesian damping ratio per axis (6D)

        ext_force_threshold: External TCP force threshold for collision detection [N]
        ext_torque_threshold: External joint torque threshold for collision detection [Nm]

        go_to_start: Move to start_position_degree after connect
        start_position_degree: Joint positions in degrees for start pose
        start_vel_scale: Joint velocity scale for MoveJ (1-100)
        zero_ft_sensor_on_connect: Zero force-torque sensors on connect

        connect_retries: Number of connection retries for flexiv_rt.Robot
        retry_interval_sec: Seconds between connection retries
    """

    # Robot identification
    robot_sn: str = "Rizon4-063423"

    # Force control
    use_force: bool = False
    use_joint_observation: bool = False

    # Python-side frequency (RT thread is always 1 kHz internally)
    control_frequency: float = 100.0  # Hz

    # Connection behavior
    go_to_start: bool = True

    # Camera configurations (external cameras, e.g., scene cameras)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Cartesian impedance
    stiffness_ratio: float = 0.2
    damping_ratio: list[float] = field(default_factory=lambda: [0.7] * 6)

    # Force control settings
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

    # Start position parameters
    start_position_degree: list[float] = field(
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

    # ========== Flare Gripper (end-effector) settings ==========
    use_gripper: bool = True
    flare_gripper_mac_addr: str = "e2b26adbb104"
    flare_gripper_cam_size: tuple[int, int] = (640, 480)
    flare_gripper_rectify_size: tuple[int, int] = (400, 700)
    flare_gripper_sensor_output_type: SensorOutputType = SensorOutputType.RECTIFY
    flare_gripper_sensor_keys: dict[str, str] = field(
        default_factory=lambda: {
            "OG000657": "right_tactile",
            "OG000450": "left_tactile",
        }
    )
    flare_gripper_max_pos: float = 85.0
    flare_gripper_v_max: float = 80.0  # mm/s
    flare_gripper_f_max: float = 20.0  # N
    flare_gripper_init_open: bool = True

    # Auto-created in __post_init__
    flare_gripper: FlareGripperConfig | None = field(default=None, init=False)

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

        # Validate start position
        if len(self.start_position_degree) != 7:
            raise ValueError(
                f"start_position_degree must have 7 elements, got {len(self.start_position_degree)}"
            )
        if not 1 <= self.start_vel_scale <= 100:
            raise ValueError(
                f"start_vel_scale must be between 1 and 100, got {self.start_vel_scale}"
            )

        # Create FlareGripperConfig from exposed parameters
        if self.use_gripper:
            self.flare_gripper = FlareGripperConfig(
                mac_addr=self.flare_gripper_mac_addr,
                cam_size=self.flare_gripper_cam_size,
                rectify_size=self.flare_gripper_rectify_size,
                sensor_output_type=self.flare_gripper_sensor_output_type,
                sensor_keys=self.flare_gripper_sensor_keys,
                gripper_max_pos=self.flare_gripper_max_pos,
                gripper_v_max=self.flare_gripper_v_max,
                gripper_f_max=self.flare_gripper_f_max,
                init_open=self.flare_gripper_init_open,
            )
        else:
            self.flare_gripper = None

        # # Camera configuration for realsense cameras
        # self.cameras = {
        #     "top": RealSenseCameraConfig(
        #         serial_number_or_name="135522074323", fps=30, width=1280, height=720
        #     ),
        # }
