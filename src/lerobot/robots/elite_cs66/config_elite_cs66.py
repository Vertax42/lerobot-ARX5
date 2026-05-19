#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Configuration for Elite Robots CS66 arms via elite_cs_sdk."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig


class EliteCS66ControlMode(str, Enum):
    CARTESIAN_SERVO = "cartesian_servo"
    JOINT_SERVO = "joint_servo"


@RobotConfig.register_subclass("elite_cs66")
@dataclass
class EliteCS66Config(RobotConfig):
    """Configuration for a single Elite CS66 arm.

    The default mode follows the LeRobot Cartesian convention:
    actions/observations use tcp.x/y/z plus 6D rotation tcp.r1..tcp.r6, and the
    driver converts that pose to Elite's native [x, y, z, rx, ry, rz] rotvec
    format before calling writeServoj(..., cartesian=True).
    """

    robot_ip: str = "192.168.1.200"
    local_ip: str = ""
    headless_mode: bool = True
    control_mode: EliteCS66ControlMode = EliteCS66ControlMode.CARTESIAN_SERVO
    use_joint_observation: bool = False

    # Elite external control script. When unset, connect() resolves
    # elite_cs_sdk/external_control.script from the installed SDK package.
    script_file_path: str | Path | None = None

    # Servo streaming parameters. The Elite examples use 0.004 s (250 Hz).
    servoj_time: float = 0.004
    servoj_lookahead_time: float = 0.1
    servoj_gain: int = 2000
    command_timeout_ms: int = 100
    use_background_servo_loop: bool = True
    target_interpolation: bool = True
    command_stale_timeout_s: float = 0.5
    reset_duration_s: float = 3.0

    # RTSI state stream.
    rtsi_frequency: float = 250.0
    rtsi_output_recipe: str | Path | None = None
    rtsi_input_recipe: str | Path | None = None

    # Startup and shutdown behavior.
    power_on_on_connect: bool = True
    brake_release_on_connect: bool = True
    play_program_on_connect: bool = True
    start_external_control_on_connect: bool = True
    stop_control_on_disconnect: bool = True
    enable_realtime_scheduling: bool = True
    connect_timeout_s: float = 10.0

    # Optional safety clamps for Cartesian actions. Values <= 0 disable clamps.
    max_relative_translation: float = 0.05
    max_relative_rotation: float = 0.35

    # Optional gripper placeholder for dataset/action compatibility.
    use_gripper: bool = False
    gripper_min_position: float = 0.0
    gripper_max_position: float = 1.0
    initial_gripper_position: float = 0.0

    # External cameras.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        if self.servoj_time <= 0:
            raise ValueError(f"servoj_time must be > 0, got {self.servoj_time}")
        if self.servoj_lookahead_time <= 0:
            raise ValueError(f"servoj_lookahead_time must be > 0, got {self.servoj_lookahead_time}")
        if self.servoj_gain <= 0:
            raise ValueError(f"servoj_gain must be > 0, got {self.servoj_gain}")
        if self.command_timeout_ms <= 0:
            raise ValueError(f"command_timeout_ms must be > 0, got {self.command_timeout_ms}")
        if self.command_stale_timeout_s <= 0:
            raise ValueError(
                f"command_stale_timeout_s must be > 0, got {self.command_stale_timeout_s}"
            )
        if self.reset_duration_s <= 0:
            raise ValueError(f"reset_duration_s must be > 0, got {self.reset_duration_s}")
        if self.rtsi_frequency <= 0:
            raise ValueError(f"rtsi_frequency must be > 0, got {self.rtsi_frequency}")
        if self.connect_timeout_s <= 0:
            raise ValueError(f"connect_timeout_s must be > 0, got {self.connect_timeout_s}")
        if self.gripper_min_position >= self.gripper_max_position:
            raise ValueError(
                "gripper_min_position must be smaller than gripper_max_position, got "
                f"{self.gripper_min_position} >= {self.gripper_max_position}"
            )
        if not self.gripper_min_position <= self.initial_gripper_position <= self.gripper_max_position:
            raise ValueError(
                "initial_gripper_position must be in [gripper_min_position, gripper_max_position], got "
                f"{self.initial_gripper_position}"
            )
