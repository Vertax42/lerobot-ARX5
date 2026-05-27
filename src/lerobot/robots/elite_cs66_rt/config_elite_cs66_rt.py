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


class EliteCS66RTControlMode(str, Enum):
    CARTESIAN_SERVO = "cartesian_servo"
    JOINT_SERVO = "joint_servo"


@RobotConfig.register_subclass("elite_cs66_rt")
@dataclass
class EliteCS66RTConfig(RobotConfig):
    """Configuration for a single Elite CS66 arm.

    The default mode follows the LeRobot Cartesian convention:
    actions/observations use tcp.x/y/z plus 6D rotation tcp.r1..tcp.r6, and the
    driver converts that pose to Elite's native [x, y, z, rx, ry, rz] rotvec
    format before calling writeServoj(..., cartesian=True).
    """

    robot_ip: str = "192.168.1.200"
    local_ip: str = ""
    control_mode: EliteCS66RTControlMode = EliteCS66RTControlMode.CARTESIAN_SERVO

    # Observation schema is composable: enable TCP and joint state
    # independently. Use both True for multi-modal datasets (e.g. VLA
    # policies that condition on joint proprio + TCP pose).
    #   observe_tcp=True    -> tcp.x/y/z + tcp.r1..r6 (9 floats)
    #   observe_joints=True -> joint_*.pos/vel/effort (18 floats)
    # Cameras / gripper are independent of both.
    observe_tcp: bool = True
    observe_joints: bool = False

    # Elite external control script. When unset, connect() resolves
    # elite_cs_sdk/external_control.script from the installed SDK package.
    script_file_path: str | Path | None = None

    # Servo streaming parameters. servoj_time is the controller's inner
    # interpolation period (matches the SDK example at 250 Hz).  servoj_lookahead_time
    # must lie in the SDK-documented range [0.03, 0.2]; outside that range the
    # external_control script aborts and tears down all reverse sockets.
    servoj_time: float = 0.004
    servoj_lookahead_time: float = 0.1
    servoj_gain: int = 2000
    command_timeout_ms: int = 200
    use_background_servo_loop: bool = True
    command_stale_timeout_s: float = 0.5
    reset_duration_s: float = 3.0

    # RTSI state stream.
    rtsi_frequency: float = 250.0
    rtsi_output_recipe: str | Path | None = None
    rtsi_input_recipe: str | Path | None = None

    # Startup and shutdown behavior. The fleet hardcodes the canonical
    # sequence (power on → brake release → script start → clean stopControl
    # on disconnect → RT scheduling best-effort). The only knob left is the
    # overall timeout waiting for the controller-side script to handshake.
    connect_timeout_s: float = 10.0

    # Home / Start poses (J1..J6 in radians). MoveJ-style trajectory used to
    # reach these — see ``_move_j_blocking`` in EliteCS66RT.
    #   home  = safe park position; arm is moved here in disconnect() before
    #           reverse sockets are torn down. Service / shutdown pose.
    #   start = task-ready position; arm is moved here in connect() before
    #           streaming begins. Every episode starts from here.
    # The fields are kept separate by design even though our current fleet
    # uses identical values: most stations want the same candle pose for
    # both, but some workflows (overhead service position vs. workspace-
    # ready) need to differentiate. Override per station as needed.
    # The runtime ``connect(go_to_start=False)`` flag skips the start MoveJ
    # for crash-recovery / re-attach scenarios.
    home_position_rad: list[float] = field(
        default_factory=lambda: [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0]
    )
    start_position_rad: list[float] = field(
        default_factory=lambda: [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0]
    )
    start_move_duration_s: float = 4.0
    home_move_duration_s: float = 4.0
    move_j_timeout_ms: int = 200

    # The Elite SDK examples wait ~1 s after isRobotConnected() returns True
    # before issuing writeServoj; otherwise the controller script can RST the
    # reverse socket. Mirror that here.
    external_control_settle_s: float = 1.0

    # Trace every send_action and large per-step deltas to the spdlog file
    # sink (~/xenselogs). Doesn't touch console; safe to leave enabled.
    trace_servoj: bool = True
    # Per-step delta thresholds above which the trace promotes to WARNING (also
    # captured in the file log). 5 cm or 0.5 rad in a single send_action call
    # is suspicious for steady-state teleop.
    trace_translation_threshold: float = 0.05
    trace_rotation_threshold: float = 0.5

    # Optional safety clamps for Cartesian actions, applied **per send_action**.
    # Defaults to disabled (0): we mirror flexiv_rizon4_rt and let the
    # controller's lookahead + gain handle smoothing. Set >0 only when you want
    # a hard ceiling on the step size per outer-loop tick (e.g. policy roll-out
    # safety net), at the cost of a sluggish trail-off after a SpaceMouse
    # release.
    max_relative_translation: float = 0.0
    max_relative_rotation: float = 0.0

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
        if not 0.03 <= self.servoj_lookahead_time <= 0.2:
            # Elite SDK EliteDriver.hpp says lookahead time must lie in [0.03, 0.2];
            # values outside this range cause the external_control script to abort
            # and tear down all reverse sockets (50001/50003/50004), leaving the
            # Python side writing into a dead socket forever.
            raise ValueError(
                "servoj_lookahead_time must be in [0.03, 0.2] (Elite SDK requirement), "
                f"got {self.servoj_lookahead_time}"
            )
        if self.servoj_gain <= 0:
            raise ValueError(f"servoj_gain must be > 0, got {self.servoj_gain}")
        if self.command_timeout_ms < 5:
            raise ValueError(
                f"command_timeout_ms must be >= 5 (Elite SDK lower bound), got {self.command_timeout_ms}"
            )
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
        if len(self.home_position_rad) != 6:
            raise ValueError(
                f"home_position_rad must have 6 elements (J1..J6), got {len(self.home_position_rad)}"
            )
        if len(self.start_position_rad) != 6:
            raise ValueError(
                f"start_position_rad must have 6 elements (J1..J6), got {len(self.start_position_rad)}"
            )
        if self.start_move_duration_s <= 0:
            raise ValueError(
                f"start_move_duration_s must be > 0, got {self.start_move_duration_s}"
            )
        if self.home_move_duration_s <= 0:
            raise ValueError(
                f"home_move_duration_s must be > 0, got {self.home_move_duration_s}"
            )
        if (
            self.use_background_servo_loop
            and self.control_mode != EliteCS66RTControlMode.CARTESIAN_SERVO
        ):
            raise ValueError(
                "use_background_servo_loop=True is only supported with control_mode=CARTESIAN_SERVO. "
                "Set use_background_servo_loop=False for joint servo mode."
            )
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
