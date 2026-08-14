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

from dataclasses import dataclass, field, replace

import flexiv_rt

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.grippers import GripperConfig

ROBOT_TYPE = "bi_flexiv_rizon4_rt"


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

    Station hardware (arm SNs, camera SNs, home/start poses) is NOT written here:
    it is supplied by the recipe — each recipe under ``recipes/`` is
    self-contained. See ``recipes/README.md``.

    Attributes:
        left_robot_sn: Serial number of the left arm robot
        right_robot_sn: Serial number of the right arm robot
            Serial grippers self-sort left/right by board-SN parity at connect,
            so no gripper SN is configured.
        use_force: Enable force control axes (both arms)
        inner_control_hz: How often each 1 kHz RT thread consumes a new Python command (1-1000 Hz)
        interpolate_cmds: Enable linear interpolation between consumed commands
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
        commanded_actual_max_deg: Steady-state envelope (per arm) on the angle between the
            commanded TCP orientation in send_action and the live TCP orientation read from
            the RT shared memory. If exceeded, the commanded quaternion is slerp-projected
            back to within this angle of actual before being written to the RT thread.
            Default 60° keeps a 30° margin below Flexiv's 90° orientation-error safety
            (event 301005). The returned action dict carries the clamped value, so any
            dataset frame built from it records the safe envelope. Set to 180 to disable.
    """

    # Robot identification — set per bench in the recipe.
    left_robot_sn: str = ""
    right_robot_sn: str = ""
    # Force control
    use_force: bool = False

    # RT command consumption parameters (match flexiv_rt defaults)
    inner_control_hz: int = 1000
    interpolate_cmds: bool = True

    # Connection behavior
    go_to_start: bool = True

    # Camera configurations (external cameras)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    enable_tactile_sensors: bool = True
    # Head RealSense depth stream. Reserved but OFF by default: get_observation only
    # pulls (color, depth) when this is True; otherwise the head camera is color-only.
    head_camera_use_depth: bool = False

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

    # Per-arm steady-state clamp on commanded vs actual TCP orientation (degrees).
    # Below Flexiv's 90° orientation-error safety (event 301005). Set to 180 to disable.
    commanded_actual_max_deg: float = 60.0

    # Start / home joint poses (J1..J7 degrees), set per bench in the recipe.
    # Home is where each arm parks on disconnect. The zero default is not a usable
    # pose on any real bench — every recipe overrides all four.
    left_start_position_degree: list[float] = field(default_factory=lambda: [0.0] * 7)
    right_start_position_degree: list[float] = field(default_factory=lambda: [0.0] * 7)
    left_home_position_degree: list[float] = field(default_factory=lambda: [0.0] * 7)
    right_home_position_degree: list[float] = field(default_factory=lambda: [0.0] * 7)
    start_vel_scale: int = 50
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

    # ── Grippers ── One `gripper:` block in the recipe describes BOTH arms; the two
    # are always a matched pair, and each backend resolves its own side at connect
    # (serial: board-SN parity, odd → left; taccap: the firmware-burned SN). The
    # block is cloned per side in __post_init__ with `side` stamped, so nothing in
    # the recipe has to name a side.
    #
    # The block is typed (see lerobot.grippers.GripperConfig), so `type: serial`
    # with a taccap knob in it — or a typo — is rejected at parse time instead of
    # being silently ignored, which is what the old flat gripper_type + taccap_* /
    # gripper_* fields did.
    #
    # None = no gripper on either arm. Set {side}_use_gripper=False to drop just one.
    gripper: GripperConfig | None = None
    left_use_gripper: bool = True
    right_use_gripper: bool = True

    # Auto-created in __post_init__ (do not set directly)
    left_gripper: GripperConfig | None = field(default=None, init=False)
    right_gripper: GripperConfig | None = field(default=None, init=False)
    # Set in __post_init__: robot resolves wrist/tactile cameras at connect time.
    _taccap_autodiscover: bool = field(default=False, init=False)
    _serial_autodiscover: bool = field(default=False, init=False)

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.inner_control_hz, bool) or not isinstance(
            self.inner_control_hz, int
        ):
            raise TypeError(
                "inner_control_hz must be an integer in [1, 1000], "
                f"got {self.inner_control_hz!r}"
            )

        if not 1 <= self.inner_control_hz <= 1000:
            raise ValueError(
                f"inner_control_hz must be between 1 and 1000, got {self.inner_control_hz}"
            )


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

        # Build per-side gripper configs from the shared block.
        # ── Per-side gripper configs ── clone the shared block, stamping the side.
        self.left_gripper = self._side_gripper("left", self.left_use_gripper)
        self.right_gripper = self._side_gripper("right", self.right_use_gripper)

        # With auto-discovery on, the wrist + tactile cameras travel with the
        # gripper, so the robot sniffs them at connect and the recipe pins only
        # `head`. Otherwise the recipe pins every camera.
        gt = self.gripper.type if self.gripper is not None else None
        discover = self.gripper is not None and self.gripper.auto_discover_cameras
        self._taccap_autodiscover = discover and gt == "taccap_follower"
        self._serial_autodiscover = discover and gt == "serial"

    @property
    def _autodiscover_cameras(self) -> bool:
        """True when the driver, not the recipe, supplies wrist + tactile cameras.

        Either gripper backend can do it; they differ only in what anchors a side
        (taccap: firmware SN; serial: board-SN parity). Everything downstream is
        the same, so callers branch on this rather than on which backend is in use.
        """
        return self._taccap_autodiscover or self._serial_autodiscover

    def _side_gripper(self, side: str, use_gripper: bool) -> GripperConfig | None:
        """The shared ``gripper`` block with ``side`` stamped, or None for no gripper.

        Both arms always run the same backend with the same tuning — they are a
        matched pair — so the recipe writes one block and each side gets a copy.
        ``side`` is what lets each backend resolve which physical unit is which
        (serial: board-SN parity; taccap: the firmware-burned SN).
        """
        if not use_gripper or self.gripper is None:
            return None
        return replace(self.gripper, side=side)
