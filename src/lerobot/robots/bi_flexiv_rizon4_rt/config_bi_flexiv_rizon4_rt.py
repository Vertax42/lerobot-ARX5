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
from lerobot.grippers import SerialGripperConfig, TaccapFollowerConfig

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

    # Which gripper driver drives BOTH arms: "serial" (XenseSerialGripper) or
    # "taccap_follower" (xense.taccap FollowerGripper). Either way left/right is
    # auto-resolved at connect: serial by board-SN parity (odd SN → left, even SN
    # → right; see serial_discovery.find_port_by_side), taccap by firmware SN.
    # Grippers are always swapped as a matched pair, so this is a single switch
    # rather than per-side.
    gripper_type: str = "serial"

    # When gripper_type == "serial", auto-sniff the per-side wrist camera and
    # tactile sensor SNs at connect instead of the recipe's XC*/OG* SNs.
    # Works the same way as taccap_auto_discover_cameras: each arm's gripper board,
    # wrist camera and two tactile sensors share one USB hub, so the gripper (whose
    # side comes from board-SN parity) identifies the hub and the rest follows.
    # See serial_discovery.discover_serial_gripper_cameras.
    #
    # OFF by default: leaving it off keeps the recipe the single source of
    # truth, which is the verified behaviour. Turn it on per bench once you have
    # confirmed the discovered SNs match the recipe's — then swapping a sensor
    # stops being a config edit.
    serial_auto_discover_cameras: bool = False

    # ========== Left gripper settings (per-arm: enable + serial wiring) ==========
    left_use_gripper: bool = True
    left_gripper_serial_timeout: float = 1.0

    # ========== Right gripper settings (per-arm: enable + serial wiring) ==========
    right_use_gripper: bool = True
    right_gripper_serial_timeout: float = 1.0

    # ========== Shared gripper mechanical / motion params (both arms) ==========
    gripper_min_pos: float = 0.0  # mm — fully closed
    gripper_max_pos: float = 85.0  # mm — fully open
    gripper_v_max: float = 100.0  # mm/s
    gripper_f_max: float = 40.0  # N
    gripper_init_open: bool = True

    # ========== TacCap follower control params (used when gripper_type == "taccap_follower") ==========
    taccap_kp: float = 8.0          # MIT impedance stiffness (Nm/rad)
    taccap_kd: float = 1.0          # MIT impedance damping (Nm·s/rad)
    taccap_grip_ff: float = 0.0     # constant MIT feed-forward torque (Nm); NEGATIVE = clamp harder, POSITIVE = open. |ff|<=3.5
    taccap_control_hz: int = 200    # ControlLoop resubmit rate
    # When gripper_type == "taccap_follower", auto-sniff the per-side wrist camera
    # and GSPS tactile sensor SNs (via TacCap + USB topology) at robot connect time
    # instead of the recipe's XC*/OG* SNs. See grippers/taccap/discovery.py.
    taccap_auto_discover_cameras: bool = True
    # Refuse to connect a taccap_follower gripper that reports an uncalibrated GripperConfig.
    # Set False ONLY for bring-up/debug — normalized [0, 1] gripper control is then uncalibrated
    # (positions won't map to real open/close). Prefer calibrating (examples/calibrate.py).
    taccap_require_calibrated: bool = True

    # Auto-created in __post_init__ (do not set directly)
    left_gripper: SerialGripperConfig | TaccapFollowerConfig | None = field(
        default=None, init=False
    )
    right_gripper: SerialGripperConfig | TaccapFollowerConfig | None = field(
        default=None, init=False
    )
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

        # Build per-side gripper configs; both sides use the same gripper_type.
        self.left_gripper = self._make_gripper_config(
            self.left_use_gripper, self.left_gripper_serial_timeout, "left",
        )
        self.right_gripper = self._make_gripper_config(
            self.right_use_gripper, self.right_gripper_serial_timeout, "right",
        )

        # In taccap_follower + auto-discover mode, the wrist camera and GSPS tactile
        # SNs are hardware that changes with the gripper, so they are sniffed at
        # robot connect time (see taccap discovery) rather than pinned in the
        # recipe, which then only pins `head`. Any other mode expects the recipe to
        # pin every camera.
        self._taccap_autodiscover = (
            self.gripper_type == "taccap_follower" and self.taccap_auto_discover_cameras
        )
        # Serial grippers can do the same, anchored on board-SN parity instead of
        # the firmware SN. Both flags mean "the driver wires wrist + tactile at
        # connect"; only the head comes from the recipe.
        self._serial_autodiscover = (
            self.gripper_type == "serial" and self.serial_auto_discover_cameras
        )

    @property
    def _autodiscover_cameras(self) -> bool:
        """True when the driver, not the recipe, supplies wrist + tactile cameras.

        Either gripper backend can do it; they differ only in what anchors a side
        (taccap: firmware SN; serial: board-SN parity). Everything downstream is
        the same, so callers branch on this rather than on which backend is in use.
        """
        return self._taccap_autodiscover or self._serial_autodiscover

    def _make_gripper_config(
        self,
        use_gripper: bool,
        serial_timeout: float,
        side: str,
    ) -> "SerialGripperConfig | TaccapFollowerConfig | None":
        """Build the per-side nested gripper config selected by ``self.gripper_type``.

        Both drivers auto-resolve left/right at connect: "serial" by board-SN parity
        (odd → left, even → right), "taccap_follower" by the firmware-burned SN.
        Baudrate uses the SerialGripperConfig default (115200).
        """
        if not use_gripper:
            return None
        gripper_type = self.gripper_type
        if gripper_type == "serial":
            return SerialGripperConfig(
                side=side,
                serial_timeout=serial_timeout,
                gripper_min_pos=self.gripper_min_pos,
                gripper_max_pos=self.gripper_max_pos,
                gripper_v_max=self.gripper_v_max,
                gripper_f_max=self.gripper_f_max,
                init_open=self.gripper_init_open,
            )
        if gripper_type == "taccap_follower":
            return TaccapFollowerConfig(
                side=side,
                kp=self.taccap_kp,
                kd=self.taccap_kd,
                feedforward_torque=self.taccap_grip_ff,
                control_hz=self.taccap_control_hz,
                init_open=self.gripper_init_open,
                require_calibrated=self.taccap_require_calibrated,
            )
        raise ValueError(
            f"gripper_type must be 'serial' or 'taccap_follower', got {gripper_type!r}."
        )
