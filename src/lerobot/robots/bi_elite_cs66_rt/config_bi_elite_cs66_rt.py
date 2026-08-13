#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Configuration for BiEliteCS66RT dual-arm robot (two Elite CS66 controllers).

Relationship to ``EliteCS66RTConfig`` mirrors ``BiFlexivRizon4RTConfig`` vs.
``FlexivRizon4RTConfig``: shared control/servo parameters keep the single-arm
names and defaults. Bench hardware (controller IPs, start/home poses, mount
geometry, camera SNs) is supplied by the recipe — each recipe under ``recipes/``
is self-contained. Serial grippers self-sort left/right by board-SN parity at
connect, so no gripper SN is configured. Action / observation keys are
``left_``/``right_`` prefixed.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.xense import XenseOutputType, XenseTactileCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.elite_cs66_rt.config_elite_cs66_rt import _validate_singularity_params
from lerobot.grippers import (
    SerialGripperConfig,
    TaccapFollowerConfig,
)

ROBOT_TYPE = "bi_elite_cs66_rt"


class BiEliteCS66RTControlMode(str, Enum):
    CARTESIAN_SERVO = "cartesian_servo"
    JOINT_SERVO = "joint_servo"


@RobotConfig.register_subclass("bi_elite_cs66_rt")
@dataclass
class BiEliteCS66RTConfig(RobotConfig):
    """Configuration for two Elite CS66 arms via elite_cs_sdk.

    Each arm runs its own EliteDriver + RTSI stream + (optional) background
    Cartesian servo loop. ``send_action``/``get_observation`` use the same TCP /
    joint schema as the single-arm driver, ``left_``/``right_`` prefixed:
        left_tcp.x/y/z + left_tcp.r1..r6 (+ optional left_joint_*), left_gripper.pos
        right_tcp.x/y/z + right_tcp.r1..r6 (+ optional right_joint_*), right_gripper.pos
    Grippers are per-arm serial (USB) devices that self-sort left/right by board-SN
    parity at connect. Cameras (head + per-arm wrist + optional tactiles) live at the
    bimanual level; tactile images come from separate XenseTactileCamera devices, not
    the gripper.

    Bench hardware (controller IPs, start/home poses, mount geometry, camera SNs)
    is supplied by the recipe; see ``recipes/README.md``.
    """

    # ── Per-arm identity / connection ──
    left_robot_ip: str = ""
    right_robot_ip: str = ""
    left_local_ip: str = ""   # "" lets the OS route; set to pin the host NIC for RTSI
    right_local_ip: str = ""

    # ── Per-arm mounting → base↔world rotation ──
    # R_world←base = Rz(γ)·Rz(β)·Rx(α): α=tilt about base-X, β=rotate about Z (both
    # from the teach pendant, fixing only the gravity vector), γ=extra world-Z yaw
    # that aligns each arm's heading into ONE shared world frame (x=facing, y=left,
    # z=up). The driver lifts base→world in get_observation and maps world→base in
    # send_action. On the existing diagonal benches both pendants read α=45/β=90 and
    # the arms are point-symmetric, so left needs γ=180° (right γ=0° defines the
    # world) — set per bench in the recipe, not defaulted here.
    left_mount_tilt_deg: float | None = None
    left_mount_zrot_deg: float | None = None
    left_mount_world_yaw_deg: float | None = None
    right_mount_tilt_deg: float | None = None
    right_mount_zrot_deg: float | None = None
    right_mount_world_yaw_deg: float | None = None

    # Explicit per-arm world<-base rotation (3x3, rows = world X/Y/Z in base). When
    # set it OVERRIDES the angle build above for that arm — needed when the mounting
    # isn't a clean Rz·Rx (e.g. the left arm tilts about base-Y).
    #   None -> no matrix; build the rotation from the angles above
    #   []   -> same as None (accepted so a recipe can explicitly opt out)
    # Normalized to None in __post_init__, so the driver only ever sees None or 3x3.
    left_world_rotation: list[list[float]] | None = None
    right_world_rotation: list[list[float]] | None = None

    # ── Shared control mode + observation schema ──
    control_mode: BiEliteCS66RTControlMode = BiEliteCS66RTControlMode.CARTESIAN_SERVO
    observe_tcp: bool = True
    observe_joints: bool = False

    # Elite external control script (shared; resolved from the SDK when unset).
    script_file_path: str | Path | None = None

    # ── Shared servo streaming parameters (see config_elite_cs66_rt.py) ──
    servoj_time: float = 0.004
    servoj_lookahead_time: float = 0.1
    servoj_gain: int = 300
    # Larger reverse-socket read timeout than the single-arm driver (200ms): the
    # bimanual station runs 2 servo loops + 7 camera read threads + VR + the main
    # loop, so a servo-loop thread can occasionally be GIL-starved >200ms, and the
    # Elite controller would then drop external control ("socket timed out waiting
    # for command on reverse_socket") -> writeServoj fails -> teleop crash. 500ms
    # tolerance absorbs those intermittent stalls; the arm holds its last servoj
    # target meanwhile (RT thread priority does not help — the GIL is the limiter).
    command_timeout_ms: int = 500
    use_background_servo_loop: bool = True
    # Host-side stale threshold; validated as command_stale_timeout_s*1000 >=
    # command_timeout_ms so the host keeps feeding (idle) before the controller times out.
    command_stale_timeout_s: float = 1.0
    # SCHED_FIFO(99) on the per-arm servo threads. The bimanual driver runs TWO
    # servo threads (vs one single-arm). Two FIFO-99 threads + the Python GIL can
    # priority-invert: one servo thread waits on the GIL held by a normal-priority
    # camera/VR thread that the OTHER FIFO-99 servo thread keeps preempting, so one
    # arm stops feeding >command_timeout_ms and its controller drops external
    # control ("socket timed out waiting for command on reverse_socket"). RT
    # priority does not help GIL-bound Python anyway — set False to run the servo
    # loops at normal priority (recommended for the camera-heavy bimanual station).
    servo_fifo_scheduling: bool = True
    reset_duration_s: float = 3.0
    # Opt-in Cartesian velocity ceiling for the background servo loops (per arm).
    # Both None by default -> no PC-side clamp (controller envelope only). Set when
    # a jumpy leader (fast wrist rotation, VR tracking spike, clutch re-engage)
    # feeds target steps that trip a protective stop: each arm's servo loop then
    # slews its commanded TCP toward the latest target at no more than this speed,
    # turning the step into a smooth bounded ramp. Velocity ceiling (m/s, rad/s)
    # applied per servoj_time tick. See config_elite_cs66_rt.py for the rationale.
    max_lin_speed: float | None = None  # m/s; None disables the linear cap
    max_ang_speed: float | None = None  # rad/s; None disables the angular cap

    # Workspace reachability guard (per arm). Distance (m) from each arm's base
    # origin beyond which a commanded TCP target is treated as unreachable: the
    # driver HOLDS the last in-reach pose instead of sending it, preventing the
    # operator from driving an arm into the boundary singularity where the
    # controller's IK fails and external control drops (the observed
    # "writeServoj failed 251 ticks in a row" crash). Conservative spherical guard
    # from the base ORIGIN (real reach ~0.91 m for CS66); can read too tight in
    # some directions since the true workspace is offset to the shoulder.
    #
    # DISABLED by default (None) per operator request 2026-07-03: 0.85 m clipped
    # legitimate reach and held the (right) arm mid-teleop. Trade-off: with it off,
    # over-reaching DROPS external control at the boundary instead of holding. Set
    # 0.88-0.90 to re-enable. See config_elite_cs66_rt.py for rationale.
    max_reach_radius: float | None = None  # m; None disables the guard

    # Singularity-aware manipulability damping (per arm). OFF by default (w_high=None); the
    # per-arm Modified-DH is read from each controller at connect and damping disables itself
    # on any fetch / self-check failure. See config_elite_cs66_rt.py and manipulability.py.
    singularity_w_high: float | None = None  # disables damping when None
    singularity_w_low: float = 0.0  # w at/below which damping is maxed (s = s_min)
    singularity_min_scale: float = 0.05  # s_min in (0, 1]; floor so escape is always possible
    singularity_directional: bool = False  # don't damp moves that increase w (escape); needs live tuning
    dh_params: tuple[list[float], list[float], list[float]] | None = None  # (alpha, a, d) override
    log_manipulability: bool = False  # throttled debug log of w for live threshold tuning
    primary_timeout_ms: int = 1000  # one-shot DH (KinematicsInfo) fetch timeout at connect

    # Predicted joint-velocity limit scaling (per arm; single shared value). Bounds the predicted
    # joint step BEFORE the controller's internal IK spikes joint velocity past JOINT_IGNORE_SPEED
    # (30 rad/s) and drops external control — the fix for wrist-singularity rotation trips. Naturally
    # directional; needs only DH (fetched at connect) + datasheet joint limits, no live w-tuning.
    # OFF by default. See config_elite_cs66_rt.py and manipulability.py.joint_velocity_scale.
    # Official CS66 datasheet rates (J1/J2 150°/s, J3 180°/s, J4-6 230°/s)
    # -> [2.618, 2.618, 3.142, 4.014, 4.014, 4.014] rad/s.
    joint_vel_limits_rad_s: list[float] | None = None  # per-joint [J1..J6] rad/s; None disables
    joint_vel_limit_margin: float = 0.8  # enforce at this fraction of the limits (headroom, (0, 1])
    joint_vel_horizon_s: float = 0.033  # command horizon for the velocity check (~ 1/teleop fps)
    # DLS damping of the PREDICTION pseudo-inverse; keep well below the operating sigma_min so the
    # predicted dq tracks the controller's near-exact IK spike (~1/sigma_min) instead of capping it.
    # 1e-2 silently under-predicts near a wrist singularity and lets the trip through — do NOT raise.
    # See config_elite_cs66_rt.py and manipulability.py.joint_velocity_scale.
    joint_vel_dls_lambda: float = 1e-4

    # ── Shared RTSI state stream ──
    rtsi_frequency: float = 250.0
    rtsi_output_recipe: str | Path | None = None
    rtsi_input_recipe: str | Path | None = None

    # ── Shared startup / shutdown behavior ──
    connect_timeout_s: float = 10.0
    external_control_settle_s: float = 1.0
    servo_failure_tolerance_ticks: int = 250

    # ── Per-arm EliteDriver local TCP port offsets ──
    # Each EliteDriver opens host-side reverse/trajectory/script-command TCP
    # servers (SDK defaults 50001/50003/50004, +script_sender 50002). Two drivers
    # on one host must NOT share these, so the right arm's block is offset. The
    # SDK templates the offset ports into the pushed external_control.script, so
    # the controller connects back to the matching ports. Must differ between arms.
    left_driver_port_offset: int = 0
    right_driver_port_offset: int = 10

    # ── Per-arm Home / Start poses (J1..J6 radians) ──
    # Defaults are the single-arm candle pose; every real bench overrides all four
    # in its recipe (the arms are mounted at an angle, so the candle is not a
    # usable start there).
    left_start_position_rad: list[float] = field(
        default_factory=lambda: [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0]
    )
    right_start_position_rad: list[float] = field(
        default_factory=lambda: [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0]
    )
    left_home_position_rad: list[float] = field(
        default_factory=lambda: [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0]
    )
    right_home_position_rad: list[float] = field(
        default_factory=lambda: [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0]
    )
    start_move_duration_s: float = 3.0
    home_move_duration_s: float = 3.0
    # Controller-side reverse-socket recv budget for trajectory (MoveJ) commands.
    # This also arms the recv timeout that must survive the MoveJ -> servo-loop
    # HANDOFF: after the final trajectory writeIdle, the controller waits this
    # long for the servo loop's first command (thread start + FIFO-set + first
    # writeServoj/writeIdle). Must be >= the worst-case handoff latency, so keep
    # it >= command_timeout_ms (was 200ms, which under FIFO contention is shorter
    # than the handoff -> controller times out -> "socket timed out waiting for
    # command on reverse_socket" -> RST -> writeServoj fails N ticks).
    move_j_timeout_ms: int = 800

    # ── Shared servoj trace ──
    trace_servoj: bool = True
    trace_translation_threshold: float = 0.05
    trace_rotation_threshold: float = 0.5
    trace_joint_threshold: float = 0.3

    # ── Grippers: per-arm serial (USB) Xense gripper; left/right auto-resolved at
    # connect by board-SN parity (odd SN → left, even SN → right). No IP/MAC/network.
    # Set {side}_use_gripper=False to run without a gripper on that arm. Tactile
    # sensors are separate XenseTactileCamera devices (see enable_tactile_sensors),
    # not the gripper.
    left_use_gripper: bool = True
    left_gripper_serial_timeout: float = 1.0

    right_use_gripper: bool = True
    right_gripper_serial_timeout: float = 1.0

    # Shared serial-gripper mechanical / motion parameters.
    gripper_min_pos: float = 0.0   # mm — fully closed
    gripper_max_pos: float = 85.0  # mm — fully open
    gripper_v_max: float = 100.0   # mm/s
    gripper_f_max: float = 30.0    # N
    gripper_init_open: bool = True

    # ── Gripper backend ── "serial" (default; per-arm XenseSerialGripper over USB,
    # left/right auto-resolved by board-SN parity) or "taccap_follower" (xense.taccap
    # FollowerGripper, MIT impedance; left/right auto-resolved from the firmware-burned
    # SN, and the wrist + GSPS tactile cameras auto-discovered at connect). The serial
    # fields above are ignored in taccap_follower mode.
    gripper_type: str = "serial"
    # TacCap follower control params (used when gripper_type == "taccap_follower").
    taccap_kp: float = 8.0          # MIT impedance stiffness (Nm/rad)
    taccap_kd: float = 1.0          # MIT impedance damping (Nm·s/rad)
    taccap_grip_ff: float = 0.0     # constant MIT feed-forward torque (Nm); NEGATIVE = clamp harder, POSITIVE = open. |ff|<=3.5
    taccap_control_hz: int = 200    # ControlLoop resubmit rate
    taccap_auto_discover_cameras: bool = True  # sniff wrist + GSPS tactile SNs at connect
    # Same idea for serial (parallel-jaw) grippers: each arm's gripper board, wrist
    # camera and two tactile sensors share one USB hub, so the gripper — already
    # side-resolved by board-SN parity — identifies the hub and the cameras follow.
    # See serial_discovery.discover_serial_gripper_cameras.
    #
    # OFF by default: the recipe stays the single source of truth, which is
    # the verified behaviour. Turn it on per bench once the discovered SNs are
    # confirmed to match the recipe's — then swapping a sensor is no longer a
    # config edit.
    serial_auto_discover_cameras: bool = False
    taccap_require_calibrated: bool = True     # False only for bring-up (uncalibrated control)

    # Separate tactile sensors (XenseTactileCamera) attached at the bimanual
    # camera level when enabled; SNs come from the recipe.
    enable_tactile_sensors: bool = True

    # ── Robot payload (tool + gripper) ── Declared to each controller via setPayload at
    # connect. The CS66 has NO F/T sensor: collision detection is model/current-based, so an
    # UNDECLARED payload makes the controller read the tool's own weight & inertia as external
    # force and protective-stop on light contact (e.g. wiping a board). None = leave the
    # controller's existing payload (previous behavior). mass in kg; cog in m relative to the
    # flange frame. Verified fleet value: 0.7 kg @ [0, 0, 0.19].
    payload_mass: float | None = None
    payload_cog: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.19])

    # Auto-created in __post_init__. Do not set directly. None when no gripper.
    left_gripper: SerialGripperConfig | TaccapFollowerConfig | None = field(default=None, init=False)
    right_gripper: SerialGripperConfig | TaccapFollowerConfig | None = field(default=None, init=False)
    # Set in __post_init__: in taccap_follower + auto-discover mode the wrist/tactile
    # cameras are sniffed by the robot at connect rather than pinned in the recipe.
    _taccap_autodiscover: bool = field(default=False, init=False)
    _serial_autodiscover: bool = field(default=False, init=False)

    # Bimanual cameras (head + per-arm wrist + tactiles). Supplied by the recipe.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        self._validate_shared_servo_params()

        for side in ("left", "right"):
            # [] means "ignore the matrix, use the angles" — collapse it to None so
            # the driver's `is not None` check stays a two-way test.
            if getattr(self, f"{side}_world_rotation") == []:
                setattr(self, f"{side}_world_rotation", None)

        # ── Per-arm pose validation ──
        for name, pose in (
            ("left_start_position_rad", self.left_start_position_rad),
            ("right_start_position_rad", self.right_start_position_rad),
            ("left_home_position_rad", self.left_home_position_rad),
            ("right_home_position_rad", self.right_home_position_rad),
        ):
            if len(pose) != 6:
                raise ValueError(f"{name} must have 6 elements (J1..J6), got {len(pose)}")

        # ── Serial gripper config (per arm) ──
        self.left_gripper = self._build_gripper_config(
            self.left_use_gripper, self.left_gripper_serial_timeout, side="left",
        )
        self.right_gripper = self._build_gripper_config(
            self.right_use_gripper, self.right_gripper_serial_timeout, side="right",
        )

        # ── Bimanual cameras ── In taccap_follower + auto-discover mode the wrist + GSPS
        # tactile cameras change with the gripper, so the robot sniffs them at connect
        # (see _inject_taccap_cameras) and the recipe only pins the head. Any other
        # mode expects the recipe to pin every camera.
        self._taccap_autodiscover = (
            self.gripper_type == "taccap_follower" and self.taccap_auto_discover_cameras
        )
        self._serial_autodiscover = (
            self.gripper_type == "serial" and self.serial_auto_discover_cameras
        )

    def _validate_shared_servo_params(self) -> None:
        if self.payload_mass is not None:
            if self.payload_mass < 0 or self.payload_mass > 10.0:
                raise ValueError(f"payload_mass must be in [0, 10] kg when set, got {self.payload_mass}")
            if len(self.payload_cog) != 3:
                raise ValueError(f"payload_cog must have 3 elements [x, y, z] (m), got {self.payload_cog}")
        if not 0.002 <= self.servoj_time <= 0.01:
            raise ValueError(
                "servoj_time must be in [0.002, 0.01] s (CS-series RT envelope), "
                f"got {self.servoj_time}"
            )
        if not 0.03 <= self.servoj_lookahead_time <= 0.2:
            raise ValueError(
                "servoj_lookahead_time must be in [0.03, 0.2] (Elite SDK requirement), "
                f"got {self.servoj_lookahead_time}"
            )
        if not 100 <= self.servoj_gain <= 2000:
            raise ValueError(
                "servoj_gain must be in [100, 2000] (Elite SDK requirement), "
                f"got {self.servoj_gain}"
            )
        for _name, _val in (
            ("max_lin_speed", self.max_lin_speed),
            ("max_ang_speed", self.max_ang_speed),
            ("max_reach_radius", self.max_reach_radius),
        ):
            if _val is not None and _val <= 0:
                raise ValueError(f"{_name} must be > 0 when set (None disables it), got {_val}")
        _validate_singularity_params(self)
        if self.command_timeout_ms < 5:
            raise ValueError(
                f"command_timeout_ms must be >= 5 (Elite SDK lower bound), got {self.command_timeout_ms}"
            )
        if self.command_stale_timeout_s <= 0:
            raise ValueError(
                f"command_stale_timeout_s must be > 0, got {self.command_stale_timeout_s}"
            )
        if self.command_stale_timeout_s * 1000 < self.command_timeout_ms:
            raise ValueError(
                "command_stale_timeout_s * 1000 must be >= command_timeout_ms "
                f"(host stale must trigger later than controller timeout); "
                f"got command_stale_timeout_s={self.command_stale_timeout_s}s, "
                f"command_timeout_ms={self.command_timeout_ms}ms"
            )
        if self.reset_duration_s <= 0:
            raise ValueError(f"reset_duration_s must be > 0, got {self.reset_duration_s}")
        if self.rtsi_frequency <= 0:
            raise ValueError(f"rtsi_frequency must be > 0, got {self.rtsi_frequency}")
        if self.connect_timeout_s <= 0:
            raise ValueError(f"connect_timeout_s must be > 0, got {self.connect_timeout_s}")
        if self.start_move_duration_s <= 0:
            raise ValueError(
                f"start_move_duration_s must be > 0, got {self.start_move_duration_s}"
            )
        if self.home_move_duration_s <= 0:
            raise ValueError(
                f"home_move_duration_s must be > 0, got {self.home_move_duration_s}"
            )
        if self.move_j_timeout_ms < 5:
            raise ValueError(
                f"move_j_timeout_ms must be >= 5 (Elite SDK lower bound; mirrors "
                f"command_timeout_ms), got {self.move_j_timeout_ms}"
            )
        if self.external_control_settle_s < 0:
            raise ValueError(
                f"external_control_settle_s must be >= 0, got {self.external_control_settle_s}"
            )
        if self.servo_failure_tolerance_ticks < 1:
            raise ValueError(
                f"servo_failure_tolerance_ticks must be >= 1, "
                f"got {self.servo_failure_tolerance_ticks}"
            )
        # Each arm uses a 4-port block based at SDK defaults 50001..50004 plus its
        # offset. The two blocks must not overlap (|Δoffset| >= 4) and must stay in
        # the unprivileged TCP range.
        for name, off in (
            ("left_driver_port_offset", self.left_driver_port_offset),
            ("right_driver_port_offset", self.right_driver_port_offset),
        ):
            if not (0 <= 50001 + off and 50004 + off <= 65535):
                raise ValueError(
                    f"{name}={off} pushes EliteDriver ports outside the valid TCP range "
                    "(50001..50004 + offset must stay within 1024..65535)"
                )
        if abs(self.left_driver_port_offset - self.right_driver_port_offset) < 4:
            raise ValueError(
                "left_driver_port_offset and right_driver_port_offset must differ by >= 4 so the "
                "two arms' EliteDriver port blocks (reverse/sender/trajectory/script_command) do "
                f"not overlap; got left={self.left_driver_port_offset}, "
                f"right={self.right_driver_port_offset}"
            )
        if (
            self.use_background_servo_loop
            and self.control_mode != BiEliteCS66RTControlMode.CARTESIAN_SERVO
        ):
            raise ValueError(
                "use_background_servo_loop=True is only supported with control_mode=CARTESIAN_SERVO. "
                "Set use_background_servo_loop=False for joint servo mode."
            )

    def _build_gripper_config(
        self, use_gripper: bool, serial_timeout: float, side: str
    ) -> "SerialGripperConfig | TaccapFollowerConfig | None":
        # Presence is per-arm: set {side}_use_gripper=False to run without a gripper.
        if not use_gripper:
            return None
        if self.gripper_type == "taccap_follower":
            # left/right resolved from the firmware-burned SN; wrist + GSPS tactile
            # cameras auto-discovered by the robot at connect.
            return TaccapFollowerConfig(
                side=side,
                kp=self.taccap_kp,
                kd=self.taccap_kd,
                feedforward_torque=self.taccap_grip_ff,
                control_hz=self.taccap_control_hz,
                init_open=self.gripper_init_open,
                require_calibrated=self.taccap_require_calibrated,
            )
        if self.gripper_type != "serial":
            raise ValueError(
                f"gripper_type must be 'serial' or 'taccap_follower', got {self.gripper_type!r}"
            )
        if self.gripper_min_pos >= self.gripper_max_pos:
            raise ValueError(
                "gripper_min_pos must be smaller than gripper_max_pos, got "
                f"{self.gripper_min_pos} >= {self.gripper_max_pos}"
            )
        # serial (default): left/right self-sorts by board-SN parity (odd → left,
        # even → right); baudrate uses the SerialGripperConfig default (115200).
        return SerialGripperConfig(
            side=side,
            serial_timeout=serial_timeout,
            gripper_min_pos=self.gripper_min_pos,
            gripper_max_pos=self.gripper_max_pos,
            gripper_v_max=self.gripper_v_max,
            gripper_f_max=self.gripper_f_max,
            init_open=self.gripper_init_open,
        )

    @property
    def _autodiscover_cameras(self) -> bool:
        """True when the driver, not the recipe, supplies wrist + tactile cameras.

        Either gripper backend can do it; they differ only in what anchors a side
        (taccap: firmware SN; serial: board-SN parity). Everything downstream is
        the same, so callers branch on this rather than on which backend is in use.
        """
        return self._taccap_autodiscover or self._serial_autodiscover
