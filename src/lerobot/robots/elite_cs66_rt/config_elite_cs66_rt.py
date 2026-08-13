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
from lerobot.grippers.xense.configuration_xense import (
    SensorOutputType,
    XenseGripperConfig,
)


class EliteCS66RTControlMode(str, Enum):
    CARTESIAN_SERVO = "cartesian_servo"
    JOINT_SERVO = "joint_servo"


def _validate_singularity_params(cfg) -> None:
    """Validate the singularity-damping fields. Shared by the single-arm and bimanual configs
    (duck-typed on the common attribute names)."""
    if cfg.singularity_w_high is not None:
        if cfg.singularity_w_low < 0 or cfg.singularity_w_high <= cfg.singularity_w_low:
            raise ValueError(
                "require singularity_w_high > singularity_w_low >= 0, got "
                f"w_high={cfg.singularity_w_high}, w_low={cfg.singularity_w_low}"
            )
    # min_scale is the damping floor. A NON-directional guard must keep it > 0 so the operator can
    # always creep out of a singularity. A directional guard handles escape separately (it returns
    # s=1 for any move that raises w), so it may use min_scale == 0 for a TRUE hold — the fix for the
    # observed creep-through, where a 0.05 floor let the arm crawl past the IK-refusal boundary.
    if cfg.singularity_directional:
        if not (0.0 <= cfg.singularity_min_scale <= 1.0):
            raise ValueError(
                f"singularity_min_scale must be in [0, 1], got {cfg.singularity_min_scale}"
            )
    elif not (0.0 < cfg.singularity_min_scale <= 1.0):
        raise ValueError(
            "singularity_min_scale must be in (0, 1] for a non-directional guard (0 would trap the "
            f"arm; set singularity_directional=true to allow a full hold), got {cfg.singularity_min_scale}"
        )
    if cfg.dh_params is not None:
        if len(cfg.dh_params) != 3 or any(len(v) != 6 for v in cfg.dh_params):
            raise ValueError("dh_params must be a 3-tuple (alpha, a, d) of length-6 lists")
    if cfg.primary_timeout_ms < 5:
        raise ValueError(f"primary_timeout_ms must be >= 5, got {cfg.primary_timeout_ms}")
    if cfg.joint_vel_limits_rad_s is not None:
        if len(cfg.joint_vel_limits_rad_s) != 6 or any(v <= 0 for v in cfg.joint_vel_limits_rad_s):
            raise ValueError(
                "joint_vel_limits_rad_s must be a length-6 list of positive per-joint velocity "
                f"limits (rad/s), got {cfg.joint_vel_limits_rad_s}"
            )
    if not (0.0 < cfg.joint_vel_limit_margin <= 1.0):
        raise ValueError(
            f"joint_vel_limit_margin must be in (0, 1], got {cfg.joint_vel_limit_margin}"
        )
    if cfg.joint_vel_horizon_s <= 0:
        raise ValueError(f"joint_vel_horizon_s must be > 0, got {cfg.joint_vel_horizon_s}")
    if cfg.joint_vel_dls_lambda <= 0:
        raise ValueError(f"joint_vel_dls_lambda must be > 0, got {cfg.joint_vel_dls_lambda}")


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
    #
    # servoj_gain is the urscript servoj() position-following P gain. Elite CS
    # does NOT have native Cartesian impedance (unlike flexiv_rizon4_rt's
    # stiffness_ratio), so lowering gain is the only knob we have to make the
    # arm yield under external force. Reference points:
    #   2000  - SDK example default, industrial stiff
    #    500  - moderate, noticeable yield but still tracks target
    #    300  - compliant (default here); rough equivalent of flexiv's
    #           stiffness_ratio=0.2 in feel, NOT in dynamics
    #   <100  - too soft; gravity drift visible on heavy payloads
    # SDK-documented range is [100, 2000]; values outside are checked in
    # __post_init__.
    servoj_time: float = 0.004
    servoj_lookahead_time: float = 0.1
    servoj_gain: int = 300
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
    start_move_duration_s: float = 3.0
    home_move_duration_s: float = 3.0
    move_j_timeout_ms: int = 200

    # Minimum total wall time between EliteDriver() construction and the
    # first writeServoj. SDK example sleeps ~1 s after isRobotConnected()
    # returns True; otherwise the controller-side script can RST the reverse
    # socket if hit too early. We collapse this to an elapsed-time check —
    # fast handshakes don't pay the full sleep.
    external_control_settle_s: float = 1.0

    # Number of consecutive writeServoj() failures the background servo
    # loop tolerates before declaring itself dead. SDK reverse-socket writes
    # can fail transiently right after the script comes up. Decoupled from
    # external_control_settle_s so shrinking the connect-time settle doesn't
    # also shrink the failure tolerance.
    # 250 ticks * 0.004 s = 1 s of failure tolerance, matching the original
    # tied-to-settle behavior.
    servo_failure_tolerance_ticks: int = 250

    # Trace every send_action and large per-step deltas to the spdlog file
    # sink (~/xenselogs). Doesn't touch console; safe to leave enabled.
    trace_servoj: bool = True
    # Per-step delta thresholds above which the trace promotes to WARNING (also
    # captured in the file log). 5 cm or 0.5 rad in a single send_action call
    # is suspicious for steady-state teleop.
    trace_translation_threshold: float = 0.05
    trace_rotation_threshold: float = 0.5
    # Joint-mode trace: max per-joint delta (rad) above which the trace
    # promotes to WARNING. 0.3 rad ≈ 17° per send_action tick — at 50 Hz outer
    # loop that's ~855°/s, well above normal leader-follower joint speeds.
    trace_joint_threshold: float = 0.3

    # Opt-in Cartesian velocity ceiling for the background servo loop. Both None
    # by default -> no PC-side clamp: the controller's pendant safety config and
    # external_control.script's JOINT_IGNORE_SPEED=30 rad/s already bound the
    # hardware envelope, and a clamp risks masking real safety incidents behind a
    # "looks smooth" behavior. Set these when a jumpy leader (fast wrist rotation,
    # VR tracking spike, clutch re-engage) feeds target steps that imply joint
    # speed above that bound and trip a protective stop: the servo loop then slews
    # the commanded TCP toward the latest target at no more than this speed,
    # turning the step into a smooth bounded ramp. Expressed as a velocity ceiling
    # (m/s, rad/s) applied per servoj_time tick — NOT a per-tick delta. Only the
    # background servo loop honors these; the direct-write path is unaffected.
    max_lin_speed: float | None = None  # m/s; None disables the linear cap
    max_ang_speed: float | None = None  # rad/s; None disables the angular cap

    # Workspace reachability guard. Distance (m) from the base-frame origin beyond
    # which a commanded TCP target is treated as unreachable: the driver then HOLDS
    # the last in-reach pose instead of sending it, so the operator can't drive the
    # arm into the boundary singularity where the controller's IK fails and drops
    # external control. Conservative spherical guard from the base ORIGIN (not an
    # exact reachability test — the true workspace is offset to the shoulder, so it
    # can read too tight in some directions). Real reach ~0.91 m for CS66.
    #
    # DISABLED by default (None) per operator request 2026-07-03: 0.85 m was clipping
    # legitimate reach and holding the arm mid-teleop. Trade-off to be aware of: with
    # the guard off, over-reaching no longer HOLDS gracefully — the command is sent,
    # the controller hits the boundary singularity, and external control DROPS. Set a
    # value (e.g. 0.88-0.90, below the ~0.91 physical reach) to re-enable if that
    # dropout becomes a problem. Applies to both the background and direct servo paths.
    max_reach_radius: float | None = None  # m; None disables the guard

    # Singularity-aware manipulability damping (model-based). Detects proximity to ANY
    # kinematic singularity (wrist q5≈0, shoulder, elbow/boundary) from the arm's Modified-DH
    # kinematics and smoothly slows the Cartesian command before the controller's IK spikes
    # joint velocity past JOINT_IGNORE_SPEED and drops external control. The metric is
    # w = |det(J)| at the current joints (tool- and frame-invariant). When w drops below
    # singularity_w_high, the target is pulled toward the last commanded pose by a scale
    # s = clamp((w-w_low)/(w_high-w_low), s_min, 1); s_min>0 always permits slow escape.
    #
    # OFF by default (w_high=None): the per-arm DH must be read from the live controller and
    # w_low/w_high tuned from logged values, so enable only after that. Set log_manipulability
    # to print w (throttled) while teleoperating near a singular pose to read off the band.
    # If the DH fetch or the connect-time FK self-check fails, damping disables itself
    # (fail-safe to the no-damping behavior). See manipulability.py for the kinematics.
    singularity_w_high: float | None = None  # disables damping when None
    singularity_w_low: float = 0.0  # w at/below which damping is maxed (s = s_min)
    singularity_min_scale: float = 0.05  # s_min in (0, 1]; floor so escape is always possible
    singularity_directional: bool = False  # don't damp moves that increase w (escape); needs live tuning
    # Optional (alpha, a, d) Modified-DH override (each length-6) if the controller fetch is
    # unavailable; None -> read from the controller via getPrimaryPackage at connect.
    dh_params: tuple[list[float], list[float], list[float]] | None = None
    log_manipulability: bool = False  # throttled debug log of w for live threshold tuning
    primary_timeout_ms: int = 1000  # one-shot DH (KinematicsInfo) fetch timeout at connect

    # Predicted joint-velocity limit scaling (model-based, MoveIt-Servo style). Complements the
    # w-band damping above and shares the same DH (fetched at connect; the guard self-disables on
    # any DH fetch / self-check failure, exactly like the damping). Before a Cartesian target is
    # sent, the DLS pseudo-inverse of the geometric Jacobian predicts the joint step dq for the
    # commanded step; if any |dq_i| would exceed joint_vel_limits_rad_s[i] * joint_vel_limit_margin
    # over joint_vel_horizon_s, the whole Cartesian step is scaled down uniformly (preserving EE
    # direction). This bounds joint velocity BEFORE the controller's internal IK spikes it past
    # JOINT_IGNORE_SPEED (30 rad/s) and drops external control — the fix for wrist-singularity
    # rotation trips, where a small TCP rotation maps to a huge joint velocity. It is naturally
    # directional (holds moves into a singularity, allows escape) and needs only DH + datasheet
    # joint limits — no live w-band tuning. OFF by default (limits None). Official CS66 datasheet
    # rates (Elite CS-Series): J1/J2 150°/s, J3 180°/s, J4-6 230°/s
    # -> [2.618, 2.618, 3.142, 4.014, 4.014, 4.014] rad/s.
    joint_vel_limits_rad_s: list[float] | None = None  # per-joint [J1..J6] rad/s; None disables
    joint_vel_limit_margin: float = 0.8  # enforce at this fraction of the limits (headroom, (0, 1])
    joint_vel_horizon_s: float = 0.033  # command horizon for the velocity check (~ 1/teleop fps)
    # DLS damping of the PREDICTION pseudo-inverse. Must stay a couple orders below the operating
    # sigma_min (~4e-3 entering the trip) so the predicted dq tracks the controller's near-exact IK
    # (~1/sigma_min) instead of capping the spike at 1/(2*lambda); a lambda ~ sigma_min under-predicts
    # and lets the trip through. 1e-4 verified vs the datasheet limits; do NOT raise toward 1e-2.
    joint_vel_dls_lambda: float = 1e-4

    # Gripper backend dispatch. Mirrors the flexiv_rizon4_rt pattern: a single
    # string selects the driver, the gripper-specific config is auto-built from
    # exposed gripper_* fields in __post_init__.
    #   "none"          - no gripper attached; gripper.pos absent from features
    #   "xense_gripper" - XenseGripper (USB/network, independent of arm)
    #   "dahuan_rs485"  - planned: Dahuan industrial gripper over CS66 tool RS485
    #                     (raises NotImplementedError until driver lands)
    gripper_type: str = "none"

    # XenseGripper-specific parameters (used when gripper_type=="xense_gripper").
    # Mirror flexiv's gripper_* field set so a Xense gripper can move between
    # arms without reconfiguring.
    gripper_mac_addr: str = ""
    gripper_enable_sensor: bool = True
    gripper_rectify_size: tuple[int, int] = (96, 160)
    gripper_sensor_output_type: SensorOutputType = SensorOutputType.RECTIFY
    gripper_sensor_keys: dict[str, str] = field(default_factory=dict)
    gripper_min_pos: float = 0.0
    gripper_max_pos: float = 85.0
    gripper_v_max: float = 100.0   # mm/s
    gripper_f_max: float = 30.0    # N
    gripper_init_open: bool = True

    # Auto-created in __post_init__ from gripper_* parameters. Do not set
    # directly. None when gripper_type=="none".
    gripper: XenseGripperConfig | None = field(default=None, init=False)

    # External cameras.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        if not 0.002 <= self.servoj_time <= 0.01:
            # Below 2 ms the controller's inner servo loop can't keep up; above
            # 10 ms the script's per-tick extrapolation window grows large and
            # target updates feel laggy. Default 4 ms matches the SDK example
            # (250 Hz, the CS-series RT controller's native rate).
            raise ValueError(
                "servoj_time must be in [0.002, 0.01] s (CS-series RT envelope), "
                f"got {self.servoj_time}"
            )
        if not 0.03 <= self.servoj_lookahead_time <= 0.2:
            # Elite SDK EliteDriver.hpp says lookahead time must lie in [0.03, 0.2];
            # values outside this range cause the external_control script to abort
            # and tear down all reverse sockets (50001/50003/50004), leaving the
            # Python side writing into a dead socket forever.
            raise ValueError(
                "servoj_lookahead_time must be in [0.03, 0.2] (Elite SDK requirement), "
                f"got {self.servoj_lookahead_time}"
            )
        if not 100 <= self.servoj_gain <= 2000:
            # SDK-documented range for servoj() position-following gain.
            # Outside the range the controller's IK behavior is unspecified
            # (very low: unable to track even slow targets; very high: IK
            # oscillation / velocity-limit trips).
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
            # The host-side stale gate must trip later than the controller's
            # own command timeout. Otherwise, during the window between
            # command_timeout_ms (controller stops) and command_stale_timeout_s
            # (host writes idle), nobody is sending anything and the controller
            # has already entered its self-protect stop, causing surprise
            # halts during transient outer-loop hiccups. Keeping
            # command_stale_timeout_s >> command_timeout_ms ensures the host
            # always remains the active party in the control channel.
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
        if (
            self.use_background_servo_loop
            and self.control_mode != EliteCS66RTControlMode.CARTESIAN_SERVO
        ):
            raise ValueError(
                "use_background_servo_loop=True is only supported with control_mode=CARTESIAN_SERVO. "
                "Set use_background_servo_loop=False for joint servo mode."
            )
        # Gripper dispatch
        if self.gripper_type == "xense_gripper":
            if not self.gripper_mac_addr:
                raise ValueError(
                    "gripper_type='xense_gripper' requires gripper_mac_addr to be set."
                )
            if self.gripper_min_pos >= self.gripper_max_pos:
                raise ValueError(
                    "gripper_min_pos must be smaller than gripper_max_pos, got "
                    f"{self.gripper_min_pos} >= {self.gripper_max_pos}"
                )
            self.gripper = XenseGripperConfig(
                mac_addr=self.gripper_mac_addr,
                enable_sensor=self.gripper_enable_sensor,
                rectify_size=self.gripper_rectify_size,
                sensor_output_type=self.gripper_sensor_output_type,
                sensor_keys=self.gripper_sensor_keys,
                gripper_min_pos=self.gripper_min_pos,
                gripper_max_pos=self.gripper_max_pos,
                gripper_v_max=self.gripper_v_max,
                gripper_f_max=self.gripper_f_max,
                init_open=self.gripper_init_open,
            )
        elif self.gripper_type == "dahuan_rs485":
            raise NotImplementedError(
                "Dahuan RS485 gripper driver (over CS66 tool RS485 via Elite SDK "
                "ScriptCommandInterface) is planned but not yet implemented."
            )
        elif self.gripper_type != "none":
            raise ValueError(
                f"gripper_type must be one of 'none' / 'xense_gripper' / "
                f"'dahuan_rs485', got {self.gripper_type!r}"
            )

        # Cross-check: gripper sensor names and camera names land in the same
        # observation/features dict (see EliteCS66RT.observation_features). A
        # collision silently lets one entry overwrite the other in dict
        # assignment order, producing a corrupt dataset where the same key
        # alternates between a tactile rectify and a camera frame from step
        # to step. Fail loud at config time instead.
        if self.gripper is not None:
            sensor_names = set(self.gripper.sensor_keys.values())
            camera_names = set(self.cameras.keys())
            overlap = sensor_names & camera_names
            if overlap:
                raise ValueError(
                    f"Feature key collision between gripper sensor_keys and "
                    f"cameras: {sorted(overlap)}. Rename one side."
                )
