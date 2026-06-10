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
names and defaults, while station-specific values (controller IPs, serial gripper
SNs, start/home poses, camera SNs) are bundled per-arm and selected by
``bi_mount_type`` preset. Action / observation keys are ``left_``/``right_`` prefixed.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.xense import XenseOutputType, XenseTactileCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.grippers import SerialGripperConfig


class BiEliteCS66RTControlMode(str, Enum):
    CARTESIAN_SERVO = "cartesian_servo"
    JOINT_SERVO = "joint_servo"


# Per-station presets. Each bundles both controllers' IPs, serial gripper board
# SNs, the MoveJ start/home poses (J1..J6, radians), and camera SNs (head, per-arm
# wrist, per-arm tactile). Tactile camera labels are pre-namespaced
# (left_tactile_* / right_tactile_*) so the flat observation dict stays unique.
#
# NOTE: the "diagonal" preset below describes the real station — the two Elite
# CS66 arms are mounted diagonally/opposed (tilt about base-X + rotate about Z,
# see the bi-arm mounting transform). It has real controller IPs and per-arm
# start/home joint poses (measured on the station). Grippers are serial (USB,
# addressed by board SN — no IP/MAC). Gripper SNs and tactile sensor SNs are
# still PLACEHOLDER — replace the TODO-marked values before deployment.
#
# Per-arm mounting → the world←base rotation R = Rz(γ)·Rz(β)·Rx(α) that lifts
# base-frame TCP poses into a SHARED gravity-aligned world frame (x = facing,
# y = left, z = up), matching the Flexiv convention so data is comparable.
#   α = tilt about base-X, β = rotate about Z  — BOTH from the teach pendant
#       (both controllers read α=45°, β=90°). These two only fix the gravity
#       vector in base (i.e. recover "z up"); they do NOT fix the heading.
#   γ = extra yaw about world-Z to align each arm's heading into the ONE shared
#       world frame. The two arms are mounted symmetrically/diagonally (point-
#       symmetric), so they face opposite directions: right γ=0° (defines the
#       world), left γ=180°. The teach pendant cannot show this 180° (it's the
#       residual yaw-about-gravity; see the bi-arm mounting-transform note).
# Right (Rz(90°)·Rx(45°)) is validated empirically (base-X 130° pose → tool +Z
# approach axis straight down). Verify the left 180° on-station via RTSI reads.
_CANDLE_POSE = [0.0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0]

# Measured station start/home joint poses (J1..J6, radians).
#   left  = [-80, -30,  90, -30,  90, 0] deg
#   right = [-100, -150, -90, -150, -90, 0] deg
_LEFT_START_POSE = [-1.3963, -0.5236, 1.5708, -0.5236, 1.5708, 0.0]
_RIGHT_START_POSE = [-1.7453, -2.6180, -1.5708, -2.6180, -1.5708, 0.0]

_PRESETS: dict[str, dict] = {
    "diagonal": {
        "left_ip": "192.168.8.53",
        "right_ip": "192.168.8.223",
        "left_local_ip": "",
        "right_local_ip": "",
        # Serial-gripper board SNs (read over CH340; same unit numbering as the
        # wrist cameras XC0000xx). Empty -> set *_use_gripper=False.
        "left_gripper_sn": "000045",
        "right_gripper_sn": "000046",
        "left_start": list(_LEFT_START_POSE),
        "right_start": list(_RIGHT_START_POSE),
        "left_home": list(_LEFT_START_POSE),
        "right_home": list(_RIGHT_START_POSE),
        # Per-arm mounting → R = Rz(γ)·Rz(β)·Rx(α). tilt/zrot are the teach-pendant
        # readings (both arms 45/90, fix the gravity vector only); world_yaw γ aligns
        # headings into one shared world frame: right 0° (reference), left 180°
        # (arms are point-symmetric / face opposite ways). See note above.
        "right_tilt_deg": 45.0,
        "right_zrot_deg": 90.0,
        "right_world_yaw_deg": 0.0,
        "left_tilt_deg": 45.0,
        "left_zrot_deg": 90.0,
        "left_world_yaw_deg": 180.0,
        # Explicit world<-base rotation (rows = world X/Y/Z axes in base). When set
        # it OVERRIDES the tilt/zrot/world_yaw angle build for that arm. The teach-
        # pendant 45/90 angles assumed a tilt about base-X, but the LEFT arm is
        # actually tilted 45° about base-Y (verified: freedrive-probe measurement +
        # the base-link geometry). This matrix == base rotated about Z by 90° then
        # about Y by 45°  ->  world X(fwd)=base+Y, world Z(up)=[0.707,0,0.707].
        # Right is the point-symmetric partner of the (validated) left:
        # R_right = Rz(180)·R_left  (forward flips to base-Y, up stays [0.707,0,0.707]).
        # This matches the user's right-arm recipe (Z clockwise instead of left's CCW,
        # Y 45°). VERIFY on-station with the axis test before trusting; if the heading
        # is yawed, re-measure with the freedrive probe like the left.
        "left_world_R": [
            [0.0, 1.0, 0.0],
            [-0.70710678, 0.0, 0.70710678],
            [0.70710678, 0.0, 0.70710678],
        ],
        "right_world_R": [
            [0.0, -1.0, 0.0],
            [0.70710678, 0.0, -0.70710678],
            [0.70710678, 0.0, 0.70710678],
        ],
        "head_camera_sn": "346522074942",
        "left_wrist_camera_sn": "XC000045",
        "right_wrist_camera_sn": "XC000046",
        # Tactile sensor SNs (XenseTactileCamera). left = OG001349/OG001350,
        # right = OG001351/OG001352.
        "left_tactile_camera_sn_0": "OG001349",
        "left_tactile_camera_sn_1": "OG001350",
        "right_tactile_camera_sn_0": "OG001351",
        "right_tactile_camera_sn_1": "OG001352",
    },
}


@RobotConfig.register_subclass("bi_elite_cs66_rt")
@dataclass
class BiEliteCS66RTConfig(RobotConfig):
    """Configuration for two Elite CS66 arms via elite_cs_sdk.

    Each arm runs its own EliteDriver + RTSI stream + (optional) background
    Cartesian servo loop. ``send_action``/``get_observation`` use the same TCP /
    joint schema as the single-arm driver, ``left_``/``right_`` prefixed:
        left_tcp.x/y/z + left_tcp.r1..r6 (+ optional left_joint_*), left_gripper.pos
        right_tcp.x/y/z + right_tcp.r1..r6 (+ optional right_joint_*), right_gripper.pos
    Grippers are per-arm serial (USB) devices addressed by board SN. Cameras
    (head + per-arm wrist + optional tactiles) live at the bimanual level; tactile
    images come from separate XenseTactileCamera devices, not the gripper.
    """

    # ── Per-arm identity / connection ──
    # robot IPs default to "" (unset) → filled from the bi_mount_type preset in
    # __post_init__. An explicitly-passed --robot.{left,right}_robot_ip OVERRIDES
    # the preset. local_ip is taken from the preset.
    left_robot_ip: str = ""
    right_robot_ip: str = ""
    left_local_ip: str = ""
    right_local_ip: str = ""
    bi_mount_type: str = "diagonal"

    # ── Per-arm mounting → base↔world rotation (overwritten from the preset) ──
    # R_world←base = Rz(γ)·Rz(β)·Rx(α): α=tilt about base-X, β=rotate about Z (both
    # from the teach pendant, fixing only the gravity vector), γ=extra world-Z yaw
    # that aligns each arm's heading into ONE shared world frame (x=facing, y=left,
    # z=up). The driver lifts base→world in get_observation and maps world→base in
    # send_action. Both pendants read α=45/β=90; the arms are point-symmetric so
    # left needs γ=180° (right γ=0° defines the world). Right is validated.
    left_mount_tilt_deg: float = 45.0
    left_mount_zrot_deg: float = 90.0
    left_mount_world_yaw_deg: float = 180.0
    right_mount_tilt_deg: float = 45.0
    right_mount_zrot_deg: float = 90.0
    right_mount_world_yaw_deg: float = 0.0

    # Optional explicit per-arm world<-base rotation (3x3, rows = world X/Y/Z in
    # base). When not None it OVERRIDES the angle build above for that arm — used
    # when the mounting isn't a clean Rz·Rx (e.g. the left arm tilts about base-Y).
    # Populated from the preset's {side}_world_R.
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

    # ── Per-arm Home / Start poses (J1..J6 radians; overwritten from preset) ──
    left_start_position_rad: list[float] = field(default_factory=lambda: list(_CANDLE_POSE))
    right_start_position_rad: list[float] = field(default_factory=lambda: list(_CANDLE_POSE))
    left_home_position_rad: list[float] = field(default_factory=lambda: list(_CANDLE_POSE))
    right_home_position_rad: list[float] = field(default_factory=lambda: list(_CANDLE_POSE))
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

    # ── Grippers: per-arm serial (USB) Xense gripper, addressed by board SN ──
    # No IP/MAC/network. Set {side}_use_gripper=False (or leave the preset SN
    # empty) to run without a gripper on that arm. Tactile sensors are separate
    # XenseTactileCamera devices (see enable_tactile_sensors), not the gripper.
    left_use_gripper: bool = True
    left_gripper_sn: str = ""               # board SN (overwritten from preset)
    left_gripper_baudrate: int = 115200
    left_gripper_serial_timeout: float = 1.0

    right_use_gripper: bool = True
    right_gripper_sn: str = ""              # board SN (overwritten from preset)
    right_gripper_baudrate: int = 115200
    right_gripper_serial_timeout: float = 1.0

    # Shared serial-gripper mechanical / motion parameters.
    gripper_min_pos: float = 0.0   # mm — fully closed
    gripper_max_pos: float = 85.0  # mm — fully open
    gripper_v_max: float = 100.0   # mm/s
    gripper_f_max: float = 30.0    # N
    gripper_init_open: bool = True

    # Separate tactile sensors (XenseTactileCamera) attached at the bimanual
    # camera level when enabled; SNs come from the preset.
    enable_tactile_sensors: bool = True

    # Auto-created in __post_init__. Do not set directly. None when no gripper.
    left_gripper: SerialGripperConfig | None = field(default=None, init=False)
    right_gripper: SerialGripperConfig | None = field(default=None, init=False)

    # Bimanual cameras (head + per-arm wrist + tactiles). Auto-populated from the preset.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        self._validate_shared_servo_params()

        # ── Apply preset ──
        if self.bi_mount_type not in _PRESETS:
            raise ValueError(
                f"Unknown bi_mount_type {self.bi_mount_type!r}, expected one of {list(_PRESETS)}"
            )
        preset = _PRESETS[self.bi_mount_type]
        # CLI-passed IPs win; fall back to the preset when left unset ("").
        self.left_robot_ip = self.left_robot_ip or preset["left_ip"]
        self.right_robot_ip = self.right_robot_ip or preset["right_ip"]
        self.left_local_ip = preset["left_local_ip"]
        self.right_local_ip = preset["right_local_ip"]
        self.left_gripper_sn = preset["left_gripper_sn"]
        self.right_gripper_sn = preset["right_gripper_sn"]
        self.left_start_position_rad = list(preset["left_start"])
        self.right_start_position_rad = list(preset["right_start"])
        self.left_home_position_rad = list(preset["left_home"])
        self.right_home_position_rad = list(preset["right_home"])
        self.left_mount_tilt_deg = preset["left_tilt_deg"]
        self.left_mount_zrot_deg = preset["left_zrot_deg"]
        self.left_mount_world_yaw_deg = preset["left_world_yaw_deg"]
        self.right_mount_tilt_deg = preset["right_tilt_deg"]
        self.right_mount_zrot_deg = preset["right_zrot_deg"]
        self.right_mount_world_yaw_deg = preset["right_world_yaw_deg"]
        self.left_world_rotation = preset.get("left_world_R")
        self.right_world_rotation = preset.get("right_world_R")

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
            self.left_use_gripper, self.left_gripper_sn,
            self.left_gripper_baudrate, self.left_gripper_serial_timeout, side="left",
        )
        self.right_gripper = self._build_gripper_config(
            self.right_use_gripper, self.right_gripper_sn,
            self.right_gripper_baudrate, self.right_gripper_serial_timeout, side="right",
        )

        # ── Bimanual cameras from preset (head + wrists + optional tactiles) ──
        self._build_cameras(preset)

    def _validate_shared_servo_params(self) -> None:
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
        self, use_gripper: bool, sn: str, baudrate: int, serial_timeout: float, side: str
    ) -> SerialGripperConfig | None:
        # No gripper when disabled or when no board SN is configured yet (mirrors
        # the camera handling: an empty preset SN simply means "not attached").
        if not use_gripper or not sn:
            return None
        if self.gripper_min_pos >= self.gripper_max_pos:
            raise ValueError(
                "gripper_min_pos must be smaller than gripper_max_pos, got "
                f"{self.gripper_min_pos} >= {self.gripper_max_pos}"
            )
        return SerialGripperConfig(
            sn=sn,
            baudrate=baudrate,
            serial_timeout=serial_timeout,
            gripper_min_pos=self.gripper_min_pos,
            gripper_max_pos=self.gripper_max_pos,
            gripper_v_max=self.gripper_v_max,
            gripper_f_max=self.gripper_f_max,
            init_open=self.gripper_init_open,
        )

    def _build_cameras(self, preset: dict) -> None:
        cameras: dict[str, CameraConfig] = {}
        if preset.get("head_camera_sn"):
            cameras["head"] = RealSenseCameraConfig(
                serial_number_or_name=preset["head_camera_sn"],
                fps=30,
                width=640,
                height=480,
                warmup_s=1.0,
            )
        if preset.get("left_wrist_camera_sn"):
            cameras["left_wrist"] = OpenCVCameraConfig(
                index_or_path=preset["left_wrist_camera_sn"],
                fourcc="MJPG",
                width=640,
                height=480,
                fps=30,
                warmup_s=1.0,
            )
        if preset.get("right_wrist_camera_sn"):
            cameras["right_wrist"] = OpenCVCameraConfig(
                index_or_path=preset["right_wrist_camera_sn"],
                fourcc="MJPG",
                width=640,
                height=480,
                fps=30,
                warmup_s=1.0,
            )
        # Tactile sensors are separate XenseTactileCamera devices (not the
        # serial gripper). Namespaced left_/right_ so the flat obs dict stays unique.
        if self.enable_tactile_sensors:
            for label, sn_key in (
                ("left_tactile_0", "left_tactile_camera_sn_0"),
                ("left_tactile_1", "left_tactile_camera_sn_1"),
                ("right_tactile_0", "right_tactile_camera_sn_0"),
                ("right_tactile_1", "right_tactile_camera_sn_1"),
            ):
                if preset.get(sn_key):
                    cameras[label] = XenseTactileCameraConfig(
                        serial_number=preset[sn_key],
                        fps=30,
                        output_types=[XenseOutputType.RECTIFY],
                        warmup_s=0.05,
                    )
        # Only override the (empty) default when the preset actually supplies
        # camera SNs, so a user-provided cameras dict isn't silently wiped.
        if cameras:
            self.cameras = cameras
