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
names and defaults, while station-specific values (controller IPs, gripper MACs,
start/home poses, camera SNs) are bundled per-arm and selected by ``bi_mount_type``
preset. Action / observation keys are ``left_``/``right_`` prefixed.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.grippers.config_xense_gripper import (
    SensorOutputType,
    XenseGripperConfig,
)


class BiEliteCS66RTControlMode(str, Enum):
    CARTESIAN_SERVO = "cartesian_servo"
    JOINT_SERVO = "joint_servo"


# Per-station presets. Each bundles both controllers' IPs, gripper MACs, the
# MoveJ start/home poses (J1..J6, radians), camera SNs, and the per-arm tactile
# sensor SN -> observation-label maps. Tactile labels are pre-namespaced
# (left_tactile_* / right_tactile_*) so the flat observation dict stays unique
# without any extra prefixing at runtime.
#
# NOTE: the "diagonal" preset below describes the real station — the two Elite
# CS66 arms are mounted diagonally/opposed (45° tilt + 90° Z, see the bi-arm
# mounting transform). It has real controller IPs and per-arm start/home joint
# poses (measured on the station). Gripper MACs, camera SNs, and tactile sensor
# SNs are still PLACEHOLDER — replace the TODO-marked values before deployment.
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
        # TODO: real XenseGripper MAC addresses (empty -> gripper_type must be "none").
        "left_gripper_mac": "",
        "right_gripper_mac": "",
        "left_start": list(_LEFT_START_POSE),
        "right_start": list(_RIGHT_START_POSE),
        "left_home": list(_LEFT_START_POSE),
        "right_home": list(_RIGHT_START_POSE),
        # TODO: real camera SNs.
        "head_camera_sn": "",
        "left_wrist_camera_sn": "",
        "right_wrist_camera_sn": "",
        # TODO: real tactile sensor SNs (values are the namespaced obs labels).
        "left_gripper_sensor_keys": {
            "OG_LEFT_0": "left_tactile_0",
            "OG_LEFT_1": "left_tactile_1",
        },
        "right_gripper_sensor_keys": {
            "OG_RIGHT_0": "right_tactile_0",
            "OG_RIGHT_1": "right_tactile_1",
        },
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
    Cameras (head + per-arm wrist) live at the bimanual level; tactile images
    arrive through each arm's XenseGripper.
    """

    # ── Per-arm identity / connection (overwritten from the preset) ──
    left_robot_ip: str = "192.168.1.200"
    right_robot_ip: str = "192.168.1.201"
    left_local_ip: str = ""
    right_local_ip: str = ""
    bi_mount_type: str = "diagonal"

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
    command_timeout_ms: int = 200
    use_background_servo_loop: bool = True
    command_stale_timeout_s: float = 0.5
    reset_duration_s: float = 3.0

    # ── Shared RTSI state stream ──
    rtsi_frequency: float = 250.0
    rtsi_output_recipe: str | Path | None = None
    rtsi_input_recipe: str | Path | None = None

    # ── Shared startup / shutdown behavior ──
    connect_timeout_s: float = 10.0
    external_control_settle_s: float = 1.0
    servo_failure_tolerance_ticks: int = 250

    # ── Per-arm Home / Start poses (J1..J6 radians; overwritten from preset) ──
    left_start_position_rad: list[float] = field(default_factory=lambda: list(_CANDLE_POSE))
    right_start_position_rad: list[float] = field(default_factory=lambda: list(_CANDLE_POSE))
    left_home_position_rad: list[float] = field(default_factory=lambda: list(_CANDLE_POSE))
    right_home_position_rad: list[float] = field(default_factory=lambda: list(_CANDLE_POSE))
    start_move_duration_s: float = 3.0
    home_move_duration_s: float = 3.0
    move_j_timeout_ms: int = 200

    # ── Shared servoj trace ──
    trace_servoj: bool = True
    trace_translation_threshold: float = 0.05
    trace_rotation_threshold: float = 0.5
    trace_joint_threshold: float = 0.3

    # ── Gripper backend (shared dispatch, per-arm device identifiers) ──
    #   "none"          - no grippers attached; gripper.pos absent from features
    #   "xense_gripper" - XenseGripper per arm (USB/network, independent of arms)
    #   "dahuan_rs485"  - planned (raises NotImplementedError until driver lands)
    gripper_type: str = "none"

    left_gripper_mac_addr: str = ""
    right_gripper_mac_addr: str = ""
    # Per-arm tactile sensor SN -> observation label maps (overwritten from preset).
    left_gripper_sensor_keys: dict[str, str] = field(default_factory=dict)
    right_gripper_sensor_keys: dict[str, str] = field(default_factory=dict)

    # Shared XenseGripper motion / sensor parameters (used when gripper_type=="xense_gripper").
    gripper_enable_sensor: bool = True
    gripper_rectify_size: tuple[int, int] = (96, 160)
    gripper_sensor_output_type: SensorOutputType = SensorOutputType.RECTIFY
    gripper_min_pos: float = 0.0
    gripper_max_pos: float = 85.0
    gripper_v_max: float = 100.0   # mm/s
    gripper_f_max: float = 30.0    # N
    gripper_init_open: bool = True

    # Auto-created in __post_init__. Do not set directly. None when no gripper.
    left_gripper: XenseGripperConfig | None = field(default=None, init=False)
    right_gripper: XenseGripperConfig | None = field(default=None, init=False)

    # Bimanual cameras (head + per-arm wrist). Auto-populated from the preset.
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
        self.left_robot_ip = preset["left_ip"]
        self.right_robot_ip = preset["right_ip"]
        self.left_local_ip = preset["left_local_ip"]
        self.right_local_ip = preset["right_local_ip"]
        self.left_gripper_mac_addr = preset["left_gripper_mac"]
        self.right_gripper_mac_addr = preset["right_gripper_mac"]
        self.left_start_position_rad = list(preset["left_start"])
        self.right_start_position_rad = list(preset["right_start"])
        self.left_home_position_rad = list(preset["left_home"])
        self.right_home_position_rad = list(preset["right_home"])
        self.left_gripper_sensor_keys = dict(preset["left_gripper_sensor_keys"])
        self.right_gripper_sensor_keys = dict(preset["right_gripper_sensor_keys"])

        # ── Per-arm pose validation ──
        for name, pose in (
            ("left_start_position_rad", self.left_start_position_rad),
            ("right_start_position_rad", self.right_start_position_rad),
            ("left_home_position_rad", self.left_home_position_rad),
            ("right_home_position_rad", self.right_home_position_rad),
        ):
            if len(pose) != 6:
                raise ValueError(f"{name} must have 6 elements (J1..J6), got {len(pose)}")

        # ── Gripper dispatch (per arm) ──
        self.left_gripper = self._build_gripper_config(
            self.left_gripper_mac_addr, self.left_gripper_sensor_keys, side="left"
        )
        self.right_gripper = self._build_gripper_config(
            self.right_gripper_mac_addr, self.right_gripper_sensor_keys, side="right"
        )

        # ── Bimanual cameras from preset (tactiles come via grippers) ──
        self._build_cameras(preset)

        # ── Feature-key collision check across both arms ──
        sensor_names: set[str] = set()
        for gripper in (self.left_gripper, self.right_gripper):
            if gripper is not None:
                sensor_names |= set(gripper.sensor_keys.values())
        overlap = sensor_names & set(self.cameras.keys())
        if overlap:
            raise ValueError(
                f"Feature key collision between gripper sensor_keys and cameras: "
                f"{sorted(overlap)}. Rename one side."
            )

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
        if (
            self.use_background_servo_loop
            and self.control_mode != BiEliteCS66RTControlMode.CARTESIAN_SERVO
        ):
            raise ValueError(
                "use_background_servo_loop=True is only supported with control_mode=CARTESIAN_SERVO. "
                "Set use_background_servo_loop=False for joint servo mode."
            )

    def _build_gripper_config(
        self, mac_addr: str, sensor_keys: dict[str, str], side: str
    ) -> XenseGripperConfig | None:
        if self.gripper_type == "none":
            return None
        if self.gripper_type == "dahuan_rs485":
            raise NotImplementedError(
                "Dahuan RS485 gripper driver (over CS66 tool RS485 via Elite SDK "
                "ScriptCommandInterface) is planned but not yet implemented."
            )
        if self.gripper_type != "xense_gripper":
            raise ValueError(
                f"gripper_type must be one of 'none' / 'xense_gripper' / "
                f"'dahuan_rs485', got {self.gripper_type!r}"
            )
        if not mac_addr:
            raise ValueError(
                f"gripper_type='xense_gripper' requires {side}_gripper_mac_addr to be set."
            )
        if self.gripper_min_pos >= self.gripper_max_pos:
            raise ValueError(
                "gripper_min_pos must be smaller than gripper_max_pos, got "
                f"{self.gripper_min_pos} >= {self.gripper_max_pos}"
            )
        return XenseGripperConfig(
            mac_addr=mac_addr,
            enable_sensor=self.gripper_enable_sensor,
            rectify_size=self.gripper_rectify_size,
            sensor_output_type=self.gripper_sensor_output_type,
            sensor_keys=dict(sensor_keys),
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
        # Only override the (empty) default when the preset actually supplies
        # camera SNs, so a user-provided cameras dict isn't silently wiped.
        if cameras:
            self.cameras = cameras
