#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Configuration for the bimanual TacCap-Gripper handheld data-collection rig.

Two independent TacCap-Gripper units (left + right), each = a motor-driven jaw
(encoder read-only), two embedded visuotactile sensors, a wrist UVC camera, an
IMU, plus a Pico4 Ultra motion tracker mounted on top for 6-DoF pose.

This is the bimanual analogue of ``taccap_gripper`` (the single unit). It follows
the *reimplement-with-prefixes* convention used by ``bi_elite_cs66_rt``: a single
flat config with ``left_``/``right_`` prefixed fields, and observation/action keys
prefixed the same way so the flat dict stays unique.

Per-side identity / cameras come from the operator (the MCU-only SDK no longer
reports them):
- ``{side}_firmware_sn`` pins the gripper (None => find_one, errors on 0 or >1).
- ``{side}_tracker_sn`` pins the Pico4 tracker (None => first available — ambiguous
  when two trackers are present, so pin both for a real bimanual rig).
- ``{side}_wrist_camera_index_or_path`` is the wrist UVC V4L2 path.
- tactile sensors go in ``cameras`` with pre-prefixed keys
  (``left_tactile_0``/``left_tactile_1``/``right_tactile_0``/``right_tactile_1``).
"""

from dataclasses import dataclass, field

from lerobot.cameras.utils import CameraConfig
from lerobot.cameras.xense.configuration_xense import XenseTactileCameraConfig

from ..config import RobotConfig

_SIDES = ("left", "right")

_DEFAULT_INIT_TCP_POSE = (
    0.693307,
    -0.114902,
    0.14589,
    0.004567,
    0.003238,
    0.999984,
    0.001246,
)


@RobotConfig.register_subclass("bi_taccap_gripper")
@dataclass
class BiTaccapGripperConfig(RobotConfig):
    """Configuration for the bimanual TacCap-Gripper data-collection rig.

    See ``TaccapGripperConfig`` for the per-unit semantics; every field below is
    the ``left_``/``right_`` prefixed version of a single-unit field. Gripper
    position is normalised via ``clip(position_rad / {side}_gripper_open_rad, 0, 1)``
    (0 = closed, fixed by the SDK's ``Encoder.set_zero()``; 1 = mechanical max).
    """

    # ---- Left TacCap unit -------------------------------------------------
    left_firmware_sn: str | None = None
    left_enable_gripper: bool = True
    left_enable_imu: bool = False
    left_gripper_open_rad: float = 1.7

    left_enable_tracker: bool = True
    left_tracker_sn: str | None = None
    left_tracker_to_ee_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_tracker_to_ee_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    left_enable_init_pose_alignment: bool = False
    left_init_tcp_pose: tuple[float, float, float, float, float, float, float] = (
        _DEFAULT_INIT_TCP_POSE
    )

    left_tactile_serials: list[str] = field(default_factory=list)
    """Left-hand Xense tactile serials, e.g. ["GSPS01A24Z0003","GSPS01A24Z0004"]
    → obs keys left_tactile_0 / left_tactile_1 (opened by serial via xensesdk)."""

    left_enable_wrist_camera: bool = True
    left_wrist_camera_serial: str = ""
    """Left wrist UVC serial, e.g. "XCA24Z0003m" (resolved via /dev/v4l/by-id).
    Use this OR left_wrist_camera_index_or_path (the latter wins)."""
    left_wrist_camera_index_or_path: str = ""
    left_wrist_camera_width: int = 640
    left_wrist_camera_height: int = 480
    left_wrist_camera_fps: int = 30

    # ---- Right TacCap unit ------------------------------------------------
    right_firmware_sn: str | None = None
    right_enable_gripper: bool = True
    right_enable_imu: bool = False
    right_gripper_open_rad: float = 1.7

    right_enable_tracker: bool = True
    right_tracker_sn: str | None = None
    right_tracker_to_ee_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_tracker_to_ee_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    right_enable_init_pose_alignment: bool = False
    right_init_tcp_pose: tuple[float, float, float, float, float, float, float] = (
        _DEFAULT_INIT_TCP_POSE
    )

    right_tactile_serials: list[str] = field(default_factory=list)
    """Right-hand Xense tactile serials, e.g. ["GSPS01A24Z0005","GSPS01A24Z0006"]
    → obs keys right_tactile_0 / right_tactile_1 (opened by serial via xensesdk)."""

    right_enable_wrist_camera: bool = True
    right_wrist_camera_serial: str = ""
    """Right wrist UVC serial, e.g. "XCA24Z0004m" (resolved via /dev/v4l/by-id).
    Use this OR right_wrist_camera_index_or_path (the latter wins)."""
    right_wrist_camera_index_or_path: str = ""
    right_wrist_camera_width: int = 640
    right_wrist_camera_height: int = 480
    right_wrist_camera_fps: int = 30

    # ---- Shared -----------------------------------------------------------
    tracker_wait_timeout: float = 10.0
    """Seconds to wait for the first valid tracker pose at connect time (both sides)."""

    tactile_fps: int = 30
    tactile_output_types: list[str] = field(default_factory=lambda: ["rectify"])
    """Defaults applied to every tactile serial. Single output type → one
    (H, W, 3) image (rectify is inference-free, landscape (400, 700, 3) like
    v0.4.4). Width/height auto-derive from the SDK rectify_size — don't hard-code."""

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    """Tactile camera configs keyed by *pre-prefixed* feature name, e.g.
    ``left_tactile_0`` / ``left_tactile_1`` / ``right_tactile_0`` / ``right_tactile_1``,
    each an ``XenseTactileCameraConfig(serial_number="GSPS...")``. The wrist UVC
    cameras do NOT belong here — they are wired per side via
    ``{side}_enable_wrist_camera`` + ``{side}_wrist_camera_index_or_path`` and appear
    as observation keys ``left_wrist`` / ``right_wrist``."""

    def __post_init__(self):
        super().__post_init__()
        for side in _SIDES:
            if getattr(self, f"{side}_enable_gripper") and getattr(
                self, f"{side}_gripper_open_rad"
            ) <= 0:
                raise ValueError(
                    f"{side}_gripper_open_rad must be positive, got "
                    f"{getattr(self, f'{side}_gripper_open_rad')}. Closed=0 is fixed by the "
                    "SDK's Encoder.set_zero(); open_rad is the mechanical-max angle "
                    "(TC-GU-01 default 1.7)."
                )
            # Build this side's tactile camera configs from serials. Keyed
            # {side}_tactile_0/1; width/height auto-derive (landscape (400,700,3)).
            for i, sn in enumerate(getattr(self, f"{side}_tactile_serials")):
                key = f"{side}_tactile_{i}"
                if key not in self.cameras:
                    self.cameras[key] = XenseTactileCameraConfig(
                        serial_number=sn,
                        fps=self.tactile_fps,
                        output_types=list(self.tactile_output_types),
                    )

            wrist_key = f"{side}_wrist"
            if getattr(self, f"{side}_enable_wrist_camera"):
                if wrist_key in self.cameras:
                    raise ValueError(
                        f"{wrist_key} is wired by {side}_enable_wrist_camera=True; remove it "
                        f"from `cameras` or set {side}_enable_wrist_camera=False."
                    )
                if not (
                    getattr(self, f"{side}_wrist_camera_serial")
                    or getattr(self, f"{side}_wrist_camera_index_or_path")
                ):
                    raise ValueError(
                        f"{side}_enable_wrist_camera=True requires {side}_wrist_camera_serial "
                        f"(e.g. 'XCA24Z0003m') or {side}_wrist_camera_index_or_path. Set one, "
                        f"or disable with {side}_enable_wrist_camera=false."
                    )
