#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Configuration for TacCap-Gripper handheld data-collection device.

Hardware:
- TacCap-Gripper handheld unit (XenseRobotics): motor-driven jaw, two
  embedded visuotactile sensors, wrist UVC camera, encoder, IMU.
  Driven by the ``xense.taccap`` SDK (``taccap-gripper`` PyPI package).
- Pico4 Ultra independent motion tracker physically mounted on top to
  provide 6-DoF pose. Reached via ``xensevr_pc_service_sdk``.

Tactile and wrist cameras are wired through the standard LeRobot
``cameras`` framework (not the SDK's bundled streams), so they appear
as normal swappable ``cameras.<name>=...`` CLI entries.

Recorded frame: raw Pico4 native (X right, Y up, Z toward the headset
operator at Unity-launch time). The recorded pose is **not** in any
robot's base frame; downstream policies must reframe explicitly.
"""

from dataclasses import dataclass, field

from lerobot.cameras.utils import CameraConfig
from lerobot.cameras.xense.configuration_xense import XenseTactileCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("taccap_gripper")
@dataclass
class TaccapGripperConfig(RobotConfig):
    """Configuration for the TacCap-Gripper handheld data-collection device.

    Discovery uses the firmware-burned serial (stable across CH343 chip
    swaps, which the MCU serial is not):
    - ``firmware_sn=None`` => ``xense.taccap.find_one()`` (errors on 0 or >1).
    - ``firmware_sn="SN000..."`` => ``scan_grippers()`` filtered to that SN.

    Pose is sourced from a single Pico4 Ultra motion tracker:
    - ``tracker_sn=None`` => the first tracker the service reports.
    - ``tracker_sn="..."`` => match by serial; fails fast with the available
      SNs if not found.

    Gripper position is normalised via ``clip(position_rad / open_rad, 0, 1)``.
    ``position_rad`` is the SDK's cooked (post-zero, >=0-clamped) reading.
    The closed endpoint is **always 0** -- the SDK's ``Encoder.set_zero()``
    command latches the closed pose into firmware. Run the SDK's
    ``examples/calibrate.py`` once per device to set the zero; then this
    config only needs the mechanical-max ``gripper_open_rad`` (default
    1.7 = TC-GU-01 hardware stop).
    """

    # ---- TacCap gripper ---------------------------------------------------
    firmware_sn: str | None = None
    """Firmware SN reported by ``GripperEndpoints.firmware_sn`` (None =
    find_one). Stable across CH343 chip swaps."""

    enable_gripper: bool = True
    enable_imu: bool = False
    """If True, also publish ``imu.{accel,gyro,mag}.{x,y,z}`` per observation."""

    gripper_open_rad: float = 1.7
    """Encoder reading (rad) when the jaw is fully open (gripper.pos = 1).
    Default 1.7 ~= TC-GU-01 mechanical limit (~97 deg). Override per unit
    if your sample varies. Closed (gripper.pos = 0) is always 0 rad --
    set via the SDK's Encoder.set_zero() (run calibrate.py once)."""

    # ---- Pico4 Ultra tracker ---------------------------------------------
    enable_tracker: bool = True

    tracker_sn: str | None = None
    """Pico4 motion-tracker serial number. None = first available."""

    tracker_to_ee_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation from the tracker frame to the gripper end-effector frame
    (meters). Defaults to zero (i.e. EE coincides with tracker)."""

    tracker_to_ee_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Rotation from the tracker frame to the gripper end-effector frame,
    [qw, qx, qy, qz]. Defaults to identity."""

    tracker_wait_timeout: float = 10.0
    """Seconds to wait for the first valid tracker pose at connect time."""

    # ---- UMI-style init-pose alignment (reserved, off by default) --------
    enable_init_pose_alignment: bool = False
    """If True, snapshot the first valid tracker pose at connect time
    and compute a rigid transform so all subsequent recorded poses are
    in the same frame as ``init_tcp_pose`` (typically the deployment
    robot's base frame at the home configuration). Mirrors
    ``vive_tracker``'s UMI behaviour. Off by default
    pending live verification on real Flexiv hardware."""

    init_tcp_pose: tuple[float, float, float, float, float, float, float] = (
        0.693307, -0.114902, 0.14589, 0.004567, 0.003238, 0.999984, 0.001246,
    )
    """Robot TCP pose at the operator's "init" stance, as
    ``[x, y, z, qw, qx, qy, qz]``. Default is Flexiv Rizon4's home
    pose. Only consumed when
    ``enable_init_pose_alignment`` is True."""

    # ---- Tactile sensors (Xense; opened by serial via xensesdk) ----------
    tactile_serials: list[str] = field(default_factory=list)
    """Xense tactile sensor serial numbers, e.g.
    ``["GSPS01A24Z0003", "GSPS01A24Z0004"]``. Each becomes an observation key
    ``tactile_0`` / ``tactile_1`` / … . xensesdk's ``Sensor.create(serial)``
    resolves the V4L2 video port from the serial — no device path needed."""

    tactile_fps: int = 30
    tactile_output_types: list[str] = field(default_factory=lambda: ["rectify"])
    """Defaults applied to every ``tactile_serials`` entry. A single output type
    yields one (H, W, 3) image; ``rectify`` is inference-free. Width/height are
    auto-derived from the SDK's rectify_size (do not hard-code them — the rectify
    array is (400, 700, 3))."""

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    """Advanced/extra camera configs, merged after the ones built from
    ``tactile_serials``. Normally leave empty and use ``tactile_serials``.
    The wrist UVC camera does NOT belong here — see ``wrist_camera_serial``."""

    # ---- Wrist camera (OpenCV UVC; opened by serial or explicit path) ----
    enable_wrist_camera: bool = True
    """Wire the wrist UVC camera under observation key ``wrist_cam``. Requires
    ``wrist_camera_serial`` (preferred) or ``wrist_camera_index_or_path``."""

    wrist_camera_serial: str = ""
    """Wrist UVC camera serial, e.g. ``"XCA24Z0003m"``. Resolved to its V4L2
    device at connect via ``/dev/v4l/by-id/*<serial>*-video-index0``. Use this
    OR ``wrist_camera_index_or_path``."""

    wrist_camera_index_or_path: str = ""
    """Explicit V4L2 device path/index override (wins over ``wrist_camera_serial``),
    e.g. ``/dev/v4l/by-id/usb-...-index0`` or ``"4"``."""

    wrist_camera_width: int = 640
    wrist_camera_height: int = 480
    wrist_camera_fps: int = 30

    def __post_init__(self):
        super().__post_init__()
        if self.enable_gripper and self.gripper_open_rad <= 0:
            raise ValueError(
                f"gripper_open_rad must be positive, got {self.gripper_open_rad}. "
                "Closed=0 is fixed by the SDK's Encoder.set_zero(); open_rad "
                "is the mechanical-max angle (TC-GU-01 default 1.7)."
            )

        # Build tactile camera configs from serials (xensesdk resolves the video
        # port from the serial). Keyed tactile_0, tactile_1, … . width/height are
        # left to the SDK config to auto-derive (correct rectify orientation).
        for i, sn in enumerate(self.tactile_serials):
            key = f"tactile_{i}"
            if key not in self.cameras:
                self.cameras[key] = XenseTactileCameraConfig(
                    serial_number=sn,
                    fps=self.tactile_fps,
                    output_types=list(self.tactile_output_types),
                )

        if self.enable_wrist_camera and "wrist_cam" in self.cameras:
            raise ValueError(
                "wrist_cam is wired by enable_wrist_camera=True; "
                "remove it from `cameras` or set enable_wrist_camera=False."
            )
        if self.enable_wrist_camera and not (
            self.wrist_camera_serial or self.wrist_camera_index_or_path
        ):
            raise ValueError(
                "enable_wrist_camera=True requires wrist_camera_serial "
                "(e.g. 'XCA24Z0003m') or wrist_camera_index_or_path. Set one, or "
                "disable with enable_wrist_camera=false."
            )
