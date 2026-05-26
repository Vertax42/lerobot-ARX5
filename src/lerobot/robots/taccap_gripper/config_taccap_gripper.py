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

    # ---- Cameras (tactile + extras) --------------------------------------
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    """Camera configs keyed by feature name. Typical entries for the
    TacCap-Gripper:
    - tactile: ``XenseTactileCameraConfig`` with the OG serials reported by
      ``find_one()`` (``tactile_left_serial``, ``tactile_right_serial``).

    The wrist UVC camera does NOT belong here — it is auto-wired via
    ``enable_wrist_camera`` (below) using the V4L2 path the SDK reports."""

    # ---- Wrist camera (auto-discovered via GripperEndpoints.wrist_video) -
    enable_wrist_camera: bool = True
    """Auto-wire the wrist UVC camera using ``GripperEndpoints.wrist_video``.
    Surfaced as observation key ``wrist_cam``. Set False to suppress."""

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
        if self.enable_wrist_camera and "wrist_cam" in self.cameras:
            raise ValueError(
                "wrist_cam is auto-wired by enable_wrist_camera=True; "
                "remove it from `cameras` or set enable_wrist_camera=False."
            )
