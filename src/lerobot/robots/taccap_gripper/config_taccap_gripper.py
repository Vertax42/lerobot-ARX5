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

    Discovery is serial-based:
    - ``mcu_serial=None`` ⇒ ``xense.taccap.find_one()`` (errors on 0 or >1).
    - ``mcu_serial="MCU…"`` ⇒ ``scan_grippers()`` filtered to that serial.

    Pose is sourced from a single Pico4 Ultra motion tracker:
    - ``tracker_sn=None`` ⇒ the first tracker the service reports.
    - ``tracker_sn="…"`` ⇒ match by serial; fails fast with the available
      SNs if not found.

    Gripper position is normalised from radians via
    ``(position_rad - closed_rad) / (open_rad - closed_rad)`` clipped to
    [0, 1]. Run ``calibrate_gripper_range.py`` to find the two endpoints
    on a fresh device.
    """

    # ---- TacCap gripper ---------------------------------------------------
    mcu_serial: str | None = None
    """MCU serial reported by ``GripperEndpoints.mcu_serial`` (None = find_one)."""

    enable_gripper: bool = True
    enable_imu: bool = False
    """If True, also publish ``imu.{accel,gyro,mag}.{x,y,z}`` per observation."""

    gripper_closed_rad: float = 0.0
    """Encoder reading (rad) when the jaw is fully closed (gripper.pos = 0)."""

    gripper_open_rad: float = 1.0
    """Encoder reading (rad) when the jaw is fully open (gripper.pos = 1).
    Must differ from ``gripper_closed_rad``."""

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

    # ---- Cameras (tactile + wrist) ---------------------------------------
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    """Camera configs keyed by feature name. For the TacCap-Gripper:
    - tactile: ``XenseTactileCameraConfig`` with the OG serials reported by
      ``find_one()`` (e.g. ``tactile_left_serial``, ``tactile_right_serial``).
    - wrist: ``OpenCVCameraConfig`` pointing at the V4L2 UVC device that
      enumerates as the wrist camera.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.enable_gripper and self.gripper_open_rad == self.gripper_closed_rad:
            raise ValueError(
                "gripper_open_rad must differ from gripper_closed_rad. "
                "Run calibrate_gripper_range.py and copy the two endpoints."
            )
