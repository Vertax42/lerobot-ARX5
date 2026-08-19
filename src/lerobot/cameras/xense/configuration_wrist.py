#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Configuration for a TacCap wrist camera, with optional fisheye rectification."""

from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# The firmware record stores intrinsics but no image size, so they are only
# meaningful at the resolution they were calibrated at. Rescaling would be a
# guess, and the SDK refuses it — so do we, with a message that says why.
FISHEYE_CALIB_WIDTH = 640
FISHEYE_CALIB_HEIGHT = 480


@CameraConfig.register_subclass("xense_wrist")
@dataclass
class XenseWristCameraConfig(OpenCVCameraConfig):
    """A TacCap wrist camera: an ordinary UVC device that can rectify its own frames.

    The wrist lens is a fisheye, and its intrinsics live in the gripper's MCU
    flash. The TacCap SDK can apply them itself, but only when it owns the UVC
    device — here lerobot's camera layer owns it, so the rectification has to
    happen on this side. The calibration is read off the MCU at connect time and
    handed to the camera by the arm; see ``attach_wrist_fisheye_calibration``.

    Attributes:
        undistort: Rectify every frame using the intrinsics the firmware holds.
            Off by default, so nothing changes for a rig that has not opted in.
            When on and the firmware has no calibration, connect fails rather
            than quietly recording fisheye frames — a dataset that silently
            mixes rectified and raw frames is discovered at training time.
        fisheye_balance: 0.0 keeps the calibrated focal length (natural view,
            matching the PC tool's default); 1.0 uses 0.70x for the widest field
            of view, at the cost of more black border. Clamped to [0, 1].
    """

    undistort: bool = False
    fisheye_balance: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if not self.undistort:
            return
        if not 0.0 <= self.fisheye_balance <= 1.0:
            raise ValueError(f"fisheye_balance must be in [0, 1], got {self.fisheye_balance}.")
        if (self.width, self.height) != (FISHEYE_CALIB_WIDTH, FISHEYE_CALIB_HEIGHT):
            raise ValueError(
                f"undistort needs the calibrated resolution "
                f"{FISHEYE_CALIB_WIDTH}x{FISHEYE_CALIB_HEIGHT}, but this camera is "
                f"configured for {self.width}x{self.height}. The firmware record "
                "carries no image size, so the intrinsics cannot be rescaled — "
                "either capture at the calibrated size or turn undistort off."
            )
