#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""TacCap wrist camera: an OpenCV camera that can rectify its own fisheye frames.

The rectification hooks into ``_postprocess_image``, which receives the frame as
OpenCV delivered it — BGR, before any colour conversion or rotation. That matches
what ``FisheyeUndistorter.apply`` expects and returns, so no extra conversion is
needed and the frame reaches the colour/rotation steps already rectified.

The intrinsics come from the gripper's MCU, not from this class: the camera is
handed a calibration before it connects (see ``attach_wrist_fisheye_calibration``
in ``lerobot.grippers.camera_injection``). Keeping the read out here means the
camera never has to hold a gripper handle or reason about who owns the serial
transport.
"""

from typing import Any

from numpy.typing import NDArray

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.utils.robot_utils import get_logger

from .configuration_wrist import XenseWristCameraConfig

logger = get_logger("XenseWristCamera")


class XenseWristCamera(OpenCVCamera):
    """An OpenCV camera that applies the wrist lens' fisheye rectification."""

    def __init__(self, config: XenseWristCameraConfig):
        super().__init__(config)
        self.undistort = config.undistort
        self.fisheye_balance = config.fisheye_balance
        self._calibration = None  # xense.taccap.CameraFisheyeCal
        self._calibration_is_reference = False
        self._undistorter = None  # xense.taccap.FisheyeUndistorter

    # -- calibration handed in from the gripper ------------------------------
    def set_fisheye_calibration(self, calibration, *, is_reference: bool = False) -> None:
        """Give the camera the intrinsics to rectify with.

        Must be called before ``connect()`` when ``undistort`` is on. Building
        the remap tables is deferred to connect so a rig that never opens this
        camera does not pay for them.

        Args:
            calibration: A ``xense.taccap.CameraFisheyeCal``.
            is_reference: True when these are the SDK's shared reference values
                rather than this unit's own. Recorded so the connect line says
                which was used — a dataset rectified with reference intrinsics is
                otherwise indistinguishable from one rectified properly.
        """
        self._calibration = calibration
        self._calibration_is_reference = is_reference

    def connect(self, warmup: bool = True) -> None:
        super().connect(warmup=warmup)
        if not self.undistort:
            return
        if self._calibration is None:
            raise RuntimeError(
                f"{self} has undistort enabled but no fisheye calibration was "
                "supplied. The arm reads it off the gripper's MCU before "
                "connecting cameras; a wrist camera configured by hand, or one "
                "whose gripper is disabled, will not receive it. Turn undistort "
                "off, or wire the camera through gripper auto-discovery."
            )
        from xense.taccap import FisheyeUndistorter

        self._undistorter = FisheyeUndistorter(
            self._calibration,
            width=self.capture_width,
            height=self.capture_height,
            balance=self.fisheye_balance,
        )
        source = "SDK reference" if self._calibration_is_reference else "this unit"
        logger.info(
            f"{self} fisheye rectification on (balance={self.fisheye_balance}, "
            f"{self.capture_width}x{self.capture_height}, calibration={source})"
        )

    def disconnect(self) -> None:
        super().disconnect()
        self._undistorter = None

    # -- the data path -------------------------------------------------------
    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        """Rectify before colour conversion and rotation.

        ``image`` arrives as OpenCV read it — BGR, at capture resolution — which
        is exactly the contract of FisheyeUndistorter.apply(). Applying here
        rather than after conversion keeps the frame in the layout the SDK
        expects and leaves the base class's validation and rotation untouched.
        """
        if self._undistorter is not None:
            image = self._undistorter.apply(image)
        return super()._postprocess_image(image)
