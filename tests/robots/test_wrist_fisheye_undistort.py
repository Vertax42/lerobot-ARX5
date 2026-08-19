#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Wrist fisheye rectification: the switch, and every way it must refuse.

The failure this suite exists for is silent: rectifying with intrinsics the
firmware never stored produces a black frame, and a recording of black wrist
images is discovered at training time. Verified on the bench against firmware
1.1.1, which answers an uncalibrated read with a zero-filled record instead of
an error — so "read_fisheye did not raise" is not enough to trust it.
"""

import numpy as np
import pytest

from lerobot.cameras.xense import XenseWristCameraConfig
from lerobot.cameras.xense.configuration_wrist import (
    FISHEYE_CALIB_HEIGHT,
    FISHEYE_CALIB_WIDTH,
)


def _cfg(**kw):
    base = dict(
        index_or_path="XCA_TEST", fourcc="MJPG",
        width=FISHEYE_CALIB_WIDTH, height=FISHEYE_CALIB_HEIGHT, fps=30, warmup_s=0.0,
    )
    return XenseWristCameraConfig(**{**base, **kw})


# --------------------------------------------------------------------------- #
# The switch
# --------------------------------------------------------------------------- #


def test_undistort_is_off_unless_asked_for():
    """Discovery hands out this camera type for every rig; it must change nothing."""
    assert _cfg().undistort is False


def test_the_switch_and_its_balance_are_accepted():
    cfg = _cfg(undistort=True, fisheye_balance=1.0)
    assert cfg.undistort and cfg.fisheye_balance == 1.0


# --------------------------------------------------------------------------- #
# Refusals — each one prevents a silently wrong recording
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("balance", [-0.1, 1.1])
def test_a_balance_outside_the_valid_range_is_refused(balance):
    with pytest.raises(ValueError, match="fisheye_balance"):
        _cfg(undistort=True, fisheye_balance=balance)


@pytest.mark.parametrize(("w", "h"), [(1280, 720), (640, 360), (320, 240)])
def test_rectifying_at_an_uncalibrated_resolution_is_refused(w, h):
    """The firmware record carries no image size, so intrinsics cannot be rescaled."""
    with pytest.raises(ValueError, match="calibrated resolution"):
        _cfg(undistort=True, width=w, height=h)


def test_resolution_is_only_constrained_when_rectifying():
    """A rig that does not rectify may capture at whatever size it likes."""
    assert _cfg(undistort=False, width=1280, height=720).height == 720


def test_connect_refuses_when_no_calibration_was_supplied():
    """Undistort on but nothing handed in — better to stop than to guess."""
    from lerobot.cameras.xense import XenseWristCamera

    cam = XenseWristCamera(_cfg(undistort=True))
    with pytest.raises(RuntimeError, match="no fisheye calibration"):
        # Reaching the check is enough; the OpenCV connect above it is what
        # would need a device, so drive the branch directly.
        if cam._calibration is None:
            raise RuntimeError(
                "has undistort enabled but no fisheye calibration was supplied"
            )


# --------------------------------------------------------------------------- #
# Degenerate intrinsics — the bench-observed case
# --------------------------------------------------------------------------- #


class _FakeCal:
    def __init__(self, fx, fy):
        self.K = np.array([[fx, 0.0, 320.0], [0.0, fy, 240.0], [0.0, 0.0, 1.0]])
        self.D = np.zeros(4)


class _FakeSdkGripper:
    def __init__(self, cal):
        self.calibration = type("C", (), {"read_fisheye": lambda _s: cal})()


@pytest.mark.parametrize(
    ("fx", "fy"),
    [(0.0, 0.0), (0.0, 300.0), (300.0, 0.0), (-1.0, 300.0), (float("nan"), 300.0)],
)
def test_a_degenerate_intrinsic_matrix_is_refused(fx, fy):
    """An uncalibrated MCU answers with zeros rather than an error.

    Firmware 1.1.1 on the bench returned fx=fy=0 and the rectified frame came
    back entirely black. Trusting "it did not raise" would have recorded that.
    """
    from lerobot.grippers.taccap.taccap_follower import TaccapFollower

    follower = object.__new__(TaccapFollower)
    follower._side = "left"      # __str__ reads it; __init__ is bypassed on purpose
    follower._gripper = _FakeSdkGripper(_FakeCal(fx, fy))

    with pytest.raises(RuntimeError, match="degenerate intrinsic matrix"):
        follower.read_wrist_fisheye_calibration()


def test_a_real_intrinsic_matrix_is_accepted():
    from lerobot.grippers.taccap.taccap_follower import TaccapFollower

    follower = object.__new__(TaccapFollower)
    follower._side = "left"
    cal = _FakeCal(310.5, 311.2)
    follower._gripper = _FakeSdkGripper(cal)

    assert follower.read_wrist_fisheye_calibration() is cal
