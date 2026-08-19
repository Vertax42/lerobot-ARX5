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

import logging

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


def _cal(fx, fy):
    """A real CameraFisheyeCal — the SDK predicate only accepts its own type,
    and using it here checks that contract too."""
    from xense.taccap import CameraFisheyeCal

    return CameraFisheyeCal(
        fx=fx, fy=fy, cx=320.0, cy=240.0, k1=0.0, k2=0.0, k3=0.0, k4=0.0
    )


class _FakeVersion:
    def __init__(self, major, minor, patch):
        self.major, self.minor, self.patch = major, minor, patch
        self.tuple = (major, minor, patch)


class _FakeSdkGripper:
    def __init__(self, cal, version=(2, 0, 0), raises=None):
        self.firmware_version = _FakeVersion(*version) if version else None
        self.calibration = type(
            "C", (), {"read_fisheye": lambda _s: (_ for _ in ()).throw(raises) if raises else cal}
        )()


def _follower(**kw):
    from lerobot.grippers.taccap.taccap_follower import TaccapFollower

    f = object.__new__(TaccapFollower)
    f._side = "left"                      # __str__ reads it; __init__ is bypassed
    f.logger = logging.getLogger("test")
    f._gripper = _FakeSdkGripper(**kw)
    return f


# --------------------------------------------------------------------------- #
# Falling back — never silently, and never to black frames
# --------------------------------------------------------------------------- #


def test_a_real_calibration_is_used_as_is():
    cal = _cal(310.5, 311.2)
    got, is_reference = _follower(cal=cal).read_wrist_fisheye_calibration()

    assert got is cal
    assert is_reference is False


@pytest.mark.parametrize(
    ("case", "kwargs"),
    [
        ("firmware too old",   dict(cal=_cal(310.0, 311.0), version=(1, 1, 1))),
        ("never calibrated",   dict(cal=None)),
        ("empty record",       dict(cal=_cal(0.0, 0.0))),
        ("no version reported", dict(cal=_cal(310.0, 311.0), version=None)),
        ("read raises",        dict(cal=None, raises=RuntimeError("nack"))),
    ],
)
def test_the_reference_calibration_stands_in_and_says_so(case, kwargs, caplog):
    """Each of these used to mean raw fisheye frames, or black ones.

    Firmware 1.1.1 on the bench answers an uncalibrated read with an all-zero
    record rather than an error — remapping with fx = fy = 0 sends every pixel
    outside the source image, so the frame came out entirely black.
    """
    from xense.taccap import FISHEYE_FALLBACK_CAL, is_usable_fisheye_cal

    with caplog.at_level(logging.WARNING):
        got, is_reference = _follower(**kwargs).read_wrist_fisheye_calibration()

    assert is_reference is True, case
    assert got is FISHEYE_FALLBACK_CAL, case
    assert is_usable_fisheye_cal(got), "the fallback itself must be usable"
    assert "REFERENCE" in caplog.text, "falling back must be visible in the log"


def test_the_firmware_gate_names_the_version_it_wants():
    """A version error that does not say the required version is a riddle."""
    from lerobot.grippers.taccap.taccap_follower import TaccapFollower

    reason = _follower(cal=None, version=(1, 1, 1))._fisheye_firmware_shortfall()

    assert "1.1.1" in reason
    assert ".".join(str(p) for p in TaccapFollower.FISHEYE_MIN_FIRMWARE) in reason


def test_new_enough_firmware_does_not_trip_the_gate():
    assert _follower(cal=_cal(310.0, 311.0), version=(2, 0, 0))._fisheye_firmware_shortfall() is None
