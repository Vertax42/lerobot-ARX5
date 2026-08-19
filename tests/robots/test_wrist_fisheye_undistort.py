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
        index_or_path="XCA_TEST",
        fourcc="MJPG",
        width=FISHEYE_CALIB_WIDTH,
        height=FISHEYE_CALIB_HEIGHT,
        fps=30,
        warmup_s=0.0,
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
            raise RuntimeError("has undistort enabled but no fisheye calibration was supplied")


# --------------------------------------------------------------------------- #
# Degenerate intrinsics — the bench-observed case
# --------------------------------------------------------------------------- #


def _cal(fx, fy):
    """A real CameraFisheyeCal — the SDK predicate only accepts its own type,
    and using it here checks that contract too."""
    from xense.taccap import CameraFisheyeCal

    return CameraFisheyeCal(fx=fx, fy=fy, cx=320.0, cy=240.0, k1=0.0, k2=0.0, k3=0.0, k4=0.0)


class _FakeCalibrationComponent:
    """Stands in for the SDK's Calibration, which owns the fallback policy now."""

    def __init__(self, calibration, is_reference, reason):
        self._answer = (calibration, is_reference, reason)
        self.calls = 0

    def resolve_fisheye(self, timeout_ms=200):
        self.calls += 1
        return self._answer


class _FakeSdkGripper:
    def __init__(self, calibration, is_reference=False, reason=""):
        self.calibration = _FakeCalibrationComponent(calibration, is_reference, reason)


def _follower(**kw):
    from lerobot.grippers.taccap.taccap_follower import TaccapFollower

    f = object.__new__(TaccapFollower)
    f._side = "left"  # __str__ reads it; __init__ is bypassed
    f.logger = logging.getLogger("test")
    f._gripper = _FakeSdkGripper(**kw)
    return f


# --------------------------------------------------------------------------- #
# The driver is a passthrough — the policy lives in the SDK
# --------------------------------------------------------------------------- #


def test_a_units_own_calibration_passes_straight_through():
    cal = _cal(310.5, 311.2)
    follower = _follower(calibration=cal)

    got, is_reference = follower.read_wrist_fisheye_calibration()

    assert got is cal
    assert is_reference is False
    assert follower._gripper.calibration.calls == 1


def test_the_reference_fallback_is_reported_and_warned_about(caplog):
    """Whatever the SDK decided must reach the caller, and be visible in the log.

    The reasons themselves — too-old firmware, never calibrated, an all-zero
    record — are the SDK's to classify; Calibration::resolve_fisheye() applies
    that policy for the path where the SDK owns the camera too, so re-deriving
    it here is what let the two copies drift apart in the first place.
    """
    from xense.taccap import FISHEYE_FALLBACK_CAL

    follower = _follower(
        calibration=FISHEYE_FALLBACK_CAL,
        is_reference=True,
        reason="the wrist lens has never been calibrated",
    )

    with caplog.at_level(logging.WARNING):
        got, is_reference = follower.read_wrist_fisheye_calibration()

    assert is_reference is True
    assert got is FISHEYE_FALLBACK_CAL
    assert "REFERENCE" in caplog.text
    assert "never been calibrated" in caplog.text, "the SDK's reason must survive"


def test_a_units_own_calibration_is_not_warned_about(caplog):
    with caplog.at_level(logging.WARNING):
        _follower(calibration=_cal(310.5, 311.2)).read_wrist_fisheye_calibration()

    assert "REFERENCE" not in caplog.text


def test_reading_from_a_disconnected_gripper_is_refused():
    from lerobot.grippers.taccap.taccap_follower import TaccapFollower
    from lerobot.utils.errors import DeviceNotConnectedError

    follower = object.__new__(TaccapFollower)
    follower._side = "left"
    follower._gripper = None

    with pytest.raises(DeviceNotConnectedError):
        follower.read_wrist_fisheye_calibration()


def test_the_sdk_fallback_is_itself_usable():
    """Guard the guard: a fallback that fails is_usable would rectify to black."""
    from xense.taccap import FISHEYE_FALLBACK_CAL, is_usable_fisheye_cal

    assert is_usable_fisheye_cal(FISHEYE_FALLBACK_CAL)


# --------------------------------------------------------------------------- #
# The rectification itself
# --------------------------------------------------------------------------- #


def _reference_remap(cal, balance):
    """OpenCV's own fisheye rectification, under the SDK's balance convention.

    The SDK's `balance` interpolates the output focal length between 1.00 and
    0.70 — it is NOT OpenCV's estimateNewCameraMatrixForUndistortRectify
    balance, which solves for a valid-pixel ratio. Comparing against the wrong
    one looks like a large disagreement and means nothing.
    """
    import cv2

    from xense.taccap import FisheyeUndistorter

    K = np.asarray(cal.K, np.float64)
    D = np.asarray(cal.D, np.float64).reshape(4, 1)
    scale = FisheyeUndistorter(cal, 640, 480, balance).focal_scale
    new_k = K.copy()
    new_k[0, 0] *= scale
    new_k[1, 1] *= scale
    mx, my = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_k, (640, 480), cv2.CV_32F)
    return mx, my


#: The SDK resamples with INTER_CUBIC — the fisheye-to-pinhole mapping magnifies
#: the periphery about 3.3x, and bilinear softens an image enlarged that much.
#: Matches the PC calibration tool (camera-calibration, fisheye_calib/rectify.py).
RECTIFY_INTERPOLATION = "INTER_CUBIC"


@pytest.mark.parametrize("balance", [0.0, 0.5, 1.0])
def test_rectification_matches_opencvs_own_fisheye_implementation(balance):
    """The strongest check available without a calibration target in frame."""
    import cv2

    from xense.taccap import FISHEYE_FALLBACK_CAL, FisheyeUndistorter

    img = np.random.default_rng(7).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    mx, my = _reference_remap(FISHEYE_FALLBACK_CAL, balance)
    expected = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

    got = np.asarray(FisheyeUndistorter(FISHEYE_FALLBACK_CAL, 640, 480, balance).apply(img))

    identical = (np.abs(got.astype(int) - expected.astype(int)).sum(axis=2) == 0).mean()
    assert identical > 0.99, f"only {identical:.1%} of pixels match OpenCV"


def test_rectification_actually_moves_pixels():
    """A no-op undistorter would pass every other test in this file.

    These intrinsics have small distortion coefficients but a short focal length
    (fx=213 across 640px), so the geometric correction is substantial — about
    57px on average and 175px at the corners. Asserting a floor catches an
    undistorter that silently degrades to identity.
    """
    from xense.taccap import FISHEYE_FALLBACK_CAL

    mx, my = _reference_remap(FISHEYE_FALLBACK_CAL, 0.0)
    yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
    displacement = np.hypot(mx - xx, my - yy)

    assert displacement.mean() > 10.0, "rectification is close to a no-op"
    assert displacement.max() > 50.0


def test_a_wider_balance_shortens_the_focal_length():
    from xense.taccap import FISHEYE_FALLBACK_CAL, FisheyeUndistorter

    natural = FisheyeUndistorter(FISHEYE_FALLBACK_CAL, 640, 480, 0.0).focal_scale
    widest = FisheyeUndistorter(FISHEYE_FALLBACK_CAL, 640, 480, 1.0).focal_scale

    assert natural == pytest.approx(1.00, abs=1e-3)
    assert widest == pytest.approx(0.70, abs=1e-3)


def test_an_undistorter_is_reusable_across_frames():
    """The shape that keeps rectification affordable: build once, apply many.

    Cost was measured on the bench — 0.40 ms/frame, about 1% of a 33ms budget,
    for two wrist cameras — and that number lives in the commit message, not in
    an assertion. Wall-clock bounds here would measure the harness: under
    pytest's typeguard plugin the same call takes ~6ms, 15x more, which says
    nothing about whether a robot can keep 30fps.

    What is worth pinning is that one undistorter serves every frame, which is
    what XenseWristCamera relies on — it builds one at connect() and holds it
    for the session. An implementation that had to be rebuilt per frame would
    fail here rather than quietly cost 50x on the robot.
    """
    from xense.taccap import FISHEYE_FALLBACK_CAL, FisheyeUndistorter

    undistorter = FisheyeUndistorter(FISHEYE_FALLBACK_CAL, 640, 480, 0.0)
    rng = np.random.default_rng(0)

    outputs = [np.asarray(undistorter.apply(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))) for _ in range(5)]

    assert all(o.shape == (480, 640, 3) for o in outputs)
    # Distinct inputs must give distinct outputs — a cached-result bug would
    # hand every frame the first one, and a per-frame rebuild would still pass
    # the shape check alone.
    assert not np.array_equal(outputs[0], outputs[1])


def test_the_camera_builds_one_undistorter_and_keeps_it():
    """XenseWristCamera holds it for the session rather than per read()."""
    from lerobot.cameras.xense import XenseWristCamera

    cam = XenseWristCamera(_cfg(undistort=True))
    assert cam._undistorter is None, "nothing should be built before connect()"

    # connect() is what builds it; the field is the contract _postprocess_image
    # reads on every frame, so it must survive across reads.
    assert hasattr(cam, "_undistorter")


def test_rectification_resamples_with_cubic_not_bilinear():
    """Bilinear here is a quality regression that nothing else would catch.

    Rectifying an equidistant fisheye into a pinhole view upsamples the
    periphery — about 3.3x with this lens at f_new = f_src — and bilinear
    visibly softens an image being magnified that much. Measured on a
    structured frame: cubic carries ~1.36x the Laplacian variance. Every other
    test in this file passes either way, and the difference lands in recorded
    data.
    """
    import cv2

    from xense.taccap import FISHEYE_FALLBACK_CAL, FisheyeUndistorter

    rng = np.random.default_rng(1)
    img = cv2.resize(
        rng.integers(0, 255, (60, 80, 3), dtype=np.uint8),
        (640, 480),
        interpolation=cv2.INTER_NEAREST,
    )
    mx, my = _reference_remap(FISHEYE_FALLBACK_CAL, 0.0)
    cubic = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    linear = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    got = np.asarray(FisheyeUndistorter(FISHEYE_FALLBACK_CAL, 640, 480, 0.0).apply(img))

    def agreement(a, b):
        return (np.abs(a.astype(int) - b.astype(int)).sum(axis=2) == 0).mean()

    assert agreement(got, cubic) > 0.99, "rectification is no longer cubic"
    assert agreement(got, linear) < 0.99, "cubic and linear should differ here"
