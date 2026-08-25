#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pin down what gripper auto-discovery writes into ``config.cameras``.

All four arms — both bimanual and both single — share one implementation of this
wiring, and it only ever executes with hardware attached, so without these tests
a change to the key scheme or the camera settings would first show up as a
dataset recorded with the wrong keys. Faking the discovery sweep is what makes
the mapping testable on a bare host.

The single-arm cases matter for the same reason the bimanual ones do, plus one
of their own: a single arm has no side, so its camera keys are named after the
*gripper's* side. Get that wrong and a single-arm dataset stops lining up with a
bimanual one.

Tactile keys are ``<side>_tactile_<finger>``: the arm the gripper is on, then
which jaw the pad sits on, the latter read off the sensor serial's trailing-digit
parity. That is the scheme the sister repo ``xense-taccap-lerobot`` records under,
so these assertions are also what keeps the two fleets' datasets interchangeable.
"""

from dataclasses import dataclass, field
from logging import getLogger

import pytest

from lerobot.grippers import camera_injection as ci
from lerobot.utils.errors import DeviceNotConnectedError
from tests.utils import require_package

LOGGER = getLogger("test_gripper_camera_injection")


@dataclass
class FakeDevice:
    """Stands in for TaccapSideDevices / SerialGripperCameras alike."""

    wrist_camera_name: str
    tactile_sns: list[str] = field(default_factory=list)
    mcu_device: str = "/dev/serial/by-id/fake"
    usb_hub: str = "1-3"


# Tactile serials carry the finger in their trailing digit (odd -> left jaw,
# even -> right), so the fakes have to be numbered like real ones: a positional
# placeholder would let a regression back to enumeration-order keys pass.
def _two_sides() -> dict[str, FakeDevice]:
    return {
        "left": FakeDevice("XCA_LEFT", ["OG001349", "OG001350"], mcu_device="/dev/left"),
        "right": FakeDevice("XCA_RIGHT", ["OG001351", "OG001352"], mcu_device="/dev/right"),
    }


@pytest.fixture
def fake_taccap(monkeypatch):
    def _install(found):
        monkeypatch.setattr(
            "lerobot.grippers.taccap.discovery.discover_taccap_sides",
            lambda: found,
        )

    return _install


@pytest.fixture
def fake_serial(monkeypatch):
    def _install(found):
        monkeypatch.setattr(
            "lerobot.grippers.serial.discovery.discover_serial_gripper_cameras",
            lambda sides=(), **kw: {s: found[s] for s in sides if s in found},
        )

    return _install


# --------------------------------------------------------------------------- #
# Key scheme
# --------------------------------------------------------------------------- #


def test_taccap_injects_wrist_and_tactile_keys(fake_taccap):
    fake_taccap(_two_sides())
    cameras: dict = {}

    ci.inject_taccap_cameras(cameras, sides=("left", "right"), enable_tactile=True, logger=LOGGER)

    assert sorted(cameras) == [
        "left_tactile_left",
        "left_tactile_right",
        "left_wrist",
        "right_tactile_left",
        "right_tactile_right",
        "right_wrist",
    ]


@require_package("xgripper")
def test_serial_injects_the_same_key_scheme_as_taccap(fake_taccap, fake_serial):
    """Both gripper families must land on identical keys, or a dataset recorded
    on one rig stops loading against a policy trained on the other."""
    fake_taccap(_two_sides())
    fake_serial(_two_sides())
    taccap_cameras: dict = {}
    serial_cameras: dict = {}

    ci.inject_taccap_cameras(taccap_cameras, sides=("left", "right"), enable_tactile=True, logger=LOGGER)
    ci.inject_serial_gripper_cameras(serial_cameras, sides=("left", "right"), enable_tactile=True, logger=LOGGER)

    assert sorted(taccap_cameras) == sorted(serial_cameras)


def test_the_finger_comes_from_the_serial_not_the_discovery_order(fake_taccap):
    """Reversing the order discovery hands the sensors back must not move a key.

    USB port order is a stable *ordering*, not an identity — re-cabling the two
    pads swaps it. The key has to follow the serial, or a dataset silently ends
    up with the jaws transposed against every earlier recording.
    """
    forward: dict = {}
    reversed_: dict = {}
    fake_taccap({"left": FakeDevice("XCA_LEFT", ["OG001349", "OG001350"])})
    ci.inject_taccap_cameras(forward, sides=("left",), enable_tactile=True, logger=LOGGER)
    fake_taccap({"left": FakeDevice("XCA_LEFT", ["OG001350", "OG001349"])})
    ci.inject_taccap_cameras(reversed_, sides=("left",), enable_tactile=True, logger=LOGGER)

    assert forward["left_tactile_left"].serial_number == "OG001349"
    assert forward["left_tactile_right"].serial_number == "OG001350"
    assert {k: v.serial_number for k, v in reversed_.items() if "tactile" in k} == {
        k: v.serial_number for k, v in forward.items() if "tactile" in k
    }


def test_two_sensors_on_the_same_finger_raise(fake_taccap):
    """Both pads odd-numbered is a mis-burned or mis-installed sensor. Keeping
    one would drop half the tactile stream from the schema with nothing to see."""
    fake_taccap({"left": FakeDevice("XCA_LEFT", ["OG001349", "OG001351"])})

    with pytest.raises(ValueError, match="resolve to the left finger"):
        ci.inject_taccap_cameras({}, sides=("left",), enable_tactile=True, logger=LOGGER)


def test_a_serial_with_no_digits_raises(fake_taccap):
    fake_taccap({"left": FakeDevice("XCA_LEFT", ["OG00134A", "OG001350"])})

    with pytest.raises(ValueError, match="no trailing digits"):
        ci.inject_taccap_cameras({}, sides=("left",), enable_tactile=True, logger=LOGGER)


@require_package("xgripper")
def test_tactile_sensors_are_skipped_when_disabled(fake_serial):
    fake_serial(_two_sides())
    cameras: dict = {}

    ci.inject_serial_gripper_cameras(cameras, sides=("left", "right"), enable_tactile=False, logger=LOGGER)

    assert sorted(cameras) == ["left_wrist", "right_wrist"]


# --------------------------------------------------------------------------- #
# Camera settings — frozen so enabling discovery cannot change recorded pixels
# --------------------------------------------------------------------------- #


def test_wrist_camera_settings_are_frozen(fake_taccap):
    fake_taccap(_two_sides())
    cameras: dict = {}

    ci.inject_taccap_cameras(cameras, sides=("left",), enable_tactile=False, logger=LOGGER)

    wrist = cameras["left_wrist"]
    assert wrist.index_or_path == "XCA_LEFT"
    assert (wrist.width, wrist.height, wrist.fps) == (640, 480, 30)
    assert wrist.fourcc == "MJPG"
    assert wrist.warmup_s == 1.0


def test_tactile_camera_settings_are_frozen(fake_taccap):
    fake_taccap(_two_sides())
    cameras: dict = {}

    ci.inject_taccap_cameras(cameras, sides=("left",), enable_tactile=True, logger=LOGGER)

    tactile = cameras["left_tactile_left"]
    assert tactile.serial_number == "OG001349"
    assert tactile.fps == 30
    assert tactile.warmup_s == 0.05


# --------------------------------------------------------------------------- #
# Partial rigs and failures
# --------------------------------------------------------------------------- #


@require_package("xgripper")
def test_serial_one_armed_bench_does_not_touch_the_other_side(fake_serial):
    """A bench with a gripper on one arm only must not fail for the other."""
    fake_serial({"left": FakeDevice("XCA_LEFT", ["OG001349", "OG001350"])})
    cameras: dict = {}

    ci.inject_serial_gripper_cameras(cameras, sides=("left",), enable_tactile=True, logger=LOGGER)

    assert sorted(cameras) == ["left_tactile_left", "left_tactile_right", "left_wrist"]


@require_package("xgripper")
def test_serial_no_gripper_at_all_is_a_no_op(fake_serial):
    fake_serial({})
    cameras: dict = {}

    ci.inject_serial_gripper_cameras(cameras, sides=(), enable_tactile=True, logger=LOGGER)

    assert cameras == {}


def test_missing_side_raises(fake_taccap):
    fake_taccap({"left": FakeDevice("XCA_LEFT", ["OG001349", "OG001350"])})

    with pytest.raises(DeviceNotConnectedError, match="no right gripper"):
        ci.inject_taccap_cameras({}, sides=("left", "right"), enable_tactile=True, logger=LOGGER)


@require_package("xgripper")
def test_too_few_tactile_sensors_raises(fake_serial):
    fake_serial({"left": FakeDevice("XCA_LEFT", ["OG001349"], usb_hub="3-1")})

    with pytest.raises(DeviceNotConnectedError, match="expected 2 tactile sensors"):
        ci.inject_serial_gripper_cameras({}, sides=("left",), enable_tactile=True, logger=LOGGER)


@require_package("xgripper")
def test_too_few_tactile_sensors_is_ignored_when_tactile_disabled(fake_serial):
    fake_serial({"left": FakeDevice("XCA_LEFT", [])})
    cameras: dict = {}

    ci.inject_serial_gripper_cameras(cameras, sides=("left",), enable_tactile=False, logger=LOGGER)

    assert sorted(cameras) == ["left_wrist"]


# --------------------------------------------------------------------------- #
# MCU adoption
# --------------------------------------------------------------------------- #


def test_taccap_returns_mcu_devices_for_adoption(fake_taccap):
    fake_taccap(_two_sides())

    mcu = ci.inject_taccap_cameras({}, sides=("left", "right"), enable_tactile=False, logger=LOGGER)

    assert mcu == {"left": "/dev/left", "right": "/dev/right"}


class FakeGripper:
    def __init__(self, mcu_device=None):
        self._mcu_device = mcu_device


def test_adopt_pins_the_discovered_path():
    gripper = FakeGripper()

    ci.adopt_taccap_mcu_device(gripper, "left", "/dev/discovered", LOGGER)

    assert gripper._mcu_device == "/dev/discovered"


def test_adopt_never_overrides_an_explicit_config_value():
    gripper = FakeGripper(mcu_device="/dev/pinned-by-operator")

    ci.adopt_taccap_mcu_device(gripper, "left", "/dev/discovered", LOGGER)

    assert gripper._mcu_device == "/dev/pinned-by-operator"


def test_adopt_on_a_side_without_a_gripper_is_a_no_op():
    ci.adopt_taccap_mcu_device(None, "left", "/dev/discovered", LOGGER)


# --------------------------------------------------------------------------- #
# Single-arm shape — one side, named after the gripper
# --------------------------------------------------------------------------- #


def _one_side(side: str) -> dict[str, FakeDevice]:
    """Discovery on a bench where only ``side`` is plugged in."""
    return {side: FakeDevice(f"XCA_{side.upper()}", ["OG001349", "OG001350"])}


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_single_arm_gets_exactly_its_own_gripper_side(fake_taccap, side):
    """The keys follow the gripper's side, not a hardcoded 'left'."""
    other = "right" if side == "left" else "left"
    fake_taccap(_two_sides())
    cameras: dict = {}

    ci.inject_taccap_cameras(cameras, sides=(side,), enable_tactile=True, logger=LOGGER)

    assert set(cameras) == {f"{side}_wrist", f"{side}_tactile_left", f"{side}_tactile_right"}
    assert not [k for k in cameras if k.startswith(other)]


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_single_arm_bench_needs_only_its_own_side_present(fake_taccap, side):
    """A one-armed bench has one gripper on the bus; asking for it must not
    require the other to be there."""
    fake_taccap(_one_side(side))
    cameras: dict = {}

    ci.inject_taccap_cameras(cameras, sides=(side,), enable_tactile=True, logger=LOGGER)

    assert f"{side}_wrist" in cameras


def test_a_single_arm_asking_for_the_side_that_is_not_plugged_in_raises(fake_taccap):
    fake_taccap(_one_side("left"))

    with pytest.raises(DeviceNotConnectedError):
        ci.inject_taccap_cameras({}, sides=("right",), enable_tactile=True, logger=LOGGER)


def test_the_single_arm_key_scheme_matches_the_bimanual_one(fake_taccap):
    """A single-arm recording and one arm of a bimanual recording must be
    interchangeable, which they are only if the keys are identical."""
    fake_taccap(_two_sides())
    single: dict = {}
    bimanual: dict = {}

    ci.inject_taccap_cameras(single, sides=("left",), enable_tactile=True, logger=LOGGER)
    ci.inject_taccap_cameras(bimanual, sides=("left", "right"), enable_tactile=True, logger=LOGGER)

    assert set(single) == {k for k in bimanual if k.startswith("left")}
    for key in single:
        assert single[key] == bimanual[key]


def test_a_single_arm_undistort_flag_reaches_its_wrist_camera(fake_taccap):
    fake_taccap(_two_sides())
    cameras: dict = {}

    ci.inject_taccap_cameras(
        cameras,
        sides=("right",),
        enable_tactile=False,
        logger=LOGGER,
        undistort_wrist=True,
        fisheye_balance=0.25,
    )

    assert cameras["right_wrist"].undistort is True
    assert cameras["right_wrist"].fisheye_balance == 0.25


# --------------------------------------------------------------------------- #
# Handing the calibration over — the single-arm dict has one entry
# --------------------------------------------------------------------------- #


class FakeWristCamera:
    def __init__(self, undistort: bool = True):
        self.undistort = undistort
        self.calibration = None
        self.is_reference = None

    def set_fisheye_calibration(self, calibration, *, is_reference: bool = False) -> None:
        self.calibration = calibration
        self.is_reference = is_reference


class FakeCalibratedGripper:
    def __init__(self, cal="CAL", is_reference=False):
        self._cal = cal
        self._is_reference = is_reference

    def read_wrist_fisheye_calibration(self):
        return self._cal, self._is_reference


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_one_entry_gripper_dict_still_reaches_the_camera(side):
    cam = FakeWristCamera()

    ci.attach_wrist_fisheye_calibration({f"{side}_wrist": cam}, {side: FakeCalibratedGripper()}, LOGGER)

    assert cam.calibration == "CAL"
    assert cam.is_reference is False


def test_the_reference_fallback_flag_is_carried_through():
    """A dataset rectified from shared reference intrinsics has to be
    distinguishable from one rectified from the unit's own."""
    cam = FakeWristCamera()

    ci.attach_wrist_fisheye_calibration({"left_wrist": cam}, {"left": FakeCalibratedGripper(is_reference=True)}, LOGGER)

    assert cam.is_reference is True


def test_a_camera_that_did_not_ask_for_undistort_is_left_alone():
    cam = FakeWristCamera(undistort=False)

    ci.attach_wrist_fisheye_calibration({"left_wrist": cam}, {"left": FakeCalibratedGripper()}, LOGGER)

    assert cam.calibration is None


def test_a_serial_gripper_has_no_calibration_to_hand_over():
    """The serial family holds no firmware intrinsics. The camera is left for
    its own connect() to refuse, where the error can name the camera."""
    cam = FakeWristCamera()

    ci.attach_wrist_fisheye_calibration({"left_wrist": cam}, {"left": FakeGripper()}, LOGGER)

    assert cam.calibration is None


def test_a_side_without_a_gripper_at_all_is_left_alone():
    cam = FakeWristCamera()

    ci.attach_wrist_fisheye_calibration({"left_wrist": cam}, {"left": None}, LOGGER)

    assert cam.calibration is None
