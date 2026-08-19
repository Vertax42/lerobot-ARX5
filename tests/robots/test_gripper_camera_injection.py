#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pin down what gripper auto-discovery writes into ``config.cameras``.

Both bimanual arms now share one implementation of this wiring, and it only ever
executes with hardware attached — so without these tests a change to the key
scheme or the camera settings would first show up as a dataset recorded with the
wrong keys. Faking the discovery sweep is what makes the mapping testable on a
bare host.
"""

from dataclasses import dataclass, field
from logging import getLogger

import pytest

from lerobot.grippers import camera_injection as ci
from lerobot.utils.errors import DeviceNotConnectedError

LOGGER = getLogger("test_gripper_camera_injection")


@dataclass
class FakeDevice:
    """Stands in for TaccapSideDevices / SerialGripperCameras alike."""

    wrist_camera_name: str
    tactile_sns: list[str] = field(default_factory=list)
    mcu_device: str = "/dev/serial/by-id/fake"
    usb_hub: str = "1-3"


def _two_sides() -> dict[str, FakeDevice]:
    return {
        "left": FakeDevice("XCA_LEFT", ["GSPS_L0", "GSPS_L1"], mcu_device="/dev/left"),
        "right": FakeDevice("XCA_RIGHT", ["GSPS_R0", "GSPS_R1"], mcu_device="/dev/right"),
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

    ci.inject_taccap_cameras(
        cameras, sides=("left", "right"), enable_tactile=True, logger=LOGGER
    )

    assert sorted(cameras) == [
        "left_tactile_0",
        "left_tactile_1",
        "left_wrist",
        "right_tactile_0",
        "right_tactile_1",
        "right_wrist",
    ]


def test_serial_injects_the_same_key_scheme_as_taccap(fake_taccap, fake_serial):
    """Both gripper families must land on identical keys, or a dataset recorded
    on one rig stops loading against a policy trained on the other."""
    fake_taccap(_two_sides())
    fake_serial(_two_sides())
    taccap_cameras: dict = {}
    serial_cameras: dict = {}

    ci.inject_taccap_cameras(
        taccap_cameras, sides=("left", "right"), enable_tactile=True, logger=LOGGER
    )
    ci.inject_serial_gripper_cameras(
        serial_cameras, sides=("left", "right"), enable_tactile=True, logger=LOGGER
    )

    assert sorted(taccap_cameras) == sorted(serial_cameras)


def test_tactile_sensors_are_skipped_when_disabled(fake_serial):
    fake_serial(_two_sides())
    cameras: dict = {}

    ci.inject_serial_gripper_cameras(
        cameras, sides=("left", "right"), enable_tactile=False, logger=LOGGER
    )

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

    tactile = cameras["left_tactile_0"]
    assert tactile.serial_number == "GSPS_L0"
    assert tactile.fps == 30
    assert tactile.warmup_s == 0.05


# --------------------------------------------------------------------------- #
# Partial rigs and failures
# --------------------------------------------------------------------------- #


def test_serial_one_armed_bench_does_not_touch_the_other_side(fake_serial):
    """A bench with a gripper on one arm only must not fail for the other."""
    fake_serial({"left": FakeDevice("XCA_LEFT", ["GSPS_L0", "GSPS_L1"])})
    cameras: dict = {}

    ci.inject_serial_gripper_cameras(
        cameras, sides=("left",), enable_tactile=True, logger=LOGGER
    )

    assert sorted(cameras) == ["left_tactile_0", "left_tactile_1", "left_wrist"]


def test_serial_no_gripper_at_all_is_a_no_op(fake_serial):
    fake_serial({})
    cameras: dict = {}

    ci.inject_serial_gripper_cameras(cameras, sides=(), enable_tactile=True, logger=LOGGER)

    assert cameras == {}


def test_missing_side_raises(fake_taccap):
    fake_taccap({"left": FakeDevice("XCA_LEFT", ["GSPS_L0", "GSPS_L1"])})

    with pytest.raises(DeviceNotConnectedError, match="no right gripper"):
        ci.inject_taccap_cameras(
            {}, sides=("left", "right"), enable_tactile=True, logger=LOGGER
        )


def test_too_few_tactile_sensors_raises(fake_serial):
    fake_serial({"left": FakeDevice("XCA_LEFT", ["GSPS_L0"], usb_hub="3-1")})

    with pytest.raises(DeviceNotConnectedError, match="expected 2 tactile sensors"):
        ci.inject_serial_gripper_cameras(
            {}, sides=("left",), enable_tactile=True, logger=LOGGER
        )


def test_too_few_tactile_sensors_is_ignored_when_tactile_disabled(fake_serial):
    fake_serial({"left": FakeDevice("XCA_LEFT", [])})
    cameras: dict = {}

    ci.inject_serial_gripper_cameras(
        cameras, sides=("left",), enable_tactile=False, logger=LOGGER
    )

    assert sorted(cameras) == ["left_wrist"]


# --------------------------------------------------------------------------- #
# MCU adoption
# --------------------------------------------------------------------------- #


def test_taccap_returns_mcu_devices_for_adoption(fake_taccap):
    fake_taccap(_two_sides())

    mcu = ci.inject_taccap_cameras(
        {}, sides=("left", "right"), enable_tactile=False, logger=LOGGER
    )

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
