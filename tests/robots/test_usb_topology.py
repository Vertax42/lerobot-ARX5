#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""USB-hub grouping, and the serial-gripper camera discovery built on it.

The whole point of this machinery is that a camera never has to be named in a
config: the gripper resolves its own side, its USB hub identifies the arm, and
whatever else hangs off that hub belongs to the same arm. These tests pin the two
parts that could silently produce a *plausible but wrong* answer — sensor
ordering, and the "everything on the hub that is not tactile is the wrist camera"
inference — because either one getting it wrong mislabels a dataset in a way that
only shows up long after the recording.

Topology is faked here (no hardware). The end-to-end check runs on a bench; see
stations/README.md.
"""

from importlib.util import find_spec

import pytest

from lerobot.grippers import usb_topology as topo

HAS_FLEXIV = find_spec("flexiv_rt") is not None
HAS_ELITE = find_spec("elite_cs_sdk") is not None


# ── Path parsing ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("sysfs", "hub", "port"),
    [
        # Observed shapes: a gripper MCU behind a hub, and a camera two levels down.
        ("/sys/devices/pci0000:00/0000:00:14.0/usb3/3-1/3-1.1/3-1.1:1.2/tty/ttyACM1", "3-1", "3-1.1"),
        ("/sys/devices/pci0000:80/0000:80:14.0/usb3/3-7/3-7.4/3-7.4:1.0/video4linux/video6", "3-7", "3-7.4"),
        # Deeper nesting still reports the TOP hub — that is what groups one arm.
        ("/sys/devices/pci0000:00/usb1/1-3/1-3.2/1-3.2.4/1-3.2.4:1.0/video4linux/video9", "1-3", "1-3.2"),
        # Straight into a root port, no intervening hub: the root port itself is
        # the group token, and there is no sub-port to order by. Devices sharing a
        # root port would therefore group together — fine here, because a gripper
        # arm is only ever anchored via its own hub.
        ("/sys/devices/pci0000:00/usb2/2-4/2-4:1.0/video4linux/video0", "2-4", None),
    ],
)
def test_usb_hub_and_port_parsing(sysfs, hub, port):
    got_hub, got_port = topo.usb_hub_and_port(sysfs)
    assert got_hub == hub
    if port is None:
        assert got_port is None
    else:
        # The port token must start at the hub — otherwise devices on different
        # arms could be ordered against each other.
        assert got_port is not None and got_port.startswith(hub)


@pytest.mark.parametrize(
    ("cam", "expected"),
    [(6, "/dev/video6"), ("/dev/video12", "/dev/video12"), ("4", "/dev/video4")],
)
def test_video_node_accepts_both_sdk_shapes(cam, expected):
    """scanSerialNumber() returns an int index on some SDK builds, a path on others."""
    assert topo.video_node(cam) == expected


# ── Grouping and ordering ───────────────────────────────────────────────────

# Two arms, each a hub with 2 tactile sensors + 1 wrist camera, plus an unrelated
# webcam straight on a root port. Deliberately declared out of port order.
FAKE_SYSFS = {
    "/dev/video20": "/sys/devices/pci0000:00/usb3/3-1/3-1.4/3-1.4:1.0/video4linux/video20",
    "/dev/video16": "/sys/devices/pci0000:00/usb3/3-1/3-1.2/3-1.2:1.0/video4linux/video16",
    "/dev/video19": "/sys/devices/pci0000:00/usb3/3-1/3-1.3/3-1.3:1.0/video4linux/video19",
    "/dev/video4": "/sys/devices/pci0000:00/usb3/3-7/3-7.2/3-7.2:1.0/video4linux/video4",
    "/dev/video8": "/sys/devices/pci0000:00/usb3/3-7/3-7.4/3-7.4:1.0/video4linux/video8",
    "/dev/video2": "/sys/devices/pci0000:00/usb3/3-8/3-8:1.0/video4linux/video2",
}
FAKE_TACTILE = {"TAC_B": 20, "TAC_A": 16, "TAC_D": 4, "TAC_C": 8}
FAKE_VIDEO_NAMES = {
    "TAC_B": ["/dev/video20"],
    "TAC_A": ["/dev/video16"],
    "WRIST_L": ["/dev/video19"],
    "TAC_D": ["/dev/video4"],
    "TAC_C": ["/dev/video8"],
    "Some USB webcam": ["/dev/video2"],
}


@pytest.fixture
def fake_topology(monkeypatch):
    """Patch the two enumerations and the sysfs walk with the layout above."""
    monkeypatch.setattr(
        topo, "hub_of_video_node", lambda node: topo.usb_hub_and_port(FAKE_SYSFS[node])
    )

    class _Sensor:
        @staticmethod
        def scanSerialNumber():  # noqa: N802 - mirrors the SDK's name
            return dict(FAKE_TACTILE)

    monkeypatch.setitem(__import__("sys").modules, "xensesdk", type("M", (), {"Sensor": _Sensor}))
    monkeypatch.setattr(
        "lerobot.cameras.opencv.camera_opencv._parse_v4l2_devices",
        lambda: dict(FAKE_VIDEO_NAMES),
    )


def test_tactile_grouped_by_hub_and_ordered_by_port(fake_topology):
    """Sensor order must follow USB port, not enumeration order — that is what
    makes *_tactile_0 the same physical pad on every run."""
    by_hub = topo.tactile_sns_by_hub()
    assert set(by_hub) == {"3-1", "3-7"}
    assert [sn for _, sn in by_hub["3-1"]] == ["TAC_A", "TAC_B"]  # 3-1.2 before 3-1.4
    assert [sn for _, sn in by_hub["3-7"]] == ["TAC_D", "TAC_C"]  # 3-7.2 before 3-7.4


def test_video_names_grouped_by_hub(fake_topology):
    by_hub = topo.video_names_by_hub()
    assert [n for _, n in by_hub["3-1"]] == ["TAC_A", "WRIST_L", "TAC_B"]
    assert by_hub["3-8"] == [("/dev/video2", "Some USB webcam")], (
        "a device on a root port has no hub token; it must fall back to its node "
        "rather than being grouped with an arm"
    )


# ── The wrist-camera inference ──────────────────────────────────────────────


def _infer_wrist(hub, tactile_by_hub, video_by_hub):
    """The inference under test, mirroring discover_serial_gripper_cameras."""
    tactile = {sn for _, sn in tactile_by_hub.get(hub, [])}
    return [name for _, name in video_by_hub.get(hub, []) if name not in tactile]


def test_wrist_inference_converges_to_one(fake_topology):
    t, v = topo.tactile_sns_by_hub(), topo.video_names_by_hub()
    assert _infer_wrist("3-1", t, v) == ["WRIST_L"]


def test_wrist_inference_detects_missing_and_ambiguous(fake_topology):
    """Both failure shapes must be distinguishable from success, so the caller can
    refuse rather than pick one."""
    t, v = topo.tactile_sns_by_hub(), topo.video_names_by_hub()
    # 3-7 has two tactile sensors and no other video device -> nothing to pick.
    assert _infer_wrist("3-7", t, v) == []
    # An extra camera on the arm's hub -> two candidates, not one.
    v["3-1"].append(("3-1.5", "AN_EXTRA_CAM"))
    assert sorted(_infer_wrist("3-1", t, v)) == ["AN_EXTRA_CAM", "WRIST_L"]


def test_discovery_refuses_when_not_exactly_one_candidate(monkeypatch, fake_topology):
    """A wrong-but-plausible camera is worse than a failure: it silently mislabels
    a dataset. Discovery must raise, naming what it found."""
    from lerobot.grippers.serial import discovery as sd

    monkeypatch.setattr(sd, "find_port_by_side", lambda side, **kw: "/dev/ttyUSB0")
    monkeypatch.setattr(sd, "_scan_port_sns", lambda **kw: {"/dev/ttyUSB0": "000031"})
    monkeypatch.setattr(sd, "hub_of_serial_device", lambda dev: "3-7")  # hub with no wrist

    with pytest.raises(RuntimeError, match="exactly one non-tactile video device"):
        sd.discover_serial_gripper_cameras(sides=("left",))


def test_discovery_resolves_a_well_formed_hub(monkeypatch, fake_topology):
    from lerobot.grippers.serial import discovery as sd

    monkeypatch.setattr(sd, "find_port_by_side", lambda side, **kw: "/dev/ttyUSB0")
    monkeypatch.setattr(sd, "_scan_port_sns", lambda **kw: {"/dev/ttyUSB0": "000031"})
    monkeypatch.setattr(sd, "hub_of_serial_device", lambda dev: "3-1")

    got = sd.discover_serial_gripper_cameras(sides=("left",))
    assert set(got) == {"left"}
    dev = got["left"]
    assert dev.usb_hub == "3-1"
    assert dev.wrist_camera_name == "WRIST_L"
    assert dev.tactile_sns == ["TAC_A", "TAC_B"]
    assert dev.gripper_sn == "000031"


def test_discovery_fails_loudly_when_hub_unresolvable(monkeypatch, fake_topology):
    """A gripper plugged straight into the host has no hub to group by; that is a
    wiring problem the operator must know about, not something to work around."""
    from lerobot.grippers.serial import discovery as sd

    monkeypatch.setattr(sd, "find_port_by_side", lambda side, **kw: "/dev/ttyUSB0")
    monkeypatch.setattr(sd, "_scan_port_sns", lambda **kw: {"/dev/ttyUSB0": "000031"})
    monkeypatch.setattr(sd, "hub_of_serial_device", lambda dev: None)

    with pytest.raises(RuntimeError, match="could not resolve the USB hub"):
        sd.discover_serial_gripper_cameras(sides=("left",))


# ── Config wiring ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("robot_type", "station"),
    [("bi_flexiv_rizon4_rt", "forward-06"), ("bi_elite_cs66_rt", "diagonal-08")],
)
def test_serial_autodiscover_is_off_by_default(robot_type, station):
    """Default off is what keeps the station file the verified source of truth —
    the golden snapshots in test_stations.py depend on this."""
    cfg = _config(robot_type)(bi_mount_type=station)
    assert cfg.serial_auto_discover_cameras is False
    assert cfg._serial_autodiscover is False
    assert cfg._autodiscover_cameras is False
    assert "left_wrist" in cfg.cameras, "station must still supply the cameras"


@pytest.mark.parametrize(
    ("robot_type", "station"),
    [("bi_flexiv_rizon4_rt", "forward-06"), ("bi_elite_cs66_rt", "diagonal-08")],
)
def test_serial_autodiscover_leaves_only_head(robot_type, station):
    cfg = _config(robot_type)(bi_mount_type=station, serial_auto_discover_cameras=True)
    assert cfg._serial_autodiscover is True
    assert sorted(cfg.cameras) == ["head"], "the driver injects wrist + tactile at connect"


@pytest.mark.parametrize(
    ("robot_type", "station"),
    [("bi_flexiv_rizon4_rt", "forward-06"), ("bi_elite_cs66_rt", "diagonal-08")],
)
def test_serial_flag_is_inert_under_taccap(robot_type, station):
    """The flag names the serial backend; it must not change taccap behaviour."""
    cfg = _config(robot_type)(
        bi_mount_type=station, gripper_type="taccap_follower", serial_auto_discover_cameras=True
    )
    assert cfg._serial_autodiscover is False


def _config(robot_type: str):
    if robot_type == "bi_flexiv_rizon4_rt":
        if not HAS_FLEXIV:
            pytest.skip("flexiv_rt not importable")
        from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import (
            BiFlexivRizon4RTConfig,
        )

        return BiFlexivRizon4RTConfig
    if not HAS_ELITE:
        pytest.skip("elite_cs_sdk not importable")
    from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig

    return BiEliteCS66RTConfig
