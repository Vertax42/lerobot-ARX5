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
"""Station loading, and the guard that ``stations/*.yaml`` still resolves to what
the old ``_PRESETS`` dicts did.

The golden fixtures in ``tests/artifacts/stations/`` were generated from the
_PRESETS literals before they were deleted (see ``_dump_snapshot.py``), so they
are an independent record of the pre-migration behaviour. A failure here means a
station file drifted from the hardware it is supposed to describe — regenerating
the fixtures to make it pass would destroy the only evidence.

The schema/loader tests need no robot SDK; the snapshot tests import the config
modules and so need ``flexiv_rt`` / ``elite_cs_sdk``, and skip without them.
"""

import dataclasses
import json
import textwrap
from importlib.util import find_spec
from pathlib import Path

import pytest

from lerobot.robots.bi_flexiv_rizon4_rt.station import BiFlexivStationSpec
from lerobot.robots.stations import list_stations, load_station, resolve_station_file
from lerobot.robots.stations.spec import StationSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIONS_ROOT = REPO_ROOT / "stations"
ARTIFACTS = REPO_ROOT / "tests" / "artifacts" / "stations"

HAS_FLEXIV = find_spec("flexiv_rt") is not None
HAS_ELITE = find_spec("elite_cs_sdk") is not None

# robot_type -> (spec class, whether its SDK is importable)
ROBOTS = {
    "bi_flexiv_rizon4_rt": HAS_FLEXIV,
    "bi_elite_cs66_rt": HAS_ELITE,
}


def _spec_class(robot_type: str) -> type[StationSpec]:
    if robot_type == "bi_flexiv_rizon4_rt":
        return BiFlexivStationSpec
    if robot_type == "bi_elite_cs66_rt":
        from lerobot.robots.bi_elite_cs66_rt.station import BiEliteStationSpec

        return BiEliteStationSpec
    raise AssertionError(robot_type)  # pragma: no cover - guarded by parametrization


def _config_class(robot_type: str):
    """Imported lazily: these modules pull in the vendor SDK, which is absent in
    environments that can still run the schema/loader tests."""
    if robot_type == "bi_flexiv_rizon4_rt":
        from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import (
            BiFlexivRizon4RTConfig,
        )

        return BiFlexivRizon4RTConfig
    from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig

    return BiEliteCS66RTConfig


def _all_station_files() -> list[Path]:
    return sorted(STATIONS_ROOT.glob("*/*.yaml"))


def _jsonable(value):
    """Mirror of ``_dump_snapshot._jsonable`` — see that module for the why."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if hasattr(value, "value") and hasattr(type(value), "__members__"):
        return value.value
    if dataclasses.is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    return value


@pytest.fixture
def stations_dir(tmp_path, monkeypatch):
    """An isolated stations tree that resolve_station_file will search first."""
    monkeypatch.setenv("LEROBOT_STATIONS_DIR", str(tmp_path))
    (tmp_path / "bi_flexiv_rizon4_rt").mkdir()
    return tmp_path


def write_station(stations_dir: Path, name: str, body: str) -> None:
    (stations_dir / "bi_flexiv_rizon4_rt" / f"{name}.yaml").write_text(textwrap.dedent(body))


VALID_STATION = """
    name: {name}
    robot_type: bi_flexiv_rizon4_rt
    arms:
      left:  {{serial_number: L1, start_deg: [1,2,3,4,5,6,7], home_deg: [1,2,3,4,5,6,7]}}
      right: {{serial_number: R1, start_deg: [1,2,3,4,5,6,7], home_deg: [1,2,3,4,5,6,7]}}
    cameras:
      head: {{type: realsense, serial: "123456"}}
"""


# ── The committed station files ─────────────────────────────────────────────


def test_repo_has_station_files():
    """Guards against the whole tree going missing, which would make the
    per-file tests below vacuously pass."""
    assert _all_station_files(), f"no station YAML found under {STATIONS_ROOT}"


@pytest.mark.parametrize("path", _all_station_files(), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_committed_station_loads(path):
    """Every committed station decodes, validates, and agrees with its location."""
    robot_type = path.parent.name
    assert robot_type in ROBOTS, f"unexpected robot directory {robot_type!r}"
    station = load_station(_spec_class(robot_type), robot_type, path.stem)
    assert station.name == path.stem
    assert station.robot_type == robot_type
    assert set(station.arms) == {"left", "right"}
    assert station.cameras, "a station with no cameras is almost certainly a mistake"


@pytest.mark.parametrize("robot_type", sorted(ROBOTS))
def test_list_stations_finds_committed_files(robot_type):
    listed = list_stations(robot_type)
    on_disk = sorted(p.stem for p in (STATIONS_ROOT / robot_type).glob("*.yaml"))
    assert set(on_disk).issubset(set(listed))


# ── Loader behaviour ────────────────────────────────────────────────────────


def test_unknown_station_error_lists_alternatives(stations_dir):
    write_station(stations_dir, "real", VALID_STATION.format(name="real"))
    with pytest.raises(FileNotFoundError) as exc:
        load_station(BiFlexivStationSpec, "bi_flexiv_rizon4_rt", "nope")
    message = str(exc.value)
    assert "nope" in message
    assert "real" in message, "the error should list what IS available"
    assert "tried:" in message, "the error should show the paths it searched"


def test_path_like_name_is_used_verbatim(tmp_path):
    """A bench not worth committing can be passed as a path."""
    path = tmp_path / "scratch-bench.yaml"
    path.write_text(textwrap.dedent(VALID_STATION.format(name="anything-at-all")))
    station = load_station(BiFlexivStationSpec, "bi_flexiv_rizon4_rt", str(path))
    # A path-like name skips the filename/name cross-check on purpose.
    assert station.name == "anything-at-all"


def test_unknown_field_is_rejected(stations_dir):
    body = VALID_STATION.format(name="typo").replace("serial_number: L1", "serial_numbr: L1")
    write_station(stations_dir, "typo", body)
    with pytest.raises(ValueError, match="serial_numbr"):
        load_station(BiFlexivStationSpec, "bi_flexiv_rizon4_rt", "typo")


def test_name_must_match_filename(stations_dir):
    write_station(stations_dir, "filed-as", VALID_STATION.format(name="declared-as"))
    with pytest.raises(ValueError, match="declared-as"):
        load_station(BiFlexivStationSpec, "bi_flexiv_rizon4_rt", "filed-as")


def test_robot_type_must_match_directory(stations_dir):
    body = VALID_STATION.format(name="wrongdir").replace(
        "robot_type: bi_flexiv_rizon4_rt", "robot_type: bi_elite_cs66_rt"
    )
    write_station(stations_dir, "wrongdir", body)
    with pytest.raises(ValueError, match="wrong directory"):
        load_station(BiFlexivStationSpec, "bi_flexiv_rizon4_rt", "wrongdir")


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (("[1,2,3,4,5,6,7], home_deg", "[1,2,3], home_deg"), "must have 7 elements"),
        (("type: realsense", "type: webcam"), "camera type must be one of"),
        (("serial_number: L1", 'serial_number: ""'), "serial_number is empty"),
        (('serial: "123456"', 'serial: ""'), "empty serial"),
        (
            ('serial: "123456"}', 'serial: "123456", camera_properties: {a: 1}}'),
            "only applies to xense_tactile",
        ),
    ],
    ids=["short-pose", "bad-camera-type", "empty-arm-serial", "empty-camera-serial", "misplaced-override"],
)
def test_validation_errors_name_the_problem(stations_dir, mutation, expected):
    """Validation runs post-decode so its message survives draccus, which
    otherwise reports only a generic 'Couldn't instantiate class'."""
    old, new = mutation
    body = VALID_STATION.format(name="bad").replace(old, new, 1)
    write_station(stations_dir, "bad", body)
    with pytest.raises(ValueError, match=expected):
        load_station(BiFlexivStationSpec, "bi_flexiv_rizon4_rt", "bad")


def test_camera_overrides_round_trip(stations_dir):
    """camera_properties is an opaque SDK property bag — values must reach the
    camera config untouched. A narrower union type than dict[str, Any] silently
    turned `auto_wb: false` into 0, and `object` had no draccus decoder at all."""
    # Appended raw, not dedent()ed: write_station dedents the whole body, so this
    # block must already sit at the indent it needs under `cameras:`.
    extra_camera = (
        "      left_tactile_0:\n"
        "        type: xense_tactile\n"
        "        serial: OG9999\n"
        "        warmup_s: 0.25\n"
        "        fps: 15\n"
        "        camera_properties: {auto_exposure: 1, auto_wb: false, exposure: 250}\n"
    )
    write_station(stations_dir, "overrides", VALID_STATION.format(name="overrides") + extra_camera)
    station = load_station(BiFlexivStationSpec, "bi_flexiv_rizon4_rt", "overrides")
    spec = station.cameras["left_tactile_0"]
    assert spec.camera_properties == {"auto_exposure": 1, "auto_wb": False, "exposure": 250}
    assert spec.camera_properties["auto_wb"] is False, "bool must not be coerced to int"
    assert spec.overrides == {
        "fps": 15,
        "warmup_s": 0.25,
        "camera_properties": {"auto_exposure": 1, "auto_wb": False, "exposure": 250},
    }
    # A camera that overrides nothing contributes no keys, so the robot's own
    # defaults apply untouched.
    assert station.cameras["head"].overrides == {}


def test_missing_arm_is_rejected(stations_dir):
    body = VALID_STATION.format(name="onearm")
    body = "\n".join(line for line in body.splitlines() if "right:" not in line)
    write_station(stations_dir, "onearm", body)
    with pytest.raises(ValueError, match="right"):
        load_station(BiFlexivStationSpec, "bi_flexiv_rizon4_rt", "onearm")


def test_env_var_takes_precedence_over_repo(stations_dir):
    """A bench-local override shadows the committed file of the same name."""
    name = list_stations("bi_flexiv_rizon4_rt")[0]
    write_station(stations_dir, name, VALID_STATION.format(name=name))
    resolved = resolve_station_file("bi_flexiv_rizon4_rt", name)
    assert resolved.parent.parent == stations_dir
    station = load_station(BiFlexivStationSpec, "bi_flexiv_rizon4_rt", name)
    assert station.arms["left"].serial_number == "L1"


# ── Golden snapshot: the migration preserved behaviour ──────────────────────


def _snapshot_cases() -> list[tuple[str, str, str]]:
    cases = []
    for path in sorted(ARTIFACTS.glob("*.json")):
        robot_type, station, gripper_type = path.stem.split("__")
        cases.append((robot_type, station, gripper_type))
    return cases


@pytest.mark.parametrize(
    ("robot_type", "station", "gripper_type"),
    _snapshot_cases(),
    ids=lambda v: v,
)
def test_station_matches_frozen_snapshot(robot_type, station, gripper_type):
    if not ROBOTS[robot_type]:
        pytest.skip(f"{robot_type} SDK not importable in this environment")

    gold = json.loads((ARTIFACTS / f"{robot_type}__{station}__{gripper_type}.json").read_text())
    cfg = _config_class(robot_type)(bi_mount_type=station, gripper_type=gripper_type)

    for name, expected in gold["config"].items():
        assert _jsonable(getattr(cfg, name)) == expected, f"config.{name} drifted"

    assert set(cfg.cameras) == set(gold["cameras"]), "camera set drifted"
    for label, expected_cam in gold["cameras"].items():
        actual = cfg.cameras[label]
        assert type(actual).__name__ == expected_cam["__class__"]
        for field_name, expected_value in expected_cam.items():
            if field_name == "__class__":
                continue
            assert _jsonable(getattr(actual, field_name)) == expected_value, (
                f"cameras.{label}.{field_name} drifted"
            )


# ── Explicit values win over the station ────────────────────────────────────


@pytest.mark.skipif(not HAS_FLEXIV, reason="flexiv_rt not importable")
def test_flexiv_explicit_value_overrides_station():
    """Before the migration the config overwrote these unconditionally, so a
    recipe setting left_robot_sn was silently discarded."""
    cls = _config_class("bi_flexiv_rizon4_rt")
    from_station = cls(bi_mount_type="forward-05")
    assert from_station.left_robot_sn == "Rizon4-063786"

    overridden = cls(bi_mount_type="forward-05", left_robot_sn="SPARE-ARM")
    assert overridden.left_robot_sn == "SPARE-ARM"
    # the side that was not overridden still follows the station
    assert overridden.right_robot_sn == from_station.right_robot_sn


@pytest.mark.skipif(not HAS_FLEXIV, reason="flexiv_rt not importable")
def test_flexiv_explicit_pose_overrides_station():
    cls = _config_class("bi_flexiv_rizon4_rt")
    pose = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    cfg = cls(bi_mount_type="forward-06", left_start_position_degree=pose)
    assert cfg.left_start_position_degree == pose
    assert cfg.right_start_position_degree == cls(bi_mount_type="forward-06").right_start_position_degree


@pytest.mark.skipif(not HAS_ELITE, reason="elite_cs_sdk not importable")
def test_elite_explicit_values_override_station():
    cls = _config_class("bi_elite_cs66_rt")
    baseline = cls(bi_mount_type="diagonal-08")
    cfg = cls(bi_mount_type="diagonal-08", right_robot_ip="10.0.0.5", left_mount_world_yaw_deg=90.0)
    assert cfg.right_robot_ip == "10.0.0.5"
    assert cfg.left_robot_ip == baseline.left_robot_ip
    assert cfg.left_mount_world_yaw_deg == 90.0
    assert cfg.right_mount_world_yaw_deg == baseline.right_mount_world_yaw_deg


@pytest.mark.skipif(not HAS_ELITE, reason="elite_cs_sdk not importable")
def test_elite_empty_world_rotation_falls_back_to_angles():
    """[] is the only way to say "ignore the station's matrix" now that None
    means "inherit it". It collapses to None so the driver's `is not None`
    check keeps working unchanged."""
    cls = _config_class("bi_elite_cs66_rt")
    cfg = cls(bi_mount_type="diagonal-08", left_world_rotation=[])
    assert cfg.left_world_rotation is None
    assert cfg.right_world_rotation is not None, "the other arm still inherits"


@pytest.mark.skipif(not HAS_FLEXIV, reason="flexiv_rt not importable")
def test_explicit_cameras_dict_wins_over_station():
    from lerobot.cameras.opencv import OpenCVCameraConfig

    cls = _config_class("bi_flexiv_rizon4_rt")
    cfg = cls(
        bi_mount_type="forward-06",
        cameras={"only": OpenCVCameraConfig(index_or_path="X", width=64, height=48, fps=30)},
    )
    assert list(cfg.cameras) == ["only"]


@pytest.mark.skipif(not HAS_FLEXIV, reason="flexiv_rt not importable")
def test_unknown_station_fails_config_construction():
    cls = _config_class("bi_flexiv_rizon4_rt")
    with pytest.raises(FileNotFoundError, match="forward-05"):
        cls(bi_mount_type="no-such-bench")
