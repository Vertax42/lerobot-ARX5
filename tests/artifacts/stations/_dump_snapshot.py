#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dump the fully-resolved station fields of the bimanual configs to JSON.

Run this to (re)generate the golden fixtures consumed by
``tests/robots/test_stations.py``. It only constructs config dataclasses, so it
touches no hardware and needs no robot on the network.

    ~/miniforge3/envs/lerobot-xense/bin/python tests/artifacts/stations/_dump_snapshot.py

The fixtures were first generated from the pre-migration ``_PRESETS`` dicts and
are what proves the ``stations/*.yaml`` migration is behaviour-preserving: do
NOT regenerate them to make a failing test pass. Regenerate only when a station's
hardware genuinely changes, and review the resulting diff as a hardware change.
"""

import dataclasses
import json
from pathlib import Path

from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig
from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import BiFlexivRizon4RTConfig

OUT_DIR = Path(__file__).parent

# Every field each robot resolves from its station, plus the knobs that steer how
# the station is applied. Anything listed here is compared field-by-field.
FLEXIV_FIELDS = [
    "bi_mount_type",
    "left_robot_sn",
    "right_robot_sn",
    "left_start_position_degree",
    "right_start_position_degree",
    "left_home_position_degree",
    "right_home_position_degree",
    "enable_tactile_sensors",
    "head_camera_use_depth",
    "gripper_type",
    "_taccap_autodiscover",
]

ELITE_FIELDS = [
    "bi_mount_type",
    "left_robot_ip",
    "right_robot_ip",
    "left_local_ip",
    "right_local_ip",
    "left_start_position_rad",
    "right_start_position_rad",
    "left_home_position_rad",
    "right_home_position_rad",
    "left_mount_tilt_deg",
    "left_mount_zrot_deg",
    "left_mount_world_yaw_deg",
    "right_mount_tilt_deg",
    "right_mount_zrot_deg",
    "right_mount_world_yaw_deg",
    "left_world_rotation",
    "right_world_rotation",
    "enable_tactile_sensors",
    "gripper_type",
    "_taccap_autodiscover",
]

CASES = [
    (
        "bi_flexiv_rizon4_rt",
        BiFlexivRizon4RTConfig,
        FLEXIV_FIELDS,
        ["forward-04", "forward-05", "forward-06", "forward-dewu", "diagonal-02"],
    ),
    (
        "bi_elite_cs66_rt",
        BiEliteCS66RTConfig,
        ELITE_FIELDS,
        ["diagonal-07", "diagonal-08"],
    ),
]

# Both gripper backends: "serial" wires every camera from the station, while
# "taccap_follower" + auto-discover leaves only the head (the wrist/tactile SNs
# are sniffed at connect). The snapshot must pin both branches.
GRIPPER_TYPES = ["serial", "taccap_follower"]


def snapshot_camera(cam) -> dict:
    """Every field of a camera config, plus its class name, JSON-safe."""
    out = {"__class__": type(cam).__name__}
    for f in dataclasses.fields(cam):
        out[f.name] = _jsonable(getattr(cam, f.name))
    return out


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if hasattr(value, "value") and hasattr(type(value), "__members__"):  # Enum
        return value.value
    if dataclasses.is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    return value


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for robot_type, cfg_cls, fields, stations in CASES:
        for station in stations:
            for gripper_type in GRIPPER_TYPES:
                cfg = cfg_cls(bi_mount_type=station, gripper_type=gripper_type)
                snap = {
                    "robot_type": robot_type,
                    "station": station,
                    "gripper_type": gripper_type,
                    "config": {name: _jsonable(getattr(cfg, name)) for name in fields},
                    "cameras": {
                        label: snapshot_camera(cam) for label, cam in cfg.cameras.items()
                    },
                }
                path = OUT_DIR / f"{robot_type}__{station}__{gripper_type}.json"
                path.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n")
                written += 1
                print(f"wrote {path.relative_to(OUT_DIR.parents[2])} ({len(snap['cameras'])} cameras)")
    print(f"\n{written} snapshots written to {OUT_DIR}")


if __name__ == "__main__":
    main()
