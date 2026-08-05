#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Station descriptions: the per-bench hardware a robot config is wired to.

See ``stations/README.md`` at the repo root for the file format and how to add
a bench.
"""

from lerobot.robots.stations.loader import (
    STATIONS_DIR_ENV,
    list_stations,
    load_station,
    resolve_station_file,
    station_search_dirs,
)
from lerobot.robots.stations.spec import (
    CAMERA_TYPES,
    CameraSpec,
    StationSpec,
    validate_pose,
    validate_rotation,
)

__all__ = [
    "CAMERA_TYPES",
    "STATIONS_DIR_ENV",
    "CameraSpec",
    "StationSpec",
    "list_stations",
    "load_station",
    "resolve_station_file",
    "station_search_dirs",
    "validate_pose",
    "validate_rotation",
]
