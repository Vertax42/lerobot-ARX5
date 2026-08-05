#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Find and decode ``stations/<robot_type>/<name>.yaml``.

Decoding goes through ``draccus.decode``, which rejects unknown keys — a typo'd
field is an error at load time naming the offending key, not a silently-missing
camera at connect time.
"""

import os
from dataclasses import is_dataclass
from pathlib import Path
from typing import TypeVar

import draccus
import yaml

from lerobot.robots.stations.spec import StationSpec

T = TypeVar("T", bound=StationSpec)

STATIONS_DIR_ENV = "LEROBOT_STATIONS_DIR"
_STATIONS_DIRNAME = "stations"


def _repo_stations_dir() -> Path | None:
    """``<repo>/stations`` inferred from the installed package location.

    Works for an editable install (``src/lerobot/robots/stations/loader.py`` ->
    four parents up is ``src/``'s parent, the repo root). Returns None for a
    plain wheel install, where the directory does not ship — see the module
    docstring of ``stations/README.md`` and set ``LEROBOT_STATIONS_DIR`` instead.
    """
    candidate = Path(__file__).resolve().parents[4] / _STATIONS_DIRNAME
    return candidate if candidate.is_dir() else None


def station_search_dirs(robot_type: str) -> list[Path]:
    """Directories searched for a station file, highest precedence first."""
    dirs: list[Path] = []
    env_dir = os.environ.get(STATIONS_DIR_ENV)
    if env_dir:
        dirs.append(Path(env_dir).expanduser() / robot_type)
    dirs.append(Path.cwd() / _STATIONS_DIRNAME / robot_type)
    repo_dir = _repo_stations_dir()
    if repo_dir is not None:
        dirs.append(repo_dir / robot_type)
    # Preserve order while dropping duplicates (cwd == repo root is the common case).
    seen: set[Path] = set()
    unique: list[Path] = []
    for d in dirs:
        resolved = d.resolve() if d.exists() else d
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(d)
    return unique


def _is_path_like(name: str) -> bool:
    """True when ``name`` should be treated as a filesystem path, not a station id."""
    return "/" in name or name.endswith((".yaml", ".yml"))


def list_stations(robot_type: str) -> list[str]:
    """Station names discoverable for ``robot_type``, deduplicated and sorted."""
    names: set[str] = set()
    for directory in station_search_dirs(robot_type):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.suffix in (".yaml", ".yml") and not path.name.startswith("_"):
                names.add(path.stem)
    return sorted(names)


def resolve_station_file(robot_type: str, name: str) -> Path:
    """Locate the YAML describing station ``name`` of ``robot_type``.

    ``name`` may also be a path (contains ``/`` or ends in ``.yaml``/``.yml``),
    which is used verbatim — handy for a one-off bench file that is not worth
    committing yet.
    """
    if _is_path_like(name):
        path = Path(name).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"station file {name!r} (read as a path because it contains '/' or a "
                f"YAML suffix) does not exist: {path}"
            )
        return path

    tried = []
    for directory in station_search_dirs(robot_type):
        for suffix in (".yaml", ".yml"):
            candidate = directory / f"{name}{suffix}"
            tried.append(candidate)
            if candidate.is_file():
                return candidate

    available = list_stations(robot_type)
    hint = (
        f"available {robot_type} stations: {available}"
        if available
        else (
            f"no station files found at all. If lerobot is installed as a plain wheel "
            f"the repo's stations/ directory is not present — point {STATIONS_DIR_ENV} "
            f"at a checkout's stations/ directory."
        )
    )
    tried_list = "\n  ".join(str(p) for p in tried)
    raise FileNotFoundError(
        f"unknown {robot_type} station {name!r}; {hint}\ntried:\n  {tried_list}"
    )


def load_station(spec_cls: type[T], robot_type: str, name: str) -> T:
    """Load, decode and cross-check one station file.

    Raises with the file path attached on malformed YAML, unknown fields, a
    ``robot_type`` that disagrees with the directory the file was found in, or
    any validation the spec's ``__post_init__`` performs.
    """
    if not is_dataclass(spec_cls):
        raise TypeError(f"spec_cls must be a dataclass, got {spec_cls!r}")

    path = resolve_station_file(robot_type, name)
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"station file {path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"station file {path} must contain a YAML mapping at the top level, "
            f"got {type(raw).__name__}"
        )

    try:
        station = draccus.decode(spec_cls, raw)
    except Exception as exc:
        raise ValueError(f"station file {path} failed to decode: {exc}") from exc

    # Validation is a post-decode step, not __post_init__: draccus reports only a
    # generic "Couldn't instantiate class" for anything raised during construction,
    # which would throw away the message naming the offending field.
    try:
        station.validate()
    except ValueError as exc:
        raise ValueError(f"station file {path} is invalid: {exc}") from exc

    if station.robot_type != robot_type:
        raise ValueError(
            f"station file {path} declares robot_type={station.robot_type!r} but was "
            f"loaded for {robot_type!r} — the file is in the wrong directory, or its "
            f"robot_type is wrong"
        )
    # Only meaningful when the file was found by station id; a path-like name may
    # legitimately be called anything.
    if not _is_path_like(name) and station.name != name:
        raise ValueError(
            f"station file {path} declares name={station.name!r} but is filed as "
            f"{name!r} — rename the file or fix the 'name' field so recipes and "
            f"files agree"
        )
    return station
