#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Robot-agnostic station schema.

A *station* is one physical bench: which arms are bolted to it, how they are
mounted, and which cameras are wired to it. It lives in
``stations/<robot_type>/<name>.yaml`` and is selected by a robot config's
``bi_mount_type``.

Each robot subclasses :class:`StationSpec` with its own ``arms`` type — arm
identity differs per vendor (Flexiv addresses arms by serial number, Elite by
controller IP) and so do pose units (degrees vs radians), so there is nothing
useful to share beyond ``name`` / ``robot_type`` / ``cameras``.

Everything here describes *hardware*, never tuning. Control knobs stay in the
robot config dataclass and are set per-run from a recipe.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar

# CameraSpec.type -> the camera config class the robot builds. The robot owns the
# defaults (resolution, fps, warmup) because they differ per robot; the station
# only supplies the serial plus any per-camera override.
CAMERA_TYPES = ("realsense", "opencv", "xense_tactile")


@dataclass
class CameraSpec:
    """One camera wired to a station.

    The mapping key this is stored under is the observation key the camera's
    frames appear as (``head``, ``left_wrist``, ``right_tactile_0``, ...), so
    renaming a key renames the dataset column.

    Every field other than ``type`` / ``serial`` is an optional override: left
    unset (``None``) the robot's own default applies. Set one only when this
    station's camera genuinely differs from the fleet — an override that merely
    restates the default is noise that will silently rot when the default moves.
    """

    # NOTE: validation lives in validate(), not __post_init__, because draccus
    # swallows exceptions raised during construction and reports only a generic
    # "Couldn't instantiate class ...". load_station() calls validate() after
    # decoding so the pointed message survives.
    type: str
    serial: str

    fps: int | None = None
    width: int | None = None
    height: int | None = None
    warmup_s: float | None = None
    # RealSense only: pull the depth stream alongside color.
    use_depth: bool | None = None
    # XenseTactileCamera only: overrides the OG xpack camera template. The SDK
    # replaces the whole camera.<os> dict, so a partial dict here DROPS the keys
    # it omits — always write the full set.
    #
    # Typed dict[str, Any] rather than a narrower union on purpose: draccus has no
    # decoder for `object`, and a `int | float | bool | str` union coerces bools to
    # ints (auto_wb: false would reach the SDK as 0). Any passes values through
    # untouched, which is what an opaque SDK property bag needs.
    camera_properties: dict[str, Any] | None = None

    def validate(self, where: str) -> None:
        if self.type not in CAMERA_TYPES:
            raise ValueError(
                f"{where}: camera type must be one of {list(CAMERA_TYPES)}, got {self.type!r}"
            )
        if not self.serial:
            raise ValueError(f"{where}: camera of type {self.type!r} has an empty serial")
        if self.use_depth is not None and self.type != "realsense":
            raise ValueError(
                f"{where}: 'use_depth' only applies to realsense cameras, not {self.type!r}"
            )
        if self.camera_properties is not None and self.type != "xense_tactile":
            raise ValueError(
                f"{where}: 'camera_properties' only applies to xense_tactile cameras, "
                f"not {self.type!r}"
            )

    @property
    def overrides(self) -> dict[str, Any]:
        """The explicitly-set override fields, ready to splat into a camera config."""
        return {
            name: value
            for name, value in (
                ("fps", self.fps),
                ("width", self.width),
                ("height", self.height),
                ("warmup_s", self.warmup_s),
                ("use_depth", self.use_depth),
                ("camera_properties", self.camera_properties),
            )
            if value is not None
        }


@dataclass
class StationSpec:
    """Base for a per-robot station description.

    Subclasses add an ``arms`` field. ``name`` and ``robot_type`` are redundant
    with the file's location on disk on purpose: the loader cross-checks them, so
    a file copied to the wrong directory fails loudly instead of quietly
    describing the wrong bench.

    Subclasses extend :meth:`validate` (calling ``super().validate()``) rather
    than defining ``__post_init__`` — see the note on :class:`CameraSpec`.
    """

    name: str
    robot_type: str
    cameras: dict[str, CameraSpec] = field(default_factory=dict)

    # Subclasses set this to the arm keys they require, e.g. ("left", "right").
    # ClassVar, so it stays out of the dataclass fields draccus decodes from YAML.
    REQUIRED_ARMS: ClassVar[tuple[str, ...]] = ()

    def validate(self) -> None:
        """Check everything this spec can check on its own. Called by ``load_station``."""
        if not self.name:
            raise ValueError("station is missing a 'name'")
        for label, camera in self.cameras.items():
            camera.validate(f"station {self.name!r}: cameras.{label}")

    def validate_arms(self, arms: dict) -> None:
        """Check ``arms`` has exactly ``REQUIRED_ARMS`` as keys."""
        expected = set(self.REQUIRED_ARMS)
        got = set(arms)
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            detail = []
            if missing:
                detail.append(f"missing {missing}")
            if extra:
                detail.append(f"unexpected {extra}")
            raise ValueError(
                f"station {self.name!r}: 'arms' must have exactly "
                f"{sorted(expected)} ({', '.join(detail)})"
            )

    def tactile_camera_keys(self) -> list[str]:
        """Observation keys of the tactile cameras, in declaration order."""
        return [k for k, cam in self.cameras.items() if cam.type == "xense_tactile"]


def validate_pose(station_name: str, label: str, pose: list[float], n_joints: int) -> None:
    """Check a joint pose has the arm's joint count."""
    if len(pose) != n_joints:
        raise ValueError(
            f"station {station_name!r}: {label} must have {n_joints} elements "
            f"(J1..J{n_joints}), got {len(pose)}"
        )


def validate_rotation(station_name: str, label: str, rotation: list[list[float]] | None) -> None:
    """Check an explicit world<-base rotation is 3x3."""
    if rotation is None:
        return
    if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
        shape = f"{len(rotation)}x{[len(r) for r in rotation]}"
        raise ValueError(
            f"station {station_name!r}: {label} must be a 3x3 matrix "
            f"(rows = world X/Y/Z axes in base), got {shape}"
        )
