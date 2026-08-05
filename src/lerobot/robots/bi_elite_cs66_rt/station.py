#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Station schema for the bimanual Elite CS66 RT bench.

One file per physical bench in ``stations/bi_elite_cs66_rt/<name>.yaml``,
selected by ``BiEliteCS66RTConfig.bi_mount_type``.

Unlike Flexiv, an Elite arm is addressed by controller IP and carries a mounting
description: the two CS66 arms are bolted diagonally/opposed, so each needs a
world<-base rotation to lift its base-frame TCP poses into ONE shared,
gravity-aligned world frame (x = facing, y = left, z = up) matching the Flexiv
convention, so recorded data is comparable across robots.

Grippers are absent for the same reason as Flexiv — left/right self-sorts at
connect, so there is no per-bench gripper identity.
"""

from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.robots.stations.spec import StationSpec, validate_pose, validate_rotation

N_JOINTS = 6


@dataclass
class EliteMountSpec:
    """How one CS66 is bolted to the bench, as a world<-base rotation.

    Two ways to express it, and ``world_rotation`` wins when present:

    ``tilt_deg`` / ``zrot_deg`` / ``world_yaw_deg`` build
    ``R = Rz(world_yaw)·Rz(zrot)·Rx(tilt)``, where:
      - ``tilt_deg`` (α) = tilt about base-X and ``zrot_deg`` (β) = rotation about
        Z are BOTH read off the teach pendant. Together they only fix the gravity
        vector in base (i.e. recover "z up"); they do NOT fix the heading.
      - ``world_yaw_deg`` (γ) is the extra yaw about world-Z that aligns each
        arm's heading into the one shared world frame. The teach pendant cannot
        show it — it is the residual yaw about gravity.

    ``world_rotation`` is the rotation written out directly (rows = world X/Y/Z
    axes expressed in base). Needed when the mounting is not a clean Rz·Rx, which
    is the case on the existing diagonal benches: the pendant's 45/90 reading
    assumes a tilt about base-X, but the LEFT arm is actually tilted 45° about
    base-Y (verified by freedrive-probe measurement plus the base-link geometry).

    Both are kept per bench even when ``world_rotation`` makes the angles inert,
    because the angles record what the pendant actually reads — useful when
    re-deriving the mounting on a new bench.
    """

    tilt_deg: float
    zrot_deg: float
    world_yaw_deg: float
    world_rotation: list[list[float]] | None = None


@dataclass
class EliteArmSpec:
    """One CS66 arm on a bench.

    Poses are joint angles in RADIANS (J1..J6), matching the units the Elite SDK
    takes. ``local_ip`` is the host-side interface to bind for RTSI; "" lets the
    OS route.
    """

    ip: str
    start_rad: list[float]
    home_rad: list[float]
    mount: EliteMountSpec
    local_ip: str = ""


@dataclass
class BiEliteStationSpec(StationSpec):
    arms: dict[str, EliteArmSpec] = field(default_factory=dict)

    REQUIRED_ARMS: ClassVar[tuple[str, ...]] = ("left", "right")

    def validate(self) -> None:
        super().validate()
        self.validate_arms(self.arms)
        for side, arm in self.arms.items():
            if not arm.ip:
                raise ValueError(f"station {self.name!r}: arms.{side}.ip is empty")
            validate_pose(self.name, f"arms.{side}.start_rad", arm.start_rad, N_JOINTS)
            validate_pose(self.name, f"arms.{side}.home_rad", arm.home_rad, N_JOINTS)
            validate_rotation(
                self.name, f"arms.{side}.mount.world_rotation", arm.mount.world_rotation
            )
        if self.arms["left"].ip == self.arms["right"].ip:
            raise ValueError(
                f"station {self.name!r}: both arms have ip {self.arms['left'].ip!r}; "
                "the two controllers must be distinct hosts"
            )
