#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Station schema for the bimanual Flexiv Rizon4 RT bench.

One file per physical bench in ``stations/bi_flexiv_rizon4_rt/<name>.yaml``,
selected by ``BiFlexivRizon4RTConfig.bi_mount_type``.

Grippers are deliberately absent: both backends resolve left/right at connect
(serial by board-SN parity, taccap by the firmware-burned SN), so there is no
per-bench gripper identity to pin.
"""

from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.robots.stations.spec import StationSpec, validate_pose

N_JOINTS = 7


@dataclass
class FlexivArmSpec:
    """One Rizon4 arm on a bench.

    Poses are joint angles in DEGREES (J1..J7), matching the units the driver's
    MoveJ takes. ``home_deg`` is where the arm parks on disconnect; it is usually
    identical to ``start_deg`` but is kept separate so a bench can differ.
    """

    serial_number: str
    start_deg: list[float]
    home_deg: list[float]


@dataclass
class BiFlexivStationSpec(StationSpec):
    arms: dict[str, FlexivArmSpec] = field(default_factory=dict)

    REQUIRED_ARMS: ClassVar[tuple[str, ...]] = ("left", "right")

    def validate(self) -> None:
        super().validate()
        self.validate_arms(self.arms)
        for side, arm in self.arms.items():
            if not arm.serial_number:
                raise ValueError(f"station {self.name!r}: arms.{side}.serial_number is empty")
            validate_pose(self.name, f"arms.{side}.start_deg", arm.start_deg, N_JOINTS)
            validate_pose(self.name, f"arms.{side}.home_deg", arm.home_deg, N_JOINTS)
