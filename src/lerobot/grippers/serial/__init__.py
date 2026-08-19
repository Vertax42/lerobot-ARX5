#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Serial (XGripper) parallel-jaw gripper backend.

Only the config class is imported eagerly. The driver and the discovery helpers
both reach the ``xgripper`` SDK, and importing a submodule runs this file first —
so an eager import here would drag that SDK in for anyone who merely parsed an
arm config, which is exactly what ``lerobot.grippers`` promises not to do.
They resolve on first attribute access instead.
"""

from .configuration_serial import SerialGripperConfig  # noqa: F401

__all__ = [
    "SerialGripperConfig",
    "SerialGripper",
    # Bring-up helpers. discover_serial_gripper_cameras is what the arms call;
    # discover_serial_gripper_sides is the one to run by hand when a gripper does
    # not show up — it reports which board SN answered on which port, and how
    # parity classified it.
    "discover_serial_gripper_cameras",
    "discover_serial_gripper_sides",
]


def __getattr__(name: str):
    if name == "SerialGripper":
        from .serial_gripper import SerialGripper

        return SerialGripper
    if name in ("discover_serial_gripper_cameras", "discover_serial_gripper_sides"):
        from . import discovery

        return getattr(discovery, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
