#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""TacCap follower (actuated) gripper backend.

Only the config class is imported eagerly — see the note in the sibling
``serial`` package: the driver and the discovery sweep both reach a hardware SDK,
and importing a submodule runs this file first.
"""

from .configuration_taccap import TaccapFollowerConfig  # noqa: F401

__all__ = [
    "TaccapFollowerConfig",
    "TaccapFollower",
    # Bring-up helper: what the arms call to wire the per-side wrist + GSPS
    # cameras, and the one to run by hand when a gripper does not show up.
    "discover_taccap_sides",
]


def __getattr__(name: str):
    if name == "TaccapFollower":
        from .taccap_follower import TaccapFollower

        return TaccapFollower
    if name == "discover_taccap_sides":
        from .discovery import discover_taccap_sides

        return discover_taccap_sides
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
