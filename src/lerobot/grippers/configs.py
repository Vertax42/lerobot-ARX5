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

"""Gripper configuration base class and shared enums.

Mirrors ``lerobot.cameras.configs``: a ``draccus.ChoiceRegistry`` base so each
backend registers under a stable string name (``"serial"``, ``"taccap_follower"``,
``"xense"``, ``"flare"``) and ``make_gripper_from_config`` can dispatch on
``cfg.type`` instead of a chain of ``isinstance`` checks duplicated per arm.

Config classes deliberately carry no hardware-SDK import, so an arm's config
module stays importable on a host where that gripper's SDK was never built.
"""

import abc
from dataclasses import dataclass
from typing import ClassVar

import draccus  # type: ignore  # TODO: add type stubs for draccus


@dataclass(kw_only=True)
class GripperConfig(draccus.ChoiceRegistry, abc.ABC):  # type: ignore  # TODO: add type stubs for draccus
    """Base class for gripper configurations.

    Subclasses register themselves with ``@GripperConfig.register_subclass("<name>")``;
    that name is what ``type`` returns and what the factory dispatches on. A robot's
    ``gripper:`` block in a recipe is decoded through that registry, so draccus
    rejects a knob that belongs to a different backend instead of silently ignoring
    it — which is what the old flat ``gripper_type`` + ``taccap_*``/``gripper_*``
    fields did.
    """

    #: Fields a recipe should not bother listing: pinned by the wire protocol or
    #: the wiring, with no command to change them. They stay configurable (a CLI
    #: override still works, and odd hardware may need one) but are left out of
    #: the commented reference block in recipes, so that block only carries knobs
    #: worth turning. ``tests/robots/test_recipe_gripper_blocks.py`` reads this.
    protocol_fixed_fields: ClassVar[frozenset[str]] = frozenset()

    auto_discover_cameras: bool = False
    """Sniff this gripper's wrist + tactile cameras off its own USB hub at connect.

    Cameras only. Finding the *gripper* is always automatic and not switchable:
    each backend resolves its own side at connect (serial by board-SN parity,
    taccap by the firmware-burned SN), which is why no recipe pins a gripper SN.

    On means the recipe pins just the head camera and the robot injects the rest.
    TacCap defaults to on — its wrist and GSPS sensors travel with the gripper, so
    swapping one swaps them. Serial defaults to off — those cameras are mounted on
    the bench, so the recipe stays their source of truth until discovery has been
    checked against it there.
    """

    enable_tactile: bool = True
    """Wire this gripper's two tactile sensors as cameras alongside its wrist.

    Lives here for the same reason ``auto_discover_cameras`` does: what a gripper
    carries is a property of the gripper, not of the arm holding it. Swapping a
    TacCap gripper for an XGripper changes which sensors are on the hub; the arm
    is unchanged.

    Only consulted when the cameras are discovered — a recipe that pins them by
    hand simply lists the ones it wants.
    """

    @property
    def type(self) -> str:
        return str(self.get_choice_name(self.__class__))
