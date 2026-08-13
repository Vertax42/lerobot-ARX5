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
from enum import Enum

import draccus  # type: ignore  # TODO: add type stubs for draccus


class SensorOutputType(Enum):
    """Output type for the visuotactile sensors embedded in a gripper.

    Single definition shared by the Xense and Flare backends — these were
    previously two byte-identical enums in separate config modules.
    """

    RECTIFY = "rectify"
    DIFFERENCE = "difference"


@dataclass(kw_only=True)
class GripperConfig(draccus.ChoiceRegistry, abc.ABC):  # type: ignore  # TODO: add type stubs for draccus
    """Base class for gripper configurations.

    Subclasses register themselves with ``@GripperConfig.register_subclass("<name>")``;
    that name is what ``type`` returns and what the factory dispatches on.
    """

    @property
    def type(self) -> str:
        return str(self.get_choice_name(self.__class__))
