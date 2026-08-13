#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""TacCap follower (actuated) gripper backend."""

from .configuration_taccap import TaccapFollowerConfig  # noqa: F401
from .taccap_follower import TaccapFollower  # noqa: F401
