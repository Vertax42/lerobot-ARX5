#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Arm-agnostic gripper drivers, shared across robot packages.

Layout mirrors ``lerobot.cameras``: a ``Gripper`` ABC and a ``GripperConfig``
choice registry at the top, one subpackage per backend underneath.

Config classes are imported eagerly — they carry no hardware-SDK import, so arm
configs and ``--robot.type`` registration keep working on a host without the
Xense/TacCap SDKs. Driver classes are NOT imported here; get one through
``make_gripper_from_config``, which imports only the branch it needs.
"""

from .configs import GripperConfig  # noqa: F401
from .gripper import Gripper  # noqa: F401
from .utils import make_gripper_from_config  # noqa: F401

# Config-only imports: safe without any hardware SDK present, and required so the
# @register_subclass side effects run before a config is parsed or dispatched.
from .serial.configuration_serial import SerialGripperConfig  # noqa: F401
from .taccap.configuration_taccap import TaccapFollowerConfig  # noqa: F401
