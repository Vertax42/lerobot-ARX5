#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Serial (XGripper) parallel-jaw gripper backend."""

from .configuration_serial import SerialGripperConfig  # noqa: F401
from .serial_gripper import SerialGripper  # noqa: F401

# Bring-up helpers. discover_serial_gripper_cameras is what the arms call;
# discover_serial_gripper_sides is the one to run by hand when a gripper does not
# show up — it reports which board SN answered on which port, and how parity
# classified it.
from .discovery import (  # noqa: F401
    discover_serial_gripper_cameras,
    discover_serial_gripper_sides,
)
