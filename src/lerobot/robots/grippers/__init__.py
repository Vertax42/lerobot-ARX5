#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Arm-agnostic gripper drivers shared across robot packages."""

# Config classes carry no hardware-SDK import — always available so robot configs and
# --robot.type registration keep working even when the Xense SDKs are not installed.
from .config_serial_gripper import SerialGripperConfig  # noqa: F401
from .config_taccap_follower_gripper import TaccapFollowerGripperConfig  # noqa: F401
from .config_xense_gripper import SensorOutputType, XenseGripperConfig  # noqa: F401

# taccap_follower_gripper guards its own `xense.taccap` import internally, so the driver
# class is always importable (it raises with rebuild guidance at connect() if absent).
from .taccap_follower_gripper import TaccapFollowerGripper  # noqa: F401

# The serial / xense gripper drivers import xensegripper (XGripper) and xensesdk at
# module top. Guard them so a host without the Xense SDK can still import this package
# (and the arm robots that pull it in); the driver classes are simply absent there.
try:
    from .serial_discovery import (  # noqa: F401
        SerialGripperSideDevice,
        discover_serial_gripper_sides,
        find_port_by_side,
        find_port_by_sn,
        sn_side,
    )
    from .serial_gripper import SerialGripper  # noqa: F401
    from .xense_gripper import XenseGripper  # noqa: F401
except ImportError:
    pass
