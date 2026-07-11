#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from .config_elite_cs66_rt import EliteCS66RTConfig, EliteCS66RTControlMode  # noqa: F401

# Guarded: the driver pulls in the xense gripper stack (xensegripper/xensesdk) and
# elite_cs_sdk, so importing the CLI on a host without those must not crash. The
# SDK-free config above stays registered as an --robot.type choice regardless.
try:
    from .elite_cs66_rt import EliteCS66RT  # noqa: F401
except ImportError:
    pass
