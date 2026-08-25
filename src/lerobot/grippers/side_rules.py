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

"""The fleet's one serial-number → side rule: **odd → left, even → right**.

Xense burns the same convention (``单左双右``) into every serial that has to name
a side, and this module is the single place it is spelled out:

- **gripper board SN → which arm** (``000031`` → left) — see
  ``serial.discovery.sn_side``, which is this function under its historical name.
- **visuotactile sensor SN → which jaw** (``OG001349`` → the left finger) — see
  ``camera_injection.tactile_finger``, which turns it into the ``left`` / ``right``
  half of a ``<side>_tactile_<finger>`` camera key.

The parity is read off the **trailing run of digits**, so both bare numeric SNs
(``"000031"``) and prefixed ones (``"OG001349"``, ``"GSPS01A24Z0003"``) classify
by the same rule.

The sister repo ``xense-taccap-lerobot`` derives its ``{side}_tactile_{finger}``
keys from this identical rule (``serial_discovery.side_of_sequence``); a dataset
recorded here has to line up with one recorded there, which is why the two must
not drift.
"""

import re

# Trailing run of digits in a serial, e.g. "000031" -> "000031", "XG0042" -> "0042".
_TRAILING_DIGITS_RE = re.compile(r"(\d+)\s*$")


def side_of_serial(sn: str) -> str | None:
    """Classify a serial number to a side by parity: odd → left, even → right.

    Returns ``None`` when the serial has no trailing digits to classify — callers
    decide whether that is a skip (scan a bus, ignore what does not answer the
    rule) or a hard error (a device that *must* be identified).
    """
    if not sn:
        return None
    match = _TRAILING_DIGITS_RE.search(sn.strip())
    if match is None:
        return None
    return "left" if int(match.group(1)) % 2 == 1 else "right"
