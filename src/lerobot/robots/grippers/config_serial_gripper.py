#!/usr/bin/env python

# Copyright 2025 The XenseRobotics Inc. team. All rights reserved.
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

"""Configuration for pure-serial Xense gripper (no ezros / xensesdk required)."""

from dataclasses import dataclass


@dataclass
class SerialGripperConfig:
    """Configuration for the serial-port Xense gripper (XenseSerialGripper).

    This gripper communicates directly over a USB-serial port and does not
    require the ezros / xensesdk stack.

    Identification (provide one):
        sn:              Board serial number (e.g. ``"000001"``).  When set,
                         the driver scans available serial ports at connect()
                         time and picks the one whose READ_BOARD_SN response
                         matches.  Takes priority over ``port``.
        port:            Fallback explicit serial port path (e.g.
                         ``"/dev/ttyUSB0"``).  Used when ``sn`` is None.

    Attributes:
        baudrate:        Serial baud rate (default: 115200).
        serial_timeout:  Read timeout in seconds for the serial port (default: 1.0).

        gripper_min_pos: Minimum gripper position in mm (0 = fully closed).
        gripper_max_pos: Maximum gripper position in mm (85 = fully open).
        gripper_v_max:   Maximum jaw velocity in mm/s [0, 350].
        gripper_f_max:   Maximum jaw force in N [0, 60].

        init_open:       If True, fully open the gripper on ``connect()``.
    """

    # ── Identification ─────────────────────────────────────────────────────────
    sn: str | None = None          # board SN — preferred over port
    port: str = ""                 # fallback explicit port path

    # ── Serial connection ──────────────────────────────────────────────────────
    baudrate: int = 115200
    serial_timeout: float = 1.0
    device_id: int = 1  # XenseSerialGripper device ID on the RS-485 bus

    # ── Mechanical limits ──────────────────────────────────────────────────────
    gripper_min_pos: float = 0.0   # mm — fully closed
    gripper_max_pos: float = 85.0  # mm — fully open

    # ── Motion parameters ──────────────────────────────────────────────────────
    gripper_v_max: float = 80.0  # mm/s  (range: 0–350)
    gripper_f_max: float = 27.0  # N     (range: 0–60)

    # ── Initialization ─────────────────────────────────────────────────────────
    init_open: bool = True

    def __post_init__(self):
        if not self.sn and not self.port:
            raise ValueError("SerialGripperConfig: provide either 'sn' or 'port'.")
        if not 0 < self.baudrate:
            raise ValueError(f"SerialGripperConfig: baudrate must be positive, got {self.baudrate}.")
        if not 0.0 <= self.gripper_min_pos < self.gripper_max_pos:
            raise ValueError(
                f"SerialGripperConfig: gripper_min_pos ({self.gripper_min_pos}) must be "
                f"< gripper_max_pos ({self.gripper_max_pos})."
            )
        if not 0.0 < self.gripper_v_max <= 350.0:
            raise ValueError(
                f"SerialGripperConfig: gripper_v_max {self.gripper_v_max} out of range (0, 350] mm/s."
            )
        if not 0.0 < self.gripper_f_max <= 60.0:
            raise ValueError(
                f"SerialGripperConfig: gripper_f_max {self.gripper_f_max} out of range (0, 60] N."
            )
