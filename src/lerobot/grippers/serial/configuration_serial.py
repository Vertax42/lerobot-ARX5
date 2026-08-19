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

"""Configuration for the pure-serial Xense gripper.

Importing this module needs no SDK at all — that is what lets an arm config parse
on a bare host. The driver it configures needs ``xgripper`` (but not the ezros /
xensesdk stack); see ``serial_gripper``.
"""

from dataclasses import dataclass
from typing import ClassVar

from ..configs import GripperConfig


@GripperConfig.register_subclass("serial")
@dataclass(kw_only=True)
class SerialGripperConfig(GripperConfig):
    """Configuration for the serial-port Xense gripper (XenseSerialGripper).

    This gripper communicates directly over a USB-serial port and does not
    require the ezros / xensesdk stack.

    Identification (provide one; resolved at connect() in this priority):
        port:            Explicit serial port path (e.g. ``"/dev/ttyUSB0"``).
                         Highest priority — bypasses all scanning.
        sn:              Board serial number (e.g. ``"000001"``).  When set,
                         the driver scans available serial ports at connect()
                         time and picks the one whose READ_BOARD_SN response
                         matches.
        side:            ``"left"`` / ``"right"`` — auto-discover the port by
                         board-SN parity (odd SN → left, even SN → right; see
                         ``serial_discovery.find_port_by_side``). Lets a
                         correctly-numbered gripper self-sort without pinning an
                         exact ``sn``. Used only when ``port`` and ``sn`` are unset.

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
    port: str = ""  # explicit port path — highest priority
    sn: str | None = None  # board SN — scanned/matched when port unset
    side: str | None = None  # "left"/"right" — parity auto-discover when port+sn unset

    # ── Serial connection ──────────────────────────────────────────────────────
    # baudrate and device_id are fixed in practice, hence protocol_fixed_fields:
    #   - 115200 is the only rate the firmware speaks, and the protocol has no
    #     command to change it (see Command in xgripper.xense_gripper) — only
    #     a firmware reflash would.
    #   - device_id is the RS-485 address byte in each packet. The protocol carries
    #     it because a bus can hold several devices; our wiring gives every gripper
    #     its own USB serial port, so it is always the default 1. There is no
    #     command to change it either.
    baudrate: int = 115200
    device_id: int = 1
    serial_timeout: float = 1.0

    protocol_fixed_fields: ClassVar[frozenset[str]] = frozenset({"baudrate", "device_id"})

    # On, like the TacCap backend. A serial gripper is the same shape of device: one
    # USB hub carrying the gripper board, its wrist camera and its two tactile
    # sensors, with the board SN's parity naming the side (odd → left). So the
    # cameras follow the gripper and there is nothing a recipe can pin that
    # discovery cannot work out — pinning SNs by hand was the older, worse way.
    auto_discover_cameras: bool = True

    # ── Mechanical limits ──────────────────────────────────────────────────────
    gripper_min_pos: float = 0.0  # mm — fully closed
    gripper_max_pos: float = 85.0  # mm — fully open

    # ── Motion parameters ──────────────────────────────────────────────────────
    gripper_v_max: float = 80.0  # mm/s  (range: 0–350)
    gripper_f_max: float = 27.0  # N     (range: 0–60)

    # ── Initialization ─────────────────────────────────────────────────────────
    init_open: bool = True

    def __post_init__(self):
        # NOTE: "at least one of port / sn / side" is deliberately NOT checked here.
        # A bimanual recipe writes ONE shared gripper block and the arm clones it
        # per side, so the block as written legitimately has no identity yet.
        # SerialGripper.connect() raises if it still cannot resolve a port.
        if self.side is not None and self.side not in ("left", "right"):
            raise ValueError(f"SerialGripperConfig: side must be 'left' or 'right', got {self.side!r}.")
        if not self.baudrate > 0:
            raise ValueError(f"SerialGripperConfig: baudrate must be positive, got {self.baudrate}.")
        if not 0.0 <= self.gripper_min_pos < self.gripper_max_pos:
            raise ValueError(
                f"SerialGripperConfig: gripper_min_pos ({self.gripper_min_pos}) must be "
                f"< gripper_max_pos ({self.gripper_max_pos})."
            )
        if not 0.0 < self.gripper_v_max <= 350.0:
            raise ValueError(f"SerialGripperConfig: gripper_v_max {self.gripper_v_max} out of range (0, 350] mm/s.")
        if not 0.0 < self.gripper_f_max <= 60.0:
            raise ValueError(f"SerialGripperConfig: gripper_f_max {self.gripper_f_max} out of range (0, 60] N.")
