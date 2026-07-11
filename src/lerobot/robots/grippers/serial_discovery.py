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

"""Auto-discovery of the left / right side for pure-serial Xense grippers.

The serial gripper (``XenseSerialGripper``) has no firmware-burned "side" the way
the TacCap follower does (cf. ``taccap_discovery.py`` + the SDK's
``find_left`` / ``find_right``). Instead the fleet follows a numbering
convention: **odd board SN → left, even board SN → right** (单数=left, 双数=right).

This module scans every ``/dev/ttyUSB*`` / ``/dev/ttyACM*`` port, reads each
board SN via ``read_board_sn`` (the same query ``find_port_by_sn`` already uses),
and maps ports to sides by that parity — so any correctly-numbered gripper
self-sorts regardless of which USB port it lands on, with no per-station SN
pinned in the ``bi_mount_type`` preset.

It also owns the shared serial-scan lock and the low-level port-scan primitive so
that side discovery and exact-SN lookup (``find_port_by_sn``) serialize against
each other on the RS-485 bus (parallel per-side ``connect()`` calls must not
interleave their status queries).
"""

import re
from dataclasses import dataclass
from glob import glob
from threading import Lock

from xensegripper import read_board_sn

from lerobot.utils.robot_utils import get_logger

logger = get_logger("SerialGripperDiscovery")

# Serialize port scans so parallel gripper connect() calls (and find_port_by_sn /
# find_port_by_side) don't interfere with each other's serial read_board_sn() queries.
_scan_lock = Lock()

_SIDES = ("left", "right")

# Trailing run of digits in a board SN, e.g. "000031" -> "000031", "XG0042" -> "0042".
_TRAILING_DIGITS_RE = re.compile(r"(\d+)\s*$")


@dataclass
class SerialGripperSideDevice:
    side: str   # "left" | "right"
    sn: str     # board serial number, e.g. "000031"
    port: str   # e.g. "/dev/ttyUSB0"


def sn_side(sn: str) -> str | None:
    """Classify a board SN to a side by parity: odd → left, even → right.

    The parity is taken from the trailing run of digits in the SN, so both
    pure-numeric SNs (``"000031"``) and prefixed SNs (``"XG0042"``) work. Returns
    ``None`` when the SN has no digits to classify.
    """
    if not sn:
        return None
    match = _TRAILING_DIGITS_RE.search(sn.strip())
    if match is None:
        return None
    return "left" if int(match.group(1)) % 2 == 1 else "right"


def _scan_port_sns(baudrate: int = 115200, device_id: int = 1) -> dict[str, str]:
    """Read the board SN from every candidate serial port (serialized).

    Ports that don't answer the READ_BOARD_SN query (non-gripper devices, or a
    gripper that's busy/unplugged) are simply omitted.

    Returns:
        {port: sn} for every port that returned a non-empty board SN.
    """
    found: dict[str, str] = {}
    with _scan_lock:
        candidates = sorted(glob("/dev/ttyUSB*") + glob("/dev/ttyACM*"))
        for port in candidates:
            try:
                sn = read_board_sn(port, baudrate=baudrate, device_id=device_id)
            except Exception:
                sn = None
            if sn and sn.strip():
                found[port] = sn.strip()
    return found


def find_port_by_sn(sn: str, baudrate: int = 115200, device_id: int = 1) -> str:
    """Scan all ttyUSB/ttyACM ports and return the one whose board SN matches.

    Args:
        sn:        Target board serial number string (e.g. ``"000001"``).
        baudrate:  Baud rate to use when querying each port.
        device_id: Device ID to use when querying each port.

    Returns:
        Matched port path (e.g. ``"/dev/ttyUSB3"``).

    Raises:
        RuntimeError: If no port with the given SN is found.
    """
    ports = _scan_port_sns(baudrate=baudrate, device_id=device_id)
    for port, found in ports.items():
        if found == sn.strip():
            return port
    raise RuntimeError(
        f"SerialGripper: could not find a port with SN={sn!r}. "
        f"Scanned SNs: {ports or '(none responded)'}"
    )


def find_port_by_side(side: str, baudrate: int = 115200, device_id: int = 1) -> str:
    """Return the serial port of the gripper whose SN parity matches ``side``.

    Odd SN → left, even SN → right (see :func:`sn_side`). Only the requested side
    is resolved, so a per-side ``connect()`` never fails because of the *other*
    side's wiring.

    Raises:
        ValueError:   If ``side`` is not ``"left"`` / ``"right"``.
        RuntimeError: If no gripper (or more than one) matches ``side``'s parity.
    """
    if side not in _SIDES:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}.")
    ports = _scan_port_sns(baudrate=baudrate, device_id=device_id)
    matches = {port: sn for port, sn in ports.items() if sn_side(sn) == side}
    if not matches:
        raise RuntimeError(
            f"SerialGripper: no {side} gripper found (odd SN → left, even SN → right). "
            f"Scanned SNs: {ports or '(none responded)'}. Check USB connection, "
            "dialout permissions, and ModemManager not grabbing the serial ports."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"SerialGripper: ambiguous {side} side — multiple {side}-parity grippers "
            f"on the bus: {matches}. Only one gripper per side is supported by parity "
            "discovery; pin an exact port/sn on SerialGripperConfig instead."
        )
    port, sn = next(iter(matches.items()))
    logger.info(f"[{side}] gripper SN={sn} on {port} (parity discovery).")
    return port


def discover_serial_gripper_sides(
    baudrate: int = 115200, device_id: int = 1
) -> dict[str, SerialGripperSideDevice]:
    """Discover both serial-gripper sides at once (diagnostics / find-port helper).

    Returns:
        ``{"left": SerialGripperSideDevice, "right": SerialGripperSideDevice}`` —
        only the sides that resolve to exactly one gripper are included.
    """
    ports = _scan_port_sns(baudrate=baudrate, device_id=device_id)
    by_side: dict[str, list[tuple[str, str]]] = {s: [] for s in _SIDES}
    for port, sn in ports.items():
        side = sn_side(sn)
        if side is None:
            logger.warn(f"Board SN {sn!r} on {port} has no digits to classify; skipping.")
            continue
        by_side[side].append((port, sn))

    sides: dict[str, SerialGripperSideDevice] = {}
    for side in _SIDES:
        entries = by_side[side]
        if not entries:
            logger.info(f"No {side} serial gripper found.")
            continue
        if len(entries) > 1:
            logger.warn(
                f"Multiple {side}-parity serial grippers found ({entries}); "
                "side is ambiguous, omitting."
            )
            continue
        port, sn = entries[0]
        sides[side] = SerialGripperSideDevice(side=side, sn=sn, port=port)
        logger.info(f"[{side}] gripper SN={sn} on {port}.")
    return sides
