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

"""Pure-serial Xense gripper driver for bi_flexiv_rizon4_rt.

Uses XenseSerialGripper (from the XGripper submodule) directly over a
USB-serial port.  No ezros / xensesdk stack required.
"""

import time
from threading import Thread, Lock
from glob import glob

from xensegripper import XenseSerialGripper, read_board_sn

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.bi_flexiv_rizon4_rt.config_serial_gripper import SerialGripperConfig
from lerobot.utils.robot_utils import get_logger

# Serialize port scans so parallel gripper connect() calls don't
# interfere with each other's serial read_board_sn() queries.
_scan_lock = Lock()


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
    with _scan_lock:
        candidates = sorted(glob("/dev/ttyUSB*") + glob("/dev/ttyACM*"))
        for port in candidates:
            try:
                found = read_board_sn(port, baudrate=baudrate, device_id=device_id)
                if found and found.strip() == sn.strip():
                    return port
            except Exception:
                pass
    raise RuntimeError(
        f"SerialGripper: could not find a port with SN={sn!r}. "
        f"Scanned: {candidates}"
    )


class SerialGripper:
    """Wrapper around XenseSerialGripper for use inside BiFlexivRizon4RT.

    Normalized position convention:
        0.0  →  fully closed (SDK position = gripper_min_pos, e.g.  0 mm)
        1.0  →  fully open   (SDK position = gripper_max_pos, e.g. 85 mm)

    Note: XenseSerialGripper uses 0 mm = fully closed, 85 mm = fully open internally.
    Normalized maps directly: normalized * span + min_pos → SDK position.

    Example::

        cfg = SerialGripperConfig(port="/dev/ttyUSB0")
        g = SerialGripper(cfg)
        g.connect()
        g.set_gripper_position(0.5)   # half-closed
        print(g.get_gripper_position())
        g.disconnect()
    """

    config_class = SerialGripperConfig

    def __init__(self, config: SerialGripperConfig):
        self._config = config
        self._gripper_min_pos = config.gripper_min_pos
        self._gripper_max_pos = config.gripper_max_pos
        self._gripper_v_max = config.gripper_v_max
        self._gripper_f_max = config.gripper_f_max
        self._init_open = config.init_open

        # Resolved at connect() time (SN lookup or explicit port)
        self._port: str = config.port

        label = config.sn or config.port.split("/")[-1]
        self._logger = get_logger(f"SerialGripper-{label}")
        self._is_connected: bool = False
        self._gripper: XenseSerialGripper | None = None

        # Cached position updated by background poller — avoids blocking get_observation()
        self._cached_position: float = 0.0
        self._poll_thread: Thread | None = None
        self._poll_running: bool = False

    # ── Connection lifecycle ───────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the serial port and start the background receive thread."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # Resolve port from SN if not explicitly set
        if self._config.sn:
            self._logger.info(f"Scanning serial ports for gripper SN={self._config.sn!r}...")
            self._port = find_port_by_sn(
                self._config.sn,
                baudrate=self._config.baudrate,
                device_id=self._config.device_id,
            )
            self._logger.info(f"Found SN={self._config.sn!r} on {self._port}.")

        self._logger.info(
            f"Connecting serial gripper on {self._port} "
            f"(baud={self._config.baudrate}, id={self._config.device_id})..."
        )
        try:
            self._gripper = XenseSerialGripper(
                port=self._port,
                device_id=self._config.device_id,
                baudrate=self._config.baudrate,
                timeout=self._config.serial_timeout,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to open serial gripper on {self._port}: {e}") from e

        self._is_connected = True
        self._logger.info(f"Serial gripper connected on {self._port}.")

        if self._init_open:
            self._logger.info("Initializing gripper to fully open position (non-blocking)...")
            try:
                # Use non-blocking set_position: MCU does not respond to status queries
                # while idle, so set_position_sync would spin for ~4s even when already open.
                self._gripper.set_position(
                    self._gripper_max_pos,
                    vmax=self._gripper_v_max,
                    fmax=self._gripper_f_max / 2,
                )
                self._logger.info("Gripper open command sent.")
            except Exception as e:
                self._logger.warn(f"Gripper init-open failed (non-fatal): {e}")

        # Start background position poller so get_gripper_position() never blocks
        self._poll_running = True
        self._poll_thread = Thread(target=self._position_poll_loop, daemon=True)
        self._poll_thread.start()

    def _position_poll_loop(self) -> None:
        """Background thread: continuously refresh _cached_position via serial query."""
        span = self._gripper_max_pos - self._gripper_min_pos
        while self._poll_running:
            gripper = self._gripper
            if gripper is None:
                break
            try:
                status = gripper.get_gripper_status(timeout=0.05)
                if status is not None:
                    raw_pos = float(status.get("position", 0.0))
                    raw_pos = max(self._gripper_min_pos, min(raw_pos, self._gripper_max_pos))
                    self._cached_position = (raw_pos - self._gripper_min_pos) / span
            except Exception:
                pass

    def disconnect(self) -> None:
        """Stop the background thread and close the serial port."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._poll_running = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=0.5)
            self._poll_thread = None

        self._logger.info("Disconnecting serial gripper...")
        if self._gripper is not None:
            try:
                self._gripper.release()
            except Exception as e:
                self._logger.debug(f"Error releasing serial gripper: {e}")
            self._gripper = None

        self._is_connected = False
        self._logger.info("Serial gripper disconnected.")

    # ── Position interface ─────────────────────────────────────────────────────

    def get_gripper_position(self) -> float:
        """Return normalized gripper position in [0, 1] from the background cache.

        Returns immediately (non-blocking). The value is refreshed by a background
        poller thread at ~serial-round-trip rate (~10ms), so latency is at most one
        poll cycle behind reality — acceptable for 30 Hz teleoperation.

        Returns:
            0.0 = fully open, 1.0 = fully closed.
        """
        if not self._is_connected:
            return 0.0
        return self._cached_position

    def set_gripper_position(self, normalized_pos: float) -> None:
        """Send a position command to the gripper.

        Args:
            normalized_pos: Target position in [0, 1].
                            0.0 = fully open, 1.0 = fully closed.
        """
        if not self._is_connected or self._gripper is None:
            raise DeviceNotConnectedError("Serial gripper is not connected.")
        if not 0.0 <= normalized_pos <= 1.0:
            raise ValueError(
                f"normalized_pos must be in [0, 1], got {normalized_pos}."
            )
        span = self._gripper_max_pos - self._gripper_min_pos
        target_mm = self._gripper_min_pos + normalized_pos * span
        self._gripper.set_position(
            target_mm,
            vmax=self._gripper_v_max,
            fmax=self._gripper_f_max,
        )

    def set_gripper_position_sync(
        self,
        normalized_pos: float,
        timeout: float = 10.0,
        vmax: float | None = None,
        fmax: float | None = None,
    ) -> None:
        """Send a position command and block until the gripper reaches the target.

        Args:
            normalized_pos: Target position in [0, 1] (0.0 = open, 1.0 = closed).
            timeout:        Maximum wait time in seconds (default: 10.0).
            vmax:           Override velocity limit mm/s; uses config default if None.
            fmax:           Override force limit N; uses config default if None.
        """
        if not self._is_connected or self._gripper is None:
            raise DeviceNotConnectedError("Serial gripper is not connected.")
        if not 0.0 <= normalized_pos <= 1.0:
            raise ValueError(f"normalized_pos must be in [0, 1], got {normalized_pos}.")
        span = self._gripper_max_pos - self._gripper_min_pos
        target_mm = self._gripper_min_pos + normalized_pos * span
        self._gripper.set_position_sync(
            target_mm,
            vmax=vmax if vmax is not None else self._gripper_v_max,
            fmax=fmax if fmax is not None else self._gripper_f_max,
            timeout=timeout,
        )
