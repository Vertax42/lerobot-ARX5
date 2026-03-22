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

from xensegripper import XenseSerialGripper

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.bi_flexiv_rizon4_rt.config_serial_gripper import SerialGripperConfig
from lerobot.utils.robot_utils import get_logger


class SerialGripper:
    """Wrapper around XenseSerialGripper for use inside BiFlexivRizon4RT.

    Normalized position convention:
        0.0  →  fully open   (SDK position = gripper_max_pos, e.g. 85 mm)
        1.0  →  fully closed (SDK position = gripper_min_pos, e.g.  0 mm)

    Note: XenseSerialGripper uses 0 = closed, 85 = open internally.
    This wrapper inverts the mapping so normalized 0.0 always means open.

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

        self._logger = get_logger(f"SerialGripper-{config.port.split('/')[-1]}")
        self._is_connected: bool = False
        self._gripper: XenseSerialGripper | None = None

    # ── Connection lifecycle ───────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the serial port and start the background receive thread."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        self._logger.info(
            f"Connecting serial gripper on {self._config.port} "
            f"(baud={self._config.baudrate}, id={self._config.device_id})..."
        )
        try:
            self._gripper = XenseSerialGripper(
                port=self._config.port,
                device_id=self._config.device_id,
                baudrate=self._config.baudrate,
                timeout=self._config.serial_timeout,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to open serial gripper on {self._config.port}: {e}") from e

        self._is_connected = True
        self._logger.info(f"Serial gripper connected on {self._config.port}.")

        if self._init_open:
            self._logger.info("Initializing gripper to fully open position...")
            try:
                self._gripper.set_position_sync(
                    position=self._gripper_max_pos,
                    vmax=self._gripper_v_max / 2,
                    fmax=self._gripper_f_max / 2,
                    timeout=10.0,
                )
                self._logger.info("Gripper initialized to open position.")
            except Exception as e:
                self._logger.warn(f"Gripper init-open failed (non-fatal): {e}")

    def disconnect(self) -> None:
        """Stop the background thread and close the serial port."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

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
        """Return normalized gripper position in [0, 1].

        Returns:
            0.0 = fully open, 1.0 = fully closed.
            Returns 0.0 if not connected or status unavailable.
        """
        if not self._is_connected or self._gripper is None:
            return 0.0
        try:
            status = self._gripper.get_gripper_status(timeout=0.1)
            if status is None:
                return 0.0
            raw_pos = float(status.get("position", 0.0))
            raw_pos = max(self._gripper_min_pos, min(raw_pos, self._gripper_max_pos))
            span = self._gripper_max_pos - self._gripper_min_pos
            # SDK convention: position=85 means open, position=0 means closed.
            # Normalized: 0.0 = open, 1.0 = closed → invert.
            return 1.0 - (raw_pos - self._gripper_min_pos) / span
        except Exception:
            return 0.0

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
        # SDK convention: position=85 opens, position=0 closes → invert normalized mapping.
        target_mm = self._gripper_max_pos - normalized_pos * span
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
        target_mm = self._gripper_max_pos - normalized_pos * span
        self._gripper.set_position_sync(
            target_mm,
            vmax=vmax if vmax is not None else self._gripper_v_max,
            fmax=fmax if fmax is not None else self._gripper_f_max,
            timeout=timeout,
        )
