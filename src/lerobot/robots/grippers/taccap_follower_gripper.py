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

"""TacCap follower (actuated) gripper driver — arm-agnostic, shared across robots.

Wraps ``xense.taccap.FollowerGripper``. The follower drives an FDCAN motor via
the MIT force-position (impedance) primitive. A background ``ControlLoop`` (a C++
thread) resubmits the latest normalized target at a fixed rate and keeps a
thread-safe observation fresh, so ``get_gripper_position`` / ``set_gripper_position``
are non-blocking and match the duck-typed gripper contract used by the arm robots
(same as ``SerialGripper``):

    connect() / disconnect()
    get_gripper_position() -> float in [0, 1]   (0 = closed, 1 = open)
    set_gripper_position(float in [0, 1])
    initialize_gripper_position(float in [0, 1])

Left/right units are told apart by the firmware-burned serial number via the
SDK's side-aware discovery (``find_left`` / ``find_right``).

The ``xense.taccap`` SDK is imported lazily (guarded) so importing the grippers
package does not hard-fail on hosts where the native extension is not built —
the error is raised with rebuild guidance at ``connect()`` time instead.
"""

import time

from lerobot.robots.grippers.config_taccap_follower_gripper import TaccapFollowerGripperConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import get_logger

try:
    import xense.taccap as taccap  # type: ignore

    _TACCAP_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - depends on native build being present
    taccap = None  # type: ignore
    _TACCAP_IMPORT_ERROR = e

# GripperConfig.flags bit 0 = calibrated (see SDK gripper_control_test.py).
_CALIBRATED_FLAG = 0x0001

# initialize_gripper_position() reach-wait defaults.
_INIT_POSITION_TOLERANCE = 0.03
_INIT_TIMEOUT_S = 3.0


class TaccapFollowerGripper:
    """Wrapper around ``xense.taccap.FollowerGripper`` for arm robots.

    Normalized position convention (matches ``SerialGripper``):
        0.0  →  fully closed
        1.0  →  fully open

    Example::

        cfg = TaccapFollowerGripperConfig(side="left")
        g = TaccapFollowerGripper(cfg)
        g.connect()
        g.set_gripper_position(0.5)
        print(g.get_gripper_position())
        g.disconnect()
    """

    config_class = TaccapFollowerGripperConfig

    def __init__(self, config: TaccapFollowerGripperConfig):
        self._config = config
        self._side = config.side
        self._mcu_device = config.mcu_device
        self._kp = config.kp
        self._kd = config.kd
        self._control_hz = config.control_hz
        self._init_open = config.init_open
        self._require_calibrated = config.require_calibrated

        self._logger = get_logger(f"TaccapFollowerGripper-{config.side}")
        self._is_connected: bool = False
        self._gripper = None  # xense.taccap.FollowerGripper
        self._loop = None     # xense.taccap.ControlLoop

        # Seed so get_gripper_position() returns something sane before the loop
        # produces its first observation.
        self._cached_position: float = 1.0 if config.init_open else 0.0

    def __str__(self) -> str:
        return f"TaccapFollowerGripper({self._side})"

    # ── Connection lifecycle ───────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        """Return the MCU device path for this side (explicit override or by SN)."""
        if self._mcu_device:
            return self._mcu_device
        finder = taccap.find_left if self._side == "left" else taccap.find_right
        eps = finder()  # throws if the requested side isn't visible on the bus
        return eps.mcu_device

    def connect(self) -> None:
        """Discover the follower, verify calibration, enable the motor, and start
        the background control loop."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        if taccap is None:
            raise RuntimeError(
                "xense.taccap is not importable — the TacCap SDK native extension is "
                "missing or stale. Rebuild it with `bash setup_env.sh --install` (or "
                "`pip install third_party/taccap-gripper`). "
                f"Original import error: {_TACCAP_IMPORT_ERROR!r}"
            )

        device = self._resolve_device()
        self._logger.info(f"Opening follower gripper (side={self._side}) on {device}...")
        try:
            self._gripper = taccap.FollowerGripper(device)
        except Exception as e:
            self._gripper = None
            raise RuntimeError(
                f"Failed to open TacCap follower (side={self._side}) on {device}: {e}"
            ) from e

        # Calibration is required for normalized [0, 1] control.
        try:
            cfg = self._gripper.get_gripper_config()
            calibrated = bool(cfg.flags & _CALIBRATED_FLAG)
        except Exception as e:
            calibrated = False
            self._logger.warn(f"Could not read gripper config (assuming uncalibrated): {e}")
        if not calibrated and self._require_calibrated:
            self._gripper = None
            raise RuntimeError(
                f"TacCap follower (side={self._side}) is not calibrated; normalized [0, 1] "
                "control requires calibration. Calibrate first (zero at full close, then "
                "write max_open), or enable firmware power-on auto-cal. See "
                "third_party/taccap-gripper/python/examples/calibrate.py."
            )

        # Enable the motor before any motion.
        self._gripper.motor.clear_fault()
        self._gripper.motor.enable()

        # Background control loop: resubmits latest target at control_hz, keeps a
        # fresh thread-safe observation. start() seeds target = current pos (no jump).
        self._loop = taccap.ControlLoop(self._gripper, hz=self._control_hz, kp=self._kp, kd=self._kd)
        self._loop.start()

        self._is_connected = True
        try:
            self._cached_position = float(self._loop.observation().position)
        except Exception:
            self._cached_position = 1.0 if self._init_open else 0.0
        self._logger.info(f"TacCap follower connected (side={self._side}) on {device}.")

        if self._init_open:
            try:
                self.initialize_gripper_position(1.0)
            except Exception as e:
                self._logger.warn(f"Gripper init-open failed (non-fatal): {e}")

    def disconnect(self) -> None:
        """Stop the control loop, disable the motor, and release the device."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._loop is not None:
            try:
                self._loop.stop()
            except Exception as e:
                self._logger.debug(f"Error stopping control loop: {e}")
            self._loop = None

        if self._gripper is not None:
            try:
                self._gripper.motor.disable()
            except Exception as e:
                self._logger.debug(f"Error disabling follower motor: {e}")
            self._gripper = None

        self._is_connected = False
        self._logger.info(f"TacCap follower disconnected (side={self._side}).")

    # ── Position interface ─────────────────────────────────────────────────────

    def get_gripper_position(self) -> float:
        """Return normalized position in [0, 1] from the control loop's latest
        observation (non-blocking). Returns 0.0 if not connected."""
        if not self._is_connected or self._loop is None:
            return 0.0
        try:
            self._cached_position = _clamp01(float(self._loop.observation().position))
        except Exception as e:
            self._logger.debug(f"observation() read failed, returning cached: {e}")
        return self._cached_position

    def set_gripper_position(self, normalized_pos: float) -> None:
        """Command a normalized target in [0, 1] (0 = closed, 1 = open)."""
        if not self._is_connected or self._loop is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not 0.0 <= normalized_pos <= 1.0:
            raise ValueError(f"normalized_pos must be in [0, 1], got {normalized_pos}.")
        self._loop.set_target(normalized_pos)

    def initialize_gripper_position(
        self,
        normalized_pos: float,
        tolerance: float = _INIT_POSITION_TOLERANCE,
        timeout: float = _INIT_TIMEOUT_S,
    ) -> None:
        """Command a target and block until the gripper reaches it (or timeout).

        Used by the arm during homing; kept name-compatible with SerialGripper so
        the arm's gripper wiring works unchanged.
        """
        self.set_gripper_position(normalized_pos)
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if abs(self.get_gripper_position() - normalized_pos) <= tolerance:
                return
            time.sleep(0.02)
        self._logger.warn(
            f"Gripper did not reach init target {normalized_pos:.3f} within {timeout:.1f}s "
            f"(current={self.get_gripper_position():.3f})."
        )


def _clamp01(x: float) -> float:
    return max(0.0, min(x, 1.0))
