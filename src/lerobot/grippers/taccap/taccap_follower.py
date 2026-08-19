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
are non-blocking.

Left/right units are told apart by the firmware-burned serial number via the
SDK's side-aware discovery (``find_left`` / ``find_right``). When the arm has
already run ``discover_taccap_sides()`` — it does, to wire the wrist + tactile
cameras — it passes the resolved device path through ``config.mcu_device`` so
this driver does not re-enumerate the bus a second time per side.

The ``xense.taccap`` SDK is imported lazily (guarded) so importing the grippers
package does not hard-fail on hosts where the native extension is not built —
the error is raised with rebuild guidance at ``connect()`` time instead.
"""

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import get_logger

from ..gripper import Gripper
from .configuration_taccap import TaccapFollowerConfig

try:
    import xense.taccap as taccap  # type: ignore

    _TACCAP_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - depends on native build being present
    taccap = None  # type: ignore
    _TACCAP_IMPORT_ERROR = e

# GripperConfig.flags bit 0 = calibrated (see SDK gripper_control_test.py).
_CALIBRATED_FLAG = 0x0001


class TaccapFollower(Gripper):
    """Wrapper around ``xense.taccap.FollowerGripper`` for arm robots.

    Normalized position convention (matches ``SerialGripper``):
        0.0  →  fully closed
        1.0  →  fully open

    Example::

        cfg = TaccapFollowerConfig(side="left")
        g = TaccapFollower(cfg)
        g.connect()
        g.set_gripper_position(0.5)
        print(g.get_gripper_position())
        g.disconnect()
    """

    config_class = TaccapFollowerConfig

    def __init__(self, config: TaccapFollowerConfig):
        super().__init__(config)
        self._config = config
        self._side = config.side
        self._mcu_device = config.mcu_device
        self._kp = config.kp
        self._kd = config.kd
        self._feedforward_torque = config.feedforward_torque
        self._control_hz = config.control_hz
        self._init_open = config.init_open
        self._require_calibrated = config.require_calibrated

        self.logger = get_logger(f"TaccapFollower-{config.side}")
        self._is_connected: bool = False
        self._gripper = None  # xense.taccap.FollowerGripper
        self._loop = None     # xense.taccap.ControlLoop

        # Seed so get_gripper_position() returns something sane before the loop
        # produces its first observation.
        self._cached_position: float = 1.0 if config.init_open else 0.0

    def __str__(self) -> str:
        return f"TaccapFollower({self._side})"

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # ── Connection lifecycle ───────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        """Return the MCU device path for this side.

        Prefers ``config.mcu_device`` — the arm fills it in from the
        ``discover_taccap_sides()`` sweep it already ran for the cameras, so the
        common bimanual path enumerates the bus once instead of once per side
        again here. Falls back to a side-aware scan when nothing was injected.
        """
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
        self.logger.info(f"Opening follower gripper (side={self._side}) on {device}...")
        try:
            self._gripper = taccap.FollowerGripper(device)
        except Exception as e:
            self._gripper = None
            raise RuntimeError(
                f"Failed to open TacCap follower (side={self._side}) on {device}: {e}"
            ) from e

        # From here the device handle is open. Anything that throws must release it
        # before propagating: _is_connected is still False, so disconnect() would
        # refuse to run and the handle would be stranded until GC.
        try:
            # Calibration is required for normalized [0, 1] control.
            try:
                cfg = self._gripper.get_gripper_config()
                calibrated = bool(cfg.flags & _CALIBRATED_FLAG)
            except Exception as e:
                calibrated = False
                self.logger.warn(f"Could not read gripper config (assuming uncalibrated): {e}")
            if not calibrated and self._require_calibrated:
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
            # feedforward_torque is a CONSTANT bias added to every MIT frame (negative =
            # closing/clamp); with a non-zero clamp bias the jaw settles ~ff/kp rad short
            # of any commanded open, and init_open below will warn on the timeout — expected.
            self._loop = taccap.ControlLoop(
                self._gripper,
                hz=self._control_hz,
                kp=self._kp,
                kd=self._kd,
                feedforward_torque=self._feedforward_torque,
            )
            if self._feedforward_torque != 0.0:
                self.logger.info(
                    f"MIT feed-forward torque bias = {self._feedforward_torque:+.2f} Nm "
                    f"({'closing/clamp' if self._feedforward_torque < 0 else 'opening'})."
                )
            self._loop.start()
        except Exception:
            self._release_after_failed_connect()
            raise

        self._is_connected = True
        try:
            self._cached_position = float(self._loop.observation().position)
        except Exception:
            self._cached_position = 1.0 if self._init_open else 0.0
        self.logger.info(f"TacCap follower connected (side={self._side}) on {device}.")

        if self._init_open:
            try:
                self.initialize_gripper_position(1.0)
            except Exception as e:
                self.logger.warn(f"Gripper init-open failed (non-fatal): {e}")

    def _release_after_failed_connect(self) -> None:
        """Best-effort teardown of a partially-opened device.

        Mirrors disconnect(), but tolerates every step being absent — we may be
        anywhere between "handle open" and "loop running" when this fires.
        """
        if self._loop is not None:
            try:
                self._loop.stop()
            except Exception as e:  # pragma: no cover — best-effort
                self.logger.debug(f"Error stopping control loop during rollback: {e}")
            self._loop = None
        if self._gripper is not None:
            try:
                self._gripper.motor.disable()
            except Exception as e:  # pragma: no cover — best-effort
                self.logger.debug(f"Error disabling motor during rollback: {e}")
            self._gripper = None

    #: Follower firmware that first answered Cmd::CameraFisheyeCal (0x2B).
    #:
    #: The command belongs to command set V2.0, but command-set numbers and
    #: firmware numbers are different sequences — follower 1.1.0 already carries
    #: V2.1. Comparing a firmware version against "2.0.0" would call every
    #: shipping unit too old, and point the reader at an upgrade that changes
    #: nothing here. Verified on the bench: 1.1.1 answers the read.
    FISHEYE_MIN_FIRMWARE = (1, 1, 0)

    def read_wrist_fisheye_calibration(self):
        """The wrist lens' fisheye intrinsics for this gripper.

        Prefers what this unit's firmware holds. Falls back to the SDK's
        reference calibration — with a warning — when the firmware is too old to
        carry one, has never been calibrated, or answers with an empty record.
        Every TC-GU-01 shares the lens and the 640x480 sensor, so the reference
        numbers are much closer to correct than raw fisheye; they are not a
        substitute for calibrating a unit, since lens placement varies per
        assembly and the principal point drifts with it.

        Returns:
            ``(calibration, is_reference)`` — the second value is True when the
            fallback was used, so callers can label or refuse it themselves.
        """
        if self._gripper is None:
            raise DeviceNotConnectedError(
                f"{self} is not connected; cannot read the wrist fisheye calibration."
            )
        from xense.taccap import FISHEYE_FALLBACK_CAL, is_usable_fisheye_cal

        reason = self._fisheye_firmware_shortfall()
        if reason is None:
            try:
                cal = self._gripper.calibration.read_fisheye()
            except Exception as e:
                reason = f"the firmware would not answer the fisheye read ({e!r})"
            else:
                if cal is None:
                    reason = "the firmware has never been calibrated"
                elif not is_usable_fisheye_cal(cal):
                    # An uncalibrated unit answers with an all-zero record rather
                    # than a NACK; remapping with fx = fy = 0 yields a black frame.
                    reason = (
                        "the wrist lens has never been calibrated — the firmware "
                        "answered the read with an empty record"
                    )
                else:
                    return cal, False

        self.logger.warn(
            f"{self}: using the SDK's REFERENCE wrist fisheye intrinsics because "
            f"{reason}. Rectification will be approximate — lens placement varies "
            f"per assembly, so the principal point drifts. Calibrate this unit's "
            f"wrist lens with the PC tool for measurements taken off these frames."
        )
        return FISHEYE_FALLBACK_CAL, True

    def _fisheye_firmware_shortfall(self) -> str | None:
        """Why this firmware cannot carry a fisheye record, or None if it can."""
        version = getattr(self._gripper, "firmware_version", None)
        if version is None:
            return "the MCU did not report a firmware version"
        needed = ".".join(str(p) for p in self.FISHEYE_MIN_FIRMWARE)
        if tuple(version.tuple) < self.FISHEYE_MIN_FIRMWARE:
            return (
                f"its firmware is {version.major}.{version.minor}.{version.patch}, "
                f"older than the {needed} that first answered the fisheye read"
            )
        return None

    def disconnect(self) -> None:
        """Stop the control loop, disable the motor, and release the device."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._loop is not None:
            try:
                self._loop.stop()
            except Exception as e:
                self.logger.debug(f"Error stopping control loop: {e}")
            self._loop = None

        if self._gripper is not None:
            try:
                self._gripper.motor.disable()
            except Exception as e:
                self.logger.debug(f"Error disabling follower motor: {e}")
            self._gripper = None

        self._is_connected = False
        self.logger.info(f"TacCap follower disconnected (side={self._side}).")

    # ── Position interface ─────────────────────────────────────────────────────

    def get_gripper_position(self) -> float:
        """Return normalized position in [0, 1] from the control loop's latest
        observation (non-blocking).

        Raises:
            DeviceNotConnectedError: If not connected. (This used to return 0.0,
                which is indistinguishable from a genuinely closed jaw — a
                disconnected gripper read as "fully closed" to every caller.)
        """
        if not self._is_connected or self._loop is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        try:
            self._cached_position = _clamp01(float(self._loop.observation().position))
        except Exception as e:
            self.logger.debug(f"observation() read failed, returning cached: {e}")
        return self._cached_position

    def set_gripper_position(self, normalized_pos: float) -> None:
        """Command a normalized target in [0, 1] (0 = closed, 1 = open)."""
        if not self._is_connected or self._loop is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not 0.0 <= normalized_pos <= 1.0:
            raise ValueError(f"normalized_pos must be in [0, 1], got {normalized_pos}.")
        self._loop.set_target(normalized_pos)


def _clamp01(x: float) -> float:
    return max(0.0, min(x, 1.0))
