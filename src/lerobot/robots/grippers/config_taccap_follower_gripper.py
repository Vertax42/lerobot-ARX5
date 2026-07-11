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

"""Configuration for the TacCap follower (actuated) gripper.

Wraps ``xense.taccap.FollowerGripper``. The follower drives an FDCAN motor via
the MIT force-position (impedance) primitive; the driver runs a background
``ControlLoop`` so reads/writes are non-blocking. Left/right units are told
apart automatically by the firmware-burned serial number (``side``), so no
per-unit SN/port needs configuring in the common case.
"""

from dataclasses import dataclass

# Hard bound on the constant feed-forward torque, to catch sign/scale typos before
# they reach the motor. The MIT impedance path applies feed-forward with NO firmware
# max_torque clamp (only position/velocity modes clamp), and ~3.5 Nm is the top of the
# motor's usable envelope (cf. the max_torque values in the codec tests). This is a
# safety rail, not a recommendation — the gentle-grasp example aborts at 0.30 Nm.
MAX_FEEDFORWARD_TORQUE_NM = 3.5


@dataclass
class TaccapFollowerGripperConfig:
    """Configuration for a single TacCap follower gripper.

    Identification:
        side:        Which physical unit to drive, ``"left"`` or ``"right"``.
                     Resolved at connect() time via the SDK's side-aware device
                     discovery (``find_left()`` / ``find_right()``), which reads
                     the firmware-burned serial number. Ignored when
                     ``mcu_device`` is set.
        mcu_device:  Optional explicit MCU device path (e.g. ``"/dev/ttyACM0"``)
                     to bypass side-based discovery. Use only when auto-discovery
                     is not viable.

    Control (MIT impedance):
        kp:          Position-tracking stiffness gain (Nm/rad).
        kd:          Velocity damping gain (Nm·s/rad).
        feedforward_torque: Constant torque bias (Nm) added to every MIT frame,
                     on top of the kp/kd position term. SIGN: negative = closing
                     (clamps harder), positive = opening. Default 0.0. This is a
                     *constant* bias — it acts even with an empty jaw (holds the
                     mechanical stop and biases the open pose), unlike kp which
                     only produces force when there is a position error. Bounded
                     to |ff| <= MAX_FEEDFORWARD_TORQUE_NM to catch sign/scale
                     typos; the SDK's own gentle-grasp example treats 0.30 Nm as
                     an abort threshold, so values past ~1 Nm are a hard crush.
        control_hz:  Rate of the background ControlLoop that resubmits the latest
                     normalized target to the firmware.

    Behavior:
        init_open:       If True, drive fully open on ``connect()``.
        require_calibrated: If True, refuse to connect when the gripper reports
                     an uncalibrated ``GripperConfig`` (normalized [0, 1] control
                     needs calibration). Set False only for bring-up/debug.
    """

    # ── Identification ─────────────────────────────────────────────────────────
    side: str = "left"                 # "left" | "right" (firmware-SN auto side)
    mcu_device: str | None = None      # optional explicit device path override

    # ── Control (MIT impedance) ────────────────────────────────────────────────
    kp: float = 8.0                    # Nm/rad
    kd: float = 1.0                    # Nm·s/rad
    feedforward_torque: float = 0.0    # Nm; NEGATIVE = closing/clamp, POSITIVE = opening
    control_hz: int = 200              # ControlLoop resubmit rate

    # ── Behavior ───────────────────────────────────────────────────────────────
    init_open: bool = True
    require_calibrated: bool = True

    def __post_init__(self):
        if self.side not in ("left", "right"):
            raise ValueError(
                f"TaccapFollowerGripperConfig: side must be 'left' or 'right', got {self.side!r}."
            )
        if not self.kp > 0.0:
            raise ValueError(f"TaccapFollowerGripperConfig: kp must be positive, got {self.kp}.")
        if not self.kd >= 0.0:
            raise ValueError(f"TaccapFollowerGripperConfig: kd must be non-negative, got {self.kd}.")
        if abs(self.feedforward_torque) > MAX_FEEDFORWARD_TORQUE_NM:
            raise ValueError(
                f"TaccapFollowerGripperConfig: |feedforward_torque| must be <= "
                f"{MAX_FEEDFORWARD_TORQUE_NM} Nm, got {self.feedforward_torque}. "
                "Sign: negative = closing/clamp, positive = opening. Values past "
                "~1 Nm are a hard crush (the SDK's gentle-grasp example aborts at 0.30 Nm)."
            )
        if not 0 < self.control_hz <= 500:
            raise ValueError(
                f"TaccapFollowerGripperConfig: control_hz must be in (0, 500], got {self.control_hz}."
            )
