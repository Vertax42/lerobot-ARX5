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

from ..configs import GripperConfig

# Hard bound on the constant feed-forward torque, to catch sign/scale typos before
# they reach the motor. The MIT impedance path applies feed-forward with NO firmware
# max_torque clamp (only position/velocity modes clamp), and ~3.5 Nm is the top of the
# motor's usable envelope (cf. the max_torque values in the codec tests). This is a
# safety rail, not a recommendation — the gentle-grasp example aborts at 0.30 Nm.
MAX_FEEDFORWARD_TORQUE_NM = 3.5


@GripperConfig.register_subclass("taccap_follower")
@dataclass(kw_only=True)
class TaccapFollowerConfig(GripperConfig):
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
    side: str = "left"  # "left" | "right" (firmware-SN auto side)
    mcu_device: str | None = None  # optional explicit device path override

    # ── Control (MIT impedance) ────────────────────────────────────────────────
    kp: float = 8.0  # Nm/rad
    kd: float = 1.0  # Nm·s/rad
    feedforward_torque: float = 0.0  # Nm; NEGATIVE = closing/clamp, POSITIVE = opening
    control_hz: int = 200  # ControlLoop resubmit rate

    # ── Behavior ───────────────────────────────────────────────────────────────
    init_open: bool = True
    require_calibrated: bool = True
    # On by default: a TacCap gripper is a self-contained USB hub carrying its own
    # wrist camera and two GSPS sensors, so they travel with the gripper and are
    # cheaper to sniff than to pin per bench.
    auto_discover_cameras: bool = True

    # ── Wrist fisheye rectification ────────────────────────────────────────────
    # The wrist lens is a fisheye and its intrinsics are burned into this
    # gripper's own MCU flash, which is why the switch belongs here rather than
    # on the arm: swap the gripper and both the lens and its calibration go with
    # it. The serial (XGripper) family holds no such record, so it has neither
    # field — a recipe that writes them on an XGripper block is refused at parse
    # rather than quietly ignored.
    #
    # Off by default: nothing changes for a rig that has not opted in. When a
    # unit's firmware holds no calibration, the SDK's shared reference intrinsics
    # are used with a warning — close, but not this unit's, so calibrate before
    # measuring in pixels off a rectified frame.
    undistort_wrist: bool = False
    # 0.0 keeps the calibrated focal length (natural view, the PC tool's default);
    # 1.0 shortens it to 0.70x for the widest field of view, with more black
    # border. Only fx/fy move — the principal point stays put, so the view does
    # not drift as this turns.
    fisheye_balance: float = 0.0

    def __post_init__(self):
        if self.side not in ("left", "right"):
            raise ValueError(f"TaccapFollowerConfig: side must be 'left' or 'right', got {self.side!r}.")
        if not self.kp > 0.0:
            raise ValueError(f"TaccapFollowerConfig: kp must be positive, got {self.kp}.")
        if not self.kd >= 0.0:
            raise ValueError(f"TaccapFollowerConfig: kd must be non-negative, got {self.kd}.")
        if abs(self.feedforward_torque) > MAX_FEEDFORWARD_TORQUE_NM:
            raise ValueError(
                f"TaccapFollowerConfig: |feedforward_torque| must be <= "
                f"{MAX_FEEDFORWARD_TORQUE_NM} Nm, got {self.feedforward_torque}. "
                "Sign: negative = closing/clamp, positive = opening. Values past "
                "~1 Nm are a hard crush (the SDK's gentle-grasp example aborts at 0.30 Nm)."
            )
        if not 0 < self.control_hz <= 500:
            raise ValueError(f"TaccapFollowerConfig: control_hz must be in (0, 500], got {self.control_hz}.")
        if not 0.0 <= self.fisheye_balance <= 1.0:
            raise ValueError(f"TaccapFollowerConfig: fisheye_balance must be in [0, 1], got {self.fisheye_balance}.")
        if self.undistort_wrist and not self.auto_discover_cameras:
            # This combination used to be accepted and do nothing at all: the
            # switch is applied to the wrist camera as it is discovered, so with
            # discovery off there is no camera for it to reach and the rig
            # recorded raw fisheye frames with the knob reading as on. A recipe
            # that pins its cameras by hand sets `undistort` on the wrist camera
            # block instead, where it is next to the resolution it constrains.
            raise ValueError(
                "TaccapFollowerConfig: undistort_wrist=True needs "
                "auto_discover_cameras=True — it is applied to the wrist camera "
                "as it is discovered. With cameras pinned by hand, set "
                "`undistort: true` on the wrist camera block itself."
            )
