#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Pico4 Ultra independent motion-tracker reader.

This is a *thin* reader, not a full Teleoperator. It reads one motion
tracker from the XenseVR PC Service and emits the same 9-D pose dict
(``tcp.x/y/z`` + ``tcp.r1-r6`` in 6D rotation representation) that
``ViveTrackerTeleop.get_action()`` produces — so a robot can swap a Vive
reader for a Pico4 reader without changing its observation schema.

The XenseVR service exposes up to three independent trackers. We pin to
one tracker by serial number; if no SN is given we take whichever one
the service reports at index 0.

Coordinate frame
----------------
The tracker pose this reader emits is in the **same xrt-native frame
as the controller poses** that ``teleop_pico4.get_*_controller_pose()``
returns — both come from the same XenseVR PC Service via the same
pybind, just different endpoint calls (``get_motion_tracker_pose`` vs.
``get_left/right_controller_pose``). So if you want this reader to act
as a drop-in replacement for the controller in a teleop flow, apply
the same coordinate transform that the controller-based teleop applies
(e.g. ``teleop_pico4`` does a Pico→Flexiv remap at line ~447-513 of
``teleop_pico4.py``; you'd reuse that on top of ``get_pose_raw()``).

For pure handheld data collection (taccap_gripper) there is no arm,
so we don't remap — we emit the raw xrt frame and leave reframing to
post-processing.

The world origin is the headset position the moment the Unity VR app
started — *not* when xrt.init() ran, *not* when you clicked Connect in
Unity. Restarting Unity mid-session relocates the origin.

The axis convention (handedness, Z direction) is documented inconsistently
upstream — Pico docs say right-handed, SDK example notes left-handed.
Treat this as **TBD pending live verification on real hardware**. The
reader returns what xrt gives us, untouched apart from the configurable
rigid mount transform (``tracker_to_ee_pos`` / ``tracker_to_ee_quat``).

Concurrency
-----------
``xrt.init()`` is a process-level singleton. This module guards it with
a class-level flag so multiple readers in one process share the same
SDK instance. ``xrt.close()`` is intentionally NOT called on disconnect
— if a second subscriber is still alive, closing tears it down too.
Rely on process exit.

Threading model — background poller
-----------------------------------
A daemon thread polls ``xrt.get_motion_tracker_pose()`` at the Pico
tracker's native rate (~90 Hz) and caches the latest pose under a
lock. ``get_pose_raw()`` / ``get_pose_ee()`` / ``get_action()`` read
from this cache, never block the SDK. Benefits:

  * A hung PC Service cannot stall the observation thread.
  * The tracker runs at its native rate regardless of how slowly the
    consumer polls (slower → simply gets the freshest sample).
  * Hemisphere-continuity fix is applied in-thread sequentially, so
    even very slow consumers see a continuous quaternion stream.

Matches the architecture of the SDK's
``examples/rerun_dual_with_tracker.py`` ``TrackerPoller``.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import (
    get_logger,
    matrix_to_pose7d,
    quaternion_to_matrix,
    quaternion_to_rotation_6d,
)

# Pico tracker's native sampling rate. Polling slower wastes data;
# faster just returns duplicates. SDK example uses the same constant.
TRACKER_POLL_HZ = 90.0

# How long to remember the last known pose after the tracker drops out
# before logging a stale-pose warning. Below this threshold,
# get_pose_ee() returns the cached pose silently; above, it warns once
# and continues returning the stale value (callers can decide what to
# do with it).
STALE_WARN_THRESHOLD_S = 0.5

# Minimum interval between "tracker SN X not visible" warnings to keep
# the log readable when the service is up but a tracker is unplugged.
MISSING_SN_WARN_INTERVAL_S = 5.0


class Pico4TrackerReader:
    """Read a single Pico4 Ultra motion tracker via the XenseVR PC Service.

    Args:
        tracker_sn: Tracker serial number. ``None`` picks index 0 from
            ``xrt.get_motion_tracker_serial_numbers()``.
        tracker_to_ee_pos: Rigid offset from the tracker frame to the
            end-effector frame, [x, y, z] in meters. Default: zero.
        tracker_to_ee_quat: Rigid rotation from the tracker frame to the
            end-effector frame, [qw, qx, qy, qz]. Default: identity.
        device_wait_timeout: Seconds to wait for the service to report
            non-zero tracker data at connect time. Raises
            ``DeviceNotConnectedError`` on timeout.
        hemisphere_fix: If True, flip the sign of incoming quaternions
            so the dot product with the previous frame's quaternion is
            non-negative. Prevents discontinuities in the 6D rotation
            representation when the quaternion crosses a hemisphere
            boundary. See commit af2b2939.
        poll_hz: Background poll rate. Default ``TRACKER_POLL_HZ`` (90).
        logger_name: Optional logger name suffix.
    """

    # Class-level singleton guard so multiple readers don't re-init xrt.
    _xrt_initialized: bool = False
    _xrt = None  # module reference, set on first connect
    _init_lock = threading.Lock()
    # spdlog rejects duplicate logger names; use an instance counter to
    # disambiguate when multiple readers share the same logger_name (e.g.
    # both default to "auto").
    _instance_counter: int = 0
    _counter_lock = threading.Lock()

    def __init__(
        self,
        tracker_sn: str | None = None,
        tracker_to_ee_pos: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
        tracker_to_ee_quat: tuple[float, float, float, float] | list[float] = (1.0, 0.0, 0.0, 0.0),
        device_wait_timeout: float = 10.0,
        hemisphere_fix: bool = True,
        poll_hz: float = TRACKER_POLL_HZ,
        logger_name: str | None = None,
    ):
        self.tracker_sn = tracker_sn
        self.tracker_to_ee_pos = np.asarray(tracker_to_ee_pos, dtype=np.float64)
        self.tracker_to_ee_quat = np.asarray(tracker_to_ee_quat, dtype=np.float64)
        self.device_wait_timeout = float(device_wait_timeout)
        self.hemisphere_fix = bool(hemisphere_fix)
        self._poll_period = 1.0 / max(1e-3, float(poll_hz))

        suffix = logger_name or (tracker_sn if tracker_sn else "auto")
        with Pico4TrackerReader._counter_lock:
            Pico4TrackerReader._instance_counter += 1
            instance_id = Pico4TrackerReader._instance_counter
        self.logger = get_logger(f"Pico4TrackerReader-{suffix}-{instance_id}")

        # Pre-compute the 4x4 tracker→EE rigid transform.
        tracker_to_ee_pose = np.concatenate([self.tracker_to_ee_pos, self.tracker_to_ee_quat])
        self._tracker_to_ee_matrix = quaternion_to_matrix(tracker_to_ee_pose, input_format="wxyz")

        # Resolved at connect time.
        self._is_connected: bool = False
        self._tracker_index: int | None = None
        self._resolved_sn: str | None = None

        # Cached state, guarded by _pose_lock and produced by _poller_thread.
        self._pose_lock = threading.Lock()
        self._latest_raw_wxyz: np.ndarray | None = None   # [x,y,z,qw,qx,qy,qz]
        self._latest_ee_wxyz: np.ndarray | None = None    # [x,y,z,qw,qx,qy,qz] after rigid + hemisphere
        self._latest_ts: float | None = None              # time.monotonic() of latest valid pose
        self._stale_warned: bool = False                  # gate stale-pose warning to once per drop-out
        self._missing_warned_at: float = 0.0              # throttle missing-SN warning

        # Poller thread.
        self._poller_thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        """Initialise the SDK, pin the requested tracker, and start the
        background poll thread.

        Raises:
            DeviceAlreadyConnectedError: if already connected.
            ImportError: if ``xensevr_pc_service_sdk`` is not installed.
            DeviceNotConnectedError: if the service reports no tracker
                data within ``device_wait_timeout`` seconds.
            ValueError: if ``tracker_sn`` is set but doesn't match any
                tracker advertised by the service.
        """
        if self._is_connected:
            raise DeviceAlreadyConnectedError("Pico4TrackerReader already connected")

        with Pico4TrackerReader._init_lock:
            if not Pico4TrackerReader._xrt_initialized:
                try:
                    import xensevr_pc_service_sdk as xrt
                except ImportError as e:
                    raise ImportError(
                        "xensevr_pc_service_sdk is required for Pico4TrackerReader. "
                        "Build the pybind under src/lerobot/teleoperators/pico4/xensevr-pc-service-pybind/."
                    ) from e
                xrt.init()
                Pico4TrackerReader._xrt = xrt
                Pico4TrackerReader._xrt_initialized = True
                self.logger.info("XenseVR SDK initialized.")
            else:
                self.logger.info("XenseVR SDK already initialized; reusing singleton.")

        xrt = Pico4TrackerReader._xrt

        # IMPORTANT: the Pico4 coordinate origin is set by Unity at app
        # launch — not by xrt.init(), and not by re-clicking Connect in
        # Unity. If Unity restarts mid-session, all subsequent poses
        # will be expressed in a different origin frame. Warn loudly.
        self.logger.warn(
            "Pico4 origin is fixed at Unity-app launch time. "
            "Do NOT restart the Unity client between episodes, or recorded "
            "episodes will be expressed in mismatched origin frames."
        )

        deadline = time.monotonic() + self.device_wait_timeout
        attempt = 0
        while time.monotonic() < deadline:
            n_trackers = xrt.num_motion_data_available()
            if n_trackers > 0:
                poses = xrt.get_motion_tracker_pose()
                sns = xrt.get_motion_tracker_serial_numbers()
                if poses and any(abs(v) > 1e-6 for v in poses[0][:3]):
                    self._resolve_tracker_index(poses, sns, n_trackers)
                    break
            time.sleep(0.1)
            attempt += 1
        else:
            raise DeviceNotConnectedError(
                f"No Pico4 motion-tracker data after {self.device_wait_timeout:.1f}s. "
                "Check: 1) the VR Client app is running on the Pico4 headset, "
                "2) the PC service is up, 3) the tracker is powered on and paired."
            )

        # Spin up the poller. Use is_connected as the flag *before* the
        # thread starts so it can run its first tick immediately without
        # racing against connect()'s return.
        self._is_connected = True
        self._stop_evt.clear()
        self._poller_thread = threading.Thread(
            target=self._poller_loop,
            name=f"pico4-tracker-poller-{self._resolved_sn or 'auto'}",
            daemon=True,
        )
        self._poller_thread.start()
        self.logger.info(
            f"Pico4TrackerReader connected to tracker idx={self._tracker_index} "
            f"sn={self._resolved_sn!r} after {attempt + 1} polls; "
            f"poller started at {1.0 / self._poll_period:.0f} Hz."
        )

    def _resolve_tracker_index(
        self,
        poses: list[list[float]],
        sns: list[str],
        n_trackers: int,
    ) -> None:
        """Pick the tracker matching ``tracker_sn`` (or index 0)."""
        if self.tracker_sn is None:
            self._tracker_index = 0
            self._resolved_sn = sns[0] if sns else "<unknown>"
            return

        for idx in range(min(n_trackers, len(sns))):
            if sns[idx] == self.tracker_sn:
                self._tracker_index = idx
                self._resolved_sn = sns[idx]
                return

        raise ValueError(
            f"Tracker SN {self.tracker_sn!r} not found. "
            f"Available SNs: {sns[:n_trackers]!r}."
        )

    def disconnect(self) -> None:
        """Stop the poller thread and mark the reader disconnected.

        NOTE: we deliberately do NOT call ``xrt.close()`` — the service
        is a process-level singleton and other readers in the same
        process may still need it. The OS reclaims the service
        connection at process exit.
        """
        if not self._is_connected:
            return

        self._stop_evt.set()
        if self._poller_thread is not None:
            self._poller_thread.join(timeout=2.0)
            if self._poller_thread.is_alive():
                self.logger.warn("Poller thread did not exit within 2s.")
            self._poller_thread = None

        self._is_connected = False
        self._tracker_index = None
        self._resolved_sn = None
        with self._pose_lock:
            self._latest_raw_wxyz = None
            self._latest_ee_wxyz = None
            self._latest_ts = None
        self.logger.info("Pico4TrackerReader disconnected (xrt left open for other subscribers).")

    # ---- Background poller --------------------------------------------------

    def _poller_loop(self) -> None:
        """Body of the background thread. Polls xrt at the requested rate
        and updates the cached pose under the lock. Lifecycle is driven
        by ``_stop_evt`` (set by ``disconnect()``)."""
        xrt = Pico4TrackerReader._xrt
        prev_ee_quat: np.ndarray | None = None

        while not self._stop_evt.wait(self._poll_period):
            try:
                n = xrt.num_motion_data_available()
            except Exception as e:  # pragma: no cover — defensive
                self.logger.warn(f"num_motion_data_available failed: {e}")
                continue

            if n == 0:
                self._maybe_warn_missing()
                continue

            try:
                poses = xrt.get_motion_tracker_pose()
                sns = xrt.get_motion_tracker_serial_numbers()
            except Exception as e:  # pragma: no cover — defensive
                self.logger.warn(f"tracker pose query failed: {e}")
                continue

            # The tracker index might have shifted if other trackers
            # plugged/unplugged between connect and now. Re-resolve.
            idx = self._resolve_current_index(sns, n)
            if idx is None:
                self._maybe_warn_missing()
                continue
            if idx >= len(poses):
                continue

            raw_xyzw = np.asarray(poses[idx], dtype=np.float64)
            # Reorder xyzw → wxyz so downstream utilities (all wxyz) match.
            raw_wxyz = np.array(
                [raw_xyzw[0], raw_xyzw[1], raw_xyzw[2],
                 raw_xyzw[6], raw_xyzw[3], raw_xyzw[4], raw_xyzw[5]],
                dtype=np.float64,
            )

            # Apply tracker→EE rigid transform.
            t_world_tracker = quaternion_to_matrix(raw_wxyz, input_format="wxyz")
            t_world_ee = t_world_tracker @ self._tracker_to_ee_matrix
            ee_pose = matrix_to_pose7d(t_world_ee, output_format="wxyz")

            # Hemisphere continuity — applied here (sequential), so even
            # slow consumers see no sign flips.
            if self.hemisphere_fix and prev_ee_quat is not None:
                q_new = ee_pose[3:7]
                if float(np.dot(q_new, prev_ee_quat)) < 0.0:
                    ee_pose[3:7] = -q_new
            prev_ee_quat = ee_pose[3:7].copy()

            with self._pose_lock:
                self._latest_raw_wxyz = raw_wxyz
                self._latest_ee_wxyz = ee_pose
                self._latest_ts = time.monotonic()
                self._stale_warned = False

    def _resolve_current_index(self, sns: list, n: int) -> int | None:
        """Locate the current index of our pinned tracker. Returns None
        if it has dropped out."""
        if self.tracker_sn is None:
            # Anonymous: just keep using the same slot; if it shrunk to
            # 0 trackers we already returned None above.
            return self._tracker_index if (self._tracker_index or 0) < n else None
        for idx in range(min(n, len(sns))):
            sn = sns[idx]
            if isinstance(sn, bytes):
                sn = sn.decode()
            if sn == self.tracker_sn:
                if idx != self._tracker_index:
                    self.logger.info(
                        f"Tracker SN {self.tracker_sn!r} index shifted "
                        f"{self._tracker_index} -> {idx}"
                    )
                    self._tracker_index = idx
                return idx
        return None

    def _maybe_warn_missing(self) -> None:
        """Log 'tracker SN not visible' at most every ``MISSING_SN_WARN_INTERVAL_S``."""
        now = time.monotonic()
        if now - self._missing_warned_at < MISSING_SN_WARN_INTERVAL_S:
            return
        self._missing_warned_at = now
        sn = self.tracker_sn or "<index 0>"
        self.logger.warn(
            f"Tracker SN={sn!r} not currently visible to the PC Service. "
            "Check that it is paired and powered on."
        )

    # ---- Public accessors (read from cache) --------------------------------

    def get_pose_raw(self) -> np.ndarray | None:
        """Raw Pico4 tracker pose in scalar-first (wxyz) convention.

        Returns ``[x, y, z, qw, qx, qy, qz]`` or ``None`` if the
        tracker has dropped out and there is no cached value."""
        if not self._is_connected:
            raise DeviceNotConnectedError("Pico4TrackerReader is not connected")
        with self._pose_lock:
            if self._latest_raw_wxyz is None:
                return None
            return self._latest_raw_wxyz.copy()

    def get_pose_ee(self) -> np.ndarray | None:
        """Latest pose of the end-effector frame (rigid-transformed,
        hemisphere-corrected) as ``[x, y, z, qw, qx, qy, qz]``.

        Logs a stale-pose warning once if the cached pose is older than
        ``STALE_WARN_THRESHOLD_S`` (the consumer keeps getting the last
        good value)."""
        if not self._is_connected:
            raise DeviceNotConnectedError("Pico4TrackerReader is not connected")
        with self._pose_lock:
            if self._latest_ee_wxyz is None:
                return None
            age = time.monotonic() - (self._latest_ts or 0.0)
            if age > STALE_WARN_THRESHOLD_S and not self._stale_warned:
                self._stale_warned = True
                self.logger.warn(
                    f"Pose stale by {age:.2f}s — tracker may have dropped out. "
                    "Returning last-known pose."
                )
            return self._latest_ee_wxyz.copy()

    def get_action(self) -> dict[str, Any]:
        """Return the same 9-field dict that ``ViveTrackerTeleop.get_action()``
        produces (no gripper field — callers add it from their own source).

        Keys: ``tcp.x``, ``tcp.y``, ``tcp.z``, ``tcp.r1``..``tcp.r6``.

        If the tracker never produced a pose, returns an identity-rotation
        zero pose so the observation schema stays well-formed."""
        ee_pose = self.get_pose_ee()
        if ee_pose is None:
            return {
                "tcp.x": 0.0,
                "tcp.y": 0.0,
                "tcp.z": 0.0,
                "tcp.r1": 1.0,
                "tcp.r2": 0.0,
                "tcp.r3": 0.0,
                "tcp.r4": 0.0,
                "tcp.r5": 1.0,
                "tcp.r6": 0.0,
            }

        qw, qx, qy, qz = ee_pose[3], ee_pose[4], ee_pose[5], ee_pose[6]
        r6d = quaternion_to_rotation_6d(qw, qx, qy, qz)
        return {
            "tcp.x": float(ee_pose[0]),
            "tcp.y": float(ee_pose[1]),
            "tcp.z": float(ee_pose[2]),
            "tcp.r1": float(r6d[0]),
            "tcp.r2": float(r6d[1]),
            "tcp.r3": float(r6d[2]),
            "tcp.r4": float(r6d[3]),
            "tcp.r5": float(r6d[4]),
            "tcp.r6": float(r6d[5]),
        }


def _smoke_test() -> None:
    """Manual smoke test: print 50 raw + EE poses then exit.

    Usage:
        python -m lerobot.teleoperators.pico4.tracker
        python -m lerobot.teleoperators.pico4.tracker <tracker_sn>
    """
    import sys

    sn = sys.argv[1] if len(sys.argv) > 1 else None
    reader = Pico4TrackerReader(tracker_sn=sn)
    reader.connect()
    try:
        for i in range(50):
            raw = reader.get_pose_raw()
            ee = reader.get_pose_ee()
            if raw is not None:
                print(
                    f"[{i:03d}] raw xyz={raw[:3]} wxyz={raw[3:]} "
                    f"ee xyz={ee[:3]} wxyz={ee[3:]}"
                )
            else:
                print(f"[{i:03d}] (no pose yet)")
            time.sleep(0.1)
    finally:
        reader.disconnect()


if __name__ == "__main__":
    _smoke_test()
