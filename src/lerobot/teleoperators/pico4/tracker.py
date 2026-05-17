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

Coordinate frame: raw Pico4 native (X right, Y up, Z toward the headset
operator at Unity-launch time). Callers that need a different frame
should apply ``tracker_to_ee_pos`` / ``tracker_to_ee_quat`` (passed in
at construction) for the rigid mount transform, or a separate world-
frame transform downstream.

Concurrency: ``xrt.init()`` is a process-level singleton. This module
guards it with a class-level flag so multiple readers (e.g. a robot
and a teleop sharing the same process) won't double-init. We never
call ``xrt.close()`` — relying on process exit — because tearing down
the service while a second subscriber is still active crashes the
first.
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
            non-zero tracker data after ``xrt.init()``. Raises
            ``DeviceNotConnectedError`` on timeout.
        hemisphere_fix: If True, flip the sign of incoming quaternions
            so the dot product with the previous frame's quaternion is
            non-negative. Prevents discontinuities in the 6D rotation
            representation when the quaternion crosses a hemisphere
            boundary. See commit af2b2939.
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
        logger_name: str | None = None,
    ):
        self.tracker_sn = tracker_sn
        self.tracker_to_ee_pos = np.asarray(tracker_to_ee_pos, dtype=np.float64)
        self.tracker_to_ee_quat = np.asarray(tracker_to_ee_quat, dtype=np.float64)
        self.device_wait_timeout = float(device_wait_timeout)
        self.hemisphere_fix = bool(hemisphere_fix)

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
        self._prev_quat_wxyz: np.ndarray | None = None  # for hemisphere continuity

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        """Connect to the XenseVR service and pin the requested tracker.

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
                # Wait one more poll to make sure data is real, not just
                # an enumeration flicker.
                poses = xrt.get_motion_tracker_pose()
                sns = xrt.get_motion_tracker_serial_numbers()
                if poses and any(abs(v) > 1e-6 for v in poses[0][:3]):
                    self._resolve_tracker_index(poses, sns, n_trackers)
                    self._is_connected = True
                    self.logger.info(
                        f"Pico4TrackerReader connected to tracker idx={self._tracker_index} "
                        f"sn={self._resolved_sn!r} after {attempt + 1} polls."
                    )
                    return
            time.sleep(0.1)
            attempt += 1

        raise DeviceNotConnectedError(
            f"No Pico4 motion-tracker data after {self.device_wait_timeout:.1f}s. "
            "Check: 1) the VR Client app is running on the Pico4 headset, "
            "2) the PC service is up, 3) the tracker is powered on and paired."
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
        """Mark the reader disconnected.

        NOTE: we deliberately do NOT call ``xrt.close()`` — the service
        is a process-level singleton and other readers (e.g. a teleop
        running in the same process) may still need it. The OS reclaims
        the service connection at process exit.
        """
        if not self._is_connected:
            return
        self._is_connected = False
        self._tracker_index = None
        self._resolved_sn = None
        self._prev_quat_wxyz = None
        self.logger.info("Pico4TrackerReader disconnected (xrt left open for other subscribers).")

    def _raw_pose_xyzw(self) -> np.ndarray | None:
        """Return the raw tracker pose [x, y, z, qx, qy, qz, qw] (xyzw)
        from the SDK, or ``None`` if the tracker dropped out."""
        xrt = Pico4TrackerReader._xrt
        n = xrt.num_motion_data_available()
        if n == 0 or self._tracker_index is None or self._tracker_index >= n:
            return None
        poses = xrt.get_motion_tracker_pose()
        if not poses or self._tracker_index >= len(poses):
            return None
        return np.asarray(poses[self._tracker_index], dtype=np.float64)

    def get_pose_raw(self) -> np.ndarray | None:
        """Raw Pico4 tracker pose in scalar-first (wxyz) convention.

        Returns ``[x, y, z, qw, qx, qy, qz]`` or ``None`` if the tracker
        dropped out this poll.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("Pico4TrackerReader is not connected")
        raw = self._raw_pose_xyzw()
        if raw is None:
            return None
        # Reorder xyzw → wxyz so downstream utilities (all wxyz) match.
        return np.array(
            [raw[0], raw[1], raw[2], raw[6], raw[3], raw[4], raw[5]],
            dtype=np.float64,
        )

    def get_pose_ee(self) -> np.ndarray | None:
        """Pose of the end-effector frame, computed as
        ``T_world_tracker @ T_tracker_ee`` and re-extracted to a
        7-vector ``[x, y, z, qw, qx, qy, qz]``. Applies the hemisphere
        continuity fix if enabled. Returns ``None`` if the tracker
        dropped out."""
        raw_wxyz = self.get_pose_raw()
        if raw_wxyz is None:
            return None

        t_world_tracker = quaternion_to_matrix(raw_wxyz, input_format="wxyz")
        t_world_ee = t_world_tracker @ self._tracker_to_ee_matrix
        ee_pose = matrix_to_pose7d(t_world_ee, output_format="wxyz")

        if self.hemisphere_fix and self._prev_quat_wxyz is not None:
            q_new = ee_pose[3:7]
            if float(np.dot(q_new, self._prev_quat_wxyz)) < 0.0:
                ee_pose[3:7] = -q_new
        self._prev_quat_wxyz = ee_pose[3:7].copy()
        return ee_pose

    def get_action(self) -> dict[str, Any]:
        """Return the same 9-field dict that ``ViveTrackerTeleop.get_action()``
        produces (no gripper field — callers add it from their own source).

        Keys: ``tcp.x``, ``tcp.y``, ``tcp.z``, ``tcp.r1``..``tcp.r6``.

        If the tracker drops out, returns the last-known 6D rotation with
        position zeroed — callers should treat this as a stale-pose hint
        and log/warn appropriately.
        """
        ee_pose = self.get_pose_ee()
        if ee_pose is None:
            self.logger.warn("Tracker dropped out; returning identity-rotation zero-pose.")
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
                print(f"[{i:03d}] (tracker dropped out)")
            time.sleep(0.1)
    finally:
        reader.disconnect()


if __name__ == "__main__":
    _smoke_test()
