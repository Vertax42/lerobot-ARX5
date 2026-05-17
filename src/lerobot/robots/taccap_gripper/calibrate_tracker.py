#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Sanity-check helper for the Pico4 motion tracker.

Prints raw + EE pose at 10 Hz so you can:
    - confirm the tracker is alive and the SN matches what you expect,
    - eyeball the ``tracker_to_ee_pos`` / ``tracker_to_ee_quat`` rigid
      mount transform (defaults to identity = EE coincident with tracker),
    - watch for hemisphere flips in the quaternion (the reader applies
      a continuity fix; if you still see sign jumps, file a bug).

Usage:
    python -m lerobot.robots.taccap_gripper.calibrate_tracker
    python -m lerobot.robots.taccap_gripper.calibrate_tracker <tracker_sn>
"""

from __future__ import annotations

import argparse
import time

from lerobot.teleoperators.pico4.tracker import Pico4TrackerReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracker_sn", nargs="?", default=None,
                        help="Pico4 tracker serial. Default = first available.")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Run for N seconds, then exit. 0 = until Ctrl+C.")
    args = parser.parse_args()

    reader = Pico4TrackerReader(tracker_sn=args.tracker_sn)
    reader.connect()
    print(f"[calibrate] tracker connected. Press Ctrl+C to stop.")

    t_start = time.monotonic()
    try:
        while True:
            raw = reader.get_pose_raw()
            ee = reader.get_pose_ee()
            t = time.monotonic() - t_start
            if raw is None or ee is None:
                print(f"[{t:7.2f}s] tracker dropped out", end="\r", flush=True)
            else:
                print(
                    f"[{t:7.2f}s] "
                    f"raw xyz=({raw[0]:+7.3f},{raw[1]:+7.3f},{raw[2]:+7.3f}) "
                    f"wxyz=({raw[3]:+5.2f},{raw[4]:+5.2f},{raw[5]:+5.2f},{raw[6]:+5.2f})  "
                    f"ee xyz=({ee[0]:+7.3f},{ee[1]:+7.3f},{ee[2]:+7.3f})",
                    end="\r",
                    flush=True,
                )
            if args.duration > 0 and t >= args.duration:
                print()
                print(f"[calibrate] reached duration={args.duration}s")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        reader.disconnect()


if __name__ == "__main__":
    main()
