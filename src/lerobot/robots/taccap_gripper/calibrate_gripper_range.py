#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Find the encoder endpoints (rad) for a TacCap-Gripper jaw.

Workflow:
    1. Plug in exactly one TacCap-Gripper.
    2. Run this script (no args needed for single-gripper rigs):
           python -m lerobot.robots.taccap_gripper.calibrate_gripper_range
       It prints ``position_rad`` at ~10 Hz, with a running min/max.
    3. Operate the jaw manually:
       - Squeeze it FULLY closed, hold for ~1 s, watch the min stabilise.
       - Release it FULLY open, hold for ~1 s, watch the max stabilise.
    4. Press Ctrl+C; the script prints two lines you copy into your
       ``TaccapGripperConfig``:
           gripper_closed_rad=<min>,
           gripper_open_rad=<max>,

The encoder direction is hardware-dependent — closed may correspond to
the smaller or larger value. We name the endpoints by their physical
meaning (jaw closed / open), not by numeric ordering, so the
normalisation in the robot is sign-aware.
"""

from __future__ import annotations

import argparse
import math
import time

from xense.taccap import LeaderGripper, find_one, scan_grippers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcu-serial", default=None,
                        help="MCU serial filter; required if multiple grippers are plugged in.")
    parser.add_argument("--rate-hz", type=float, default=10.0,
                        help="Print rate (Hz). Higher is noisier; default 10.")
    args = parser.parse_args()

    if args.mcu_serial is None:
        eps = find_one()
    else:
        matches = [e for e in scan_grippers() if e.mcu_serial == args.mcu_serial]
        if not matches:
            seen = [e.mcu_serial for e in scan_grippers()]
            raise SystemExit(f"No gripper with mcu_serial={args.mcu_serial}. Visible: {seen}")
        eps = matches[0]

    print(f"[calibrate] gripper side={eps.side} mcu={eps.mcu_serial}")
    g = LeaderGripper.open()

    period = 1.0 / max(args.rate_hz, 0.1)
    seen_min = math.inf
    seen_max = -math.inf

    print("[calibrate] Squeeze the jaw FULLY CLOSED, then release FULLY OPEN.")
    print("[calibrate] When the min/max stabilise at both extremes, press Ctrl+C.")
    print()
    try:
        while True:
            sample = g.encoder.read_once()
            rad = float(sample.position_rad)
            seen_min = min(seen_min, rad)
            seen_max = max(seen_max, rad)
            print(
                f"  pos={rad:+8.4f} rad   min={seen_min:+8.4f}   max={seen_max:+8.4f}   "
                f"range={seen_max - seen_min:+8.4f}",
                end="\r",
                flush=True,
            )
            time.sleep(period)
    except KeyboardInterrupt:
        pass

    print()  # newline after the carriage-return line
    print()
    if seen_min == math.inf:
        print("[calibrate] No samples observed — exiting.")
        return

    print("[calibrate] Copy these into TaccapGripperConfig:")
    print(f"    gripper_closed_rad={seen_min:.4f},")
    print(f"    gripper_open_rad={seen_max:.4f},")
    print()
    print("    NOTE: 'closed' = the value seen when the jaw was fully squeezed.")
    print("    If the encoder convention is reversed for your unit, swap the two")
    print("    field values manually.")


if __name__ == "__main__":
    main()
