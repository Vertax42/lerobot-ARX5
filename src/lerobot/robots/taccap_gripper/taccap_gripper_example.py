#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Standalone smoke test for TaccapGripper.

Usage:
    # Gripper only (encoder readings, no tracker, no cameras).
    python -m lerobot.robots.taccap_gripper.taccap_gripper_example \\
        --no-wrist-cam

    # Default: gripper + auto-wired wrist camera (V4L2 path from SDK).
    python -m lerobot.robots.taccap_gripper.taccap_gripper_example

    # Add Pico4 tracker.
    python -m lerobot.robots.taccap_gripper.taccap_gripper_example --tracker

    # Add tactile sensors too (uses SDK-reported OG serials).
    python -m lerobot.robots.taccap_gripper.taccap_gripper_example \\
        --tracker --tactile

The wrist camera path is auto-discovered from
``GripperEndpoints.wrist_video`` — no need to hard-code ``/dev/videoN``.
The script prints 10 observation frames (scalar fields + image shapes)
then disconnects.
"""

from __future__ import annotations

import argparse
import pprint
import time

from lerobot.robots.taccap_gripper import TaccapGripper, TaccapGripperConfig


def _tactile_configs(endpoints) -> dict:
    """Build XenseTactileCameraConfig entries from the live SDK endpoints."""
    from lerobot.cameras.xense.configuration_xense import (
        XenseOutputType,
        XenseTactileCameraConfig,
    )

    cameras: dict = {}
    for key, sn in (
        ("tactile_left", endpoints.tactile_left_serial),
        ("tactile_right", endpoints.tactile_right_serial),
    ):
        if not sn:
            continue
        cameras[key] = XenseTactileCameraConfig(
            serial_number=sn,
            fps=30,
            width=400,
            height=700,
            output_types=[XenseOutputType.RECTIFY],
        )
    return cameras


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcu-serial", default=None,
                        help="Pin to a specific TacCap-Gripper MCU serial.")
    parser.add_argument("--tracker", action="store_true",
                        help="Enable the Pico4 motion tracker.")
    parser.add_argument("--tracker-sn", default=None,
                        help="Pico4 tracker serial (None = first available).")
    parser.add_argument("--tactile", action="store_true",
                        help="Enable tactile cameras (left + right OG sensors).")
    parser.add_argument("--no-wrist-cam", action="store_true",
                        help="Skip the auto-wired wrist UVC camera.")
    parser.add_argument("--imu", action="store_true",
                        help="Enable IMU readings.")
    parser.add_argument("--closed-rad", type=float, default=0.0,
                        help="Encoder rad when jaw fully closed.")
    parser.add_argument("--open-rad", type=float, default=1.5,
                        help="Encoder rad when jaw fully open.")
    parser.add_argument("--frames", type=int, default=10,
                        help="How many observation frames to print.")
    args = parser.parse_args()

    cameras: dict = {}
    if args.tactile:
        # Tactile serials need the live SDK; do a quick discovery up front.
        from xense.taccap import find_one, scan_grippers

        if args.mcu_serial is None:
            eps = find_one()
        else:
            matches = [e for e in scan_grippers() if e.mcu_serial == args.mcu_serial]
            if not matches:
                raise SystemExit(f"No gripper with MCU={args.mcu_serial}")
            eps = matches[0]
        cameras = _tactile_configs(eps)

    cfg = TaccapGripperConfig(
        mcu_serial=args.mcu_serial,
        enable_gripper=True,
        enable_imu=args.imu,
        gripper_closed_rad=args.closed_rad,
        gripper_open_rad=args.open_rad,
        enable_tracker=args.tracker,
        tracker_sn=args.tracker_sn,
        enable_wrist_camera=not args.no_wrist_cam,
        cameras=cameras,
    )

    robot = TaccapGripper(cfg)

    print("[example] observation features:")
    pprint.pprint(robot.observation_features)
    print("[example] action features:")
    pprint.pprint(robot.action_features)

    robot.connect()
    try:
        for i in range(args.frames):
            obs = robot.get_observation()
            scalars = {k: v for k, v in obs.items() if not hasattr(v, "shape")}
            shapes = {k: v.shape for k, v in obs.items() if hasattr(v, "shape")}
            print(f"[{i:02d}] {scalars}  +  {shapes}")
            time.sleep(0.1)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
