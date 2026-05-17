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
    # Gripper only (minimal — verifies LeaderGripper.open() + encoder).
    python -m lerobot.robots.taccap_gripper.taccap_gripper_example

    # Add Pico4 tracker.
    python -m lerobot.robots.taccap_gripper.taccap_gripper_example --tracker

    # Add cameras (tactile + wrist) — paths must be valid for your rig.
    python -m lerobot.robots.taccap_gripper.taccap_gripper_example \\
        --tracker --cameras

This script does NOT enable the gripper motor. It only reads the
encoder, the optional tracker, and the optional cameras, prints 10
observations, then disconnects.
"""

from __future__ import annotations

import argparse
import pprint
import time

from lerobot.robots.taccap_gripper import TaccapGripper, TaccapGripperConfig


def _build_camera_configs() -> dict:
    """Build a default tactile-left/right + wrist camera config trio.

    Edit serials and the wrist V4L2 path to match your hardware before
    running with ``--cameras``. We pull tactile serials from the live
    gripper inside ``main()`` so the user doesn't have to hard-code them
    twice."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    return {
        "wrist_cam": OpenCVCameraConfig(
            index_or_path="/dev/video0",
            width=640,
            height=480,
            fps=30,
        ),
    }


def _augment_with_tactile(
    cameras: dict,
    endpoints,
) -> dict:
    """Add tactile camera configs using the SDK-reported OG serials."""
    from lerobot.cameras.xense.configuration_xense import (
        XenseOutputType,
        XenseTactileCameraConfig,
    )

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
    parser.add_argument("--cameras", action="store_true",
                        help="Enable wrist + tactile cameras.")
    parser.add_argument("--imu", action="store_true",
                        help="Enable IMU readings.")
    parser.add_argument("--closed-rad", type=float, default=0.0,
                        help="Encoder rad when jaw fully closed.")
    parser.add_argument("--open-rad", type=float, default=1.5,
                        help="Encoder rad when jaw fully open.")
    parser.add_argument("--frames", type=int, default=10,
                        help="How many observation frames to print.")
    args = parser.parse_args()

    cameras = _build_camera_configs() if args.cameras else {}

    cfg = TaccapGripperConfig(
        mcu_serial=args.mcu_serial,
        enable_gripper=True,
        enable_imu=args.imu,
        gripper_closed_rad=args.closed_rad,
        gripper_open_rad=args.open_rad,
        enable_tracker=args.tracker,
        tracker_sn=args.tracker_sn,
        cameras=cameras,
    )

    robot = TaccapGripper(cfg)

    # If we want tactile cameras, we need the SDK-reported serials AFTER
    # the SDK has discovered the gripper. So: connect with no tactile
    # cams first, augment the config with the live serials, then
    # reinstate the camera dict before continuing. Cleaner than asking
    # the operator to type two serials by hand.
    if args.cameras:
        print("[setup] partial connect to discover tactile serials...")
        # Cheat: build a temporary endpoints-only config to grab serials.
        from xense.taccap import find_one

        eps = find_one() if args.mcu_serial is None else None
        if eps is None:
            from xense.taccap import scan_grippers
            matches = [e for e in scan_grippers() if e.mcu_serial == args.mcu_serial]
            if not matches:
                raise SystemExit(f"No gripper with MCU={args.mcu_serial}")
            eps = matches[0]
        cameras = _augment_with_tactile(cameras, eps)
        # Re-build the config + robot with the augmented camera dict.
        cfg = TaccapGripperConfig(
            mcu_serial=args.mcu_serial,
            enable_gripper=True,
            enable_imu=args.imu,
            gripper_closed_rad=args.closed_rad,
            gripper_open_rad=args.open_rad,
            enable_tracker=args.tracker,
            tracker_sn=args.tracker_sn,
            cameras=cameras,
        )
        robot = TaccapGripper(cfg)

    print(f"[example] observation features:")
    pprint.pprint(robot.observation_features)
    print(f"[example] action features:")
    pprint.pprint(robot.action_features)

    robot.connect()
    try:
        for i in range(args.frames):
            obs = robot.get_observation()
            # Print only the scalar fields per frame; image shapes are
            # noisy and not very informative.
            scalars = {k: v for k, v in obs.items() if not hasattr(v, "shape")}
            shapes = {k: v.shape for k, v in obs.items() if hasattr(v, "shape")}
            print(f"[{i:02d}] {scalars}  +  {shapes}")
            time.sleep(0.1)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
