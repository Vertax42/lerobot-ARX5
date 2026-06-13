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
    python -m lerobot.robots.taccap_gripper.taccap_gripper_example

    # Add the wrist UVC camera (path supplied explicitly — the MCU-only SDK
    # no longer reports it; prefer a /dev/v4l/by-id/... path for stability).
    python -m lerobot.robots.taccap_gripper.taccap_gripper_example \\
        --wrist-cam-path /dev/v4l/by-id/usb-XenseRobotics_TacCap_Wrist-video-index0

    # Add the Pico4 tracker.
    python -m lerobot.robots.taccap_gripper.taccap_gripper_example --tracker

    # Add tactile sensors (OG serials supplied explicitly — the MCU-only SDK
    # no longer reports them):
    python -m lerobot.robots.taccap_gripper.taccap_gripper_example \\
        --tactile-left-sn OG000477 --tactile-right-sn OG000478

    # Pin a specific gripper by firmware SN (use when multiple are plugged):
    python -m lerobot.robots.taccap_gripper.taccap_gripper_example \\
        --firmware-sn SN000003

The jaw closed position is fixed at 0 rad by the SDK's ``Encoder.set_zero()``;
only the open angle (``--open-rad``, default 1.7 for TC-GU-01) is configurable.

Run the SDK's calibration once per device before using this script:
    python third_party/taccap-gripper/python/examples/calibrate.py <SN>

The script prints 10 observation frames (scalar fields + image shapes)
then disconnects.
"""

from __future__ import annotations

import argparse
import pprint
import time

from lerobot.robots.taccap_gripper import TaccapGripper, TaccapGripperConfig


def _tactile_configs(left_sn: str | None, right_sn: str | None) -> dict:
    """Build XenseTactileCameraConfig entries from explicit OG serials."""
    from lerobot.cameras.xense.configuration_xense import (
        XenseOutputType,
        XenseTactileCameraConfig,
    )

    cameras: dict = {}
    for key, sn in (("tactile_left", left_sn), ("tactile_right", right_sn)):
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
    parser.add_argument("--firmware-sn", default=None,
                        help="Pin to a TacCap-Gripper firmware SN (None = find_one).")
    parser.add_argument("--tracker", action="store_true",
                        help="Enable the Pico4 motion tracker.")
    parser.add_argument("--tracker-sn", default=None,
                        help="Pico4 tracker serial (None = first available).")
    parser.add_argument("--tactile-left-sn", default=None,
                        help="Left OG tactile sensor serial (e.g. OG000477).")
    parser.add_argument("--tactile-right-sn", default=None,
                        help="Right OG tactile sensor serial (e.g. OG000478).")
    parser.add_argument("--wrist-cam-path", default=None,
                        help="Wrist UVC V4L2 path/index; enables the wrist camera when set.")
    parser.add_argument("--imu", action="store_true",
                        help="Enable IMU readings.")
    parser.add_argument("--open-rad", type=float, default=1.7,
                        help="Encoder rad when jaw fully open (TC-GU-01 ~= 1.7).")
    parser.add_argument("--frames", type=int, default=10,
                        help="How many observation frames to print.")
    args = parser.parse_args()

    cameras = _tactile_configs(args.tactile_left_sn, args.tactile_right_sn)
    enable_wrist = args.wrist_cam_path is not None

    cfg = TaccapGripperConfig(
        firmware_sn=args.firmware_sn,
        enable_gripper=True,
        enable_imu=args.imu,
        gripper_open_rad=args.open_rad,
        enable_tracker=args.tracker,
        tracker_sn=args.tracker_sn,
        enable_wrist_camera=enable_wrist,
        wrist_camera_index_or_path=args.wrist_cam_path or "",
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
