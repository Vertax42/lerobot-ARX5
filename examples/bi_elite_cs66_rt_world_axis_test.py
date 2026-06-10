#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""World-frame axis test for ONE Elite CS66 arm via the BiEliteCS66RT interface.

Drives a single arm (default LEFT) THROUGH THE WORLD-FRAME send_action/get_observation
boundary so we can confirm the world axes match our definition (x=forward,
y=left, z=up). For each requested axis it moves the TCP +distance along that WORLD
axis, then back, keeping orientation fixed and translating only:

    X: +20 cm forward, back     Y: +20 cm left, back     Z: +20 cm up, back

Safety / behaviour:
  • THIS MOVES THE ROBOT. It refuses to move unless you type ``yes`` (or pass
    ``--yes``). Use ``--dry-run`` to print the plan without touching hardware.
  • Motion is interpolated (min-jerk) and streamed at --rate Hz; a 20 cm leg over
    --leg-duration s (default 4 s ≈ 5 cm/s). No single large servo jump.
  • Only the moved arm's keys are sent; the other arm holds its pose.
  • Orientation is locked to the pose read at start; only position changes.
  • After each leg's peak it reads back get_observation and prints the achieved
    world displacement so you can confirm it moved on the intended axis only.
  • Cameras and grippers are disabled (not needed; avoids外设 connect failures).
  • connect() powers on BOTH arms and (unless --no-go-to-start) MoveJ's both to
    their start poses first. disconnect() returns both to home.

Run on the station with a hand on the e-stop:
    python examples/bi_elite_cs66_rt_world_axis_test.py --dry-run
    python examples/bi_elite_cs66_rt_world_axis_test.py            # asks to confirm
    python examples/bi_elite_cs66_rt_world_axis_test.py --arm left --axes xyz --yes
"""

import argparse
import time

import numpy as np

from lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt import BiEliteCS66RT
from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_AXIS_DESC = {"x": "+X (forward)", "y": "+Y (left)", "z": "+Z (up)"}


def _min_jerk(alpha: float) -> float:
    alpha = min(max(alpha, 0.0), 1.0)
    return alpha * alpha * alpha * (10.0 + alpha * (-15.0 + 6.0 * alpha))


def _read_world_pose(robot: BiEliteCS66RT, side: str) -> tuple[np.ndarray, list[float]]:
    """Return (position[3] in world, rot6[6]) for ``side`` from get_observation."""
    obs = robot.get_observation()
    pos = np.array([obs[f"{side}_tcp.x"], obs[f"{side}_tcp.y"], obs[f"{side}_tcp.z"]], dtype=np.float64)
    rot6 = [float(obs[f"{side}_tcp.r{i + 1}"]) for i in range(6)]
    return pos, rot6


def _make_action(side: str, pos: np.ndarray, rot6: list[float]) -> dict:
    action = {
        f"{side}_tcp.x": float(pos[0]),
        f"{side}_tcp.y": float(pos[1]),
        f"{side}_tcp.z": float(pos[2]),
    }
    action.update({f"{side}_tcp.r{i + 1}": float(rot6[i]) for i in range(6)})
    return action


def _command(robot: BiEliteCS66RT, side: str, action: dict) -> None:
    """Command ONLY ``side``.

    We call the per-arm path directly instead of ``robot.send_action`` so that a
    fault on the *other* arm (e.g. its servo loop dying) cannot abort a
    single-arm test. ``send_action`` iterates both arms and re-raises the other
    arm's background servo error; ``_send_arm_action`` touches only this arm
    (and still surfaces THIS arm's own servo errors).
    """
    robot._send_arm_action(side, action, {})


def _ramp(robot, side, start_pos, end_pos, rot6, duration, rate) -> None:
    """Stream a min-jerk position ramp start_pos -> end_pos at ``rate`` Hz."""
    n = max(1, int(round(duration * rate)))
    dt = 1.0 / rate
    delta = end_pos - start_pos
    for i in range(1, n + 1):
        alpha = _min_jerk(i / n)
        pos = start_pos + alpha * delta
        _command(robot, side, _make_action(side, pos, rot6))
        time.sleep(dt)


def _run_axis_leg(robot, side, axis, home_pos, rot6, distance, duration, pause, rate) -> None:
    idx = _AXIS_INDEX[axis]
    target = home_pos.copy()
    target[idx] += distance
    print(f"\n--- {side} {_AXIS_DESC[axis]}  {distance * 100:.0f} cm ---")
    print(f"    home  world pos = [{home_pos[0]:+.4f}, {home_pos[1]:+.4f}, {home_pos[2]:+.4f}]")
    print(f"    target world pos= [{target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f}]")

    _ramp(robot, side, home_pos, target, rot6, duration, rate)
    time.sleep(pause)
    reached, _ = _read_world_pose(robot, side)
    d = reached - home_pos
    print(f"    reached world pos= [{reached[0]:+.4f}, {reached[1]:+.4f}, {reached[2]:+.4f}]")
    print(f"    achieved Δ (x,y,z)=[{d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}]  "
          f"(want only Δ{axis}=+{distance:.2f}, others ~0)")

    _ramp(robot, side, target, home_pos, rot6, duration, rate)
    time.sleep(pause)
    back, _ = _read_world_pose(robot, side)
    db = back - home_pos
    print(f"    back    world pos= [{back[0]:+.4f}, {back[1]:+.4f}, {back[2]:+.4f}]  "
          f"(residual |Δ|={np.linalg.norm(db) * 1000:.1f} mm)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("left", "right"), default="left")
    ap.add_argument("--axes", default="xyz", help="which world axes to sweep, in order (subset of xyz)")
    ap.add_argument("--distance", type=float, default=0.20, help="metres per leg (default 0.20)")
    ap.add_argument("--leg-duration", type=float, default=4.0, help="seconds for one 1-way leg")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds to dwell at each end")
    ap.add_argument("--rate", type=float, default=50.0, help="send_action stream rate (Hz)")
    ap.add_argument("--no-go-to-start", action="store_true", help="do not MoveJ to start on connect")
    ap.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; do not connect or move")
    args = ap.parse_args()

    axes = [a for a in args.axes.lower() if a in _AXIS_INDEX]
    if not axes:
        ap.error(f"--axes must contain at least one of x/y/z, got {args.axes!r}")

    cfg = BiEliteCS66RTConfig()
    # Disable cameras/grippers for a pure motion test (not needed; avoids外设 deps).
    cfg.cameras = {}
    cfg.left_gripper = None
    cfg.right_gripper = None

    print("World-frame axis test")
    print(f"  arm           : {args.arm}")
    print(f"  axes          : {' -> '.join(_AXIS_DESC[a] for a in axes)}")
    print(f"  distance      : {args.distance * 100:.0f} cm/leg, {args.leg_duration:.1f}s/leg "
          f"(~{args.distance / args.leg_duration * 100:.1f} cm/s), pause {args.pause:.1f}s, {args.rate:.0f} Hz")
    print(f"  IPs           : left={cfg.left_robot_ip}  right={cfg.right_robot_ip}")
    print(f"  mount ({args.arm}): tilt={getattr(cfg, f'{args.arm}_mount_tilt_deg'):g}° "
          f"zrot={getattr(cfg, f'{args.arm}_mount_zrot_deg'):g}° "
          f"world_yaw={getattr(cfg, f'{args.arm}_mount_world_yaw_deg'):g}°")

    if args.dry_run:
        print("\n[dry-run] not connecting / not moving. Plan above. "
              "Each leg: home -> home+axis -> home, orientation locked.")
        return

    if not args.yes:
        print("\n⚠️  THIS WILL MOVE THE ROBOT. Keep a hand on the e-stop.")
        if input("Type 'yes' to proceed: ").strip().lower() != "yes":
            print("Aborted.")
            return

    robot = BiEliteCS66RT(cfg)
    print("\nConnecting (powers on both arms; "
          f"{'NOT moving to start' if args.no_go_to_start else 'MoveJ both to start'})...")
    robot.connect(go_to_start=not args.no_go_to_start)
    try:
        assert robot.is_connected, "BiEliteCS66RT failed to connect"
        time.sleep(0.5)  # let the servo stream settle before first command
        home_pos, rot6 = _read_world_pose(robot, args.arm)
        print(f"[home] {args.arm} world pos = [{home_pos[0]:+.4f}, {home_pos[1]:+.4f}, {home_pos[2]:+.4f}]")
        print(f"[home] {args.arm} rot6 (locked) = [{', '.join(f'{v:+.3f}' for v in rot6)}]")

        for axis in axes:
            _run_axis_leg(
                robot, args.arm, axis, home_pos, rot6,
                args.distance, args.leg_duration, args.pause, args.rate,
            )

        print("\nAll legs done. Returning home / disconnecting...")
    finally:
        robot.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
