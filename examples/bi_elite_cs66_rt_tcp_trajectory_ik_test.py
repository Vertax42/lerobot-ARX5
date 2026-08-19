#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Continuous-TCP-trajectory IK stress test for ONE Elite CS66 arm (via BiEliteCS66RT).

Streams a smooth, continuous world-frame TCP trajectory through
``send_action`` (= Cartesian servoj = the controller's internal IK) and logs,
per tick, the MEASURED joint velocity, the manipulability ``w`` and the
requested-vs-achieved tracking error — so you can watch how the end-effector IK
behaves, and whether the joint velocity spikes toward the controller's
``JOINT_IGNORE_SPEED = 30 rad/s`` trip near a wrist singularity.

Modes (``--mode``):
  * ``wrist`` — position held; the TCP orientation swings sinusoidally about a
    tool axis with the amplitude ramping up, driving the wrist toward its
    singularity (q5 → 0). This is THE reproduction of the "rotation IK / speed
    trip" failure. Watch ``w`` fall and |qdot| rise.
  * ``circle`` — orientation held; the TCP traces a position circle through the
    start point (well-conditioned baseline; IK should stay smooth).
  * ``combo`` — circle position + orientation swing together (teleop-like).

Every mode starts and ends exactly at the current pose (the circle passes
through the start point; the swing has zero amplitude at t=0 and returns to
zero after an integer number of periods), so there is no start/stop jump.

A/B the new guard:
  * default            → guard OFF (baseline): reproduce the raw IK behaviour.
  * ``--joint-vel-guard`` → enables the predicted joint-velocity limit scaling
    with the official CS66 datasheet limits; re-run the same trajectory and
    compare max|qdot|, min w, tracking error, and whether a trip occurs.

Safety / behaviour:
  * THIS MOVES THE ROBOT. Refuses to move unless you type ``yes`` (or ``--yes``).
    ``--dry-run`` prints the plan without touching hardware.
  * Only the tested arm is commanded; the other arm holds its pose.
  * Cameras / grippers disabled (pure motion test).
  * If the background servo loop dies (e.g. the controller's "External Control
    speed limit" trip), ``send_action`` raises — the script catches it, reports
    the time / w / qdot at the trip, and returns home.

Run on the station with a hand on the e-stop:
    python examples/bi_elite_cs66_rt_tcp_trajectory_ik_test.py --dry-run
    python examples/bi_elite_cs66_rt_tcp_trajectory_ik_test.py --mode wrist            # baseline
    python examples/bi_elite_cs66_rt_tcp_trajectory_ik_test.py --mode wrist --joint-vel-guard
"""

import argparse
import contextlib
import math
import time
from pathlib import Path

import numpy as np

from lerobot.robots.bi_elite_cs66_rt.bi_elite_cs66_rt import BiEliteCS66RT
from lerobot.robots.bi_elite_cs66_rt.config_bi_elite_cs66_rt import BiEliteCS66RTConfig
from lerobot.robots.elite_cs66_rt.manipulability import manipulability
from lerobot.utils.robot_utils import quaternion_to_rotation_6d, rotation_6d_to_quaternion
from lerobot.utils.rotation import Rotation

# Official Elite CS66 datasheet per-joint speed ceilings (rad/s):
# J1/J2 150°/s, J3 180°/s, J4-6 230°/s.
CS66_QDOT_LIMIT = np.array([2.618, 2.618, 3.142, 4.014, 4.014, 4.014], dtype=np.float64)

_AXIS_VEC = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
_PLANE_AXES = {  # (u, v): circle spanned by these two tool/world axes
    "xy": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    "xz": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    "yz": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
}


def _rot6_to_matrix(rot6) -> np.ndarray:
    q = rotation_6d_to_quaternion(np.asarray(rot6, dtype=np.float64))  # wxyz
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _matrix_to_rot6(rot_mat: np.ndarray) -> list[float]:
    q = Rotation.from_matrix(rot_mat).as_quat()  # xyzw
    return list(quaternion_to_rotation_6d(float(q[3]), float(q[0]), float(q[1]), float(q[2])))


def _read_world_pose(robot: BiEliteCS66RT, side: str) -> tuple[np.ndarray, list[float]]:
    obs = robot.get_observation()
    pos = np.array([obs[f"{side}_tcp.x"], obs[f"{side}_tcp.y"], obs[f"{side}_tcp.z"]], dtype=np.float64)
    rot6 = [float(obs[f"{side}_tcp.r{i + 1}"]) for i in range(6)]
    return pos, rot6


def _make_action(side: str, pos: np.ndarray, rot6) -> dict:
    action = {f"{side}_tcp.x": float(pos[0]), f"{side}_tcp.y": float(pos[1]), f"{side}_tcp.z": float(pos[2])}
    action.update({f"{side}_tcp.r{i + 1}": float(rot6[i]) for i in range(6)})
    return action


def _amplitude(t: float, duration: float) -> float:
    """Ramp 0 -> 1 over the first 40% of the run, then hold at 1 (grow the swing/probe gradually)."""
    return min(1.0, t / (0.4 * duration)) if duration > 0 else 1.0


def _trajectory_pose(mode, t, params, pos0, rot0_mat):
    """Absolute world (pos[3], rot6[6]) at time ``t``. Starts at (pos0, rot0) at t=0."""
    period = params["period"]
    phi = 2.0 * math.pi * t / period
    pos = pos0.copy()
    rot_mat = rot0_mat

    if mode in ("circle", "combo"):
        u, v = _PLANE_AXES[params["plane"]]
        r = params["radius"]
        # (cos-1, sin) so the circle passes through pos0 at t=0 (no jump).
        pos = pos0 + r * ((math.cos(phi) - 1.0) * u + math.sin(phi) * v)

    if mode in ("wrist", "combo"):
        amp = _amplitude(t, params["duration"]) * math.radians(params["max_rot_deg"])
        theta = amp * math.sin(phi)
        # Tool-frame rotation about the chosen axis: R = R0 @ Raxis(theta).
        r_tool = Rotation.from_rotvec(_AXIS_VEC[params["wrist_axis"]] * theta).as_matrix()
        rot_mat = rot0_mat @ r_tool

    return pos, _matrix_to_rot6(rot_mat)


def _pose_err(req_pos, req_rot6, ach_pos, ach_rot6):
    dpos_mm = float(np.linalg.norm(req_pos - ach_pos)) * 1000.0
    r_req = _rot6_to_matrix(req_rot6)
    r_ach = _rot6_to_matrix(ach_rot6)
    ang = float(np.degrees(np.linalg.norm(Rotation.from_matrix(r_req.T @ r_ach).as_rotvec())))
    return dpos_mm, ang


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("left", "right"), default="left")
    ap.add_argument(
        "--recipe",
        default="recipes/teleop/bi_elite_cs66_rt/diagonal-08-xgripper.yaml",
        help="recipe YAML supplying the bench hardware (IPs, poses, mount rotation)",
    )
    ap.add_argument(
        "--mode",
        choices=("wrist", "circle", "combo", "hold"),
        default="wrist",
        help="hold = keep the start pose fixed so you can hand-push the TCP to feel compliance",
    )
    ap.add_argument(
        "--servoj-gain",
        type=int,
        default=None,
        help="override servoj position-following gain [100-2000]; lower = softer/more compliant "
        "(recipe default 300). Try 150 to feel a softer arm.",
    )
    ap.add_argument("--duration", type=float, default=20.0, help="total streaming seconds (snapped to whole periods)")
    ap.add_argument("--period", type=float, default=4.0, help="seconds per oscillation / circle revolution")
    ap.add_argument("--rate", type=float, default=50.0, help="send_action stream rate (Hz)")
    # wrist / combo
    ap.add_argument("--max-rot-deg", type=float, default=90.0, help="peak orientation swing amplitude (deg)")
    ap.add_argument("--wrist-axis", choices=("x", "y", "z"), default="y", help="tool axis to swing about")
    # circle / combo
    ap.add_argument("--radius", type=float, default=0.08, help="position circle radius (m)")
    ap.add_argument("--plane", choices=("xy", "xz", "yz"), default="xz", help="world plane for the circle")
    # guard A/B
    ap.add_argument("--joint-vel-guard", action="store_true", help="enable predicted joint-velocity limit scaling")
    ap.add_argument("--margin", type=float, default=0.8, help="joint_vel_limit_margin (fraction of datasheet limits)")
    ap.add_argument(
        "--jv-lambda",
        type=float,
        default=1e-2,
        help="joint_vel_dls_lambda; smaller -> predicts the near-singularity spike harder/earlier",
    )
    ap.add_argument("--jv-horizon", type=float, default=0.033, help="joint_vel_horizon_s (~1/fps); shorter -> tighter")
    # safety
    ap.add_argument("--warn-qdot", type=float, default=2.0, help="flag ticks whose max|qdot| exceeds this (rad/s)")
    ap.add_argument("--no-go-to-start", action="store_true", help="do not MoveJ to start on connect")
    ap.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; do not connect or move")
    args = ap.parse_args()

    # Snap duration to a whole number of periods so the trajectory ends exactly at the start pose.
    n_periods = max(1, round(args.duration / args.period))
    duration = n_periods * args.period
    params = {
        "period": args.period,
        "duration": duration,
        "radius": args.radius,
        "plane": args.plane,
        "max_rot_deg": args.max_rot_deg,
        "wrist_axis": args.wrist_axis,
    }

    print("Continuous-TCP-trajectory IK test")
    print(f"  arm / mount   : {args.arm} / {args.mount_type}")
    print(f"  mode          : {args.mode}")
    print(
        f"  servoj_gain   : {args.servoj_gain if args.servoj_gain is not None else '300 (recipe default)'}"
        + ("   (lower = softer)" if args.servoj_gain is not None else "")
    )
    if args.mode == "hold":
        print(
            "  >> HOLD: arm holds the start pose. Push the TCP by hand to feel compliance;\n"
            "     the 'track' readout = how far it yields (mm / deg). Push GENTLY."
        )
    print(f"  timing        : {duration:.1f}s = {n_periods}×{args.period:.1f}s period, {args.rate:.0f} Hz")
    if args.mode in ("wrist", "combo"):
        print(f"  orientation   : swing ±{args.max_rot_deg:.0f}° about tool-{args.wrist_axis} (amplitude ramps up)")
    if args.mode in ("circle", "combo"):
        print(f"  position      : {args.radius * 100:.0f} cm circle in world {args.plane} plane")
    print(
        f"  joint-vel guard: {'ON' if args.joint_vel_guard else 'OFF (baseline)'}"
        + (
            f"  ×margin {args.margin}, λ={args.jv_lambda:g}, horizon={args.jv_horizon:g}s"
            if args.joint_vel_guard
            else ""
        )
    )

    if args.dry_run:
        print("\n[dry-run] not connecting / not moving. Plan above.")
        return

    if not args.yes:
        print("\n⚠️  THIS WILL MOVE THE ROBOT. Keep a hand on the e-stop.")
        if input("Type 'yes' to proceed: ").strip().lower() != "yes":
            print("Aborted.")
            return

    # Bench hardware comes from the recipe (recipes are self-contained).
    import draccus
    import yaml

    raw = yaml.safe_load(Path(args.recipe).read_text())["robot"]
    raw.pop("type", None)
    if args.servoj_gain is not None:
        raw["servoj_gain"] = args.servoj_gain  # validated to [100, 2000] in __post_init__
    cfg = draccus.decode(BiEliteCS66RTConfig, raw)
    cfg.cameras = {}
    cfg.left_gripper = None
    cfg.right_gripper = None
    cfg.enable_tactile_sensors = False
    if args.joint_vel_guard:
        cfg.joint_vel_limits_rad_s = CS66_QDOT_LIMIT.tolist()
        cfg.joint_vel_limit_margin = args.margin
        cfg.joint_vel_dls_lambda = args.jv_lambda
        cfg.joint_vel_horizon_s = args.jv_horizon

    side = args.arm
    robot = BiEliteCS66RT(cfg)
    print(
        "\nConnecting (powers on both arms; "
        f"{'NOT moving to start' if args.no_go_to_start else 'MoveJ both to start'})..."
    )
    robot.connect(go_to_start=not args.no_go_to_start)

    # Capture controller-side errors/warnings host-side (e.g. "External Control speed limit",
    # protective stops) so we don't have to watch the teach pendant. The callback fires from an SDK
    # thread; each event is printed (so a Monitor catches it live) and counted for the summary.
    ctrl_events: list[str] = []

    def _on_ctrl_exception(exc):
        parts = []
        for attr in ("getErrorCode", "getSubErrorCode", "getErrorLevel", "getErrorSource", "getMessage"):
            fn = getattr(exc, attr, None)
            if fn is not None:
                # Each accessor is optional and may itself throw; a field we
                # cannot read just does not appear in the message.
                with contextlib.suppress(Exception):
                    parts.append(f"{attr[3:]}={fn()}")
        msg = "  ".join(parts) or str(exc)
        ctrl_events.append(msg)
        print(f"[CTRL-EVENT] {msg}", flush=True)

    try:
        robot._driver[side].registerRobotExceptionCallback(_on_ctrl_exception)
        print("[ctrl] robot-exception callback registered — controller errors/warnings logged host-side.")
    except Exception as e:
        print(f"[ctrl] could not register exception callback: {e}")

    tripped = False
    max_qdot_all = 0.0
    min_w = math.inf
    worst_track = (0.0, 0.0)
    try:
        assert robot.is_connected, "BiEliteCS66RT failed to connect"
        time.sleep(0.5)  # let the servo stream settle
        dh = None
        try:
            dh = robot._fetch_dh(side)  # one-shot; reused for the manipulability readout below
        except Exception:
            dh = None
        print(f"[dh] {'fetched — manipulability w will be reported' if dh else 'unavailable — w shown as n/a'}")
        if args.joint_vel_guard and not robot._damping_enabled[side]:
            print(
                f"\n[ABORT] --joint-vel-guard requested but the guard is NOT active on '{side}'\n"
                "        (connect-time DH self-check failed — see ~/xenselogs). This usually means\n"
                "        the arm was sitting on/near a singularity when connect() sampled its pose\n"
                "        (e.g. frozen there by a previous trip), making the TCP rotvec reading\n"
                "        unreliable. Refusing to drive toward the singularity UNPROTECTED.\n"
                "        Fix: bring the arm to a clean (non-singular) pose first, then re-run."
            )
            return  # finally: disconnect() returns home

        pos0, rot6_0 = _read_world_pose(robot, side)
        rot0_mat = _rot6_to_matrix(rot6_0)
        print(f"[home] {side} world pos = [{pos0[0]:+.4f}, {pos0[1]:+.4f}, {pos0[2]:+.4f}]")

        dt = 1.0 / args.rate
        n_ticks = int(round(duration * args.rate))
        next_tick = time.monotonic()
        last_log = 0.0
        for k in range(1, n_ticks + 1):
            t = k * dt
            req_pos, req_rot6 = _trajectory_pose(args.mode, t, params, pos0, rot0_mat)
            try:
                robot._send_arm_action(side, _make_action(side, req_pos, req_rot6), {})
            except Exception as exc:  # servo loop died -> controller speed-limit trip most likely
                tripped = True
                q = np.asarray(robot._rtsi[side].getActualJointVelocity(), dtype=np.float64)
                print(f"\n[TRIP] t={t:5.2f}s send_action raised: {type(exc).__name__}: {str(exc)[:90]}")
                print(f"       max|qdot| at trip = {np.max(np.abs(q)):.2f} rad/s")
                break

            # Measure: real joint velocity (the honest signal) + manipulability + tracking error.
            q = np.asarray(robot._rtsi[side].getActualJointPositions(), dtype=np.float64)
            qdot = np.asarray(robot._rtsi[side].getActualJointVelocity(), dtype=np.float64)
            max_qdot = float(np.max(np.abs(qdot)))
            j_hot = int(np.argmax(np.abs(qdot))) + 1
            max_qdot_all = max(max_qdot_all, max_qdot)
            w = manipulability(*dh, q) if dh is not None else float("nan")
            if not math.isnan(w):
                min_w = min(min_w, w)
            ach_pos, ach_rot6 = _read_world_pose(robot, side)
            track = _pose_err(req_pos, req_rot6, ach_pos, ach_rot6)
            if track[0] + track[1] > worst_track[0] + worst_track[1]:
                worst_track = track

            if t - last_log >= 0.25:  # compact status ~4 Hz
                last_log = t
                flag = "  <-- HIGH qdot" if max_qdot > args.warn_qdot else ""
                wtxt = f"{w:7.4f}" if not math.isnan(w) else "   n/a "
                print(
                    f"t={t:5.2f}s q5={math.degrees(q[4]):+6.1f}°  w={wtxt}  "
                    f"max|qdot|={max_qdot:5.2f} rad/s (J{j_hot})  "
                    f"track=({track[0]:5.1f}mm,{track[1]:4.1f}°){flag}"
                )

            next_tick += dt
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

        print("\n===== summary =====")
        print(f"  mode / guard      : {args.mode} / {'ON' if args.joint_vel_guard else 'OFF'}")
        print(f"  tripped (hard)    : {'YES (controller dropped external control)' if tripped else 'no'}")
        print(
            f"  controller events : {len(ctrl_events)}"
            + ("   <-- pendant-style errors/warnings caught host-side!" if ctrl_events else " (none)")
        )
        for m in ctrl_events[:8]:
            print(f"      - {m}")
        print(
            f"  max |qdot| seen   : {max_qdot_all:.2f} rad/s  "
            f"(datasheet min limit {CS66_QDOT_LIMIT.min():.2f}, controller trip 30.0)"
        )
        print(f"  min w (manip.)    : {'n/a' if math.isinf(min_w) else f'{min_w:.4f}'}")
        print(
            f"  worst tracking err: {worst_track[0]:.1f} mm / {worst_track[1]:.1f}°"
            + ("   (large err near low w = guard holding / IK struggling)" if worst_track[1] > 3 else "")
        )
    finally:
        print("\nReturning home / disconnecting...")
        robot.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
