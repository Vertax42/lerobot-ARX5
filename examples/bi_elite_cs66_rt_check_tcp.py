#!/usr/bin/env python3
"""Check whether an Elite CS66 controller has an active TCP offset.

`RtsiIOInterface.getActualTCPPose()` returns the pose of the *active TCP*
(flange composed with the controller's active TCP offset). `KdlKinematicsPlugin.
getPositionFK(joints)` returns the *flange* pose (the MDH chain ends at the
flange and does NOT include any tool offset). Therefore:

    actual_TCP_pose  ==  FK(joints)   ->  NO tool offset active (TCP == flange)
    actual_TCP_pose  !=  FK(joints)   ->  a tool offset IS active; the delta is it

Run this with the arm IDLE (no teleop / no EliteDriver control script holding the
ports) to read the installation/default TCP the controller applies at rest.

Usage:
    python examples/bi_elite_cs66_rt_check_tcp.py --ip 192.168.8.53
"""

import argparse
import math
import platform
import sys
from pathlib import Path

import elite_cs_sdk as cs

try:
    import elite_cs_sdk.elite_cs_sdk_python as cs_native
except Exception:
    cs_native = cs


def _resolve_plugin_path() -> Path:
    module_dir = Path(cs_native.__file__).resolve().parent
    name = {
        "Linux": "libelite_kdl_kinematics.so",
        "Windows": "elite_kdl_kinematics.dll",
        "Darwin": "libelite_kdl_kinematics.dylib",
    }.get(platform.system())
    if name and (module_dir / name).exists():
        return module_dir / name
    for pat in ("*kdl*kinematics*.so", "*kdl*kinematics*.dll", "*kdl*kinematics*.dylib"):
        hits = list(module_dir.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"kdl kinematics plugin not found under {module_dir}")


def _fmt(v):
    return "[" + ", ".join(f"{float(x):+.5f}" for x in v) + "]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--rtsi-frequency", type=float, default=250.0)
    args = ap.parse_args()

    # examples/resource recipes ship with the SDK and include actual_joint_positions
    # + actual_TCP_pose.
    res = Path(cs.__file__).resolve().parent  # placeholder; overridden below
    sdk_examples = Path(__file__).resolve()
    # Prefer the SDK's own example recipes (known-good, include both fields).
    candidates = [
        Path.home() / "Elite_Robots_CS_SDK_Python/examples/resource",
        sdk_examples.parent / "resource",
    ]
    resource_dir = next((c for c in candidates if (c / "output_recipe.txt").exists()), None)
    if resource_dir is None:
        print(f"[ERROR] no RTSI recipe dir found in {candidates}", file=sys.stderr)
        return 1
    output_recipe = str(resource_dir / "output_recipe.txt")
    input_recipe = str(resource_dir / "input_recipe.txt")

    print(f"[INFO] {args.ip}: reading MDH (primary) ...")
    primary = cs.PrimaryClientInterface()
    if not primary.connect(args.ip):
        print(f"[ERROR] primary connect failed at {args.ip}", file=sys.stderr)
        return 1
    kin = cs.KinematicsInfo()
    try:
        if not primary.getPackage(kin, 1000):
            print("[ERROR] getPackage(KinematicsInfo) failed", file=sys.stderr)
            return 1
    finally:
        primary.disconnect()

    print(f"[INFO] {args.ip}: reading joints + TCP (rtsi) ...")
    io = cs.RtsiIOInterface(output_recipe, input_recipe, args.rtsi_frequency)
    if not io.connect(args.ip):
        print(f"[ERROR] rtsi connect failed at {args.ip}:30004", file=sys.stderr)
        return 1
    try:
        joints = io.getActualJointPositions()
        tcp = list(io.getActualTCPPose())
    finally:
        io.disconnect()

    plugin = _resolve_plugin_path()
    loader = cs_native.ClassLoader(str(plugin))
    if not loader.loadLib():
        print("[ERROR] loadLib failed", file=sys.stderr)
        return 1
    solver = loader.createKinematicsInstance("ELITE::KdlKinematicsPlugin")
    solver.setMDH(kin.dh_alpha_, kin.dh_a_, kin.dh_d_)
    ok, flange = solver.getPositionFK(joints)
    if not ok:
        print("[ERROR] FK failed", file=sys.stderr)
        return 1
    flange = list(flange)

    dpos = [tcp[i] - flange[i] for i in range(3)]
    dpos_mm = math.sqrt(sum(d * d for d in dpos)) * 1000.0

    print("\n================= TCP CHECK =================")
    print(f"  active TCP  (getActualTCPPose): {_fmt(tcp)}")
    print(f"  flange      (FK of joints)    : {_fmt(flange)}")
    print(f"  delta pos (TCP-flange, base)  : {_fmt(dpos)}  |  |delta| = {dpos_mm:.2f} mm")
    print("============================================")
    if dpos_mm < 1.0:
        print(">>> NO tool offset active: getActualTCPPose == flange (within 1 mm).")
        print(">>> The pendant TCP is NOT being applied (define it AND set it as the")
        print(">>> active/default TCP, or the control script resets it).")
    else:
        print(f">>> A tool offset IS active (~{dpos_mm:.1f} mm from the flange).")
        print(">>> If teleop still behaves like flange, the headless control script")
        print(">>> resets the TCP during external control -> use an in-code offset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
