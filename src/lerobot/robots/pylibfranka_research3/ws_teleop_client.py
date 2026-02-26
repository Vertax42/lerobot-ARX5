"""
WebSocket teleoperation client for Franka robot (operator side).
Supports Cartesian (OSC) and Joint (impedance) control modes.

Cartesian mode gamepad mapping (Xbox layout):
  - Left stick X/Y: End-effector Y/X translation
  - Right stick Y: End-effector Z translation
  - Right stick X: End-effector Z rotation

Joint mode gamepad mapping:
  - Left stick X: Joint 7 rotation

Common buttons:
  - A button (or X): Exit
  - B button (or O): Reset to initial position

Usage:
    python ws_teleop_client.py --control-mode cartesian  # Cartesian end-effector control
    python ws_teleop_client.py --control-mode joint      # Joint space control
"""

import argparse
import atexit
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pygame
from scipy.spatial.transform import Rotation as R
from websockets.sync.client import connect

SERVER_SCRIPT = Path(__file__).parent / "ws_teleop_server.py"

# Cartesian control parameters
TRANSLATION_SPEED = 0.005  # m/step (maximum translation speed)
ROTATION_SPEED = 0.5       # deg/step (maximum rotation speed)

# Joint control parameters
JOINT7_SPEED = 0.01        # rad/step (Joint 7 rotation speed)

DEADZONE = 0.1             # Joystick deadzone


def launch_server(robot_ip: str, port: int, control_mode: str) -> subprocess.Popen:
    """Launch ws_teleop_server.py in the background on the local machine, returning the subprocess object."""
    cmd = [sys.executable, "-u", str(SERVER_SCRIPT),
           "--robot-ip", robot_ip, "--port", str(port),
           "--control-mode", control_mode]
    print(f"[Client] Launching server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # Automatically clean up the server on exit
    def cleanup():
        if proc.poll() is None:
            print("[Client] Shutting down server ...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    atexit.register(cleanup)

    # Wait for the server to be ready
    print("[Client] Waiting for server to be ready ...")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if line:
            print(f"  [Server] {line.rstrip()}")
            if "listening" in line.lower() or "waiting" in line.lower():
                print("[Client] Server is ready!")
                return proc
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited unexpectedly (code={proc.returncode})")
    raise TimeoutError("Server did not become ready within 60s")


def init_gamepad(control_mode: str) -> pygame.joystick.JoystickType:
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No joystick detected. Please connect a joystick and try again.")

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"[Client] Joystick connected: {joystick.get_name()}")
    print("[Client] Control instructions:")
    if control_mode == "cartesian":
        print("  Left stick: X/Y translation")
        print("  Right stick Y: Z translation")
        print("  Right stick X: Z rotation")
    else:
        print("  Left stick X: Joint 7 rotation")
    print("  A/X button: Exit")
    print("  B/O button: Reset position")
    return joystick


def apply_deadzone(value: float, deadzone: float) -> float:
    return 0.0 if abs(value) < deadzone else value


def read_joystick_axes(joystick: pygame.joystick.JoystickType, deadzone: float = DEADZONE):
    """Read raw joystick axes values (with deadzone), returning (lx, ly, rx, ry)."""
    pygame.event.pump()
    lx = apply_deadzone(joystick.get_axis(0), deadzone)
    ly = apply_deadzone(joystick.get_axis(1), deadzone)
    rx = apply_deadzone(joystick.get_axis(3), deadzone)
    ry = apply_deadzone(joystick.get_axis(4), deadzone)
    return lx, ly, rx, ry


def is_exit_pressed(joystick: pygame.joystick.JoystickType) -> bool:
    try:
        return joystick.get_button(0)
    except pygame.error:
        return False


def is_reset_pressed(joystick: pygame.joystick.JoystickType) -> bool:
    try:
        return joystick.get_button(1)
    except pygame.error:
        return False


# ─── Cartesian mode reading ─────────────────────────────────────────

def read_cartesian_delta(joystick, deadzone=DEADZONE) -> dict:
    """Cartesian delta mode."""
    lx, ly, rx, ry = read_joystick_axes(joystick, deadzone)
    if is_exit_pressed(joystick):
        return {"type": "stop"}
    if is_reset_pressed(joystick):
        return {"type": "reset"}
    return {
        "type": "cartesian_delta",
        "translation": [ly * TRANSLATION_SPEED, lx * TRANSLATION_SPEED, -ry * TRANSLATION_SPEED],
        "rotation_z": -rx * ROTATION_SPEED,
    }


def read_cartesian_absolute(joystick, ee_target: np.ndarray, deadzone=DEADZONE) -> dict:
    """Cartesian absolute mode, modifies ee_target in place."""
    lx, ly, rx, ry = read_joystick_axes(joystick, deadzone)
    if is_exit_pressed(joystick):
        return {"type": "stop"}
    if is_reset_pressed(joystick):
        return {"type": "reset"}

    ee_target[:3, 3] += np.array([ly * TRANSLATION_SPEED, lx * TRANSLATION_SPEED, -ry * TRANSLATION_SPEED])
    rot = R.from_euler('z', -rx * ROTATION_SPEED, degrees=True).as_matrix()
    ee_target[:3, :3] = rot @ ee_target[:3, :3]

    return {"type": "cartesian_absolute", "pose": ee_target.tolist()}


# ─── Joint mode reading ───────────────────────────────────────────

def read_joint_delta(joystick, deadzone=DEADZONE) -> dict:
    """Joint delta mode (only joint 7)."""
    lx, ly, rx, ry = read_joystick_axes(joystick, deadzone)
    if is_exit_pressed(joystick):
        return {"type": "stop"}
    if is_reset_pressed(joystick):
        return {"type": "reset"}
    return {
        "type": "joint_delta",
        "joint_deltas": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lx * JOINT7_SPEED],
    }


# ─── State printing ───────────────────────────────────────────────

def print_state(state: dict):
    """Print state based on control mode."""
    mode = state.get("control_mode", "unknown")
    if mode == "cartesian":
        ee = np.array(state.get("O_T_EE", []))
        wrench = state.get("ext_wrench", [])
        if ee.size == 16:
            pos = ee.reshape(4, 4).T[:3, 3]
            print(f"  End-effector position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]  |  External wrench: {np.round(wrench, 2)}")
    elif mode == "joint":
        q = state.get("q", [])
        q_desired = state.get("q_desired", [])
        if len(q) == 7 and len(q_desired) == 7:
            print(f"  q7: {q[6]:.4f}  |  q7_desired: {q_desired[6]:.4f}  |  "
                  f"Error: {abs(q[6] - q_desired[6]):.4f} rad")
    else:
        print(f"  State: {state}")


def main():
    parser = argparse.ArgumentParser(description="WebSocket Teleoperation Client (Operator Side)")
    parser.add_argument("--freq", type=float, default=50.0, help="Command frequency Hz (default: 50)")
    parser.add_argument("--deadzone", type=float, default=DEADZONE, help="Joystick deadzone (default: 0.1)")
    parser.add_argument("--robot-ip", default="192.168.99.111", help="Franka Robot IP (default: 192.168.99.111)")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port (default: 8765)")
    parser.add_argument("--control-mode", choices=["cartesian", "joint"], default="cartesian",
                        help="Control mode: cartesian or joint (default: cartesian)")
    parser.add_argument("--send-mode", choices=["delta", "absolute"], default="absolute",
                        help="Send mode: delta (incremental) or absolute (default: absolute)")
    args = parser.parse_args()

    # Automatically launch server (pass control-mode)
    server_proc = launch_server(args.robot_ip, args.port, args.control_mode)
    server_uri = f"ws://localhost:{args.port}"

    period = 1.0 / args.freq
    joystick = init_gamepad(args.control_mode)
    ee_target = None  # cartesian absolute mode

    print(f"[Client] Connecting to server: {server_uri} ...")
    with connect(server_uri) as ws:
        print(f"[Client] Connected! Control: {args.control_mode}, Send: {args.send_mode}. A to exit, B to reset.")

        # cartesian absolute mode requires initial pose
        if args.control_mode == "cartesian" and args.send_mode == "absolute":
            ws.send(json.dumps({"type": "cartesian_delta", "translation": [0, 0, 0], "rotation_z": 0.0}))
            init_state = json.loads(ws.recv(timeout=5.0))
            ee_target = np.array(init_state["ee_desired"], dtype=np.float64).reshape(4, 4)
            print("[Client] Initial pose received, starting absolute control.")

        while True:
            t0 = time.monotonic()

            # Read joystick based on mode
            if args.control_mode == "cartesian":
                if args.send_mode == "absolute":
                    cmd = read_cartesian_absolute(joystick, ee_target, args.deadzone)
                    if cmd["type"] == "cartesian_absolute":
                        pos = np.array(cmd['pose']).reshape(4, 4)[:3, 3]
                        print(f"[Client] Target position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
                else:
                    cmd = read_cartesian_delta(joystick, args.deadzone)
            else:  # joint
                cmd = read_joint_delta(joystick, args.deadzone)

            ws.send(json.dumps(cmd))

            if cmd.get("type") == "stop":
                print("[Client] Stop button detected, stopping...")
                break

            if cmd.get("type") == "reset":
                print("[Client] Resetting to initial position...")

            # Receive server state
            try:
                response = ws.recv(timeout=1.0)
                state = json.loads(response)
                print_state(state)
                # cartesian absolute mode requires syncing ee_target after reset
                if (cmd.get("type") == "reset" and args.control_mode == "cartesian"
                        and args.send_mode == "absolute" and "ee_desired" in state):
                    ee_target = np.array(state["ee_desired"], dtype=np.float64).reshape(4, 4)
            except TimeoutError:
                print("[Client] Warning: Server response timeout")

            # Control send frequency
            elapsed = time.monotonic() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    pygame.quit()
    print("[Client] Done!")


if __name__ == "__main__":
    main()
