"""
WebSocket teleoperation server for Franka Research 3 using pylibfranka.
Supports Cartesian (OSC) and Joint (impedance) control modes.

Cartesian mode: τ = J^T @ (-K @ error - D @ (J @ dq)) + coriolis
Joint mode:   τ = -Kp*(q - q_d) - Kd*dq + coriolis

Usage:
    python ws_teleop_server.py --control-mode cartesian  # Cartesian end-effector control
    python ws_teleop_server.py --control-mode joint      # Joint space control
"""

import asyncio
import argparse
import json
import time

import numpy as np
import websockets
from scipy.spatial.transform import Rotation as R

from xense_franka.robot import RobotInterface
from xense_franka import FrankaController


# 初始关节位置 (安全位置)
HOME_JOINTS = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853]

# 关节限位 (Franka FR3, rad)
JOINT_LIMITS_MIN = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.017, -2.89])
JOINT_LIMITS_MAX = np.array([2.89,  1.76,  2.89, -0.07,  2.89,  3.75,   2.89])


class TeleopServer:
    """WebSocket teleoperation server for Franka Research 3 using pylibfranka.
    Supports Cartesian (OSC) and Joint (impedance) control modes.
    """

    def __init__(self, robot_ip: str, control_mode: str = "cartesian",
                 host: str = "0.0.0.0", port: int = 8765,
                 home_joints: list = None, freq: int = 50):
        self.robot_ip = robot_ip
        self.control_mode = control_mode  # "cartesian" or "joint"
        self.host = host
        self.port = port
        self.home_joints = home_joints if home_joints is not None else list(HOME_JOINTS)
        self.freq = freq

        self.robot: RobotInterface = None
        self.controller: FrankaController = None
        self.client_connected = False

        # Initial state for reset
        self.initial_ee: np.ndarray = None
        self.initial_q: np.ndarray = None

    # ─── Robot Initialization ────────────────────────────────────────
    async def init_robot(self):
        """Initialize the robot and start the controller, switching controllers based on the mode."""
        print(f"[Server] Connecting to robot at {self.robot_ip} ...")
        self.robot = RobotInterface(self.robot_ip)
        self.controller = FrankaController(self.robot)

        await self.controller.start()

        # Move to initial pose
        await self.controller.move(self.home_joints)
        await asyncio.sleep(1.0)

        if self.control_mode == "cartesian":
            self._init_cartesian()
        else:
            self._init_joint()

        print(f"[Server] Robot ready (mode={self.control_mode}), waiting for client ...")

    def _init_cartesian(self):
        """Switch to Cartesian impedance control (OSC)."""
        self.controller.switch("osc")
        self.controller.ee_kp = np.array([600.0, 600.0, 600.0, 50.0, 50.0, 50.0])
        self.controller.ee_kd = 2.0 * np.sqrt(self.controller.ee_kp)  # Critical damping
        self.controller.set_freq(self.freq)

        self.initial_ee = self.controller.ee_desired.copy()

        print("[Server] Cartesian impedance control started")
        print(f"[Server] Stiffness: {self.controller.ee_kp}")
        print(f"[Server] Damping: {self.controller.ee_kd}")
        print(f"[Server] Initial position: {self.initial_ee[:3, 3]}")

    def _init_joint(self):
        """Switch to joint impedance control (impedance)."""
        self.controller.switch("impedance")
        self.controller.kp = np.ones(7) * 80.0
        self.controller.kd = np.ones(7) * 4.0
        self.controller.set_freq(self.freq)

        self.initial_q = self.controller.q_desired.copy()

        print("[Server] Joint impedance control started")
        print(f"[Server] Stiffness: {self.controller.kp}")
        print(f"[Server] Damping: {self.controller.kd}")
        print(f"[Server] Initial joint positions: {np.round(self.initial_q, 4)}")

    # ─── Handle Single Command ───────────────────────────────────────
    async def apply_command(self, cmd: dict):
        """Apply a command from the client to the robot.

        General commands:
          {"type": "stop"}
          {"type": "reset"}

        Cartesian mode commands:
          {"type": "cartesian_delta", "translation": [dx,dy,dz], "rotation_z": float}
          {"type": "cartesian_absolute", "pose": [[4x4 matrix]]}

        Joint mode commands:
          {"type": "joint_delta", "joint_deltas": [7 floats]}
          {"type": "joint_absolute", "q_desired": [7 floats]}
        """
        cmd_type = cmd.get("type")

        if cmd_type in ("get_state", "ee_desired"):
            # Only return state, do not perform any action
            return

        if cmd_type == "move_home":
            positions = cmd.get("positions", self.home_joints)
            print(f"[Server] Moving to specified joint positions: {np.round(positions, 4)}")
            await self.controller.move(positions)
            await asyncio.sleep(1.0)
            # Reinitialize control mode
            if self.control_mode == "cartesian":
                self._init_cartesian()
            else:
                self._init_joint()
            return

        if cmd_type == "configure":
            if "ee_kp" in cmd:
                self.controller.ee_kp = np.array(cmd["ee_kp"])
                self.controller.ee_kd = 2.0 * np.sqrt(self.controller.ee_kp)
            if "kp" in cmd:
                self.controller.kp = np.array(cmd["kp"])
            if "kd" in cmd:
                self.controller.kd = np.array(cmd["kd"])
            if "freq" in cmd:
                self.controller.set_freq(cmd["freq"])
            if "switch_mode" in cmd:
                mode = cmd["switch_mode"]
                if mode == "osc":
                    self._init_cartesian()
                elif mode == "impedance":
                    self._init_joint()
            print("[Server] Configuration update completed")
            return

        if cmd_type == "stop":
            print("[Server] Stop command received, stopping...")
            return

        if cmd_type == "reset":
            if self.control_mode == "cartesian":
                print("[Server] Resetting to initial end-effector pose...")
                await self.controller.set("ee_desired", self.initial_ee.copy())
            else:
                print("[Server] Resetting to initial joint positions...")
                await self.controller.set("q_desired", self.initial_q.copy())
            await asyncio.sleep(0.5)
            return

        # ── Cartesian commands ──
        if cmd_type == "cartesian_absolute":
            pose = np.array(cmd["pose"], dtype=np.float64).reshape(4, 4)
            await self.controller.set("ee_desired", pose)
            return

        if cmd_type == "cartesian_delta":
            translation_delta = np.array(cmd["translation"], dtype=np.float64)
            rotation_z = float(cmd.get("rotation_z", 0.0))
            rotation_delta = R.from_euler("z", rotation_z, degrees=True).as_matrix()

            with self.controller.state_lock:
                current_ee = self.controller.ee_desired.copy()

            current_ee[:3, 3] += translation_delta
            current_ee[:3, :3] = rotation_delta @ current_ee[:3, :3]

            await self.controller.set("ee_desired", current_ee)
            return

        # ── Joint commands ──
        if cmd_type == "joint_absolute":
            q = np.array(cmd["q_desired"], dtype=np.float64)
            q = np.clip(q, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX)
            await self.controller.set("q_desired", q)
            return

        if cmd_type == "joint_delta":
            joint_deltas = np.array(cmd["joint_deltas"], dtype=np.float64)

            with self.controller.state_lock:
                q_target = self.controller.q_desired.copy()

            q_target += joint_deltas
            q_target = np.clip(q_target, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX)

            await self.controller.set("q_desired", q_target)
            return

    # ─── Get robot state ─────────────────────────────────────
    def get_robot_state(self) -> dict:
        """Collect the current robot state and serialize it into a JSON-compatible dictionary."""
        state = self.robot.state

        result = {
            "control_mode": self.control_mode,
            "q": state["qpos"].tolist(),
            "dq": state["qvel"].tolist(),
            "tau": state["last_torque"].tolist(),
            "ext_wrench": state["ext_wrench"].tolist(),
            "timestamp": time.time(),
            "ee": state["ee"].tolist(),
        }

        if self.control_mode == "cartesian":
            with self.controller.state_lock:
                ee_desired = self.controller.ee_desired.copy()
            result["O_T_EE"] = state["O_T_EE"].tolist() # 16d
            result["ee_desired"] = ee_desired.tolist() # 4x4 matrix
        else:
            with self.controller.state_lock:
                q_desired = self.controller.q_desired.copy()
            result["q_desired"] = q_desired.tolist()

        return result

    # ─── WebSocket connection handling ─────────────────────────────────
    async def handler(self, websocket):
        """Handle a single WebSocket client connection."""
        remote = websocket.remote_address
        print(f"[Server] Client connected: {remote}")
        self.client_connected = True

        try:
            async for message in websocket:
                try:
                    cmd = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"error": "invalid JSON"}))
                    continue

                await self.apply_command(cmd)

                robot_state = self.get_robot_state()
                await websocket.send(json.dumps(robot_state))

        except websockets.exceptions.ConnectionClosed as e:
            print(f"[Server] Client disconnected: {remote} ({e})")
        finally:
            self.client_connected = False

    # ─── Start ──────────────────────────────────────────────
    async def run(self):
        await self.init_robot()

        async with websockets.serve(self.handler, self.host, self.port):
            print(f"[Server] WebSocket server listening on ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever


def main():
    parser = argparse.ArgumentParser(description="WebSocket Teleoperation Server (Robot Side)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    parser.add_argument("--robot-ip", default="192.168.99.111", help="Franka Robot IP")
    parser.add_argument("--control-mode", choices=["cartesian", "joint"], default="cartesian",
                        help="Control mode: cartesian or joint (default: cartesian)")
    parser.add_argument("--freq", type=int, default=50, help="Control frequency Hz (default: 50)")
    parser.add_argument("--home", nargs=7, type=float, default=None,
                        help="Initial joint positions (7 floats)")
    args = parser.parse_args()

    server = TeleopServer(
        robot_ip=args.robot_ip,
        control_mode=args.control_mode,
        host=args.host,
        port=args.port,
        home_joints=args.home,
        freq=args.freq,
    )
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
