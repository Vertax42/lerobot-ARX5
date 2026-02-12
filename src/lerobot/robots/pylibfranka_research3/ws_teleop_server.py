"""
WebSocket 遥操作服务端 — 运行在机器人侧
支持笛卡尔 (OSC) 和关节 (impedance) 两种控制模式。

笛卡尔模式: τ = J^T @ (-K @ error - D @ (J @ dq)) + coriolis
关节模式:   τ = -Kp*(q - q_d) - Kd*dq + coriolis

用法:
    python ws_teleop_server.py --control-mode cartesian  # 笛卡尔末端控制
    python ws_teleop_server.py --control-mode joint      # 关节空间控制
"""

import asyncio
import argparse
import json
import time

import numpy as np
import websockets
from scipy.spatial.transform import Rotation as R

from aiofranka.robot import RobotInterface
from aiofranka import FrankaController


# 初始关节位置 (安全位置)
HOME_JOINTS = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853]

# 关节限位 (Franka FR3, rad)
JOINT_LIMITS_MIN = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.017, -2.89])
JOINT_LIMITS_MAX = np.array([2.89,  1.76,  2.89, -0.07,  2.89,  3.75,   2.89])


class TeleopServer:
    """WebSocket 遥操作服务端，支持笛卡尔和关节两种控制模式。"""

    def __init__(self, robot_ip: str, control_mode: str = "cartesian",
                 host: str = "0.0.0.0", port: int = 8765,
                 home_joints: list = None, freq: int = 50):
        self.robot_ip = robot_ip
        self.control_mode = control_mode  # "cartesian" 或 "joint"
        self.host = host
        self.port = port
        self.home_joints = home_joints if home_joints is not None else list(HOME_JOINTS)
        self.freq = freq

        self.robot: RobotInterface = None
        self.controller: FrankaController = None
        self.client_connected = False

        # 初始状态，用于重置
        self.initial_ee: np.ndarray = None
        self.initial_q: np.ndarray = None

    # ─── 机器人初始化 ────────────────────────────────────────
    async def init_robot(self):
        """初始化机器人并启动控制器，根据模式切换控制器。"""
        print(f"[Server] Connecting to robot at {self.robot_ip} ...")
        self.robot = RobotInterface(self.robot_ip)
        self.controller = FrankaController(self.robot)

        await self.controller.start()

        # 移动到初始位姿
        await self.controller.move(self.home_joints)
        await asyncio.sleep(1.0)

        if self.control_mode == "cartesian":
            self._init_cartesian()
        else:
            self._init_joint()

        print(f"[Server] Robot ready (mode={self.control_mode}), waiting for client ...")

    def _init_cartesian(self):
        """切换到笛卡尔阻抗控制 (OSC)。"""
        self.controller.switch("osc")
        self.controller.ee_kp = np.array([600.0, 600.0, 600.0, 50.0, 50.0, 50.0])
        self.controller.ee_kd = 2.0 * np.sqrt(self.controller.ee_kp)  # 临界阻尼
        self.controller.set_freq(self.freq)

        self.initial_ee = self.controller.ee_desired.copy()

        print("[Server] 笛卡尔阻抗控制已启动")
        print(f"[Server] 刚度: {self.controller.ee_kp}")
        print(f"[Server] 阻尼: {self.controller.ee_kd}")
        print(f"[Server] 初始位置: {self.initial_ee[:3, 3]}")

    def _init_joint(self):
        """切换到关节阻抗控制 (impedance)。"""
        self.controller.switch("impedance")
        self.controller.kp = np.ones(7) * 80.0
        self.controller.kd = np.ones(7) * 4.0
        self.controller.set_freq(self.freq)

        self.initial_q = self.controller.q_desired.copy()

        print("[Server] 关节阻抗控制已启动")
        print(f"[Server] 刚度: {self.controller.kp}")
        print(f"[Server] 阻尼: {self.controller.kd}")
        print(f"[Server] 初始关节位置: {np.round(self.initial_q, 4)}")

    # ─── 处理单条指令 ───────────────────────────────────────
    async def apply_command(self, cmd: dict):
        """将客户端发来的指令应用到机器人。

        通用指令:
          {"type": "stop"}
          {"type": "reset"}

        笛卡尔模式指令:
          {"type": "cartesian_delta", "translation": [dx,dy,dz], "rotation_z": float}
          {"type": "cartesian_absolute", "pose": [[4x4 matrix]]}

        关节模式指令:
          {"type": "joint_delta", "joint_deltas": [7 floats]}
          {"type": "joint_absolute", "q_desired": [7 floats]}
        """
        cmd_type = cmd.get("type")

        if cmd_type in ("get_state", "ee_desired"):
            # 仅返回状态，不执行任何动作
            return

        if cmd_type == "move_home":
            positions = cmd.get("positions", self.home_joints)
            print(f"[Server] 移动到指定关节位置: {np.round(positions, 4)}")
            await self.controller.move(positions)
            await asyncio.sleep(1.0)
            # 重新初始化控制模式
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
            print("[Server] 配置更新完成")
            return

        if cmd_type == "stop":
            print("[Server] 收到退出指令，正在停止...")
            return

        if cmd_type == "reset":
            if self.control_mode == "cartesian":
                print("[Server] 重置到初始末端位姿...")
                await self.controller.set("ee_desired", self.initial_ee.copy())
            else:
                print("[Server] 重置到初始关节位置...")
                await self.controller.set("q_desired", self.initial_q.copy())
            await asyncio.sleep(0.5)
            return

        # ── 笛卡尔指令 ──
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

        # ── 关节指令 ──
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

    # ─── 获取机器人状态 ─────────────────────────────────────
    def get_robot_state(self) -> dict:
        """采集当前机器人状态，序列化为可 JSON 传输的字典。"""
        state = self.robot.state

        result = {
            "control_mode": self.control_mode,
            "q": state["qpos"].tolist(),
            "dq": state["qvel"].tolist(),
            "tau": state["last_torque"].tolist(),
            "ext_wrench": state["ext_wrench"].tolist(),
            "timestamp": time.time(),
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

    # ─── WebSocket 连接处理 ─────────────────────────────────
    async def handler(self, websocket):
        """处理单个 WebSocket 客户端连接。"""
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

    # ─── 启动 ──────────────────────────────────────────────
    async def run(self):
        await self.init_robot()

        async with websockets.serve(self.handler, self.host, self.port):
            print(f"[Server] WebSocket server listening on ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever


def main():
    parser = argparse.ArgumentParser(description="WebSocket 遥操作服务端 (机器人侧)")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址 (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="端口 (default: 8765)")
    parser.add_argument("--robot-ip", default="192.168.99.111", help="Franka 机器人 IP")
    parser.add_argument("--control-mode", choices=["cartesian", "joint"], default="cartesian",
                        help="控制模式: cartesian (笛卡尔) 或 joint (关节) (default: cartesian)")
    parser.add_argument("--freq", type=int, default=50, help="控制频率 Hz (default: 50)")
    parser.add_argument("--home", nargs=7, type=float, default=None,
                        help="初始关节位置 (7 个浮点数)")
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
