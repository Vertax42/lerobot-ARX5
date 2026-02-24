"""
WebSocket 关节遥操作服务端 — 运行在机器人侧
接收客户端发来的关节增量指令，使用阻抗控制驱动 Franka 机器人，
并将机器人状态回传给客户端。

控制公式: τ = -Kp*(q - q_d) - Kd*dq + coriolis

用法:
    python ws_joint_teleop_server.py [--host 0.0.0.0] [--port 8766] [--robot-ip 192.168.99.111]
"""

import asyncio
import argparse
import json
import time

import numpy as np
import websockets

from xense_franka.robot import RobotInterface
from xense_franka import FrankaController


# 初始关节位置 (安全位置)
HOME_JOINTS = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853]

# 关节限位 (Franka FR3, rad)
JOINT_LIMITS_MIN = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.017, -2.89])
JOINT_LIMITS_MAX = np.array([2.89,  1.76,  2.89, -0.07,  2.89,  3.75,   2.89])


class JointTeleopServer:
    """WebSocket 关节遥操作服务端，使用阻抗控制。"""

    def __init__(self, robot_ip: str, host: str = "0.0.0.0", port: int = 8766):
        self.robot_ip = robot_ip
        self.host = host
        self.port = port

        self.robot: RobotInterface = None
        self.controller: FrankaController = None
        self.client_connected = False
        self.initial_q: np.ndarray = None  # 初始关节位置，用于重置

    # ─── 机器人初始化 ────────────────────────────────────────
    async def init_robot(self):
        """初始化机器人并启动控制器，切换到阻抗控制模式。"""
        print(f"[Server] Connecting to robot at {self.robot_ip} ...")
        self.robot = RobotInterface(self.robot_ip)
        self.controller = FrankaController(self.robot)

        await self.controller.start()
        # 移动到初始位姿
        await self.controller.move(HOME_JOINTS, vel=np.ones(7)*0.1, acc=np.ones(7)*0.5)
        await asyncio.sleep(1.0)

        # 切换到关节阻抗控制
        self.controller.switch("impedance")
        self.controller.kp = np.ones(7) * 80.0   # 关节刚度 [Nm/rad]
        self.controller.kd = np.ones(7) * 4.0    # 关节阻尼 [Nm·s/rad]
        self.controller.set_freq(50)  # 50 Hz 指令更新

        # 记录初始关节位置用于重置
        self.initial_q = self.controller.q_desired.copy()

        print("[Server] 关节阻抗控制已启动")
        print(f"[Server] 刚度: {self.controller.kp}")
        print(f"[Server] 阻尼: {self.controller.kd}")
        print(f"[Server] 初始关节位置: {np.round(self.initial_q, 4)}")
        print("[Server] Robot ready, waiting for client ...")

    # ─── 处理单条指令 ───────────────────────────────────────
    async def apply_command(self, cmd: dict):
        """将客户端发来的指令应用到机器人。

        支持四种指令类型:
        1. delta — 增量控制 (仅第 7 关节):
           {"type": "delta", "joint_delta": float}
        2. absolute — 绝对关节位置:
           {"type": "absolute", "q_desired": [7 floats]}
        3. reset — 重置到初始关节位置:
           {"type": "reset"}
        4. stop — 停止:
           {"type": "stop"}
        """
        cmd_type = cmd.get("type")

        if cmd_type == "stop":
            print("[Server] 收到退出指令，正在停止...")
            return

        if cmd_type == "reset":
            print("[Server] 重置到初始关节位置...")
            await self.controller.set("q_desired", self.initial_q.copy())
            await asyncio.sleep(0.5)  # 等待稳定
            return

        if cmd_type == "absolute":
            q = np.array(cmd["q_desired"], dtype=np.float64)
            # 限位保护
            q = np.clip(q, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX)
            await self.controller.set("q_desired", q)
            return

        # delta (默认) — 仅修改第 7 关节
        joint_delta = float(cmd.get("joint_delta", 0.0))

        with self.controller.state_lock:
            q_target = self.controller.q_desired.copy()

        q_target[6] += joint_delta
        # 限位保护
        q_target[6] = np.clip(q_target[6], JOINT_LIMITS_MIN[6], JOINT_LIMITS_MAX[6])

        await self.controller.set("q_desired", q_target)

    # ─── 获取机器人状态 ─────────────────────────────────────
    def get_robot_state(self) -> dict:
        """采集当前机器人状态，序列化为可 JSON 传输的字典。"""
        state = self.robot.state

        with self.controller.state_lock:
            q_desired = self.controller.q_desired.copy()

        return {
            "q": state["qpos"].tolist(),
            "dq": state["qvel"].tolist(),
            "tau": state["last_torque"].tolist(),
            "ext_wrench": state["ext_wrench"].tolist(),
            "q_desired": q_desired.tolist(),
            "timestamp": time.time(),
        }

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

                # 应用指令
                await self.apply_command(cmd)

                # 回传机器人状态
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
    parser = argparse.ArgumentParser(description="WebSocket 关节遥操作服务端 (机器人侧)")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址 (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8766, help="端口 (default: 8766)")
    parser.add_argument("--robot-ip", default="192.168.99.111", help="Franka 机器人 IP")
    args = parser.parse_args()

    server = JointTeleopServer(robot_ip=args.robot_ip, host=args.host, port=args.port)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
