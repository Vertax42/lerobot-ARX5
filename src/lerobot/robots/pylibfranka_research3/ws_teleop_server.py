"""
WebSocket 遥操作服务端 — 运行在机器人侧
接收客户端发来的指令，驱动 Franka 机器人，
并将机器人状态回传给客户端。

支持的指令类型:
  - get_state   : 仅获取状态，不执行动作
  - delta        : 增量运动 {"translation": [dx,dy,dz], "rotation_euler": [rx,ry,rz]}
  - set_ee       : 设置绝对末端位姿 {"ee_desired": [[4x4 matrix]]}
  - move_home    : 移动到指定关节位置 {"positions": [7 floats]}
  - configure    : 配置控制器参数 {"ee_kp":…, "ee_kd":…, "freq":…, "switch_mode":…}
  - stop         : 停止

用法:
    python ws_teleop_server.py [--host 0.0.0.0] [--port 8765] [--robot-ip 192.168.99.111]
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


class TeleopServer:
    """WebSocket 遥操作服务端，管理机器人控制器和客户端连接。"""

    def __init__(self, robot_ip: str, host: str = "0.0.0.0", port: int = 8765,
                 home_positions: list = None, freq: int = 50):
        self.robot_ip = robot_ip
        self.host = host
        self.port = port
        self.home_positions = home_positions or [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853]
        self.freq = freq

        self.robot: RobotInterface = None
        self.controller: FrankaController = None
        self.client_connected = False

    # ─── 机器人初始化 ────────────────────────────────────────
    async def init_robot(self):
        """初始化机器人并启动控制器，切换到 OSC 模式。"""
        print(f"[Server] Connecting to robot at {self.robot_ip} ...")
        self.robot = RobotInterface(self.robot_ip)
        self.controller = FrankaController(self.robot)

        await self.controller.start()
        # 移动到初始位姿
        print(f"[Server] Moving to home: {self.home_positions}")
        await self.controller.move(self.home_positions)
        await asyncio.sleep(1.0)

        # 切换到 OSC 控制
        self.controller.switch("osc")
        self.controller.ee_kp = np.array([300.0, 300.0, 300.0, 1000.0, 1000.0, 1000.0])
        self.controller.ee_kd = np.ones(6) * 10.0
        self.controller.set_freq(self.freq)
        print(f"[Server] Robot ready (freq={self.freq}Hz)")

    # ─── 处理指令 ───────────────────────────────────────────
    async def apply_command(self, cmd: dict):
        """将客户端发来的指令应用到机器人。

        支持的指令类型:
        - "get_state"  : 仅获取状态，不执行动作
        - "delta"      : 增量运动 {"translation": [dx,dy,dz], "rotation_euler": [rx,ry,rz]}
        - "set_ee"     : 设置绝对末端位姿 {"ee_desired": [[4x4 matrix]]}
        - "move_home"  : 移动到指定关节位置 {"positions": [7 floats]}
        - "configure"  : 配置控制器参数
        - "stop"       : 停止
        """
        cmd_type = cmd.get("type", "")

        if cmd_type == "stop":
            print("[Server] Received stop command.")
            return

        elif cmd_type == "get_state":
            # 仅返回状态，不执行动作
            return

        elif cmd_type == "delta":
            translation_delta = np.array(cmd["translation"], dtype=np.float64)
            rotation_euler = np.array(cmd["rotation_euler"], dtype=np.float64)
            rotation_delta = R.from_euler("xyz", rotation_euler, degrees=True).as_matrix()

            with self.controller.state_lock:
                current_ee = self.controller.ee_desired.copy()

            current_ee[:3, 3] += translation_delta
            current_ee[:3, :3] = rotation_delta @ current_ee[:3, :3]

            await self.controller.set("ee_desired", current_ee)

        elif cmd_type == "set_ee":
            # 设置绝对末端位姿 (4x4 行主序矩阵)
            ee_desired = np.array(cmd["ee_desired"], dtype=np.float64).reshape(4, 4)

            with self.controller.state_lock:
                current_ee = self.controller.ee_desired.copy()
            current_ee[:3, 3] = ee_desired[:3, 3]

            await self.controller.set("ee_desired", ee_desired)

        elif cmd_type == "move_home":
            positions = cmd.get("positions", self.home_positions)
            print(f"[Server] Moving to home: {positions}")
            await self.controller.move(positions)
            await asyncio.sleep(0.5)
            # 重新切换回 OSC 模式
            self.controller.switch("osc")

        elif cmd_type == "configure":
            if "ee_kp" in cmd:
                self.controller.ee_kp = np.array(cmd["ee_kp"], dtype=np.float64)
            if "ee_kd" in cmd:
                self.controller.ee_kd = np.array(cmd["ee_kd"], dtype=np.float64)
            if "freq" in cmd:
                self.controller.set_freq(cmd["freq"])
            if "switch_mode" in cmd:
                self.controller.switch(cmd["switch_mode"])
            print(f"[Server] Controller configured: {cmd}")

        else:
            print(f"[Server] Unknown command type: {cmd_type}")

    # ─── 获取机器人状态 ─────────────────────────────────────
    def get_robot_state(self) -> dict:
        """采集当前机器人状态，序列化为可 JSON 传输的字典。"""
        state = self.robot.state

        with self.controller.state_lock:
            ee_desired = self.controller.ee_desired.copy()

        return {
            "O_T_EE": state["O_T_EE"].tolist(),
            "q": state["q"].tolist() if "q" in state else [],
            "dq": state["dq"].tolist() if "dq" in state else [],
            "tau": state["last_torque"].tolist() if "last_torque" in state else [],
            "ext_wrench": state["ext_wrench"].tolist(),
            "ee_desired": ee_desired.tolist(),
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
            print(f"[Server] WebSocket server listening on ws://{self.host}:{self.port}", flush=True)
            await asyncio.Future()  # run forever


def main():
    parser = argparse.ArgumentParser(description="WebSocket teleop server (robot side)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    parser.add_argument("--robot-ip", default="192.168.99.111", help="Franka robot IP")
    parser.add_argument("--home", nargs=7, type=float,
                        default=[0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853],
                        help="Home joint positions (7 values in radians)")
    parser.add_argument("--freq", type=int, default=50,
                        help="Control frequency in Hz (default: 50)")
    args = parser.parse_args()

    server = TeleopServer(
        robot_ip=args.robot_ip,
        host=args.host,
        port=args.port,
        home_positions=args.home,
        freq=args.freq,
    )
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
