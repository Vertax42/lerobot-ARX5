"""
WebSocket 关节遥操作客户端 — 运行在操作者侧
读取 gamepad（Xbox 手柄）输入，通过 WebSocket 发送第 7 关节增量指令到服务端，
并接收、打印机器人状态反馈。

手柄映射 (Xbox 布局):
  - 左摇杆 X: 第 7 关节正/负旋转
  - A 按钮 (或 X): 退出程序
  - B 按钮 (或 O): 重置到初始位置

用法:
    python ws_joint_teleop_client.py [--robot-ip 192.168.99.111] [--port 8766]
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
from websockets.sync.client import connect

SERVER_SCRIPT = Path(__file__).parent / "ws_joint_teleop_server.py"

# 控制参数 (与 gamepad_joint7_impedance_example.py 一致)
JOINT7_SPEED = 0.01   # rad/step (第 7 关节旋转速度)
DEADZONE = 0.1        # 摇杆死区


def launch_server(robot_ip: str, port: int) -> subprocess.Popen:
    """在本机后台启动 ws_joint_teleop_server.py，返回子进程对象。"""
    cmd = [sys.executable, "-u", str(SERVER_SCRIPT), "--robot-ip", robot_ip, "--port", str(port)]
    print(f"[Client] Launching server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # 退出时自动清理服务端
    def cleanup():
        if proc.poll() is None:
            print("[Client] Shutting down server ...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    atexit.register(cleanup)

    # 等待服务端就绪
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


def init_gamepad() -> pygame.joystick.JoystickType:
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise RuntimeError("未检测到手柄，请连接手柄后重试。")

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"[Client] 手柄已连接: {joystick.get_name()}")
    print("[Client] 控制说明:")
    print("  左摇杆 X: 第 7 关节旋转")
    print("  A/X 按钮: 退出")
    print("  B/O 按钮: 重置位置")
    return joystick


def apply_deadzone(value: float, deadzone: float) -> float:
    """应用死区，消除摇杆抖动"""
    return 0.0 if abs(value) < deadzone else value


def is_exit_pressed(joystick: pygame.joystick.JoystickType) -> bool:
    """检测 A 按钮 (按钮 0) 是否按下。"""
    try:
        return joystick.get_button(0)
    except pygame.error:
        return False


def is_reset_pressed(joystick: pygame.joystick.JoystickType) -> bool:
    """检测 B 按钮 (按钮 1) 是否按下。"""
    try:
        return joystick.get_button(1)
    except pygame.error:
        return False


def read_gamepad_joint_delta(joystick: pygame.joystick.JoystickType,
                             deadzone: float = DEADZONE) -> dict:
    """读取手柄输入，返回第 7 关节增量指令。"""
    pygame.event.pump()

    if is_exit_pressed(joystick):
        return {"type": "stop"}

    if is_reset_pressed(joystick):
        return {"type": "reset"}

    lx = apply_deadzone(joystick.get_axis(0), deadzone)  # 左摇杆 X
    joint_delta = lx * JOINT7_SPEED

    return {
        "type": "delta",
        "joint_delta": joint_delta,
    }


def print_state(state: dict):
    """简洁地打印从服务端收到的机器人状态。"""
    q = state.get("q", [])
    q_desired = state.get("q_desired", [])
    if len(q) == 7 and len(q_desired) == 7:
        print(f"  q7: {q[6]:.4f}  |  q7_desired: {q_desired[6]:.4f}  |  "
              f"误差: {abs(q[6] - q_desired[6]):.4f} rad")
    else:
        print(f"  状态: {state}")


def main():
    parser = argparse.ArgumentParser(description="WebSocket 关节遥操作客户端 (操作者侧)")
    parser.add_argument("--freq", type=float, default=50.0, help="指令频率 Hz (default: 50)")
    parser.add_argument("--deadzone", type=float, default=DEADZONE, help="摇杆死区 (default: 0.1)")
    parser.add_argument("--robot-ip", default="192.168.99.111", help="Franka 机器人 IP (default: 192.168.99.111)")
    parser.add_argument("--port", type=int, default=8766, help="WebSocket 端口 (default: 8766)")
    args = parser.parse_args()

    # 自动启动服务端
    server_proc = launch_server(args.robot_ip, args.port)
    server_uri = f"ws://localhost:{args.port}"

    period = 1.0 / args.freq
    joystick = init_gamepad()

    print(f"[Client] 连接服务端: {server_uri} ...")
    with connect(server_uri) as ws:
        print(f"[Client] 已连接! 使用手柄控制第 7 关节。A 退出, B 重置。")

        while True:
            t0 = time.monotonic()

            cmd = read_gamepad_joint_delta(joystick, args.deadzone)
            ws.send(json.dumps(cmd))

            if cmd.get("type") == "stop":
                print("[Client] 检测到退出按钮，正在停止...")
                break

            if cmd.get("type") == "reset":
                print("[Client] 重置到初始位置...")

            # 接收服务端回传状态
            try:
                response = ws.recv(timeout=1.0)
                state = json.loads(response)
                print_state(state)
            except TimeoutError:
                print("[Client] 警告: 服务端响应超时")

            # 控制发送频率
            elapsed = time.monotonic() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    pygame.quit()
    print("[Client] 完成!")


if __name__ == "__main__":
    main()
