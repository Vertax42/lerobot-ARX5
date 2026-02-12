"""
WebSocket 遥操作客户端 — 运行在操作者侧
读取 gamepad（Xbox 手柄）输入，通过 WebSocket 发送增量指令到服务端，
并接收、打印机器人状态反馈。

用法:
    python ws_teleop_client.py [--server ws://192.168.99.111:8765]
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

SERVER_SCRIPT = Path(__file__).parent / "ws_teleop_server.py"


def launch_server(robot_ip: str, port: int) -> subprocess.Popen:
    """在本机后台启动 ws_teleop_server.py，返回子进程对象。"""
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
        raise RuntimeError("No gamepad detected. Please connect a gamepad and try again.")

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"[Client] Gamepad connected: {joystick.get_name()}")
    return joystick


def read_gamepad(joystick: pygame.joystick.JoystickType, deadzone: float = 0.1) -> dict:
    """读取手柄摇杆，返回增量指令字典。"""
    pygame.event.pump()

    # Xbox layout: LX=0, LY=1, RX=3, RY=4
    lx = joystick.get_axis(0)
    ly = joystick.get_axis(1)
    rx = joystick.get_axis(3)
    ry = joystick.get_axis(4)

    # Apply deadzone
    lx = 0.0 if abs(lx) < deadzone else lx
    ly = 0.0 if abs(ly) < deadzone else ly
    rx = 0.0 if abs(rx) < deadzone else rx
    ry = 0.0 if abs(ry) < deadzone else ry

    # 平移增量
    translation = np.clip(np.array([-ly, -lx, -ry]) * 0.003, -0.003, 0.003).tolist()
    # 旋转增量 (欧拉角，度)
    rotation_euler = np.clip(np.array([0.0, 0.0, -rx]) * 0.5, -0.5, 0.5).tolist()

    # B 按钮 (index 1) → stop
    try:
        if joystick.get_button(1):
            return {"type": "stop"}
    except pygame.error:
        pass

    return {
        "type": "delta",
        "translation": translation,
        "rotation_euler": rotation_euler,
    }


def print_state(state: dict):
    """简洁地打印从服务端收到的机器人状态。"""
    ee = np.array(state.get("O_T_EE", []))
    wrench = state.get("ext_wrench", [])
    tau = state.get("tau", [])
    print("tau:", tau)
    if ee.size == 16:
        pos = ee.reshape(4, 4).T[:3, 3]
        print(pos)
        print(f"  EE pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]  |  wrench: {np.round(wrench, 2)}")
    else:
        print(f"  State: {state}")


def main():
    parser = argparse.ArgumentParser(description="WebSocket teleop client (operator side)")
    parser.add_argument("--freq", type=float, default=50.0, help="Command frequency in Hz (default: 50)")
    parser.add_argument("--deadzone", type=float, default=0.1, help="Joystick deadzone (default: 0.1)")
    parser.add_argument("--robot-ip", default="192.168.99.111", help="Franka robot IP (default: 192.168.99.111)")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port (default: 8765)")
    args = parser.parse_args()

    # 自动启动服务端
    server_proc = launch_server(args.robot_ip, args.port)
    server_uri = f"ws://localhost:{args.port}"

    period = 1.0 / args.freq
    joystick = init_gamepad()

    print(f"[Client] Connecting to server: {server_uri} ...")
    ws = connect(server_uri)
    with ws:
        print("[Client] Connected! Use gamepad to control robot. Press B to stop.")

        while True:
            t0 = time.monotonic()

            cmd = read_gamepad(joystick, args.deadzone)
            ws.send(json.dumps(cmd))

            if cmd.get("type") == "stop":
                print("[Client] Stop command sent. Exiting.")
                break

            # 接收服务端回传状态
            try:
                response = ws.recv(timeout=1.0)
                state = json.loads(response)
                print(state)
                print_state(state)
            except TimeoutError:
                print("[Client] Warning: server response timeout")

            # 控制发送频率
            elapsed = time.monotonic() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    pygame.quit()
    print("[Client] Done.")


if __name__ == "__main__":
    main()
