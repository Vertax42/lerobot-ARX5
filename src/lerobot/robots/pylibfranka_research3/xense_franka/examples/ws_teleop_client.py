"""
WebSocket 遥操作客户端 — 运行在操作者侧
支持笛卡尔 (OSC) 和关节 (impedance) 两种控制模式。

笛卡尔模式手柄映射 (Xbox 布局):
  - 左摇杆 X/Y: 末端 Y/X 平移
  - 右摇杆 Y: 末端 Z 平移
  - 右摇杆 X: 末端 Z 轴旋转

关节模式手柄映射:
  - 左摇杆 X: 第 7 关节旋转

通用按钮:
  - A 按钮 (或 X): 退出程序
  - B 按钮 (或 O): 重置到初始位置

用法:
    python ws_teleop_client.py --control-mode cartesian  # 笛卡尔末端控制
    python ws_teleop_client.py --control-mode joint      # 关节空间控制
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

# 笛卡尔控制参数
TRANSLATION_SPEED = 0.005  # m/step (最大平移速度)
ROTATION_SPEED = 0.5       # deg/step (最大旋转速度)

# 关节控制参数
JOINT7_SPEED = 0.01        # rad/step (第 7 关节旋转速度)

DEADZONE = 0.1             # 摇杆死区


def launch_server(robot_ip: str, port: int, control_mode: str) -> subprocess.Popen:
    """在本机后台启动 ws_teleop_server.py，返回子进程对象。"""
    cmd = [sys.executable, "-u", str(SERVER_SCRIPT),
           "--robot-ip", robot_ip, "--port", str(port),
           "--control-mode", control_mode]
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


def init_gamepad(control_mode: str) -> pygame.joystick.JoystickType:
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise RuntimeError("未检测到手柄，请连接手柄后重试。")

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"[Client] 手柄已连接: {joystick.get_name()}")
    print("[Client] 控制说明:")
    if control_mode == "cartesian":
        print("  左摇杆: X/Y 平移")
        print("  右摇杆 Y: Z 平移")
        print("  右摇杆 X: Z 轴旋转")
    else:
        print("  左摇杆 X: 第 7 关节旋转")
    print("  A/X 按钮: 退出")
    print("  B/O 按钮: 重置位置")
    return joystick


def apply_deadzone(value: float, deadzone: float) -> float:
    return 0.0 if abs(value) < deadzone else value


def read_joystick_axes(joystick: pygame.joystick.JoystickType, deadzone: float = DEADZONE):
    """读取手柄摇杆原始值（带死区），返回 (lx, ly, rx, ry)。"""
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


# ─── 笛卡尔模式读取 ─────────────────────────────────────────

def read_cartesian_delta(joystick, deadzone=DEADZONE) -> dict:
    """笛卡尔增量模式。"""
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
    """笛卡尔绝对模式，原地修改 ee_target。"""
    lx, ly, rx, ry = read_joystick_axes(joystick, deadzone)
    if is_exit_pressed(joystick):
        return {"type": "stop"}
    if is_reset_pressed(joystick):
        return {"type": "reset"}

    ee_target[:3, 3] += np.array([ly * TRANSLATION_SPEED, lx * TRANSLATION_SPEED, -ry * TRANSLATION_SPEED])
    rot = R.from_euler('z', -rx * ROTATION_SPEED, degrees=True).as_matrix()
    ee_target[:3, :3] = rot @ ee_target[:3, :3]

    return {"type": "cartesian_absolute", "pose": ee_target.tolist()}


# ─── 关节模式读取 ───────────────────────────────────────────

def read_joint_delta(joystick, deadzone=DEADZONE) -> dict:
    """关节增量模式 (仅第 7 关节)。"""
    lx, ly, rx, ry = read_joystick_axes(joystick, deadzone)
    if is_exit_pressed(joystick):
        return {"type": "stop"}
    if is_reset_pressed(joystick):
        return {"type": "reset"}
    return {
        "type": "joint_delta",
        "joint_deltas": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lx * JOINT7_SPEED],
    }


# ─── 状态打印 ───────────────────────────────────────────────

def print_state(state: dict):
    """根据控制模式打印状态。"""
    mode = state.get("control_mode", "unknown")
    if mode == "cartesian":
        ee = np.array(state.get("O_T_EE", []))
        wrench = state.get("ext_wrench", [])
        if ee.size == 16:
            pos = ee.reshape(4, 4).T[:3, 3]
            print(f"  末端位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]  |  外力: {np.round(wrench, 2)}")
    elif mode == "joint":
        q = state.get("q", [])
        q_desired = state.get("q_desired", [])
        if len(q) == 7 and len(q_desired) == 7:
            print(f"  q7: {q[6]:.4f}  |  q7_desired: {q_desired[6]:.4f}  |  "
                  f"误差: {abs(q[6] - q_desired[6]):.4f} rad")
    else:
        print(f"  状态: {state}")


def main():
    parser = argparse.ArgumentParser(description="WebSocket 遥操作客户端 (操作者侧)")
    parser.add_argument("--freq", type=float, default=50.0, help="指令频率 Hz (default: 50)")
    parser.add_argument("--deadzone", type=float, default=DEADZONE, help="摇杆死区 (default: 0.1)")
    parser.add_argument("--robot-ip", default="192.168.99.111", help="Franka 机器人 IP (default: 192.168.99.111)")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket 端口 (default: 8765)")
    parser.add_argument("--control-mode", choices=["cartesian", "joint"], default="cartesian",
                        help="控制模式: cartesian (笛卡尔) 或 joint (关节) (default: cartesian)")
    parser.add_argument("--send-mode", choices=["delta", "absolute"], default="absolute",
                        help="发送模式: delta (增量) 或 absolute (绝对) (default: absolute)")
    args = parser.parse_args()

    # 自动启动服务端 (传递 control-mode)
    server_proc = launch_server(args.robot_ip, args.port, args.control_mode)
    server_uri = f"ws://localhost:{args.port}"

    period = 1.0 / args.freq
    joystick = init_gamepad(args.control_mode)
    ee_target = None  # cartesian absolute 模式使用

    print(f"[Client] 连接服务端: {server_uri} ...")
    with connect(server_uri) as ws:
        print(f"[Client] 已连接! 控制: {args.control_mode}, 发送: {args.send_mode}。A 退出, B 重置。")

        # cartesian absolute 模式需要先获取初始位姿
        if args.control_mode == "cartesian" and args.send_mode == "absolute":
            ws.send(json.dumps({"type": "cartesian_delta", "translation": [0, 0, 0], "rotation_z": 0.0}))
            init_state = json.loads(ws.recv(timeout=5.0))
            ee_target = np.array(init_state["ee_desired"], dtype=np.float64).reshape(4, 4)
            print("[Client] 初始位姿已接收，开始绝对控制。")

        while True:
            t0 = time.monotonic()

            # 根据模式读取手柄
            if args.control_mode == "cartesian":
                if args.send_mode == "absolute":
                    cmd = read_cartesian_absolute(joystick, ee_target, args.deadzone)
                    if cmd["type"] == "cartesian_absolute":
                        pos = np.array(cmd['pose']).reshape(4, 4)[:3, 3]
                        print(f"[Client] 目标位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
                else:
                    cmd = read_cartesian_delta(joystick, args.deadzone)
            else:  # joint
                cmd = read_joint_delta(joystick, args.deadzone)

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
                # cartesian absolute 模式 reset 后同步 ee_target
                if (cmd.get("type") == "reset" and args.control_mode == "cartesian"
                        and args.send_mode == "absolute" and "ee_desired" in state):
                    ee_target = np.array(state["ee_desired"], dtype=np.float64).reshape(4, 4)
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
