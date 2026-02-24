#!/usr/bin/env python3
"""
手柄遥操作示例 — 阻抗控制模式控制第 7 关节

使用 Xbox/PlayStation 风格手柄，在关节空间阻抗控制下操纵 Franka 第 7 关节。
其余 6 个关节保持在初始位置不变（由阻抗控制维持）。

控制公式: τ = -Kp*(q - q_d) - Kd*dq + coriolis

手柄映射 (Xbox 布局):
- 左摇杆 X: 第 7 关节正/负旋转
- A 按钮 (或 X): 退出程序
- B 按钮 (或 O): 重置到初始位置
"""

import pygame
import asyncio
import numpy as np
from xense_franka.robot import RobotInterface
from xense_franka import FrankaController


# 初始关节位置 (安全位置)
HOME_JOINTS = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853]

# 控制参数
JOINT7_SPEED = 0.01   # rad/step (第 7 关节旋转速度)
DEADZONE = 0.1        # 摇杆死区

# 第 7 关节限位 (Franka FR3 joint 7: -2.8973 ~ 2.8973 rad)
JOINT7_MIN = -1
JOINT7_MAX = 1


def apply_deadzone(value, deadzone):
    """应用死区，消除摇杆抖动"""
    return 0.0 if abs(value) < deadzone else value


async def main():
    # ========== 初始化手柄 ==========
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("未检测到手柄，请连接手柄后重试。")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"手柄已连接: {joystick.get_name()}")
    print("控制说明:")
    print("  左摇杆 X: 第 7 关节旋转")
    print("  A/X 按钮: 退出")
    print("  B/O 按钮: 重置位置")
    print()

    # ========== 连接机器人 ==========
    robot = RobotInterface("192.168.99.111")
    controller = FrankaController(robot)

    await controller.start()

    # 移动到初始位置
    print("移动到初始位置...")
    await controller.move(HOME_JOINTS)
    await asyncio.sleep(0.5)

    # ========== 切换到关节阻抗控制 ==========
    controller.switch("impedance")

    # 设置刚度和阻尼
    controller.kp = np.ones(7) * 80.0   # 关节刚度 [Nm/rad]
    controller.kd = np.ones(7) * 4.0    # 关节阻尼 [Nm·s/rad]

    # 设置命令更新频率
    controller.set_freq(50)  # 50 Hz

    print("关节阻抗控制已启动")
    print(f"刚度: {controller.kp}")
    print(f"阻尼: {controller.kd}")

    # 记录初始关节位置用于重置
    initial_q = controller.q_desired.copy()
    print(f"初始第 7 关节位置: {initial_q[6]:.4f} rad")
    print("\n开始遥操作 (仅第 7 关节)...")

    # ========== 主控制循环 ==========
    running = True
    while running:
        pygame.event.pump()

        # ---- 读取按钮 ----
        # A/X 按钮退出 (按钮 0)
        if joystick.get_button(0):
            print("\n检测到退出按钮，正在停止...")
            running = False
            continue

        # B/O 按钮重置 (按钮 1)
        if joystick.get_button(1):
            print("重置到初始位置...")
            with controller.state_lock:
                controller.q_desired = initial_q.copy()
            await asyncio.sleep(0.5)
            continue

        # ---- 读取摇杆 ----
        lx = apply_deadzone(joystick.get_axis(0), DEADZONE)  # 左摇杆 X

        # ---- 计算增量 ----
        delta = lx * JOINT7_SPEED

        # ---- 更新第 7 关节目标 ----
        with controller.state_lock:
            q_target = controller.q_desired.copy()

        q_target[6] += delta
        # 限位保护
        q_target[6] = np.clip(q_target[6], JOINT7_MIN, JOINT7_MAX)

        print(f"第 7 关节目标: {q_target[6]:.4f} rad  (增量: {delta:+.4f})")

        await controller.set("q_desired", q_target)

    # ========== 清理 ==========
    print("停止控制器...")
    await controller.stop()
    pygame.quit()
    print("完成!")


if __name__ == "__main__":
    asyncio.run(main())
