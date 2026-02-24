#!/usr/bin/env python3
"""
手柄遥操作示例

使用 Xbox/PlayStation 风格手柄控制 Franka 机器人末端执行器。
力矩计算采用与官方一致的笛卡尔阻抗控制: τ = J^T @ (-K @ error - D @ (J @ dq)) + coriolis

手柄映射 (Xbox 布局):
- 左摇杆 X/Y: 末端 Y/X 平移
- 右摇杆 Y: 末端 Z 平移  
- 右摇杆 X: 末端 Z 轴旋转
- A 按钮 (或 X): 退出程序
- B 按钮 (或 O): 重置到初始位置
"""

import pygame
import asyncio
import numpy as np
from scipy.spatial.transform import Rotation as R
from xense_franka.robot import RobotInterface
from xense_franka import FrankaController


# 初始关节位置 (安全位置)
HOME_JOINTS = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853]

# 控制参数
TRANSLATION_SPEED = 0.005  # m/step (最大平移速度)
ROTATION_SPEED = 0.5       # deg/step (最大旋转速度)
DEADZONE = 0.1             # 摇杆死区


def apply_deadzone(value, deadzone):
    """应用死区，消除摇杆抖动"""
    return 0.0 if abs(value) < deadzone else value


async def main():
    # ========== 初始化手柄 ==========
    # 在连接机器人前初始化，避免通信超时
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("未检测到手柄，请连接手柄后重试。")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"手柄已连接: {joystick.get_name()}")
    print("控制说明:")
    print("  左摇杆: X/Y 平移")
    print("  右摇杆 Y: Z 平移")
    print("  右摇杆 X: Z 轴旋转")
    print("  A/X 按钮: 退出")
    print("  B/O 按钮: 重置位置")
    print()

    # ========== 连接机器人 ==========
    robot = RobotInterface("192.168.99.111")
    controller = FrankaController(robot)

    # 启动控制器
    await controller.start()

    # 移动到初始位置
    print("移动到初始位置...")
    await controller.move(HOME_JOINTS)
    await asyncio.sleep(0.5)

    # ========== 切换到笛卡尔阻抗控制 ==========
    controller.switch("osc")
    
    # 设置刚度和阻尼 (与官方示例一致的参数)
    # 平移刚度 600 N/m, 旋转刚度 50 Nm/rad
    controller.ee_kp = np.array([600.0, 600.0, 600.0, 50.0, 50.0, 50.0])
    controller.ee_kd = 2.0 * np.sqrt(controller.ee_kp)  # 临界阻尼
    
    # 设置命令更新频率
    controller.set_freq(50)  # 50 Hz

    print("笛卡尔阻抗控制已启动")
    print(f"刚度: {controller.ee_kp}")
    print(f"阻尼: {controller.ee_kd}")

    # 记录初始位姿用于重置
    initial_ee = controller.ee_desired.copy()
    print(f"初始位置: {initial_ee[:3, 3]}")
    print("\n开始遥操作...")

    # ========== 主控制循环 ==========
    running = True
    while running:
        # 处理 pygame 事件
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
                controller.ee_desired = initial_ee.copy()
            await asyncio.sleep(0.5)  # 等待稳定
            continue

        # ---- 读取摇杆 ----
        lx = apply_deadzone(joystick.get_axis(0), DEADZONE)  # 左摇杆 X
        ly = apply_deadzone(joystick.get_axis(1), DEADZONE)  # 左摇杆 Y
        rx = apply_deadzone(joystick.get_axis(3), DEADZONE)  # 右摇杆 X
        ry = apply_deadzone(joystick.get_axis(4), DEADZONE)  # 右摇杆 Y

        # ---- 计算增量 ----
        # 平移: X(前后), Y(左右), Z(上下)
        translation_delta = np.array([
            ly * TRANSLATION_SPEED,  # 左摇杆 Y -> X 轴 (前后)
            lx * TRANSLATION_SPEED,  # 左摇杆 X -> Y 轴 (左右)
            -ry * TRANSLATION_SPEED,  # 右摇杆 Y -> Z 轴 (上下)
        ])
        
        # 旋转: 只用 Z 轴旋转 (绕竖直轴)
        rotation_delta = R.from_euler('z', -rx * ROTATION_SPEED, degrees=True).as_matrix()

        # ---- 更新目标位姿 ----
        with controller.state_lock:
            current_ee = controller.ee_desired.copy()
            print("当前末端位姿:\n", current_ee)

        # 应用平移
        current_ee[:3, 3] += translation_delta
        
        # 应用旋转 (左乘，在基坐标系下旋转)
        current_ee[:3, :3] = rotation_delta @ current_ee[:3, :3]
        print("目标位置:\n",current_ee)

        # 发送新目标
        await controller.set("ee_desired", current_ee)

    # ========== 清理 ==========
    print("停止控制器...")
    await controller.stop()
    pygame.quit()
    print("完成!")


if __name__ == "__main__":
    asyncio.run(main())
