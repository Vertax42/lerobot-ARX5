#!/usr/bin/env python3
"""
笛卡尔阻抗控制示例

演示如何使用 OSC (Operational Space Control) 模式进行笛卡尔空间阻抗控制。
力矩计算公式: τ = J^T @ (-K @ error - D @ (J @ dq)) + coriolis
与官方 pylibfranka_controllers 实现一致。
"""

import asyncio
import numpy as np
from scipy.spatial.transform import Rotation as R
from xense_franka.robot import RobotInterface
from xense_franka import FrankaController


async def main():
    # 连接机器人
    robot = RobotInterface("192.168.99.111")
    controller = FrankaController(robot)

    # 启动控制器（开始 1kHz 力矩控制循环）
    await controller.start()

    # 先移动到初始位置（使用关节阻抗控制）
    print("Moving to initial position...")
    await controller.move([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853])
    await asyncio.sleep(1.0)

    # 切换到笛卡尔阻抗控制 (OSC)
    controller.switch("osc")
    
    # 设置笛卡尔刚度 [N/m for xyz, Nm/rad for rpy]
    # 与官方示例类似: 平移刚度 600 N/m, 旋转刚度 50 Nm/rad
    controller.ee_kp = np.array([600.0, 600.0, 600.0, 50.0, 50.0, 50.0])
    # 阻尼设为临界阻尼: D = 2 * sqrt(K)
    controller.ee_kd = 2.0 * np.sqrt(controller.ee_kp)
    
    # 设置命令更新频率
    controller.set_freq(50)  # 50 Hz

    print("Switched to Cartesian impedance control (OSC)")
    print(f"Stiffness: {controller.ee_kp}")
    print(f"Damping: {controller.ee_kd}")

    # 获取初始末端位姿
    initial_ee = controller.ee_desired.copy()
    print(f"Initial EE position: {initial_ee[:3, 3]}")
    print(initial_ee) 

    # 执行方形轨迹
    print("\nExecuting square trajectory in XY plane...")
    
    # 轨迹参数
    side_length = 0.05  # 5cm 边长
    steps_per_side = 50  # 每边 50 步 (1秒)
    
    # 方形四个角的偏移
    square_offsets = [
        [side_length, 0, 0],           # 右
        [side_length, side_length, 0], # 右上
        [0, side_length, 0],           # 上
        [0, 0, 0],                      # 回原点
    ]

    for i, offset in enumerate(square_offsets):
        print(f"Moving to corner {i+1}/4...")
        
        # 计算起点和终点
        if i == 0:
            start_pos = initial_ee[:3, 3].copy()
        else:
            start_pos = initial_ee[:3, 3] + np.array(square_offsets[i-1])
        
        end_pos = initial_ee[:3, 3] + np.array(offset)
        
        # 线性插值
        for step in range(steps_per_side):
            t = step / steps_per_side
            target_pos = start_pos + t * (end_pos - start_pos)
            
            # 构造目标位姿（保持姿态不变）
            target_ee = initial_ee.copy()
            target_ee[:3, 3] = target_pos
            
            await controller.set("ee_desired", target_ee)
        
        # 在角点停留
        # await asyncio.sleep(0.5)

    print("\nSquare trajectory completed!")

    # 演示柔顺性：保持位置，可以用手推动机器人
    print("\nHolding position for 5 seconds...")
    print("You can try to push the robot gently to feel the compliance.")
    
    # 降低刚度，增加柔顺性
    controller.ee_kp = np.array([200.0, 200.0, 200.0, 30.0, 30.0, 30.0])
    controller.ee_kd = 2.0 * np.sqrt(controller.ee_kp)
    print(f"Reduced stiffness: {controller.ee_kp}")
    
    await asyncio.sleep(5.0)

    # 停止控制器
    print("\nStopping controller...")
    await controller.stop()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
