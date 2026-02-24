import pygame
import time
import asyncio
import numpy as np
from xense_franka.robot import RobotInterface
from xense_franka import FrankaController
from scipy.spatial.transform import Rotation as R


async def main(): 

    # 先初始化 pygame，避免在控制循环运行中初始化导致通信超时
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No gamepad detected. Please connect a gamepad and try again.")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Gamepad connected: {joystick.get_name()}. Use it to move the robot end-effector.")

    robot = RobotInterface("192.168.99.111")
    controller = FrankaController(robot)

    await controller.start()

    await controller.move([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853])

    await asyncio.sleep(1.0)

    controller.switch("osc")
    controller.ee_kp = np.array([300.0, 300.0, 300.0, 1000.0, 1000.0, 1000.0])
    controller.ee_kd = np.ones(6) * 10.0
    controller.set_freq(50)  # Set 50Hz update rate

    deadzone = 0.1
    
    while True:
        pygame.event.pump()

        # Read joystick axes (Xbox layout: LX=0, LY=1, RX=3, RY=4)
        lx = joystick.get_axis(0)  # Left stick X
        ly = joystick.get_axis(1)  # Left stick Y
        rx = joystick.get_axis(3)  # Right stick X
        ry = joystick.get_axis(4)  # Right stick Y

        # Apply deadzone
        lx = 0 if abs(lx) < deadzone else lx
        ly = 0 if abs(ly) < deadzone else ly
        rx = 0 if abs(rx) < deadzone else rx
        ry = 0 if abs(ry) < deadzone else ry

        # Scale the inputs to get reasonable movements
        translation_delta = np.clip(np.array([-ly, -lx, -ry]) * 0.003, -0.003, 0.003) 
        rotation_delta = np.array([0, 0, -rx]) * 0.5
        rotation_delta = np.clip(rotation_delta, -0.5, 0.5)
        rotation_delta = R.from_euler('xyz', rotation_delta, degrees=True).as_matrix() 

        # Get current desired end-effector pose
        with controller.state_lock:
            current_ee = controller.ee_desired.copy()
            print("Current EE pose:\n", current_ee)
            # torques = controller.last_torque.copy()
        state = robot.state
        torques = state['last_torque']
        O_F_ext_hat_K = state['ext_wrench']
        print("Current external wrench (O_F_ext_hat_K):", O_F_ext_hat_K)
        # print("Current torques:", torques)
        O_T_EE = state['O_T_EE']
        print("Current EE from robot state:\n", O_T_EE)

        # print(current_ee)

        # Update position
        current_ee[:3, 3] += translation_delta
        # current_ee[:3, :3] = controller.initial_ee[:3, :3]
        current_ee[:3, :3] = rotation_delta @ current_ee[:3, :3]
        print("Updated EE pose:\n", current_ee)


        await controller.set("ee_desired", current_ee)
        
        # 关键：pygame.event.pump() 是非阻塞的，需要手动控制频率
        # pyspacemouse.read() 会阻塞等待数据，所以02不需要这行
        # await asyncio.sleep(1/50)  # 50Hz 与 set_freq 匹配


if __name__ == "__main__":
    asyncio.run(main())
