"""
使用同步接口控制机械臂的手柄示例。
不需要 async/await 语法。
"""
import pygame
import time
import numpy as np
from xense_franka import SyncFrankaController


def main():
    # 先初始化 pygame
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No gamepad detected. Please connect a gamepad and try again.")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Gamepad connected: {joystick.get_name()}")

    # 使用同步接口（可以使用 with 语句自动管理生命周期）
    with SyncFrankaController("192.168.99.111") as controller:
        # 移动到初始位置
        controller.move([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853])
        time.sleep(2.0)  # 等待机器人稳定

        # 切换到 OSC 模式
        controller.switch("osc")
        controller.set_gains(
            kp=np.array([300.0, 300.0, 300.0, 1000.0, 1000.0, 1000.0]),
            kd=np.ones(6) * 10.0,
            mode="osc"
        )
        controller.set_freq(30)  # 降低频率到 30Hz

        print("Use gamepad to move the robot. Ctrl+C to exit.")
        print("  Left Stick: X/Y translation")
        print("  Right Stick Y: Z translation")
        print("  Right Stick X: Rotation around Z")

        deadzone = 0.1

        try:
            while True:
                loop_start = time.time()
                pygame.event.pump()

                # 读取摇杆轴
                lx = joystick.get_axis(0)
                ly = joystick.get_axis(1)
                rx = joystick.get_axis(3)
                ry = joystick.get_axis(4)

                # 应用死区
                lx = 0 if abs(lx) < deadzone else lx
                ly = 0 if abs(ly) < deadzone else ly
                rx = 0 if abs(rx) < deadzone else rx
                ry = 0 if abs(ry) < deadzone else ry

                # 计算增量
                dx = -ly * 0.002
                dy = -lx * 0.002
                dz = -ry * 0.002
                drz = -rx * 0.3  # 度

                # 只有有输入时才发送
                if abs(dx) > 0 or abs(dy) > 0 or abs(dz) > 0 or abs(drz) > 0:
                    controller.move_delta(dx=dx, dy=dy, dz=dz, drz=drz)

                # 固定 30Hz 循环
                elapsed = time.time() - loop_start
                sleep_time = 1/30 - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nExiting...")

    pygame.quit()
    print("Done.")


if __name__ == "__main__":
    main()
