import time
import os
import sys
import threading

import click
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# Import after path setup - this is required for the SDK
import arx5_interface as arx5  # noqa: E402


# 移除了 easeInOutQuad 函数，因为不再需要运动控制


def print_state_continuously(controller, stop_event):
    """持续打印机器人状态的函数（包含夹爪数据）"""
    while not stop_event.is_set():
        try:
            joint_state = controller.get_joint_state()
            # pos = joint_state.pos()
            # vel = joint_state.vel()
            torque = joint_state.torque()
            gripper_pos = joint_state.gripper_pos
            gripper_vel = joint_state.gripper_vel
            gripper_torque = joint_state.gripper_torque

            print(
                f"\r夹爪位置: {gripper_pos} | 夹爪速度: {gripper_vel} | 夹爪力矩: {gripper_torque} | 关节力矩: {torque}",
                end="",
                flush=True,
            )
            time.sleep(0.1)  # 每100ms打印一次
        except Exception as e:
            print(f"\n状态打印错误: {e}")
            break


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def main(model: str, interface: str):
    """
    重力补偿状态监控脚本

    功能：
    - 初始化后先重置机器人到home位置
    - 设置所有gain为0，实现纯重力补偿（无位置/速度控制）
    - 进入重力补偿模式，机器人保持在当前位置
    - 实时监控并打印关节位置和速度
    - 夹爪通过设置 gripper_motor_type = MotorType.NONE 完全禁用
    - 支持 Ctrl+C 中断监控

    注意：
    - 夹爪已完全禁用，不会进行任何夹爪相关的通信或控制
    - 所有控制增益(kp, kd)设为0，只进行重力补偿
    """

    # To initialize robot with different configurations,
    # you can create RobotConfig and ControllerConfig by yourself and modify
    # based on it
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    # 设置夹爪电机类型为NONE，禁用夹爪
    # robot_config.gripper_motor_type = arx5.MotorType.NONE
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    # Modify the default configuration here
    # controller_config.controller_dt = 0.01 # etc.

    USE_MULTITHREADING = True
    if USE_MULTITHREADING:
        # Will create another thread that communicates with the arm, so each
        # send_recv_once() will take no time for the main thread to execute.
        # Otherwise (without background send/recv), send_recv_once() will block
        # the main thread until the arm responds (usually 2ms).
        controller_config.background_send_recv = True
    else:
        controller_config.background_send_recv = False

    # 夹爪已通过gripper_motor_type = MotorType.NONE完全禁用
    # 不需要设置夹爪控制参数

    # 设置所有gain为0，实现纯重力补偿
    print("设置所有gain为0，实现纯重力补偿...")
    gain = arx5.Gain(robot_config.joint_dof)
    gain.kp()[:] = 0.0  # 位置增益设为0
    gain.kd()[:] = 0.0  # 速度增益设为0
    gain.gripper_kp = 0.0  # 夹爪位置增益设为0
    gain.gripper_kd = 0.0  # 夹爪速度增益设为0

    # 打印配置信息
    print(f"机器人模型: {model}")
    print(f"CAN接口: {interface}")
    print(f"夹爪电机类型: {robot_config.gripper_motor_type}")
    print("夹爪状态: 已禁用 (MotorType.NONE)")
    print(f"关节自由度: {robot_config.joint_dof}")
    print("=" * 50)

    arx5_joint_controller = arx5.Arx5JointController(
        robot_config, controller_config, interface
    )

    # Or you can directly use the model and interface name
    # arx5_joint_controller = arx5.Arx5JointController(model, interface)

    np.set_printoptions(precision=3, suppress=True)
    arx5_joint_controller.set_log_level(arx5.LogLevel.ERROR)
    robot_config = arx5_joint_controller.get_robot_config()
    controller_config = arx5_joint_controller.get_controller_config()

    # 应用gain设置
    arx5_joint_controller.set_gain(gain)
    print(f"✓ Gain设置完成: kp={gain.kp()}, kd={gain.kd()}")

    # 移除了 step_num 变量，因为不再需要运动控制

    # 启动状态打印线程
    stop_printing = threading.Event()
    state_thread = threading.Thread(
        target=print_state_continuously, args=(arx5_joint_controller, stop_printing)
    )
    state_thread.daemon = True
    state_thread.start()

    print("开始重力补偿状态监控...")
    print("机器人将保持在当前位置，只进行状态监控")
    print("按 Ctrl+C 停止监控")
    print("=" * 50)

    # 先重置到home位置
    print("重置机器人到home位置...")
    # arx5_joint_controller.reset_to_home()
    print("✓ 机器人已重置到home位置")

    # 进入重力补偿模式，不发送运动指令
    print("进入重力补偿模式...")
    # arx5_joint_controller.set_to_damping()
    try:
        # 持续监控状态，不发送控制指令
        print("开始持续状态监控...")
        while True:
            # 只进行通信，不发送控制指令
            if not USE_MULTITHREADING:
                arx5_joint_controller.send_recv_once()
            else:
                time.sleep(controller_config.controller_dt)

    except KeyboardInterrupt:
        print("\n\nUser interrupt, resetting to home")
        # print(f"Teleop recording is terminated. Resetting to home.")
        arx5_joint_controller.reset_to_home()
        arx5_joint_controller.set_to_damping()
    except Exception as e:
        print(f"\n\n监控过程中出现错误: {e}")
    finally:
        # 停止状态打印线程
        stop_printing.set()
        print("\n状态打印已停止")
        print("重力补偿监控结束")


main()
