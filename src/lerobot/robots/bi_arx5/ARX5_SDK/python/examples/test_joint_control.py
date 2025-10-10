import os
import sys
import time

import click
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# 必须在路径设置后导入
import arx5_interface as arx5  # noqa: E402


def linear_func(t):
    return t


def easeInOutQuad(t):
    t *= 2
    if t < 1:
        return t * t / 2
    else:
        t -= 1
        return -(t * (t - 2) - 1) / 2


def easeInOutCubic(t):
    """三次贝塞尔缓动函数"""
    if t < 0.5:
        return 4 * t * t * t
    else:
        t = 2 * t - 2
        return (t * t * t + 2) / 2


def easeInSine(t):
    """正弦缓动函数 (ease-in)
    运动曲线：慢-快-慢，适合需要缓慢启动的场景, 如机械臂启动
    """
    return 1 - np.cos(t * np.pi / 2)


def easeOutSine(t):
    """正弦缓动函数 (ease-out)
    运动曲线：快-快-慢，适合需要平稳停止的场景, 如机械臂停止
    """
    return np.sin(t * np.pi / 2)


def easeInOutSine(t):
    """正弦缓动函数 (ease-in-out)
    运动曲线：慢-快-慢，适合需要平滑运动的场景
    """
    return -(np.cos(np.pi * t) - 1) / 2


def smooth_joint_interpolation(
    controller,
    target_poses,
    duration,
    control_dt,
    gripper_target=0.0,
    interpolation_func=easeInOutQuad,
    print_interval=100,
):
    """
    平滑关节插值函数

    参数:
    - controller: Arx5JointController 实例
    - target_poses: 目标关节位置 (numpy array, 6个关节)
    - duration: 插值持续时间 (秒)
    - control_dt: 控制周期 (秒)
    - gripper_target: 目标夹爪位置 (米)
    - interpolation_func: 插值函数 (默认 easeInOutQuad)
    - print_interval: 打印间隔 (步数)

    返回:
    - 最终关节状态
    """
    # 计算步数
    step_num = int(duration / control_dt)

    # 获取初始状态
    initial_state = controller.get_joint_state()
    initial_poses = initial_state.pos().copy()

    print(f"开始平滑插值: {step_num} 步, 持续时间: {duration:.3f}s")
    print(f"初始位置: {initial_poses[:3]}")
    print(f"目标位置: {target_poses[:3]}")

    # 执行插值
    for i in range(step_num):
        # 计算插值参数 (0 到 1)
        t = float(i) / (step_num - 1) if step_num > 1 else 0.0
        alpha = interpolation_func(t)

        # 计算当前目标位置
        cmd = arx5.JointState(controller.get_robot_config().joint_dof)
        cmd.pos()[:] = initial_poses + alpha * (target_poses - initial_poses)
        cmd.gripper_pos = initial_state.gripper_pos + alpha * (
            gripper_target - initial_state.gripper_pos
        )

        # 设置命令
        controller.set_joint_cmd(cmd)

        # 通信
        if not controller.get_controller_config().background_send_recv:
            controller.send_recv_once()
        else:
            time.sleep(control_dt)

        # 获取当前状态
        current_state = controller.get_joint_state()

        # 定期打印状态
        if i % print_interval == 0:
            print(
                f"步骤 {i:4d}: 目标={cmd.pos()[:3]}, 实际={current_state.pos()[:3]}, "
                f"速度={current_state.vel()[:3]}"
            )

    # 返回最终状态
    final_state = controller.get_joint_state()
    print(f"插值完成: 最终位置={final_state.pos()[:3]}")
    return final_state


def create_custom_interpolation(
    controller,
    target_poses,
    duration,
    control_dt,
    gripper_target=0.0,
    interpolation_func=easeInOutQuad,
    print_interval=100,
):
    """
    创建自定义插值函数的便捷方法

    示例用法:
    # 使用三次贝塞尔插值
    create_custom_interpolation(controller, poses, 3.0, dt, interpolation_func=easeInOutCubic)

    # 使用正弦插值
    create_custom_interpolation(controller, poses, 3.0, dt, interpolation_func=easeInOutSine)
    """
    return smooth_joint_interpolation(
        controller,
        target_poses,
        duration,
        control_dt,
        gripper_target,
        interpolation_func,
        print_interval,
    )


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def main(model: str, interface: str):

    # To initialize robot with different configurations,
    # you can create RobotConfig and ControllerConfig by yourself and modify based on it
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    robot_config.gripper_motor_type = arx5.MotorType.NONE
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    # Modify the default configuration here
    # controller_config.controller_dt = 0.01 # etc.

    USE_MULTITHREADING = True
    if USE_MULTITHREADING:
        # Will create another thread that communicates with the arm, so each send_recv_once() will take no time
        # for the main thread to execute. Otherwise (without background send/recv), send_recv_once() will block the
        # main thread until the arm responds (usually 2ms).
        controller_config.background_send_recv = True
    else:
        controller_config.background_send_recv = False

    arx5_joint_controller = arx5.Arx5JointController(
        robot_config, controller_config, interface
    )

    # Or you can directly use the model and interface name
    # arx5_joint_controller = arx5.Arx5JointController(model, interface)

    np.set_printoptions(precision=3, suppress=True)
    arx5_joint_controller.set_log_level(arx5.LogLevel.DEBUG)
    robot_config = arx5_joint_controller.get_robot_config()
    controller_config = arx5_joint_controller.get_controller_config()

    # step_num = 1500  # 不再需要，使用插值函数中的动态计算

    print("=== 调试信息 ===")
    print(
        f"controller_config.background_send_recv = {controller_config.background_send_recv}"
    )
    print(f"USE_MULTITHREADING = {USE_MULTITHREADING}")

    # 测试send_recv_once的行为来推断后台通信状态
    print("\n=== 测试send_recv_once行为 ===")
    try:
        arx5_joint_controller.send_recv_once()
        print("send_recv_once() 执行成功 - 可能后台通信未启用")
    except Exception as e:
        print(f"send_recv_once() 异常: {e}")

    # 获取初始状态
    initial_state = arx5_joint_controller.get_joint_state()
    print(f"初始关节位置: {initial_state.pos()[:3]}")

    # 获取当前增益设置
    current_gain = arx5_joint_controller.get_gain()
    print(f"当前增益 kp: {current_gain.kp()[:3]}")
    print(f"当前增益 kd: {current_gain.kd()[:3]}")

    # 设置正确的增益
    print("\n=== 设置正确的增益 ===")
    proper_gain = arx5.Gain(robot_config.joint_dof)
    proper_gain.kp()[:] = controller_config.default_kp  # 使用默认位置增益
    proper_gain.kd()[:] = controller_config.default_kd  # 使用默认速度增益
    arx5_joint_controller.set_gain(proper_gain)
    print(f"设置后增益 kp: {proper_gain.kp()[:3]}")
    print(f"设置后增益 kd: {proper_gain.kd()[:3]}")

    # 获取当前命令
    current_cmd = arx5_joint_controller.get_joint_cmd()
    print(f"当前关节命令: {current_cmd.pos()[:3]}")

    # 测试1: 使用新的插值函数
    print("\n=== 测试1: 使用线性插值函数 ===")
    target_joint_poses = np.array([0.0, 0.948, 0.858, -0.573, 0.0, 0.0])

    # 使用平滑插值函数
    final_state = smooth_joint_interpolation(
        controller=arx5_joint_controller,
        target_poses=target_joint_poses,
        duration=3.0,  # 3秒
        control_dt=controller_config.controller_dt,
        gripper_target=robot_config.gripper_width,
        interpolation_func=linear_func,
        print_interval=100,
    )

    # 测试2: 使用线性插值回到初始位置
    print("\n=== 测试2: 使用正弦插值回到初始位置 ===")
    home_poses = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 使用线性插值函数回到初始位置
    final_state = create_custom_interpolation(
        controller=arx5_joint_controller,
        target_poses=home_poses,
        duration=2.5,  # 2秒
        control_dt=controller_config.controller_dt,
        gripper_target=0.0,
        interpolation_func=easeInOutSine,
        print_interval=100,
    )

    # 获取最终状态
    print(f"最终位置: {final_state.pos()}")
    print(f"最终速度: {final_state.vel()}")

    # 获取最终增益设置
    final_gain = arx5_joint_controller.get_gain()
    print(f"最终增益 kp: {final_gain.kp()}")
    print(f"最终增益 kd: {final_gain.kd()}")

    # 注释掉reset_to_home，避免程序退出时的突然切换
    # arx5_joint_controller.reset_to_home()

    # 手动设置到阻尼模式，保持重力补偿
    # print("设置到阻尼模式，保持重力补偿...")
    # arx5_joint_controller.set_to_damping()

    # 等待一段时间让机械臂稳定
    time.sleep(2)

    # 额外示例：展示不同插值函数的使用
    print("\n=== 插值函数使用示例 ===")
    print("可用的插值函数:")
    print("1. easeInOutQuad - 二次贝塞尔缓动 (默认)")
    print("2. easeInOutCubic - 三次贝塞尔缓动")
    print("3. easeInSine - 正弦缓动 (ease-in)")
    print("4. easeOutSine - 正弦缓动 (ease-out)")
    print("5. easeInOutSine - 正弦缓动 (ease-in-out)")
    print("6. linear - 线性插值")

    print("\n使用示例:")
    print("# 平滑插值到目标位置")
    print("smooth_joint_interpolation(controller, target_poses, 3.0, control_dt)")
    print()
    print("# 线性插值")
    print("linear_joint_interpolation(controller, target_poses, 2.0, control_dt)")
    print()
    print("# 自定义插值函数")
    print(
        "create_custom_interpolation(controller, poses, 3.0, dt, interpolation_func=easeInOutCubic)"
    )


main()
