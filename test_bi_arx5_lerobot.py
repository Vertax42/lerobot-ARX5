#!/usr/bin/env python3

"""
测试 BiARX5 机器人在 LeRobot 框架下的集成
"""

import sys
import os
import logging
import signal
import atexit

# 添加 LeRobot 源码路径
sys.path.insert(0, "/home/ubuntu/lerobot-ARX5/src")

from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.bi_arx5.config_bi_arx5 import BiARX5Config

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_bi_arx5_integration():
    """测试 BiARX5 机器人在 LeRobot 框架下的集成"""

    print("=" * 60)
    print("测试 BiARX5 机器人在 LeRobot 框架下的集成")
    print("=" * 60)

    try:
        # 1. 创建配置
        print("\n1. 创建 BiARX5 配置...")
        config = BiARX5Config(
            id="test_bi_arx5",
            left_arm_model="X5",
            left_arm_port="can1",
            right_arm_model="X5",
            right_arm_port="can3",
            log_level="INFO",
            use_multithreading=True,
        )
        print(f"✓ 配置创建成功: {config.type}")
        print(f"  - 左臂模型: {config.left_arm_model}")
        print(f"  - 左臂端口: {config.left_arm_port}")
        print(f"  - 右臂模型: {config.right_arm_model}")
        print(f"  - 右臂端口: {config.right_arm_port}")
        print(f"  - 日志级别: {config.log_level}")
        print(f"  - 多线程: {config.use_multithreading}")

        # 2. 创建机器人实例
        print("\n2. 创建 BiARX5 机器人实例...")
        robot = make_robot_from_config(config)
        print(f"✓ 机器人实例创建成功: {robot}")
        print(f"  - 机器人名称: {robot.name}")
        print(f"  - 机器人类型: {robot.robot_type}")
        print(f"  - 机器人ID: {robot.id}")

        # 3. 检查连接状态
        print("\n3. 检查连接状态...")
        print(f"  - 连接状态: {robot.is_connected}")

        # 4. 打印 motors_ft 信息
        print("\n4. 打印 _motors_ft 信息...")
        motors_ft = robot._motors_ft
        print(f"✓ _motors_ft 包含 {len(motors_ft)} 个电机特征:")
        for motor_name, motor_type in motors_ft.items():
            print(f"  - {motor_name}: {motor_type}")

        # 5. 打印 observation_features 信息
        print("\n5. 打印 observation_features 信息...")
        obs_features = robot.observation_features
        print(f"✓ observation_features 包含 {len(obs_features)} 个特征:")
        for feature_name, feature_type in obs_features.items():
            print(f"  - {feature_name}: {feature_type}")

        # 6. 打印 action_features 信息
        print("\n6. 打印 action_features 信息...")
        action_features = robot.action_features
        print(f"✓ action_features 包含 {len(action_features)} 个特征:")
        for feature_name, feature_type in action_features.items():
            print(f"  - {feature_name}: {feature_type}")

        # 7. 尝试连接机器人（注意：这会尝试连接真实硬件）
        print("\n7. 尝试连接机器人...")
        print("⚠️  注意：这将尝试连接真实的 ARX5 硬件")
        user_input = input("是否继续连接？(y/N): ").strip().lower()

        if user_input == "y":
            try:
                robot.connect(calibrate=False, go_to_home=True)  # 回零
                print("✓ 机器人连接成功！")

                # 检查重力补偿模式
                if robot.is_gravity_compensation_mode():
                    print("✓ 机器人处于重力补偿模式")
                else:
                    print("⚠️  机器人未处于重力补偿模式")

                # 获取一次观测
                print("\n8. 获取机器人观测...")
                observation = robot.get_observation()
                print(f"✓ 成功获取观测，包含 {len(observation)} 个数据:")

                # 详细打印观测字典
                print("\n完整的观测字典内容:")
                print("-" * 50)
                for key, value in observation.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value}")
                    elif hasattr(value, "shape"):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                        if value.size <= 10:  # 如果数据量小，打印具体值
                            print(f"    values: {value}")
                    else:
                        print(f"  {key}: {type(value)} = {value}")
                print("-" * 50)

                # 分类显示数据
                print("\n按类型分类的观测数据:")

                # 电机数据
                motor_data = {
                    k: v
                    for k, v in observation.items()
                    if any(
                        motor in k for motor in ["left_joint", "right_joint", "gripper"]
                    )
                }
                print(f"\n📊 电机数据 ({len(motor_data)} 个):")
                for key, value in motor_data.items():
                    print(f"  - {key}: {value}")

                # 摄像头数据
                camera_data = {
                    k: v for k, v in observation.items() if k not in motors_ft
                }
                if camera_data:
                    print(f"\n📷 摄像头数据 ({len(camera_data)} 个):")
                    for key, value in camera_data.items():
                        if hasattr(value, "shape"):
                            print(
                                f"  - {key}: shape={value.shape}, dtype={value.dtype}"
                            )
                        else:
                            print(f"  - {key}: {type(value)}")

                # 断开连接
                print("\n9. 断开机器人连接...")
                robot.disconnect()
                print("✓ 机器人已断开连接")

            except Exception as e:
                print(f"❌ 连接失败: {e}")
                print("这可能是因为:")
                print("  - 硬件未连接")
                print("  - CAN 总线未配置")
                print("  - 权限问题")
        else:
            print("跳过硬件连接测试")

        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_bi_arx5_integration()
