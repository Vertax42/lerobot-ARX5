#!/usr/bin/env python3
"""
夹爪零点重新校准脚本
用于修复夹爪位置读数不正确的问题
"""
import os
import sys
import time
import click

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def recalibrate_gripper_zero(model: str, interface: str):
    """
    重新校准夹爪零点
    """
    print("🔧 夹爪零点重新校准")
    print("📋 校准步骤:")
    print("   1. 确保夹爪在完全打开位置")
    print("   2. 运行校准程序")
    print("   3. 验证位置读数")
    
    # 等待用户确认夹爪位置
    input("请确保夹爪在完全打开位置，然后按回车键继续...")
    
    joint_controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", 6
    )
    joint_controller_config.gravity_compensation = False
    
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    
    # 创建控制器
    joint_controller = arx5.Arx5JointController(
        robot_config, joint_controller_config, interface
    )
    
    print("📊 校准前状态:")
    try:
        state = joint_controller.get_joint_state()
        print(f"   位置: {state.gripper_pos:.3f}m")
        print(f"   速度: {state.gripper_vel:.3f}m/s")
        print(f"   扭矩: {state.gripper_torque:.3f}N⋅m")
    except Exception as e:
        print(f"❌ 无法获取状态: {e}")
        return
    
    print("\n🔧 开始重新校准...")
    try:
        # 运行校准
        joint_controller.calibrate_gripper()
        print("✅ 校准完成!")
        
        # 显示校准后状态
        print("\n📊 校准后状态:")
        state = joint_controller.get_joint_state()
        print(f"   位置: {state.gripper_pos:.3f}m")
        print(f"   速度: {state.gripper_vel:.3f}m/s")
        print(f"   扭矩: {state.gripper_torque:.3f}N⋅m")
        
        # 检查是否在合理范围
        if 0 <= state.gripper_pos <= robot_config.gripper_width:
            print("✅ 夹爪位置现在在安全范围内!")
        else:
            print("⚠️  夹爪位置仍然超出范围")
            print(f"   允许范围: 0 ~ {robot_config.gripper_width:.3f}m")
            
    except Exception as e:
        print(f"❌ 校准失败: {e}")
        print("💡 建议:")
        print("   1. 检查CAN总线连接")
        print("   2. 重启机器人控制器")
        print("   3. 检查夹爪机械状态")


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def test_gripper_movement(model: str, interface: str):
    """
    测试夹爪运动
    """
    print("🧪 测试夹爪运动...")
    
    joint_controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", 6
    )
    joint_controller_config.gravity_compensation = False
    
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    
    joint_controller = arx5.Arx5JointController(
        robot_config, joint_controller_config, interface
    )
    
    print("📊 当前状态:")
    try:
        state = joint_controller.get_joint_state()
        print(f"   位置: {state.gripper_pos:.3f}m")
        print(f"   速度: {state.gripper_vel:.3f}m/s")
        print(f"   扭矩: {state.gripper_torque:.3f}N⋅m")
        
        print("\n🎮 测试夹爪运动 (按Ctrl+C停止):")
        while True:
            state = joint_controller.get_joint_state()
            print(f"\r位置: {state.gripper_pos:.3f}m, 速度: {state.gripper_vel:.3f}m/s, 扭矩: {state.gripper_torque:.3f}N⋅m", end="", flush=True)
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n✅ 测试完成")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")


if __name__ == "__main__":
    # 默认运行零点校准
    recalibrate_gripper_zero()
