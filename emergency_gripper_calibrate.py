#!/usr/bin/env python3
"""
临时夹爪校准脚本 - 禁用安全检查
用于处理夹爪卡在超出安全范围位置的情况
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
def emergency_calibrate_gripper(model: str, interface: str):
    """
    紧急夹爪校准 - 禁用安全检查
    用于处理夹爪卡在超出安全范围位置的情况
    """
    print("🚨 紧急夹爪校准模式")
    print("⚠️  警告: 此模式禁用了安全检查，请确保夹爪机械安全!")
    print("📋 请手动将夹爪调整到安全位置 (0~0.085m范围内)")
    
    # 等待用户确认
    input("按回车键继续...")
    
    joint_controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", 6
    )
    joint_controller_config.gravity_compensation = False
    
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    
    # 创建控制器
    joint_controller = arx5.Arx5JointController(
        robot_config, joint_controller_config, interface
    )
    
    print("🔧 开始夹爪校准...")
    
    try:
        # 运行校准
        joint_controller.calibrate_gripper()
        print("✅ 夹爪校准完成!")
        
        # 显示当前状态
        print("📊 当前夹爪状态:")
        state = joint_controller.get_joint_state()
        print(f"   位置: {state.gripper_pos:.3f}m")
        print(f"   速度: {state.gripper_vel:.3f}m/s")
        print(f"   扭矩: {state.gripper_torque:.3f}N⋅m")
        
    except Exception as e:
        print(f"❌ 校准失败: {e}")
        print("💡 建议:")
        print("   1. 手动调整夹爪到安全位置")
        print("   2. 检查CAN总线连接")
        print("   3. 重启机器人控制器")


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def check_gripper_status(model: str, interface: str):
    """
    检查夹爪当前状态
    """
    print("📊 检查夹爪状态...")
    
    joint_controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", 6
    )
    joint_controller_config.gravity_compensation = False
    
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    
    joint_controller = arx5.Arx5JointController(
        robot_config, joint_controller_config, interface
    )
    
    try:
        state = joint_controller.get_joint_state()
        print(f"✅ 夹爪状态:")
        print(f"   位置: {state.gripper_pos:.3f}m")
        print(f"   速度: {state.gripper_vel:.3f}m/s") 
        print(f"   扭矩: {state.gripper_torque:.3f}N⋅m")
        
        # 检查是否在安全范围
        if 0 <= state.gripper_pos <= robot_config.gripper_width:
            print("✅ 夹爪位置在安全范围内")
        else:
            print("⚠️  夹爪位置超出安全范围!")
            print(f"   允许范围: 0 ~ {robot_config.gripper_width:.3f}m")
            
    except Exception as e:
        print(f"❌ 无法获取夹爪状态: {e}")


if __name__ == "__main__":
    # 默认运行紧急校准
    emergency_calibrate_gripper()
