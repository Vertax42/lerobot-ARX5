#!/usr/bin/env python3

"""
Example configuration for dual SpaceMouse teleoperation.

This demonstrates different dual-device setups for various use cases.
"""

from lerobot.teleoperators.spacemouse import (
    SpacemouseConfig,
    DeviceConfig,
    SpacemouseTeleop,
)
import numpy as np


def example_position_orientation_split():
    """
    Example 1: Left device controls position, right device controls orientation.

    This is the most common dual-hand setup for precise manipulation tasks.
    """
    config = SpacemouseConfig(
        multi_device_mode=True,
        left_device=DeviceConfig(
            device_index=0,
            enabled_axes=(
                True,
                True,
                True,
                False,
                False,
                False,
            ),  # X, Y, Z position only
            pos_sensitivity=0.8,
            ori_sensitivity=0.0,
            deadzone=0.1,
            gripper_speed=0.6,
        ),
        right_device=DeviceConfig(
            device_index=1,
            enabled_axes=(
                False,
                False,
                False,
                True,
                True,
                True,
            ),  # Roll, pitch, yaw only
            pos_sensitivity=0.0,
            ori_sensitivity=1.5,
            deadzone=0.1,
            gripper_speed=0.6,
        ),
        control_dt=0.005,  # 200 Hz control loop
        gripper_width=1.0,
    )

    return config


def example_dual_arm_control():
    """
    Example 2: Each device controls different aspects of dual-arm robots.

    Left device: Primary arm (position + Z rotation)
    Right device: Secondary arm (orientation control)
    """
    config = SpacemouseConfig(
        multi_device_mode=True,
        left_device=DeviceConfig(
            device_index=0,
            enabled_axes=(True, True, True, False, False, True),  # XYZ + Yaw
            pos_sensitivity=1.0,
            ori_sensitivity=1.0,
            deadzone=0.05,
        ),
        right_device=DeviceConfig(
            device_index=1,
            enabled_axes=(False, False, False, True, True, False),  # Roll + Pitch
            pos_sensitivity=0.0,
            ori_sensitivity=2.0,
            deadzone=0.05,
        ),
        control_dt=0.005,  # 200 Hz control loop
    )

    return config


def example_fine_coarse_control():
    """
    Example 3: Left for coarse movement, right for fine adjustment.
    """
    config = SpacemouseConfig(
        multi_device_mode=True,
        left_device=DeviceConfig(
            device_index=0,
            enabled_axes=(True, True, True, True, True, True),  # All axes
            pos_sensitivity=2.0,  # Higher sensitivity for coarse movement
            ori_sensitivity=3.0,
            deadzone=0.2,  # Higher deadzone
        ),
        right_device=DeviceConfig(
            device_index=1,
            enabled_axes=(True, True, True, True, True, True),  # All axes
            pos_sensitivity=0.1,  # Lower sensitivity for fine adjustment
            ori_sensitivity=0.2,
            deadzone=0.05,  # Lower deadzone for precision
        ),
        control_dt=0.005,  # 200 Hz control loop
    )

    return config


def demo_spacemouse_teleoperation():
    """
    Demo function showing how to use SpaceMouse teleoperation.
    """
    print("🐭 SpaceMouse Teleoperation Demo")

    # Choose configuration
    config = example_position_orientation_split()

    print(f"Multi-device mode: {config.multi_device_mode}")
    print(f"Left device axes: {config.left_device.enabled_axes}")
    print(f"Right device axes: {config.right_device.enabled_axes}")

    # Create teleoperator
    teleop = SpacemouseTeleop(config)

    print(f"Action features: {teleop.action_features}")

    # In a real application, you would:
    # 1. Connect to the spacemouse
    # teleop.connect()
    #
    # 2. Get actions in a loop
    # while True:
    #     action = teleop.get_action()
    #     print(f"Target pose: {action}")
    #     # Send action to robot...
    #
    # 3. Disconnect when done
    # teleop.disconnect()

    print(
        "✅ Demo completed. To actually use SpaceMouse, ensure devices are connected."
    )


if __name__ == "__main__":
    demo_spacemouse_teleoperation()
