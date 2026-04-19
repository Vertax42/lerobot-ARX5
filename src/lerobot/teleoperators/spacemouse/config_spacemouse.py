#!/usr/bin/env python

# Copyright 2025 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field  # noqa: F401
from typing import Tuple, Optional, Dict, Any

from ..config import TeleoperatorConfig


@dataclass
class DeviceConfig:
    """Configuration for a single SpaceMouse device."""
    device_name: Optional[str] = None  # Specific device name, None for auto-detect
    device_index: int = 0  # Device index when multiple same devices
    pos_sensitivity: float = 0.8  # Position sensitivity multiplier (m/s at max deflection)
    ori_sensitivity: float = 1.5  # Orientation sensitivity multiplier (rad/s at max deflection)
    gripper_speed: float = 0.6  # Gripper speed
    deadzone: float = 0.02  # Deadzone threshold (smaller = more responsive)
    # Axis inversion after coordinate transformation [x, y, z, roll, pitch, yaw]
    # x: True (SpaceMouse forward=-Y needs inversion to become Robot +X)
    # yaw: True (SpaceMouse yaw needs inversion)
    invert_axes: Tuple[bool, bool, bool, bool, bool, bool] = (True, False, False, False, False, True)
    swap_gripper_buttons: bool = False
    enabled_axes: Tuple[bool, bool, bool, bool, bool, bool] = (True, True, True, True, True, True)  # Which axes to use


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpacemouseConfig(TeleoperatorConfig):
    """Configuration for 3D Spacemouse teleoperator with multi-device support.

    This teleoperator provides 6-DoF absolute pose control (translation + rotation)
    suitable for end-effector teleoperation. Supports both single device and dual-hand modes.

    Single Device Mode (default):
        Uses one SpaceMouse for combined position+orientation control.

    Dual Hand Mode:
        - Left device: typically position control (enabled_axes=[True,True,True,False,False,False])
        - Right device: typically orientation control (enabled_axes=[False,False,False,True,True,True])
        - Combined output for unified control

    Attributes:
        multi_device_mode: Enable dual SpaceMouse mode
        left_device: Configuration for left hand device
        right_device: Configuration for right hand device
        filter_window_size: Moving average filter window size for smoothing
        control_dt: Control loop period in seconds
        gripper_width: Maximum gripper position ratio

    Legacy single-device attributes (used when multi_device_mode=False):
        pos_sensitivity, ori_sensitivity, gripper_speed, deadzone, invert_axes, swap_gripper_buttons
    """

    # Multi-device configuration
    multi_device_mode: bool = False
    left_device: DeviceConfig = field(default_factory=lambda: DeviceConfig(
        device_index=0,
        enabled_axes=(True, True, True, False, False, False),  # Position only
        pos_sensitivity=0.8,
        ori_sensitivity=0.0,  # Disabled
    ))
    right_device: DeviceConfig = field(default_factory=lambda: DeviceConfig(
        device_index=1,  # Second physical device (0-based index)
        enabled_axes=(False, False, False, True, True, True),  # Orientation only
        pos_sensitivity=0.0,  # Disabled
        ori_sensitivity=1.5,
    ))

    # Global settings
    filter_window_size: int = 1  # Moving average filter window size (1=disabled for best responsiveness)
    control_dt: float = 0.01  # Control loop period in seconds (should match external loop)
    gripper_width: float = 1.0  # Maximum gripper position (ratio of gripper_max_pos)

    # Legacy single-device settings (used when multi_device_mode=False)
    pos_sensitivity: float = 0.4  # default 0.4 m/s at max deflection
    ori_sensitivity: float = 0.8  # default 0.8 rad/s at max deflection
    gripper_speed: float = 0.6  # ratio of gripper_max_pos / s for gripper open/close
    deadzone: float = 0.02  # [0-1] threshold for filtering noise (smaller = more responsive)
    # Axis inversion after coordinate transformation [x, y, z, roll, pitch, yaw]
    # x: True (SpaceMouse forward=-Y needs inversion to become Robot +X)
    # yaw: True (SpaceMouse yaw needs inversion)
    invert_axes: Tuple[bool, bool, bool, bool, bool, bool] = (
        False,   # x: no inversion
        True,  # y: inversion
        False,  # z
        True,  # roll: inversion
        True,  # pitch: inversion
        True,   # yaw: inversion
    )
    swap_gripper_buttons: bool = False  # default left button to close, right button to open
