from enum import IntEnum
from typing import Any

import numpy as np

from lerobot.utils.robot_utils import (
    matrix_to_pose7d,
    quaternion_to_euler,
    rotation_6d_to_quaternion,
    xyz_rpy_to_matrix,
    normalize_quaternion,
    quaternion_to_matrix,
)
from scipy.spatial.transform import Rotation as R

pos_sensitivity = 0.005  # m/step (最大平移速度)
start_pos = np.array([0.5, 0, 0.5], dtype=np.float32) # 初始位置
start_quat = np.array([1, 0, 0, 0], dtype=np.float32) # 初始姿态 (wxyz)

# Get movement deltas from the controller
delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz = 1, 0, 0, 3.1415926, 0, 0 # Example deltas from gamepad input
rel_pos = np.array([delta_x, delta_y, delta_z], dtype=np.float32)
scaled_rel_pos = rel_pos * pos_sensitivity
target_pos = start_pos + scaled_rel_pos
start_pos = target_pos

# quaternion update
rotation_delta = np.array([delta_rx, delta_ry, delta_rz]) * pos_sensitivity
rotation_delta = R.from_euler('xyz', rotation_delta).as_matrix()
print("Rotation Delta:\n", rotation_delta)
start_matrix = quaternion_to_matrix(np.array([0,0,0,start_quat[0],start_quat[1],start_quat[2],start_quat[3]]), input_format="wxyz")
print("Start Matrix:\n", start_matrix)

current_ee_matrix = rotation_delta @ start_matrix[:3, :3] #3x3
print("Current EE Matrix:\n", current_ee_matrix)
current_ee_quat9d = np.array(current_ee_matrix).flatten() # 9d
start_quat = rotation_6d_to_quaternion(current_ee_quat9d[:6])

action_dict = {
    "tcp.x": target_pos[0],
    "tcp.y": target_pos[1],
    "tcp.z": target_pos[2],
    "tcp.r1": current_ee_matrix[0, 0],
    "tcp.r2": current_ee_matrix[1, 0],
    "tcp.r3": current_ee_matrix[2, 0],
    "tcp.r4": current_ee_matrix[0, 1],
    "tcp.r5": current_ee_matrix[1, 1],
    "tcp.r6": current_ee_matrix[2, 1],
}
print("Action Dict:", action_dict)