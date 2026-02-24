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
import time


import asyncio 
import numpy as np 
from xense_franka import RobotInterface, FrankaController

# pos_sensitivity = 0.005  # m/step (最大平移速度)
# start_pos = np.array([0.5, 0, 0.5], dtype=np.float32) # 初始位置
# start_quat = np.array([1, 0, 0, 0], dtype=np.float32) # 初始姿态 (wxyz)

# # Get movement deltas from the controller
# delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz = 1, 0, 0, 3.1415926, 0, 0 # Example deltas from gamepad input
# rel_pos = np.array([delta_x, delta_y, delta_z], dtype=np.float32)
# scaled_rel_pos = rel_pos * pos_sensitivity
# target_pos = start_pos + scaled_rel_pos
# start_pos = target_pos

# # quaternion update
# rotation_delta = np.array([delta_rx, delta_ry, delta_rz]) * pos_sensitivity
# rotation_delta = R.from_euler('xyz', rotation_delta).as_matrix()
# print("Rotation Delta:\n", rotation_delta)
# start_matrix = quaternion_to_matrix(np.array([0,0,0,start_quat[0],start_quat[1],start_quat[2],start_quat[3]]), input_format="wxyz")
# print("Start Matrix:\n", start_matrix)

# current_ee_matrix = rotation_delta @ start_matrix[:3, :3] #3x3
# print("Current EE Matrix:\n", current_ee_matrix)
# current_ee_quat9d = np.array(current_ee_matrix).flatten() # 9d
# start_quat = rotation_6d_to_quaternion(current_ee_quat9d[:6])

# action_dict = {
#     "tcp.x": target_pos[0],
#     "tcp.y": target_pos[1],
#     "tcp.z": target_pos[2],
#     "tcp.r1": current_ee_matrix[0, 0],
#     "tcp.r2": current_ee_matrix[1, 0],
#     "tcp.r3": current_ee_matrix[2, 0],
#     "tcp.r4": current_ee_matrix[0, 1],
#     "tcp.r5": current_ee_matrix[1, 1],
#     "tcp.r6": current_ee_matrix[2, 1],
# }
# print("Action Dict:", action_dict)


get_current_tcp_pose_quat = lambda: [0.5, 0, 0.5, 1, 0, 0, 0] # Example current TCP pose (x, y, z, w, x, y, z)
robot_tcp_home_position = [0.573, -0.0096, 0.5328, 1.0, 0.0, 0.0, 0.0] # TCP home position (x, y, z, w, x, y, z)

now_pose = get_current_tcp_pose_quat()[:7]  # 只取 TCP 位姿部分
target_pose = robot_tcp_home_position  # TCP 位姿目标 (x, y, z, w, x, y, z)
vel = 0.1  # m/s
de = np.linalg.norm(np.array(target_pose[:3]) - np.array(now_pose[:3]))  # 根据距离和速度计算超时时间
timeout = de / vel
print(f"Moving {now_pose[:3]} to {target_pose[:3]} with distance {de} m and velocity {vel} m/s. Estimated time: {timeout:.2f} seconds.")
hz = 30.0

"""Move the robot to the goal position with linear interpolation."""
steps = int(timeout * hz)
print(steps)
path = np.linspace(now_pose, target_pose, steps)
for p in path:
    print(f"Moving to: {p}")
    time.sleep(1 / hz)