from pylibfranka_controllers import FrankaCartesianController
import time

robot = FrankaCartesianController("192.168.99.111")
if not robot.connect():
            exit(-1)

# # 阻塞模式（默认）：等待到达目标后返回
# robot.move_to_pose([0.5, 0.0, 0.3, 1, 0, 0, 0], velocity=0.1)
# print("Moved to pose.")

# robot.move_delta([0.1, 0, 0.05], velocity=0.1)
# print("Moved delta.")
# 获取当前位姿
pose = robot.get_pose()
print(f"当前位姿: {pose}")
print(f"  位置: {pose[:3]}")
print(f"  四元数: {pose[3:]}")