# ARX5 SDK Python Package

纯ARX5 SDK Python包，不包含相机系统和机器人控制器，只提供ARX5机械臂的原生SDK功能。

## 功能特性

- ✅ **ARX5机械臂控制**: 支持X5和L5机械臂的关节控制
- ✅ **双臂支持**: 支持同时控制左右两个机械臂
- ✅ **原生SDK**: 直接使用ARX5 C++ SDK的Python绑定
- ✅ **轻量级**: 不包含额外的相机和机器人控制功能
- ✅ **即插即用**: 无需复杂安装，直接导入使用

## 快速测试

```bash
# 进入包目录
cd arx5_sdk_only

# 运行简单测试
python simple_test.py
```

## 快速开始

### 方法1: 直接使用（推荐）

```python
import sys
from pathlib import Path

# 添加SDK路径
sdk_path = Path(__file__).parent / "arx5_sdk" / "python"
sys.path.insert(0, str(sdk_path))

# 导入ARX5接口
import arx5_interface as arx5

# 创建机械臂控制器
arm = arx5.Arx5JointController("X5", "can0")
```

### 方法2: 安装为包

```bash
# 进入包目录
cd arx5_sdk_only

# 安装包
pip install -e .
```

### 方法3: 复制到项目

```bash
# 将arx5_sdk_only目录复制到你的项目中
# 然后直接使用
```

## 使用示例

### 1. 基本使用

```python
import sys
from pathlib import Path

# 添加SDK路径
sdk_path = Path(__file__).parent / "arx5_sdk" / "python"
sys.path.insert(0, str(sdk_path))

# 导入ARX5接口
import arx5_interface as arx5

# 创建机械臂控制器
left_arm = arx5.Arx5JointController("X5", "can0")

# 获取配置
robot_config = left_arm.get_robot_config()
controller_config = left_arm.get_controller_config()

# 获取当前状态
joint_state = left_arm.get_joint_state()
print(f"当前关节位置: {joint_state.pos()}")
print(f"当前夹爪位置: {joint_state.gripper_pos}")

# 控制机械臂
cmd = arx5.JointState(robot_config.joint_dof)
cmd.pos()[:] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
cmd.gripper_pos = 0.5

left_arm.set_joint_cmd(cmd)

# 清理
del left_arm
```

### 2. 双臂控制

```python
import sys
from pathlib import Path

# 添加SDK路径
sdk_path = Path(__file__).parent / "arx5_sdk" / "python"
sys.path.insert(0, str(sdk_path))

# 导入ARX5接口
import arx5_interface as arx5

# 创建双臂控制器
left_arm = arx5.Arx5JointController("X5", "can0")
right_arm = arx5.Arx5JointController("X5", "can1")

# 获取配置
robot_config = left_arm.get_robot_config()

# 创建命令
left_cmd = arx5.JointState(robot_config.joint_dof)
right_cmd = arx5.JointState(robot_config.joint_dof)

# 设置目标位置
left_cmd.pos()[:] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
right_cmd.pos()[:] = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]

# 发送命令
left_arm.set_joint_cmd(left_cmd)
right_arm.set_joint_cmd(right_cmd)

# 清理
del left_arm
del right_arm
```

## API参考

### 主要类

- `arx5.Arx5JointController(model, interface)`: 关节控制器
- `arx5.JointState(dof)`: 关节状态
- `arx5.RobotConfig`: 机器人配置
- `arx5.ControllerConfig`: 控制器配置

### 主要方法

#### Arx5JointController

- `get_robot_config()`: 获取机器人配置
- `get_controller_config()`: 获取控制器配置
- `get_joint_state()`: 获取关节状态
- `set_joint_cmd(cmd)`: 设置关节命令
- `reset_to_home()`: 重置到初始位置
- `set_log_level(level)`: 设置日志级别

#### JointState

- `pos()`: 获取/设置关节位置
- `vel()`: 获取/设置关节速度
- `gripper_pos`: 获取/设置夹爪位置

## 系统要求

### 硬件要求

- ARX5机械臂 (X5或L5)
- CAN接口支持
- Linux系统

### 软件要求

- Python 3.8+
- numpy
- CAN接口驱动

## 故障排除

### 1. CAN接口问题

```bash
# 检查CAN接口
ip link show can0
ip link show can1

# 启动CAN接口
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
```

### 2. 库文件问题

确保以下库文件在正确位置：

- `arx5_interface.cpython-*.so`
- `libhardware.so`
- `libsolver.so`

### 3. 权限问题

```bash
# 确保用户有CAN接口权限
sudo usermod -a -G dialout $USER
# 重新登录后生效
```

## 示例项目

### 简单轨迹控制

```python
import sys
from pathlib import Path
import time
import numpy as np

# 添加SDK路径
sdk_path = Path(__file__).parent / "arx5_sdk" / "python"
sys.path.insert(0, str(sdk_path))

# 导入ARX5接口
import arx5_interface as arx5

# 创建控制器
arm = arx5.Arx5JointController("X5", "can0")
robot_config = arm.get_robot_config()

# 定义轨迹点
trajectory = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]

# 执行轨迹
for point in trajectory:
    cmd = arx5.JointState(robot_config.joint_dof)
    cmd.pos()[:] = point
    arm.set_joint_cmd(cmd)
    time.sleep(1.0)

# 清理
del arm
```

### 实时控制循环

```python
import sys
from pathlib import Path
import time
import numpy as np

# 添加SDK路径
sdk_path = Path(__file__).parent / "arx5_sdk" / "python"
sys.path.insert(0, str(sdk_path))

# 导入ARX5接口
import arx5_interface as arx5

# 创建控制器
arm = arx5.Arx5JointController("X5", "can0")
robot_config = arm.get_robot_config()
controller_config = arm.get_controller_config()

# 控制循环
dt = controller_config.controller_dt
for i in range(1000):
    # 获取当前状态
    state = arm.get_joint_state()
    current_pos = state.pos()
    
    # 计算目标位置 (简单的正弦波轨迹)
    target_pos = 0.1 * np.sin(2 * np.pi * i * dt)
    cmd = arx5.JointState(robot_config.joint_dof)
    cmd.pos()[:] = [target_pos] * robot_config.joint_dof
    
    # 发送命令
    arm.set_joint_cmd(cmd)
    
    # 等待下一个控制周期
    time.sleep(dt)

# 清理
del arm
```

## 文件结构

```text
arx5_sdk_only/
├── arx5_sdk/                    # ARX5 SDK核心文件
│   ├── python/                  # Python绑定
│   │   ├── arx5_interface.cpython-*.so  # 编译的Python扩展
│   │   ├── arx5_interface.pyi           # 类型提示文件
│   │   └── examples/                    # 示例代码
│   ├── lib/                     # 库文件
│   │   ├── x86_64/              # x86_64架构库
│   │   │   ├── libhardware.so
│   │   │   └── libsolver.so
│   │   └── aarch64/             # ARM架构库
│   │       ├── libhardware.so
│   │       └── libsolver.so
│   └── models/                  # 机器人模型文件
│       ├── X5.urdf
│       ├── X7_left.urdf
│       └── meshes/
├── __init__.py                  # Python包初始化
├── setup.py                     # 安装脚本
├── example_usage.py             # 详细使用示例
├── simple_test.py               # 简单测试脚本
├── README.md                    # 使用说明
├── requirements.txt             # 依赖列表
└── install.sh                   # 安装脚本
```

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个SDK包！
