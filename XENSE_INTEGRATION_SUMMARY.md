# Xense Tactile Sensor Integration Summary

## 概述

我已经为 LeRobot 框架创建了完整的 Xense 触觉传感器集成模块，完全仿照 `OpenCVCamera` 的架构设计。

## 创建的文件

### 1. 核心模块文件
```
src/lerobot/cameras/xense/
├── __init__.py                    # 模块导出
├── configuration_xense.py         # 配置类
├── camera_xense.py               # 主实现类
└── README.md                     # 使用文档
```

### 2. 测试文件
```
test_xense_simple.py              # 简单测试脚本
test_xense_camera.py              # 完整测试套件
```

## 主要特性

### 1. 完整的 OutputType 支持

根据 SDK 文档，支持所有输出类型：

```python
class XenseOutputType(Enum):
    # 图像输出
    RECTIFY = "rectify"              # shape=(700, 400, 3), RGB
    DIFFERENCE = "difference"         # shape=(700, 400, 3), RGB
    DEPTH = "depth"                  # shape=(700, 400), 单位: mm
    
    # 力传感输出
    MARKER_2D = "marker_2d"          # shape=(35, 20, 2), 切向位移
    FORCE = "force"                  # shape=(35, 20, 3), 三维力分布
    FORCE_NORM = "force_norm"        # shape=(35, 20, 3), 法向力分量
    FORCE_RESULTANT = "force_resultant"  # shape=(6,), 六维合力
    
    # 3D 网格输出
    MESH_3D = "mesh_3d"              # shape=(35, 20, 3), 当前帧3D网格
    MESH_3D_INIT = "mesh_3d_init"    # shape=(35, 20, 3), 初始3D网格
    MESH_3D_FLOW = "mesh_3d_flow"    # shape=(35, 20, 3), 网格形变向量
```

### 2. 与 OpenCVCamera 相同的接口

```python
# 创建配置
config = XenseCameraConfig(
    serial_number="OG000344",
    fps=60,
    output_types=[XenseOutputType.FORCE, XenseOutputType.FORCE_RESULTANT]
)

# 创建相机实例
camera = XenseTactileCamera(config)

# 连接
camera.connect()

# 同步读取
data = camera.read()  # 返回 dict[str, np.ndarray]

# 异步读取（后台线程）
async_data = camera.async_read(timeout_ms=200)

# 断开连接
camera.disconnect()
```

### 3. 传感器发现

```python
# 自动发现连接的 Xense 传感器
sensors = XenseTactileCamera.find_cameras()
# 返回: [{'serial_number': 'OG000344', 'cam_id': 16, ...}, ...]
```

### 4. 异步读取支持

完全仿照 `OpenCVCamera` 的实现：
- 后台线程持续读取数据
- 线程安全的数据共享（使用 Lock）
- Event 机制通知新数据可用
- 可配置的超时时间

## 关键设计决策

### 1. 返回格式

与图像相机不同，Xense 返回 **字典格式**：

```python
data = camera.read()
# data = {
#     "force": np.ndarray(35, 20, 3),
#     "force_resultant": np.ndarray(6,)
# }
```

**原因**：
- Xense 传感器提供多种数据类型（力、深度、网格等）
- 每种类型的形状不同
- 字典格式更灵活，便于扩展

### 2. 不使用 color_mode 参数

`read()` 方法保留了 `color_mode` 参数以保持接口兼容性，但实际不使用：

```python
def read(self, color_mode=None) -> dict[str, np.ndarray]:
    # color_mode 被忽略，因为力数据不是颜色图像
```

### 3. selectSensorInfo 的正确调用

SDK 文档显示：
```python
# 单个输出：返回 np.ndarray
result = sensor.selectSensorInfo(Sensor.OutputType.Force)

# 多个输出：返回 tuple
force, resultant = sensor.selectSensorInfo(
    Sensor.OutputType.Force,
    Sensor.OutputType.ForceResultant
)
```

我们的实现正确处理了这两种情况：
```python
results = self.sensor.selectSensorInfo(*sensor_output_types)
if len(sensor_output_types) == 1:
    results = (results,)  # 统一为 tuple 格式
```

## 测试流程

### 快速测试
```bash
# 激活环境
conda activate lerobot-openpi

# 运行简单测试
python test_xense_simple.py
```

测试内容：
1. ✓ 发现传感器
2. ✓ 创建配置和实例
3. ✓ 连接传感器
4. ✓ 同步读取 (5帧)
5. ✓ 异步读取 (10帧)
6. ✓ 断开连接

### 完整测试
```bash
python test_xense_camera.py
```

测试选项：
1. 同步读取测试
2. 异步读取测试（带 FPS 统计）
3. 双传感器测试（双臂机器人）
4. 运行所有测试

## 与 bi_arx5 集成（预备）

集成方式（暂时不做，等测试通过后再集成）：

```python
# config_bi_arx5.py
from lerobot.cameras.xense import XenseCameraConfig, XenseOutputType

cameras: dict[str, CameraConfig] = field(
    default_factory=lambda: {
        "head": RealSenseCameraConfig(...),
        "left_wrist": RealSenseCameraConfig(...),
        "right_wrist": RealSenseCameraConfig(...),
        
        # 添加触觉传感器
        "right_tactile": XenseCameraConfig(
            serial_number="OG000344",
            fps=60,
            output_types=[
                XenseOutputType.FORCE,
                XenseOutputType.FORCE_RESULTANT,
            ],
        ),
        "left_tactile": XenseCameraConfig(
            serial_number="OG000352",
            fps=60,
            output_types=[
                XenseOutputType.FORCE,
                XenseOutputType.FORCE_RESULTANT,
            ],
        ),
    }
)
```

## 与图像相机的主要区别

| 特性 | OpenCV/RealSense | Xense |
|------|------------------|-------|
| 返回类型 | `np.ndarray` | `dict[str, np.ndarray]` |
| 数据格式 | 单一图像 | 多种数据类型 |
| 形状 | (H, W, 3) | 变化（35×20×3, 700×400, 6, 等） |
| color_mode | 使用 (RGB/BGR) | 不使用 |
| 主要用途 | 视觉 | 触觉力感知 |

## 依赖项

确保已安装（参考 xensesdk README.md）：
```bash
pip install xensesdk
pip install cypack cryptography pyudev assimp_py==1.0.7 qtpy PyQt5 h5py lz4
```

## 下一步

1. **立即测试**：
   ```bash
   conda activate lerobot-openpi
   python test_xense_simple.py
   ```

2. **如果测试通过**：
   - 运行完整测试套件
   - 测试双传感器（如果有两个）
   - 验证 FPS 性能

3. **测试成功后**：
   - 集成到 `bi_arx5` 配置
   - 更新机器人的 `read_observation` 方法
   - 测试数据记录和重放

## 文件清单

- ✅ `src/lerobot/cameras/xense/__init__.py`
- ✅ `src/lerobot/cameras/xense/configuration_xense.py`
- ✅ `src/lerobot/cameras/xense/camera_xense.py`
- ✅ `src/lerobot/cameras/xense/README.md`
- ✅ `test_xense_simple.py`
- ✅ `test_xense_camera.py`
- ✅ `XENSE_INTEGRATION_SUMMARY.md`

所有文件都已通过 linter 检查，没有错误！

## 注意事项

1. **SDK 版本兼容性**：代码基于 xensesdk 0.1.0 开发
2. **线程安全**：异步读取使用 Lock 保护共享数据
3. **资源清理**：确保调用 `disconnect()` 以释放传感器资源
4. **错误处理**：所有 SDK 调用都包含异常处理
5. **日志记录**：使用 Python logging 模块，与其他相机保持一致

---

**准备就绪！现在可以运行 `python test_xense_simple.py` 进行测试了。** 🚀

