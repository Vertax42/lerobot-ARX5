#!/usr/bin/env python3
"""
测试 xensesdk 返回的数据格式（检查是否需要 MJPEG 解码）
"""

import numpy as np

print("=" * 70)
print("测试 xensesdk 返回的数据格式")
print("=" * 70)
print()

try:
    import torch

    print(f"✓ Torch loaded: {torch.__version__}")
except ImportError:
    print("⚠ Torch not available")

from xensesdk import Sensor
from xensesdk import CameraSource

print()
print("【1】创建 Xense 传感器...")
sensor = Sensor.create("OG000344", use_gpu=True, api=CameraSource.AV_V4L2)
print("✓ 传感器已连接")

print()
print("【2】读取数据并检查格式...")
data = sensor.selectSensorInfo(Sensor.OutputType.Rectify)

print(f"\n数据类型: {type(data)}")
print(f"数据 dtype: {data.dtype}")
print(f"数据形状: {data.shape}")
print(f"数据范围: [{data.min()}, {data.max()}]")

# 检查数据格式
if isinstance(data, np.ndarray):
    print("\n✅ 返回的是 numpy 数组")

    if data.dtype == np.uint8:
        print("✅ uint8 格式")
        if len(data.shape) == 3 and data.shape[2] == 3:
            print("✅ RGB 图像格式 (H, W, 3)")
            print("\n结论: xensesdk 已自动解码 MJPEG → RGB")
            print("      不需要额外的解码步骤")
        else:
            print(f"⚠️  非标准 RGB 格式: {data.shape}")

    elif data.dtype == np.float32 or data.dtype == np.float64:
        print("⚠️  float 格式 - 可能是归一化后的数据")

    else:
        print(f"⚠️  未知 dtype: {data.dtype}")

elif isinstance(data, bytes):
    print("⚠️  返回的是 bytes (MJPEG 压缩数据)")
    print(f"   数据长度: {len(data)} bytes")
    print("\n需要解码: 使用 cv2.imdecode() 解码 MJPEG")

    # 尝试解码
    import cv2

    nparr = np.frombuffer(data, np.uint8)
    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if decoded is not None:
        print(f"✓ 解码成功: {decoded.shape}, dtype={decoded.dtype}")
    else:
        print("✗ 解码失败")
else:
    print(f"⚠️  未知数据类型: {type(data)}")

print()
print("【3】检查相机实际使用的格式...")
import subprocess

cam_id = sensor.getCameraID()
video_device = f"/dev/video{cam_id}"

result = subprocess.run(
    ["v4l2-ctl", "--device", video_device, "--get-fmt-video"],
    capture_output=True,
    text=True,
)

if result.returncode == 0:
    output = result.stdout
    if "MJPG" in output or "MJPEG" in output:
        print(f"✓ V4L2 格式: MJPEG")
        print("  → OpenCV 自动解码 MJPEG → RGB")
    elif "YUYV" in output:
        print(f"✓ V4L2 格式: YUYV")
        print("  → OpenCV 自动转换 YUYV → RGB")
    else:
        print(f"✓ V4L2 格式: 其他")
        print(output)

sensor.release()

print()
print("=" * 70)
print("结论:")
print()
print("如果 xensesdk 返回 numpy 数组:")
print("  → OpenCV 已经解码，lerobot 不需要修改")
print()
print("如果 xensesdk 返回 bytes:")
print("  → 需要在 camera_xense.py 中添加 cv2.imdecode()")
print("=" * 70)
