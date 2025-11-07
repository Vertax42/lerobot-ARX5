#!/usr/bin/env python3
"""
检查 Xense 相机当前使用的视频格式（MJPEG vs YUYV）
"""

import subprocess
import time
from xensesdk import Sensor

print("=" * 60)
print("Xense 相机视频格式检测")
print("=" * 60)
print()

# 导入 torch 以初始化 CUDA
try:
    import torch

    print(f"✓ Torch loaded: {torch.__version__}")
except ImportError:
    print("⚠ Torch not available")

print()
print("【1】启动 Xense 相机...")
print()

# 创建传感器（使用默认 OpenCV backend，避免 ffmpeg 警告）
sensor = Sensor.create("OG000344", use_gpu=True)
print("✓ 相机已连接")

# 等待相机初始化
print("等待 2 秒让相机稳定...")
time.sleep(2)

# 读取一帧以激活相机
print("读取测试帧...")
try:
    data = sensor.selectSensorInfo(Sensor.OutputType.Rectify)
    print(f"✓ 读取成功，图像形状: {data.shape}")
except Exception as e:
    print(f"⚠ 读取失败: {e}")

print()
print("【2】检测当前视频格式...")
print()

# 获取相机 ID
cam_id = sensor.getCameraID()
video_device = f"/dev/video{cam_id}"
print(f"相机设备: {video_device}")
print()

# 使用 v4l2-ctl 检查当前格式
try:
    result = subprocess.run(
        ["v4l2-ctl", "--device", video_device, "--get-fmt-video"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    if result.returncode == 0:
        output = result.stdout
        print("当前视频格式:")
        print(output)

        if "MJPEG" in output or "MJPG" in output:
            print("\n" + "=" * 60)
            print("✅ 结果: 相机正在使用 MJPEG 压缩")
            print("=" * 60)
            print()
            print("MJPEG 优点:")
            print("  ✓ USB 带宽占用低 (约 YUYV 的 20-30%)")
            print("  ✓ 支持高分辨率 @ 高帧率")
            print()
            print("MJPEG 缺点:")
            print("  ⚠ CPU 需要解压缩")
            print("  ⚠ 轻微图像质量损失")

        elif "YUYV" in output or "YUV" in output:
            print("\n" + "=" * 60)
            print("❌ 结果: 相机正在使用 YUYV 未压缩格式")
            print("=" * 60)
            print()
            print("这解释了为什么 USB 2.0 带宽不足！")
            print()
            print("YUYV 问题:")
            print("  ❌ USB 带宽占用高")
            print("  ❌ USB 2.0 下容易超时")
            print()
            print("建议:")
            print("  → 需要联系 Xense 厂商，询问如何配置 MJPEG 模式")
            print("  → 或者继续降低 FPS/分辨率")

    else:
        print(f"⚠ 无法获取格式信息: {result.stderr}")

except FileNotFoundError:
    print("⚠ v4l2-ctl 未安装")
    print("   安装命令: sudo apt install v4l-utils")
except Exception as e:
    print(f"⚠ 检测失败: {e}")

print()
print("【3】清理...")
sensor.release()
print("✓ 相机已释放")
print()
print("=" * 60)
