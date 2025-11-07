#!/usr/bin/env python3
"""
测试强制 xensesdk 使用 MJPEG 格式
"""

import subprocess
import time
import cv2

print("=" * 60)
print("测试：强制 Xense 相机使用 MJPEG 格式")
print("=" * 60)
print()

# 方案 1: 直接使用 OpenCV 设置 MJPEG
print("【方案 1】使用 OpenCV 直接配置 MJPEG")
print()

# 找到 Xense 相机对应的 video 设备号
result = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True)

xense_devices = []
for line in result.stdout.split("\n"):
    if "OG000344" in line or "OG000337" in line:
        # 下一行应该是 /dev/video
        idx = result.stdout.split("\n").index(line)
        next_line = result.stdout.split("\n")[idx + 1].strip()
        if "/dev/video" in next_line:
            xense_devices.append(next_line)
            print(f"发现 Xense 相机: {line.strip()} -> {next_line}")

if not xense_devices:
    print("⚠ 未找到 Xense 相机")
    exit(1)

# 测试第一个设备
test_device = xense_devices[0]
print(f"\n测试设备: {test_device}")
print()

# 提取设备号
device_id = int(test_device.split("video")[1])
print(f"设备 ID: {device_id}")
print()

# 使用 v4l2-ctl 强制设置为 MJPEG
print("【步骤 1】使用 v4l2-ctl 设置 MJPEG...")
result = subprocess.run(
    [
        "v4l2-ctl",
        "--device",
        test_device,
        "--set-fmt-video=width=640,height=480,pixelformat=MJPG",
    ],
    capture_output=True,
    text=True,
)

if result.returncode == 0:
    print("✓ 设置成功")
else:
    print(f"⚠ 设置失败: {result.stderr}")
    print("\n可能需要先关闭所有使用该相机的程序")

print()

# 验证设置
print("【步骤 2】验证当前格式...")
result = subprocess.run(
    ["v4l2-ctl", "--device", test_device, "--get-fmt-video"],
    capture_output=True,
    text=True,
)

if "MJPG" in result.stdout or "MJPEG" in result.stdout:
    print("✅ 成功！相机已切换到 MJPEG")
    print(result.stdout)
else:
    print("❌ 仍然是其他格式")
    print(result.stdout)

print()
print("【步骤 3】测试双重强制设置 MJPEG...")
print()

try:
    import torch
    from xensesdk import Sensor

    print("3.1 - 创建传感器...")
    sensor = Sensor.create("OG000344", use_gpu=True)
    print("✓ xensesdk 相机已连接")

    # CRITICAL: 立即在创建后再次强制设置 MJPEG
    # 这是在 OpenCV 读取任何帧之前完成的
    print("\n3.2 - 立即再次强制设置 MJPEG (在首次读取之前)...")
    result = subprocess.run(
        ["v4l2-ctl", "--device", test_device, "--set-fmt-video=pixelformat=MJPG"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("✓ MJPEG 二次强制设置成功")
    else:
        print(f"⚠ 二次设置失败: {result.stderr}")

    # 验证格式
    print("\n3.3 - 验证格式...")
    result = subprocess.run(
        ["v4l2-ctl", "--device", test_device, "--get-fmt-video"],
        capture_output=True,
        text=True,
    )

    print("当前格式:")
    if "MJPG" in result.stdout or "MJPEG" in result.stdout:
        print("✅ MJPEG (压缩) - 双重设置成功!")
        print(result.stdout)
    elif "YUYV" in result.stdout:
        print("❌ YUYV (未压缩) - 双重设置也失败")
        print(result.stdout)

    # 现在读取数据
    print("\n3.4 - 读取数据测试...")
    time.sleep(0.5)
    data = sensor.selectSensorInfo(Sensor.OutputType.Rectify)
    print(f"✓ 读取成功，图像形状: {data.shape}")

    # 最后再次检查格式（确认读取后格式是否改变）
    print("\n3.5 - 读取后最终格式验证...")
    result = subprocess.run(
        ["v4l2-ctl", "--device", test_device, "--get-fmt-video"],
        capture_output=True,
        text=True,
    )

    print("读取数据后的格式:")
    if "MJPG" in result.stdout:
        print("✅✅ MJPEG 保持稳定！带宽节省 80-90%!")
    elif "YUYV" in result.stdout:
        print("❌ YUYV - 读取时被切换回了")
        print("\n说明：OpenCV 在读取时强制切换回 YUYV")

    print(result.stdout)

    sensor.release()

except Exception as e:
    print(f"⚠ 测试失败: {e}")

print()
print("=" * 60)
print("结论:")
print()
print("如果 xensesdk 强制使用 YUYV:")
print("  → 需要联系 Xense 厂商")
print("  → 询问如何配置 MJPEG 模式")
print("  → 或请求更新 SDK")
print()
print("临时方案:")
print("  → 继续降低 FPS/分辨率")
print("  → 当前已优化到 10 FPS @ 160x280")
print("=" * 60)
