#!/usr/bin/env python3
"""
检查 RealSense 相机当前使用的视频格式（MJPEG vs YUYV）
"""

import subprocess
import re

print("=" * 70)
print("RealSense 相机视频格式检测")
print("=" * 70)
print()

# RealSense 相机序列号
realsense_cameras = {
    "head": "230322271365",
    "left_wrist": "230422271416",
    "right_wrist": "230322274234",
}

print("【1】扫描所有 video 设备...")
print()

result = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True)

# 解析设备列表，找到 RealSense 相机
realsense_devices = {}

lines = result.stdout.split("\n")
camera_index = 0
for i, line in enumerate(lines):
    # 查找包含 RealSense 的行
    if "RealSense" in line:
        # 下一行应该包含 /dev/video 设备
        j = i + 1
        devices = []
        while j < len(lines) and lines[j].strip().startswith("/dev/video"):
            devices.append(lines[j].strip())
            j += 1

        if devices:
            camera_index += 1
            name = f"RealSense_{camera_index}"
            realsense_devices[name] = {
                "info": line.strip(),
                "devices": devices,
            }
            print(f"✓ 找到 {name}:")
            print(f"  {line.strip()}")
            for dev in devices:
                print(f"  - {dev}")

if not realsense_devices:
    print("⚠ 未找到任何 RealSense 相机")
    print("\n可用设备列表:")
    print(result.stdout)
    exit(1)

print()
print("=" * 70)
print()

# 检查每个相机的格式
for name, info in realsense_devices.items():
    print(f"【{name}】")
    print(f"{info['info']}")
    print("-" * 70)

    for device in info["devices"]:
        print(f"\n设备: {device}")

        # 获取格式信息
        result = subprocess.run(
            ["v4l2-ctl", "--device", device, "--get-fmt-video"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            output = result.stdout

            # 提取关键信息
            width_height = re.search(r"Width/Height\s+:\s+(\d+)/(\d+)", output)
            pixel_format = re.search(r"Pixel Format\s+:\s+'(\w+)'", output)

            if width_height:
                width, height = width_height.groups()
                print(f"  分辨率: {width}x{height}")

            if pixel_format:
                fmt = pixel_format.group(1)
                print(f"  格式: {fmt}", end="")

                if fmt == "MJPG" or fmt == "MJPEG":
                    print(" ✅ (MJPEG 压缩 - 低带宽)")
                elif fmt == "YUYV":
                    print(" ⚠️  (YUYV 未压缩 - 高带宽)")
                elif fmt == "Z16":
                    print(" ℹ️  (深度流)")
                else:
                    print(f" ℹ️  ({fmt})")
        else:
            print(f"  ⚠ 无法获取格式: {result.stderr}")

    print()

print("=" * 70)
print("总结:")
print()

# 统计格式使用情况
mjpeg_count = 0
yuyv_count = 0
other_count = 0

for name, info in realsense_devices.items():
    for device in info["devices"]:
        result = subprocess.run(
            ["v4l2-ctl", "--device", device, "--get-fmt-video"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            if "MJPG" in result.stdout or "MJPEG" in result.stdout:
                mjpeg_count += 1
            elif "YUYV" in result.stdout:
                yuyv_count += 1
            else:
                other_count += 1

print(f"MJPEG (压缩): {mjpeg_count} 个设备")
print(f"YUYV (未压缩): {yuyv_count} 个设备")
print(f"其他格式: {other_count} 个设备")
print()

if yuyv_count > 0:
    print("⚠️  发现 YUYV 格式设备 - 占用高带宽")
    print()
    print("建议:")
    print("  1. RealSense 通常默认使用优化的格式")
    print("  2. 检查 RealSense SDK 配置")
    print("  3. 考虑降低分辨率或帧率")
else:
    print("✅ 所有设备都使用压缩格式或深度流")

print("=" * 70)
