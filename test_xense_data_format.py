#!/usr/bin/env python3
"""
测试 XenseTactileCamera async_read() 返回的数据格式
"""

import time
import numpy as np
from src.lerobot.cameras.xense import (
    XenseCameraConfig,
    XenseOutputType,
    XenseTactileCamera,
)


def test_xense_data_format():
    print("=" * 60)
    print("测试 Xense 触觉传感器数据格式")
    print("=" * 60)

    # 查找传感器
    sensors = XenseTactileCamera.find_cameras()
    if not sensors:
        print("❌ 未找到传感器")
        return

    serial = sensors[0]["serial_number"]
    print(f"\n使用传感器: {serial}")

    # 创建配置 - 只使用 FORCE (与 bi_arx5 配置一致)
    config = XenseCameraConfig(
        serial_number=serial,
        fps=60,
        output_types=[XenseOutputType.DIFFERENCE],  # 只有 FORCE
        warmup_s=1.0,
    )

    print(f"\n配置:")
    print(f"  serial_number: {config.serial_number}")
    print(f"  fps: {config.fps}")
    print(f"  output_types: {config.output_types}")
    print(f"  warmup_s: {config.warmup_s}")

    camera = XenseTactileCamera(config)

    print("\n连接传感器...")
    camera.connect(warmup=True)
    print("✓ 连接成功")

    # 等待后台线程稳定
    print("\n等待后台线程稳定...")
    time.sleep(1)

    print("\n" + "=" * 60)
    print("测试 async_read() 返回数据")
    print("=" * 60)

    # 读取数据
    data = camera.async_read(timeout_ms=500)

    print(f"\n返回类型: {type(data)}")
    if isinstance(data, tuple):
        print(f"返回的是元组，包含 {len(data)} 个数组")
    else:
        print("返回的是单个数组")

    print("\n" + "-" * 60)
    print("详细数据分析:")
    print("-" * 60)

    # Handle both single array and tuple of arrays
    if isinstance(data, tuple):
        data_arrays = data
    else:
        data_arrays = (data,)

    for idx, value in enumerate(data_arrays):
        key = (
            config.output_types[idx].value
            if idx < len(config.output_types)
            else f"output_{idx}"
        )
        print(f"\n[{key}]")
        print(f"  类型: {type(value)}")
        print(f"  dtype: {value.dtype}")
        print(f"  形状: {value.shape}")
        print(f"  维度: {value.ndim}D")
        print(f"  大小: {value.size} elements")
        print(f"  内存: {value.nbytes} bytes")
        print(f"  范围: [{value.min():.6f}, {value.max():.6f}]")
        print(f"  均值: {value.mean():.6f}")
        print(f"  标准差: {value.std():.6f}")

        # 打印数组的前几个值
        print(f"\n  前 3x3 切片 (shape: {value[:3, :3, :].shape}):")
        if value.ndim == 3:
            for i in range(min(3, value.shape[0])):
                print(f"    Row {i}:")
                for j in range(min(3, value.shape[1])):
                    print(
                        f"      Col {j}: [{value[i,j,0]:.4f}, {value[i,j,1]:.4f}, {value[i,j,2]:.4f}]"
                    )

    # 验证形状
    print("\n" + "=" * 60)
    print("验证数据格式")
    print("=" * 60)

    # Get first array (whether single or from tuple)
    test_array = data if not isinstance(data, tuple) else data[0]
    expected_shape = (
        (700, 400, 3)
        if config.output_types[0] == XenseOutputType.DIFFERENCE
        else (35, 20, 3)
    )

    print(f"\n✓ 找到数据")
    print(f"  预期形状: {expected_shape}")
    print(f"  实际形状: {test_array.shape}")

    if test_array.shape == expected_shape:
        print(f"  ✅ 形状匹配！")
    else:
        print(f"  ❌ 形状不匹配！")

    print(f"\n  ✓ 是 numpy.ndarray: {isinstance(test_array, np.ndarray)}")
    print(f"  ✓ 形状正确: {test_array.shape == expected_shape}")
    print(f"  ✓ 3D 数组: {test_array.ndim == 3}")
    print(f"  ✓ 浮点类型: {np.issubdtype(test_array.dtype, np.floating)}")

    # 多次读取测试
    print("\n" + "=" * 60)
    print("多次读取稳定性测试 (10次)")
    print("=" * 60)

    shapes = []
    for i in range(10):
        data = camera.async_read(timeout_ms=500)
        time.sleep(0.01)
        test_data = data if not isinstance(data, tuple) else data[0]
        shapes.append(test_data.shape)
        if i < 3:
            print(
                f"  读取 {i+1}: shape={test_data.shape}, min={test_data.min():.4f}, max={test_data.max():.4f}"
            )

    # 检查所有形状是否一致
    all_same = all(s == shapes[0] for s in shapes)
    print(f"\n  ✓ 所有读取形状一致: {all_same}")
    if all_same:
        print(f"    一致的形状: {shapes[0]}")

    # 大量数据读取测试
    print("\n" + "=" * 60)
    print("大量数据读取测试 (1000次)")
    print("=" * 60)

    print("\n开始读取 1000 个数据...")
    print("注意：在每次读取之间添加 10ms 延迟以观察后台线程更新...")
    all_forces = []
    start_time = time.time()

    for i in range(1000):
        data = camera.async_read(timeout_ms=500)
        test_data = data if not isinstance(data, tuple) else data[0]
        all_forces.append(test_data)

        # 添加延迟，给后台线程时间更新数据
        time.sleep(0.01)  # 10ms delay -> 最大 100 FPS

        # 每 100 次打印一次进度
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed
            print(f"  进度: {i+1}/1000, 用时: {elapsed:.2f}s, FPS: {fps:.1f}")

    total_time = time.time() - start_time
    avg_fps = 1000 / total_time

    print(f"\n✓ 读取完成！")
    print(f"  总时间: {total_time:.2f}s")
    print(f"  平均 FPS: {avg_fps:.1f}")
    print(f"  每帧平均耗时: {(total_time/1000)*1000:.2f}ms")

    # 分析所有数据
    print("\n数据统计分析:")
    all_forces = np.array(all_forces)  # shape: (1000, 35, 20, 3)
    print(f"  总数据形状: {all_forces.shape}")
    print(f"  总内存占用: {all_forces.nbytes / 1024 / 1024:.2f} MB")
    print(f"  全局最小值: {all_forces.min():.6f}")
    print(f"  全局最大值: {all_forces.max():.6f}")
    print(f"  全局均值: {all_forces.mean():.6f}")
    print(f"  全局标准差: {all_forces.std():.6f}")

    # 计算每个点的时间序列统计
    print("\n时间序列分析 (选取中心点 [17, 10]):")
    center_point = all_forces[:, 17, 10, :]  # shape: (1000, 3)
    print(f"  中心点形状: {center_point.shape}")
    print(
        f"  X 分量范围: [{center_point[:, 0].min():.6f}, {center_point[:, 0].max():.6f}]"
    )
    print(
        f"  Y 分量范围: [{center_point[:, 1].min():.6f}, {center_point[:, 1].max():.6f}]"
    )
    print(
        f"  Z 分量范围: [{center_point[:, 2].min():.6f}, {center_point[:, 2].max():.6f}]"
    )

    # 检查是否有异常值
    print("\n异常值检测:")
    nan_count = np.isnan(all_forces).sum()
    inf_count = np.isinf(all_forces).sum()
    print(f"  NaN 数量: {nan_count}")
    print(f"  Inf 数量: {inf_count}")
    if nan_count == 0 and inf_count == 0:
        print("  ✓ 数据完整，无异常值")

    # 断开连接
    print("\n断开传感器...")
    camera.disconnect()
    print("✓ 断开成功")

    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
    print("\n总结:")
    if len(config.output_types) == 1:
        print(f"  - async_read() 返回: np.ndarray (单个数组)")
        print(f"  - 数组类型: numpy.ndarray")
        print(f"  - 数组形状: {expected_shape}")
    else:
        print(f"  - async_read() 返回: tuple of np.ndarray")
        print(f"  - 元组长度: {len(config.output_types)}")
        print(f"  - 每个元素都是 numpy.ndarray")


if __name__ == "__main__":
    test_xense_data_format()
