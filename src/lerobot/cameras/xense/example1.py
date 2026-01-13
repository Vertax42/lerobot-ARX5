"""
使用 Rerun 可视化 Xense 传感器数据
避免 Qt/LLVM 与 MIGraphX 的冲突

安装依赖:
    pip install rerun-sdk

运行:
    python -m xensesdk.examples.example_rerun_viewer
"""

import sys
import time
import numpy as np

# 导入 rerun (不依赖 Qt，避免 LLVM 冲突)
try:
    import rerun as rr
except ImportError:
    print("请先安装 rerun-sdk: pip install rerun-sdk")
    sys.exit(1)

from xensesdk import Sensor, CameraSource
from xensesdk.xenseInterface.sensorEnum import InferType, OutputType

def init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "20%")
    rr.spawn(memory_limit=memory_limit)
    # NOTE: We do NOT send a fixed blueprint here. This lets Rerun auto-discover
    # all logged entity paths and create views dynamically. If a static blueprint
    # is sent, changing stream na
    # mes (e.g. depth -> rectify) won't update the view.

def main():
    # 初始化 Rerun
    init_rerun("xense_viewer")
    
    # 创建传感器 (使用 MIGraphX)
    print("正在连接传感器...")
    sensor = Sensor.create(
        "OG000456",
        use_gpu=True,
        infer_type=InferType.MIGraphX,
        api=CameraSource.CV2_V4L2,
    )
    print("传感器连接成功!")
    
    # 设置 Rerun 布局
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    print("开始可视化... (Ctrl+C 退出)")
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            # 获取传感器数据
            rectify, diff, depth, force, force_resultant = sensor.selectSensorInfo(
                OutputType.Rectify,
                OutputType.Difference,
                OutputType.Depth,
                OutputType.Force,
                OutputType.ForceResultant,
            )
            
            # 记录图像数据到 Rerun
            # Rectify 图像 (BGR -> RGB)
            if rectify is not None:
                rectify_rgb = rectify[..., ::-1] if rectify.ndim == 3 else rectify
                rr.log("sensor/rectify", rr.Image(rectify_rgb))
            
            # Difference 图像 (BGR -> RGB)
            if diff is not None:
                diff_rgb = diff[..., ::-1] if diff.ndim == 3 else diff
                rr.log("sensor/difference", rr.Image(diff_rgb))
            
            # Depth 图像 (归一化到 0-255 显示)
            if depth is not None:
                # 归一化深度图用于显示
                depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
                depth_display = (depth_normalized * 255).astype(np.uint8)
                rr.log("sensor/depth", rr.Image(depth_display))
                
                # 也记录原始深度值作为 DepthImage
                rr.log("sensor/depth_raw", rr.DepthImage(depth))
            
            # Force 数据 (作为热力图)
            if force is not None and force.ndim >= 2:
                # 计算力的幅值
                if force.ndim == 3 and force.shape[-1] == 3:
                    force_magnitude = np.linalg.norm(force, axis=-1)
                else:
                    force_magnitude = np.abs(force) if force.ndim == 2 else force
                
                # 归一化
                force_normalized = (force_magnitude - force_magnitude.min()) / (force_magnitude.max() - force_magnitude.min() + 1e-6)
                force_display = (force_normalized * 255).astype(np.uint8)
                rr.log("sensor/force", rr.Image(force_display))
            
            # Force Resultant (作为标量时间序列)
            if force_resultant is not None:
                if isinstance(force_resultant, np.ndarray):
                    if force_resultant.size == 6:
                        # [fx, fy, fz, tx, ty, tz]
                        rr.log("sensor/force_resultant/fx", rr.Scalar(float(force_resultant[0])))
                        rr.log("sensor/force_resultant/fy", rr.Scalar(float(force_resultant[1])))
                        rr.log("sensor/force_resultant/fz", rr.Scalar(float(force_resultant[2])))
                        rr.log("sensor/force_resultant/tx", rr.Scalar(float(force_resultant[3])))
                        rr.log("sensor/force_resultant/ty", rr.Scalar(float(force_resultant[4])))
                        rr.log("sensor/force_resultant/tz", rr.Scalar(float(force_resultant[5])))
                    elif force_resultant.size == 1:
                        rr.log("sensor/force_resultant", rr.Scalar(float(force_resultant)))
                else:
                    rr.log("sensor/force_resultant", rr.Scalar(float(force_resultant)))
            
            # 计算并显示 FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                rr.log("metrics/fps", rr.Scalar(fps))
                print(f"FPS: {fps:.1f}")
                frame_count = 0
                start_time = time.time()
            
            # 小延迟避免 CPU 过载
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n停止可视化...")
    finally:
        sensor.release()
        print("传感器已释放")


if __name__ == "__main__":
    main()
