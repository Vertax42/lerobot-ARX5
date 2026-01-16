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
            rectify, diff = sensor.selectSensorInfo(
                OutputType.Rectify,
                OutputType.Difference,
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
