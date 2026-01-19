"""
使用 LeRobot 的 XenseTactileCamera 进行多传感器 Rerun 可视化

这个示例展示了如何使用 LeRobot 的摄像头框架来：
1. 通过配置创建多个 Xense 传感器
2. 使用 make_cameras_from_configs 创建摄像头实例
3. 使用 async_read() 异步读取多个传感器数据
4. 使用 Rerun 进行实时可视化

安装依赖:
    pip install rerun-sdk

运行:
    # 基本用法（自动检测 infer_type）
    python -m lerobot.cameras.xense.example_multi_cam --sensors OG000337 OG000456
    
    # 指定 infer_type
    python -m lerobot.cameras.xense.example_multi_cam --sensors OG000337 --infer-type MIGraphX
    python -m lerobot.cameras.xense.example_multi_cam --sensors OG000337 OG000456 --infer-type ONNX
    
    # 自定义输出类型和 FPS
    python -m lerobot.cameras.xense.example_multi_cam --sensors OG000337 OG000456 --output-types difference depth force --fps 60
"""

import argparse
import os
import sys
import time

import numpy as np

# Now safe to import
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras.xense.configuration_xense import XenseCameraConfig, XenseOutputType
from lerobot.utils.robot_utils import get_logger
from lerobot.utils.visualization_utils import init_rerun

try:
    import rerun as rr
except ImportError:
    print("请先安装 rerun-sdk: pip install rerun-sdk")
    sys.exit(1)

logger = get_logger("XenseMultiSensor")


def log_sensor_data_to_rerun(sensor_key: str, data: np.ndarray | dict[str, np.ndarray]) -> None:
    """
    Log sensor data to Rerun with appropriate visualization.

    Args:
        sensor_key: Sensor identifier (e.g., "sensor_0", "sensor_OG000337")
        data: Sensor data - can be single array or dict of arrays
    """
    if isinstance(data, dict):
        # Multiple output types
        for output_type, output_data in data.items():
            if output_data is None:
                continue

            # Convert output type to string for path
            if isinstance(output_type, XenseOutputType):
                output_name = output_type.value
            else:
                output_name = str(output_type)

            path = f"{sensor_key}/{output_name}"

            # Handle different data types
            if isinstance(output_data, np.ndarray):
                _log_array_to_rerun(path, output_data)
    else:
        # Single output type
        if data is not None and isinstance(data, np.ndarray):
            _log_array_to_rerun(f"{sensor_key}/data", data)


def _log_array_to_rerun(path: str, arr: np.ndarray) -> None:
    """Helper function to log numpy array to Rerun with appropriate type."""
    path_lower = path.lower()

    # Force distribution (35, 20, 3) -> heatmap
    if arr.ndim == 3 and arr.shape[-1] == 3 and arr.dtype in (np.float32, np.float64):
        if "force" in path_lower:
            force_magnitude = np.linalg.norm(arr, axis=-1).astype(np.float32)
            rr.log(path, rr.DepthImage(force_magnitude, meter=1.0, colormap="turbo"))
            return

    # Depth images (2D float arrays)
    if arr.ndim == 2 and arr.dtype in (np.float32, np.float64):
        if "depth" in path_lower:
            rr.log(path, rr.DepthImage(arr, meter=0.001, depth_range=(0.0, 0.1), colormap="turbo"))
            return

    # 1D arrays (e.g., force_resultant with shape (6,))
    if arr.ndim == 1:
        if arr.shape[0] == 6 and arr.dtype in (np.float32, np.float64) and "force_resultant" in path_lower:
            force_labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
            for label, value in zip(force_labels, arr, strict=True):
                rr.log(f"{path}/{label}", rr.Scalars(float(value)))
            return
        else:
            # Generic 1D array
            for i, value in enumerate(arr):
                rr.log(f"{path}_{i}", rr.Scalars(float(value)))
            return

    # Images (3D arrays with 3 channels) - convert BGR to RGB if needed
    if arr.ndim == 3 and arr.shape[-1] == 3:
        # Assume BGR format from OpenCV, convert to RGB for Rerun
        arr_rgb = arr[..., ::-1] if arr.dtype == np.uint8 else arr
        rr.log(path, rr.Image(arr_rgb))
        return

    # 2D arrays (grayscale images)
    if arr.ndim == 2 and arr.dtype == np.uint8:
        rr.log(path, rr.Image(arr))
        return

    # Fallback: log as scalar series
    if arr.size == 1:
        rr.log(path, rr.Scalars(float(arr.item())))
    else:
        # Flatten and log as individual scalars
        flat = arr.flatten()
        for i, value in enumerate(flat[:100]):  # Limit to first 100 values
            rr.log(f"{path}_{i}", rr.Scalars(float(value)))


def main():
    parser = argparse.ArgumentParser(description="Multi-sensor Xense visualization with Rerun")
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=["OG000337"],
        help="Sensor serial numbers (e.g., OG000337 OG000456)",
    )
    parser.add_argument(
        "--output-types",
        nargs="+",
        default=["rectify"],
        help="Output types for each sensor (e.g., rectify, difference)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for sensor acquisition (default: 30)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds (None for infinite)",
    )
    parser.add_argument(
        "--infer-type",
        type=str,
        default=None,
        choices=["MIGraphX", "ONNX"],
        help="Inference type: MIGraphX (AMD GPU) or ONNX (default: auto-detect)",
    )
    args = parser.parse_args()

    # Initialize Rerun
    logger.info("Initializing Rerun visualizer...")
    init_rerun(session_name="xense_multi_sensor")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Create camera configurations
    logger.info(f"Creating configurations for {len(args.sensors)} sensor(s)...")
    camera_configs = {}
    output_types = [XenseOutputType(ot.lower()) for ot in args.output_types]

    for i, serial in enumerate(args.sensors):
        sensor_key = f"sensor_{i}"  # or use serial: f"sensor_{serial}"
        config = XenseCameraConfig(
            serial_number=serial,
            fps=args.fps,
            output_types=output_types,
            warmup_s=0.5,
            infer_type=args.infer_type,
        )
        camera_configs[sensor_key] = config
        infer_info = f" (infer_type: {args.infer_type})" if args.infer_type else " (infer_type: auto)"
        logger.info(f"  {sensor_key}: {serial} - {[ot.value for ot in output_types]}{infer_info}")

    # Create cameras using make_cameras_from_configs
    logger.info("Creating camera instances...")
    cameras = make_cameras_from_configs(camera_configs)

    # Initialize variables before try block to avoid UnboundLocalError in finally
    start_time = time.perf_counter()
    frame_counts = {key: 0 for key in cameras.keys()}
    connected_cameras = {}  # Track successfully connected cameras

    try:
        # Connect all cameras sequentially with delays and retries (similar to bi_arx5 but with error handling)
        logger.info("Connecting sensors...")
        for i, (cam_key, cam) in enumerate(cameras.items()):
            max_retries = 3
            retry_delay = 3.0  # seconds between retries
            connected = False
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"  Connecting {cam_key} (attempt {attempt + 1}/{max_retries})...")
                    
                    # Add longer delay between connections to avoid resource conflicts
                    # Xense sensors need more time to fully initialize
                    if i > 0 or attempt > 0:
                        delay = 5.0 if i > 0 else 2.0  # Longer delay for subsequent sensors
                        time.sleep(delay)
                    
                    cam.connect()
                    logger.info(f"  {cam_key} connected: {cam.width}x{cam.height} @ {cam.fps} FPS")
                    connected_cameras[cam_key] = cam
                    
                    # Delay after successful connection to ensure stability
                    time.sleep(1.0)
                    
                    connected = True
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warn(f"  Attempt {attempt + 1} failed for {cam_key}: {e}")
                        logger.info(f"  Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        # Try to clean up failed connection
                        try:
                            if cam.is_connected:
                                cam.disconnect()
                        except Exception:
                            pass
                    else:
                        logger.error(f"  Failed to connect {cam_key} after {max_retries} attempts: {e}")
                        logger.warn(f"  {cam_key} will be skipped.")
            
            if not connected and i < len(cameras) - 1:
                # Add delay before trying next sensor even if current one failed
                time.sleep(2.0)

        if not connected_cameras:
            logger.error("No sensors successfully connected. Exiting.")
            return

        logger.info(f"Successfully connected {len(connected_cameras)}/{len(cameras)} sensor(s)")

        logger.info("=" * 60)
        logger.info("Starting visualization... (Ctrl+C to stop)")
        logger.info("=" * 60)

        # Streaming loop
        fps_update_interval = 1.0
        last_fps_update = start_time

        while True:
            loop_start = time.perf_counter()

            # Check duration limit
            if args.duration is not None and (loop_start - start_time) >= args.duration:
                logger.info(f"Duration limit reached ({args.duration}s)")
                break

            # Read from all successfully connected sensors (similar to bi_arx5 read_observation)
            sensor_times = {}
            for cam_key, cam in connected_cameras.items():
                read_start = time.perf_counter()
                try:
                    # Use async_read() for optimal performance
                    data = cam.async_read()
                    read_end = time.perf_counter()
                    read_time_ms = (read_end - read_start) * 1000

                    # Log data to Rerun
                    log_sensor_data_to_rerun(cam_key, data)

                    # Log read latency
                    rr.log(f"{cam_key}/latency_ms", rr.Scalars(read_time_ms))

                    frame_counts[cam_key] += 1
                    sensor_times[cam_key] = read_time_ms

                except Exception as e:
                    logger.warn(f"Failed to read from {cam_key}: {e}")
                    sensor_times[cam_key] = 0

            # Update and log FPS
            current_time = time.perf_counter()
            if current_time - last_fps_update >= fps_update_interval:
                elapsed = current_time - start_time
                for cam_key in connected_cameras.keys():
                    count = frame_counts.get(cam_key, 0)
                    fps = count / elapsed if elapsed > 0 else 0
                    rr.log(f"{cam_key}/fps", rr.Scalars(fps))
                    logger.info(f"  {cam_key}: {fps:.1f} FPS, {sensor_times.get(cam_key, 0):.1f}ms latency")
                last_fps_update = current_time

    except KeyboardInterrupt:
        logger.info("Visualization stopped by user.")
    finally:
        # Disconnect all cameras (similar to bi_arx5)
        logger.info("Disconnecting sensors...")
        for cam_key, cam in cameras.items():
            try:
                if cam.is_connected:
                    cam.disconnect()
                    logger.info(f"  {cam_key} disconnected.")
            except Exception as e:
                logger.warn(f"Error disconnecting {cam_key}: {e}")

        # Summary
        end_time = time.perf_counter()
        elapsed = end_time - start_time if start_time > 0 else 0
        logger.info("=" * 60)
        logger.info("Summary:")
        for cam_key, count in frame_counts.items():
            avg_fps = count / elapsed if elapsed > 0 else 0
            status = "✓" if cam_key in connected_cameras else "✗"
            logger.info(f"  {status} {cam_key}: {count} frames, {avg_fps:.2f} FPS avg")
        logger.info(f"  Total duration: {elapsed:.1f}s")
        logger.info(f"  Successfully connected: {len(connected_cameras)}/{len(cameras)} sensor(s)")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
