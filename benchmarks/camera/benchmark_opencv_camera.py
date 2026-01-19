#!/usr/bin/env python
"""
OpenCV Camera Performance Benchmark Tool

Measures camera performance metrics including:
- Actual FPS vs requested FPS
- Frame read latency (min/max/avg/std)
- Frame drop detection
- Resolution/FPS/FOURCC capability matrix
- USB connection speed detection (USB 2.0/3.0)
- MJPG vs YUYV bandwidth comparison
- Real-time video streaming with Rerun/cv2.imshow

Usage:
    ==================== Camera Discovery ====================
    # List all cameras with device info (name, USB speed, etc.)
    python benchmark_opencv_camera.py --list

    ==================== Capability Scan ====================
    # Full capability scan (resolution, FPS, USB speed)
    python benchmark_opencv_camera.py -i 0 --scan-capabilities
    
    # Compare MJPG vs YUYV performance and bandwidth
    python benchmark_opencv_camera.py -i 0 --compare-fourcc
    python benchmark_opencv_camera.py -i 0 --compare-fourcc --fps 60 --width 1920 --height 1080

    ==================== Benchmark ====================
    # Basic benchmark (MJPG format, 1280x720@30fps by default)
    python benchmark_opencv_camera.py -i 0
    python benchmark_opencv_camera.py -i 0 --fps 60 --width 1920 --height 1080
    python benchmark_opencv_camera.py -i 0 --duration 30  # 30 second test

    # Test with YUYV format (uncompressed, higher USB bandwidth)
    python benchmark_opencv_camera.py -i 0 --fourcc YUYV

    # Test multiple cameras
    python benchmark_opencv_camera.py -i 0 1 2

    ==================== Video Stream ====================
    # Stream with Rerun visualization (default)
    python benchmark_opencv_camera.py -i 0 --video-stream
    python benchmark_opencv_camera.py -i 0 -v --fps 60 --width 1920 --height 1080

    # Stream with cv2.imshow (lowest latency, press Q to quit)
    python benchmark_opencv_camera.py -i 0 -v --cv2-display

    # Stream with LeRobot's threaded camera (recommended for low latency)
    python benchmark_opencv_camera.py -i 0 -v --use-lerobot
    python benchmark_opencv_camera.py -i 0 -v --use-lerobot --cv2-display

Defaults:
    --fourcc MJPG     (compressed, saves USB bandwidth)
    --fps 30
    --width 1280
    --height 720
    --warmup 2.0s
    --duration 10.0s

Note:
    - MJPG is recommended for USB 2.0 cameras (480 Mbps bandwidth limit)
    - YUYV requires USB 3.0 for high resolution/FPS (uncompressed)
    - Use --cv2-display instead of Rerun for better FPS during streaming
"""

import argparse
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.utils.robot_utils import get_logger

logger = get_logger("CameraBenchmark")

# Rerun memory limit (fixed 2GB for camera benchmark)
RERUN_MEMORY_LIMIT = "2GB"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    camera_index: int | str
    requested_fps: float
    requested_width: int
    requested_height: int
    actual_fps: float
    actual_width: int
    actual_height: int
    fourcc: str
    duration_s: float
    total_frames: int
    dropped_frames: int
    latency_ms_min: float
    latency_ms_max: float
    latency_ms_avg: float
    latency_ms_std: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    success: bool
    error: str | None = None


@dataclass
class CameraCapabilities:
    """Camera capability information."""

    camera_index: int | str
    name: str
    backend: str
    supported_resolutions: list[tuple[int, int]] = field(default_factory=list)
    supported_fps: list[float] = field(default_factory=list)
    supported_fourcc: list[str] = field(default_factory=list)
    # Resolution -> FPS combinations: {(width, height): [fps1, fps2, ...]}
    resolution_fps_map: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    # Resolution -> max FPS: {(width, height): max_fps}
    resolution_max_fps_map: dict[tuple[int, int], float] = field(default_factory=dict)
    # USB connection info
    usb_speed_mbps: int = 0  # USB speed in Mbps (e.g., 480 for USB 2.0, 5000 for USB 3.0)
    usb_version: str = ""  # USB version string (e.g., "USB 2.0", "USB 3.0")


@dataclass
class CameraDeviceInfo:
    """Camera device information including manufacturer details."""

    index: int
    device_path: str
    name: str
    driver: str
    bus_info: str
    vendor_id: str
    product_id: str
    manufacturer: str
    product: str
    serial: str
    width: int
    height: int
    fps: float
    fourcc: str
    backend: str


# Common resolutions to test
COMMON_RESOLUTIONS = [
    (320, 240),
    (640, 480),
    (800, 600),
    (1280, 720),
    (1280, 960),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]

# Common FPS values to test
COMMON_FPS = [15, 24, 30, 60, 90, 120]

# Common FOURCC codes
COMMON_FOURCC = ["MJPG", "YUYV", "H264", "NV12", "BGR3"]


def fourcc_to_string(fourcc_int: int) -> str:
    """Convert FOURCC integer to string."""
    return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])


def string_to_fourcc(fourcc_str: str) -> int:
    """Convert FOURCC string to integer."""
    return cv2.VideoWriter_fourcc(*fourcc_str)


def get_camera_info(cap: cv2.VideoCapture) -> dict[str, Any]:
    """Get current camera settings."""
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "fourcc": fourcc_to_string(fourcc_int),
        "backend": cap.getBackendName(),
    }


def get_v4l2_device_info(device_path: str) -> dict[str, str]:
    """Get V4L2 device info using v4l2-ctl command.

    Args:
        device_path: Path to video device (e.g., '/dev/video0')

    Returns:
        Dictionary with device name, driver, bus_info
    """
    info = {"name": "", "driver": "", "bus_info": ""}

    try:
        result = subprocess.run(
            ["v4l2-ctl", "--device", device_path, "--info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                line = line.strip()
                if line.startswith("Card type"):
                    info["name"] = line.split(":", 1)[1].strip()
                elif line.startswith("Driver name"):
                    info["driver"] = line.split(":", 1)[1].strip()
                elif line.startswith("Bus info"):
                    info["bus_info"] = line.split(":", 1)[1].strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return info


def get_usb_device_info(bus_info: str) -> dict[str, str]:
    """Get USB device info (vendor, product, manufacturer, serial).

    Args:
        bus_info: V4L2 bus info string (e.g., 'usb-0000:00:14.0-1')

    Returns:
        Dictionary with vendor_id, product_id, manufacturer, product, serial
    """
    info = {
        "vendor_id": "",
        "product_id": "",
        "manufacturer": "",
        "product": "",
        "serial": "",
    }

    # Extract USB port from bus_info (e.g., 'usb-0000:00:14.0-1' -> find device)
    if not bus_info or "usb" not in bus_info.lower():
        return info

    try:
        # Try to find USB device using lsusb or sysfs
        # First, try to find from /sys/class/video4linux/
        for video_dev in Path("/sys/class/video4linux").glob("video*"):
            device_link = video_dev / "device"
            if device_link.exists():
                real_path = device_link.resolve()

                # Check if bus_info matches
                uevent_path = real_path / "uevent"
                if uevent_path.exists():
                    uevent_content = uevent_path.read_text()
                    # Extract PRODUCT info if available
                    for line in uevent_content.split("\n"):
                        if line.startswith("PRODUCT="):
                            parts = line.split("=")[1].split("/")
                            if len(parts) >= 2:
                                info["vendor_id"] = parts[0]
                                info["product_id"] = parts[1]

                # Try to get manufacturer, product, serial from parent USB device
                usb_path = real_path
                for _ in range(5):  # Walk up to find USB device
                    usb_path = usb_path.parent
                    manufacturer_file = usb_path / "manufacturer"
                    product_file = usb_path / "product"
                    serial_file = usb_path / "serial"
                    id_vendor_file = usb_path / "idVendor"
                    id_product_file = usb_path / "idProduct"

                    if manufacturer_file.exists():
                        info["manufacturer"] = manufacturer_file.read_text().strip()
                    if product_file.exists():
                        info["product"] = product_file.read_text().strip()
                    if serial_file.exists():
                        info["serial"] = serial_file.read_text().strip()
                    if id_vendor_file.exists():
                        info["vendor_id"] = id_vendor_file.read_text().strip()
                    if id_product_file.exists():
                        info["product_id"] = id_product_file.read_text().strip()

                    if info["manufacturer"] or info["product"]:
                        break

    except Exception:
        pass

    return info


def get_device_info_from_sysfs(index: int) -> dict[str, str]:
    """Get device info directly from sysfs for a video device index.

    Args:
        index: Video device index (0 for /dev/video0)

    Returns:
        Dictionary with device info
    """
    info = {
        "name": "",
        "driver": "",
        "bus_info": "",
        "vendor_id": "",
        "product_id": "",
        "manufacturer": "",
        "product": "",
        "serial": "",
    }

    video_path = Path(f"/sys/class/video4linux/video{index}")
    if not video_path.exists():
        return info

    try:
        # Get name from sysfs
        name_file = video_path / "name"
        if name_file.exists():
            info["name"] = name_file.read_text().strip()

        # Get device link to find USB info
        device_link = video_path / "device"
        if device_link.exists():
            real_path = device_link.resolve()

            # Walk up to find USB device attributes
            usb_path = real_path
            for _ in range(6):
                usb_path = usb_path.parent

                # Check for USB attributes
                manufacturer_file = usb_path / "manufacturer"
                product_file = usb_path / "product"
                serial_file = usb_path / "serial"
                id_vendor_file = usb_path / "idVendor"
                id_product_file = usb_path / "idProduct"

                if id_vendor_file.exists() and id_product_file.exists():
                    info["vendor_id"] = id_vendor_file.read_text().strip()
                    info["product_id"] = id_product_file.read_text().strip()

                    if manufacturer_file.exists():
                        info["manufacturer"] = manufacturer_file.read_text().strip()
                    if product_file.exists():
                        info["product"] = product_file.read_text().strip()
                    if serial_file.exists():
                        info["serial"] = serial_file.read_text().strip()
                    break

    except Exception:
        pass

    return info


def get_usb_speed(index: int) -> tuple[int, str]:
    """Get USB connection speed for a video device.

    Args:
        index: Video device index (0 for /dev/video0)

    Returns:
        Tuple of (speed_mbps, version_string)
        speed_mbps: USB speed in Mbps (12=USB 1.1, 480=USB 2.0, 5000=USB 3.0, etc.)
        version_string: Human readable USB version
    """
    video_path = Path(f"/sys/class/video4linux/video{index}")
    if not video_path.exists():
        return 0, "Unknown"

    try:
        device_link = video_path / "device"
        if not device_link.exists():
            return 0, "Unknown"

        real_path = device_link.resolve()

        # Walk up the directory tree to find USB device with 'speed' attribute
        usb_path = real_path
        for _ in range(8):
            usb_path = usb_path.parent
            speed_file = usb_path / "speed"

            if speed_file.exists():
                speed_str = speed_file.read_text().strip()
                try:
                    speed_mbps = int(float(speed_str))

                    # Map speed to USB version
                    if speed_mbps >= 20000:
                        version = "USB 3.2 Gen 2x2"
                    elif speed_mbps >= 10000:
                        version = "USB 3.1/3.2 Gen 2"
                    elif speed_mbps >= 5000:
                        version = "USB 3.0/3.1 Gen 1"
                    elif speed_mbps >= 480:
                        version = "USB 2.0 High Speed"
                    elif speed_mbps >= 12:
                        version = "USB 1.1 Full Speed"
                    else:
                        version = "USB 1.0 Low Speed"

                    return speed_mbps, version
                except ValueError:
                    pass

    except Exception:
        pass

    return 0, "Unknown"


def list_cameras(max_index: int = 10) -> list[CameraDeviceInfo]:
    """List all available cameras with device information.

    Args:
        max_index: Maximum camera index to scan

    Returns:
        List of CameraDeviceInfo objects
    """
    cameras = []

    for i in range(max_index):
        device_path = f"/dev/video{i}"

        # Check if device exists
        if not Path(device_path).exists():
            continue

        # Try to open with OpenCV
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue

        # Get OpenCV camera info
        cv_info = get_camera_info(cap)
        cap.release()

        # Get sysfs device info
        sysfs_info = get_device_info_from_sysfs(i)

        # Get V4L2 info if available
        v4l2_info = get_v4l2_device_info(device_path)

        # Merge info, prefer sysfs for USB details, v4l2 for driver info
        name = sysfs_info["name"] or v4l2_info["name"] or f"Camera {i}"
        driver = v4l2_info["driver"] or ""
        bus_info = v4l2_info["bus_info"] or ""

        camera_info = CameraDeviceInfo(
            index=i,
            device_path=device_path,
            name=name,
            driver=driver,
            bus_info=bus_info,
            vendor_id=sysfs_info["vendor_id"],
            product_id=sysfs_info["product_id"],
            manufacturer=sysfs_info["manufacturer"],
            product=sysfs_info["product"],
            serial=sysfs_info["serial"],
            width=cv_info["width"],
            height=cv_info["height"],
            fps=cv_info["fps"],
            fourcc=cv_info["fourcc"],
            backend=cv_info["backend"],
        )
        cameras.append(camera_info)

    return cameras


def print_camera_list(cameras: list[CameraDeviceInfo]) -> None:
    """Print camera list with device information.

    Args:
        cameras: List of CameraDeviceInfo objects
    """
    if not cameras:
        logger.warning("No cameras found!")
        return

    logger.info("=" * 80)
    logger.info(f"Found {len(cameras)} camera(s)")
    logger.info("=" * 80)

    for cam in cameras:
        logger.info("")
        logger.info(f"📷 Camera {cam.index}: {cam.device_path}")
        logger.info("-" * 40)

        # Device identification
        if cam.manufacturer or cam.product:
            logger.info(f"  Manufacturer : {cam.manufacturer or 'N/A'}")
            logger.info(f"  Product      : {cam.product or 'N/A'}")
        elif cam.name:
            logger.info(f"  Name         : {cam.name}")

        if cam.vendor_id and cam.product_id:
            logger.info(f"  USB ID       : {cam.vendor_id}:{cam.product_id}")

        if cam.serial:
            logger.info(f"  Serial       : {cam.serial}")

        if cam.driver:
            logger.info(f"  Driver       : {cam.driver}")

        if cam.bus_info:
            logger.info(f"  Bus Info     : {cam.bus_info}")

        # Current settings
        logger.info(f"  Resolution   : {cam.width}x{cam.height}")
        logger.info(f"  FPS          : {cam.fps:.0f}")
        # Only display MJPG or YUYV, show "Other" for other formats
        fourcc_display = cam.fourcc if cam.fourcc in ["MJPG", "YUYV"] else "Other"
        logger.info(f"  FOURCC       : {fourcc_display}")
        logger.info(f"  Backend      : {cam.backend}")

    logger.info("")
    logger.info("=" * 80)

    # Summary table
    logger.info("")
    logger.info("Summary:")
    logger.info(f"{'Index':<6} {'Device':<14} {'Manufacturer':<20} {'Product':<25} {'Resolution':<12}")
    logger.info("-" * 80)
    for cam in cameras:
        manufacturer = cam.manufacturer[:18] if cam.manufacturer else "N/A"
        product = cam.product[:23] if cam.product else cam.name[:23] if cam.name else "N/A"
        res = f"{cam.width}x{cam.height}"
        logger.info(f"{cam.index:<6} {cam.device_path:<14} {manufacturer:<20} {product:<25} {res:<12}")


def stream_video_with_rerun(
    camera_index: int | str,
    fps: float = 30,
    width: int = 1280,
    height: int = 720,
    fourcc: str = "MJPG",
    duration_s: float | None = None,
    warmup_s: float = 2.0,
    use_cv2_display: bool = False,
) -> None:
    """
    Stream video from camera with real-time visualization.

    Shows live video feed with FPS overlay and performance metrics.

    Args:
        camera_index: Camera index or path
        fps: Requested FPS
        width: Requested width
        height: Requested height
        fourcc: FOURCC codec (default: MJPG for better performance)
        duration_s: Stream duration in seconds (None for infinite)
        warmup_s: Warmup time to stabilize camera before measuring FPS (default: 2.0s)
        use_cv2_display: Use cv2.imshow() for lower latency (default: False, uses Rerun)
    """
    try:
        import rerun as rr
    except ImportError:
        logger.error("Rerun is required for video streaming. Install with: pip install rerun-sdk")
        return

    display_mode = "cv2.imshow (low latency)" if use_cv2_display else "Rerun"
    logger.info("=" * 60)
    logger.info(f"Starting video stream for camera {camera_index}")
    logger.info(f"Settings: {width}x{height} @ {fps} FPS, FOURCC: {fourcc}")
    logger.info(f"Display: {display_mode}")
    logger.info("=" * 60)

    # Initialize display
    if use_cv2_display:
        window_name = f"Camera {camera_index} - Press Q to quit"
    else:
        rr.init("camera_stream")
        rr.spawn(memory_limit=RERUN_MEMORY_LIMIT)

    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_index}")
        return

    try:
        # Set FOURCC first (MJPG for better performance)
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, string_to_fourcc(fourcc))

        # Set resolution and FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        # Minimize buffer to reduce latency (get latest frame, not buffered old frames)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Get actual settings
        info = get_camera_info(cap)
        actual_buffer = int(cap.get(cv2.CAP_PROP_BUFFERSIZE))
        logger.info(
            f"Actual: {info['width']}x{info['height']} @ {info['fps']:.1f} FPS, "
            f"FOURCC: {info['fourcc']}, Buffer: {actual_buffer}"
        )

        # Warmup phase - read frames to stabilize camera
        if warmup_s > 0:
            logger.info(f"Warming up camera for {warmup_s}s...")
            warmup_start = time.perf_counter()
            warmup_frames = 0
            while time.perf_counter() - warmup_start < warmup_s:
                ret, _ = cap.read()
                if ret:
                    warmup_frames += 1
            warmup_elapsed = time.perf_counter() - warmup_start
            warmup_fps = warmup_frames / warmup_elapsed if warmup_elapsed > 0 else 0
            logger.info(
                f"Warmup complete: {warmup_frames} frames in {warmup_elapsed:.1f}s ({warmup_fps:.1f} FPS)"
            )

        # Streaming loop
        frame_count = 0
        start_time = time.perf_counter()
        fps_update_interval = 0.5  # Update FPS display every 0.5s
        last_fps_update = start_time
        recent_fps = 0.0

        logger.info("Streaming... Press Ctrl+C to stop.")

        while True:
            loop_start = time.perf_counter()

            # Check duration limit
            if duration_s is not None and (loop_start - start_time) >= duration_s:
                logger.info(f"Duration limit reached ({duration_s}s)")
                break

            # Read frame with minimal latency:
            # grab() discards buffered frames, retrieve() gets the latest
            if not cap.grab():
                logger.warning("Failed to grab frame, retrying...")
                time.sleep(0.01)
                continue
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                logger.warning("Failed to retrieve frame, retrying...")
                time.sleep(0.01)
                continue

            frame_count += 1
            current_time = time.perf_counter()
            frame_latency_ms = (current_time - loop_start) * 1000

            # Calculate FPS
            if current_time - last_fps_update >= fps_update_interval:
                elapsed = current_time - start_time
                recent_fps = frame_count / elapsed if elapsed > 0 else 0
                last_fps_update = current_time

            # Add FPS and latency overlay text (on BGR frame for cv2)
            overlay_text = f"FPS: {recent_fps:.1f} | Latency: {frame_latency_ms:.1f}ms | Frame: {frame_count}"
            cv2.putText(
                frame,
                overlay_text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            # Display frame
            if use_cv2_display:
                # Use cv2.imshow for lowest latency display
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # Q or ESC to quit
                    logger.info("Quit requested by user.")
                    break
            else:
                # Use Rerun for visualization
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rr.log("camera/image", rr.Image(frame_rgb))
                rr.log("camera/fps", rr.Scalars(recent_fps))
                rr.log("camera/latency_ms", rr.Scalars(frame_latency_ms))

            # Progress log every 100 frames
            if frame_count % 100 == 0:
                elapsed = current_time - start_time
                logger.info(
                    f"  Frames: {frame_count}, FPS: {recent_fps:.1f}, "
                    f"Latency: {frame_latency_ms:.1f}ms, Elapsed: {elapsed:.1f}s"
                )

    except KeyboardInterrupt:
        logger.info("Stream stopped by user.")
    finally:
        cap.release()
        if use_cv2_display:
            cv2.destroyAllWindows()
        elapsed = time.perf_counter() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        logger.info("=" * 60)
        logger.info("Stream Summary:")
        logger.info(f"  Total frames: {frame_count}")
        logger.info(f"  Duration: {elapsed:.1f}s")
        logger.info(f"  Average FPS: {avg_fps:.2f}")
        logger.info("=" * 60)


def stream_video_with_lerobot(
    camera_index: int | str,
    fps: float = 30,
    width: int = 1280,
    height: int = 720,
    fourcc: str = "MJPG",
    duration_s: float | None = None,
    warmup_s: float = 2.0,
    use_async: bool = True,
    use_cv2_display: bool = False,
) -> None:
    """
    Stream video using LeRobot's OpenCVCamera implementation.

    Uses LeRobot's threaded camera implementation for optimal latency.

    Args:
        camera_index: Camera index or path
        fps: Requested FPS
        width: Requested width
        height: Requested height
        fourcc: FOURCC codec (default: MJPG)
        duration_s: Stream duration in seconds (None for infinite)
        warmup_s: Warmup time before measuring FPS
        use_async: Use async_read() for lower latency (default: True)
        use_cv2_display: Use cv2.imshow() for lowest latency display (default: False)
    """
    from lerobot.cameras.opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import ColorMode, OpenCVCameraConfig

    display_mode = "cv2.imshow (low latency)" if use_cv2_display else "Rerun"
    logger.info("=" * 60)
    logger.info(f"Starting LeRobot camera stream for camera {camera_index}")
    logger.info(f"Settings: {width}x{height} @ {fps} FPS, FOURCC: {fourcc}")
    logger.info(f"Mode: {'async_read()' if use_async else 'read()'}")
    logger.info(f"Display: {display_mode}")
    logger.info("=" * 60)

    # Initialize display
    if use_cv2_display:
        window_name = f"LeRobot Camera {camera_index} - Press Q to quit"
    else:
        try:
            import rerun as rr
        except ImportError:
            logger.error("Rerun is not installed. Install with: pip install rerun-sdk")
            return
        rr.init("lerobot_camera_stream")
        rr.spawn(memory_limit=RERUN_MEMORY_LIMIT)

    # Create camera config
    config = OpenCVCameraConfig(
        index_or_path=camera_index if isinstance(camera_index, int) else Path(camera_index),
        fps=fps,
        width=width,
        height=height,
        color_mode=ColorMode.RGB,  # RGB for Rerun
        warmup_s=int(warmup_s),
        fourcc=fourcc,
    )

    camera = OpenCVCamera(config)

    # Initialize variables before try block to avoid UnboundLocalError in finally
    frame_count = 0
    start_time = None
    recent_fps = 0.0

    try:
        # Connect (includes warmup)
        logger.info(f"Connecting camera (warmup: {warmup_s}s)...")
        camera.connect(warmup=True)
        logger.info(f"Connected: {camera.width}x{camera.height} @ {camera.fps} FPS")

        # Streaming loop
        start_time = time.perf_counter()
        fps_update_interval = 0.5
        last_fps_update = start_time

        logger.info("Streaming... Press Ctrl+C to stop.")

        while True:
            loop_start = time.perf_counter()

            # Check duration limit
            if duration_s is not None and (loop_start - start_time) >= duration_s:
                logger.info(f"Duration limit reached ({duration_s}s)")
                break

            # Read frame using LeRobot camera
            frame_rgb = camera.async_read() if use_async else camera.read()

            frame_count += 1
            current_time = time.perf_counter()
            frame_latency_ms = (current_time - loop_start) * 1000

            # Calculate FPS
            if current_time - last_fps_update >= fps_update_interval:
                elapsed = current_time - start_time
                recent_fps = frame_count / elapsed if elapsed > 0 else 0
                last_fps_update = current_time

            # Add FPS and latency overlay
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            overlay_text = f"FPS: {recent_fps:.1f} | Latency: {frame_latency_ms:.1f}ms | Frame: {frame_count}"
            cv2.putText(
                frame_bgr,
                overlay_text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            # Display frame
            if use_cv2_display:
                # Use cv2.imshow for lowest latency display
                cv2.imshow(window_name, frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # Q or ESC to quit
                    logger.info("Quit requested by user.")
                    break
            else:
                # Use Rerun for visualization
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                rr.log("camera/image", rr.Image(frame_rgb))
                rr.log("camera/fps", rr.Scalars(recent_fps))
                rr.log("camera/latency_ms", rr.Scalars(frame_latency_ms))

            # Progress log every 100 frames
            if frame_count % 100 == 0:
                elapsed = current_time - start_time
                logger.info(
                    f"  Frames: {frame_count}, FPS: {recent_fps:.1f}, "
                    f"Latency: {frame_latency_ms:.1f}ms, Elapsed: {elapsed:.1f}s"
                )

    except RuntimeError as e:
        # Handle FPS validation errors or other camera connection issues
        if "failed to set fps" in str(e):
            logger.error(f"Camera FPS validation failed: {e}")
            logger.info("Tip: The camera may not support the requested FPS. Try using --scan-capabilities to see supported FPS.")
        else:
            logger.error(f"Camera error: {e}")
        raise
    except KeyboardInterrupt:
        logger.info("Stream stopped by user.")
    finally:
        camera.disconnect()
        if use_cv2_display:
            cv2.destroyAllWindows()
        if start_time is not None:
            elapsed = time.perf_counter() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            logger.info("=" * 60)
            logger.info("LeRobot Camera Stream Summary:")
            logger.info(f"  Total frames: {frame_count}")
            logger.info(f"  Duration: {elapsed:.1f}s")
            logger.info(f"  Average FPS: {avg_fps:.2f}")
            logger.info("=" * 60)


def benchmark_camera(
    camera_index: int | str,
    fps: float = 30,
    width: int = 1280,
    height: int = 720,
    fourcc: str | None = "MJPG",
    duration_s: float = 10.0,
    warmup_s: float = 1.0,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Run performance benchmark on a camera.

    Args:
        camera_index: Camera index or path (e.g., 0 or '/dev/video0')
        fps: Requested FPS
        width: Requested width
        height: Requested height
        fourcc: FOURCC codec (e.g., 'MJPG'). Default: MJPG.
        duration_s: Benchmark duration in seconds
        warmup_s: Warmup time before measuring
        verbose: Print progress

    Returns:
        BenchmarkResult with all metrics
    """
    if verbose:
        logger.info("=" * 60)
        logger.info(f"Benchmarking camera {camera_index}")
        logger.info(f"Requested: {width}x{height} @ {fps} FPS, FOURCC: {fourcc or 'MJPG'}")
        logger.info("=" * 60)

    cap = None
    try:
        # Open camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return BenchmarkResult(
                camera_index=camera_index,
                requested_fps=fps,
                requested_width=width,
                requested_height=height,
                actual_fps=0,
                actual_width=0,
                actual_height=0,
                fourcc="",
                duration_s=0,
                total_frames=0,
                dropped_frames=0,
                latency_ms_min=0,
                latency_ms_max=0,
                latency_ms_avg=0,
                latency_ms_std=0,
                latency_ms_p50=0,
                latency_ms_p95=0,
                latency_ms_p99=0,
                success=False,
                error="Failed to open camera",
            )

        # Set FOURCC first (affects available resolutions/fps)
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, string_to_fourcc(fourcc))

        # Set resolution and FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        # Get actual settings
        info = get_camera_info(cap)
        actual_width = info["width"]
        actual_height = info["height"]
        actual_fourcc = info["fourcc"]

        if verbose:
            logger.info(
                f"Actual: {actual_width}x{actual_height} @ {info['fps']:.1f} FPS, FOURCC: {actual_fourcc}"
            )
            logger.info(f"Backend: {info['backend']}")

        # Warmup
        if verbose:
            logger.info(f"Warming up for {warmup_s}s...")
        warmup_start = time.perf_counter()
        while time.perf_counter() - warmup_start < warmup_s:
            ret, _ = cap.read()
            if not ret:
                return BenchmarkResult(
                    camera_index=camera_index,
                    requested_fps=fps,
                    requested_width=width,
                    requested_height=height,
                    actual_fps=0,
                    actual_width=actual_width,
                    actual_height=actual_height,
                    fourcc=actual_fourcc,
                    duration_s=0,
                    total_frames=0,
                    dropped_frames=0,
                    latency_ms_min=0,
                    latency_ms_max=0,
                    latency_ms_avg=0,
                    latency_ms_std=0,
                    latency_ms_p50=0,
                    latency_ms_p95=0,
                    latency_ms_p99=0,
                    success=False,
                    error="Failed to read frame during warmup",
                )

        # Benchmark
        if verbose:
            logger.info(f"Running benchmark for {duration_s}s...")

        latencies: list[float] = []
        frame_times: list[float] = []
        dropped_frames = 0
        total_frames = 0

        start_time = time.perf_counter()
        last_frame_time = start_time

        while time.perf_counter() - start_time < duration_s:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t1 = time.perf_counter()

            if not ret:
                dropped_frames += 1
                continue

            total_frames += 1
            latency_ms = (t1 - t0) * 1000
            latencies.append(latency_ms)

            current_time = time.perf_counter()
            frame_times.append(current_time - last_frame_time)
            last_frame_time = current_time

            # Progress indicator
            if verbose and total_frames % 100 == 0:
                elapsed = time.perf_counter() - start_time
                current_fps = total_frames / elapsed
                logger.info(f"  Progress: {elapsed:.1f}s, Frames: {total_frames}, FPS: {current_fps:.1f}")

        end_time = time.perf_counter()
        actual_duration = end_time - start_time

        # Calculate metrics
        latencies_arr = np.array(latencies)
        actual_fps = total_frames / actual_duration if actual_duration > 0 else 0

        result = BenchmarkResult(
            camera_index=camera_index,
            requested_fps=fps,
            requested_width=width,
            requested_height=height,
            actual_fps=actual_fps,
            actual_width=actual_width,
            actual_height=actual_height,
            fourcc=actual_fourcc,
            duration_s=actual_duration,
            total_frames=total_frames,
            dropped_frames=dropped_frames,
            latency_ms_min=float(np.min(latencies_arr)) if len(latencies_arr) > 0 else 0,
            latency_ms_max=float(np.max(latencies_arr)) if len(latencies_arr) > 0 else 0,
            latency_ms_avg=float(np.mean(latencies_arr)) if len(latencies_arr) > 0 else 0,
            latency_ms_std=float(np.std(latencies_arr)) if len(latencies_arr) > 0 else 0,
            latency_ms_p50=float(np.percentile(latencies_arr, 50)) if len(latencies_arr) > 0 else 0,
            latency_ms_p95=float(np.percentile(latencies_arr, 95)) if len(latencies_arr) > 0 else 0,
            latency_ms_p99=float(np.percentile(latencies_arr, 99)) if len(latencies_arr) > 0 else 0,
            success=True,
        )

        if verbose:
            print_result(result)

        return result

    except Exception as e:
        return BenchmarkResult(
            camera_index=camera_index,
            requested_fps=fps,
            requested_width=width,
            requested_height=height,
            actual_fps=0,
            actual_width=0,
            actual_height=0,
            fourcc="",
            duration_s=0,
            total_frames=0,
            dropped_frames=0,
            latency_ms_min=0,
            latency_ms_max=0,
            latency_ms_avg=0,
            latency_ms_std=0,
            latency_ms_p50=0,
            latency_ms_p95=0,
            latency_ms_p99=0,
            success=False,
            error=str(e),
        )
    finally:
        if cap is not None:
            cap.release()


def scan_camera_capabilities(
    camera_index: int | str,
    verbose: bool = True,
) -> CameraCapabilities:
    """
    Scan camera capabilities by testing various settings.

    Args:
        camera_index: Camera index or path
        verbose: Print progress

    Returns:
        CameraCapabilities with supported modes
    """
    if verbose:
        logger.info("=" * 60)
        logger.info(f"Scanning capabilities for camera {camera_index}")
        logger.info("=" * 60)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_index}")
        return CameraCapabilities(
            camera_index=camera_index,
            name="Unknown",
            backend="Unknown",
        )

    info = get_camera_info(cap)
    
    # Get USB speed if camera index is an integer
    usb_speed_mbps = 0
    usb_version = "Unknown"
    if isinstance(camera_index, int):
        usb_speed_mbps, usb_version = get_usb_speed(camera_index)
    
    capabilities = CameraCapabilities(
        camera_index=camera_index,
        name=f"Camera {camera_index}",
        backend=info["backend"],
        usb_speed_mbps=usb_speed_mbps,
        usb_version=usb_version,
    )
    cap.release()

    # Test resolutions and measure actual FPS (using MJPG for best performance)
    if verbose:
        logger.info("Testing resolutions and measuring actual FPS (using MJPG)...")
    for width, height in COMMON_RESOLUTIONS:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            continue

        # Use MJPG for best FPS performance
        cap.set(cv2.CAP_PROP_FOURCC, string_to_fourcc("MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Request high FPS to measure maximum achievable
        cap.set(cv2.CAP_PROP_FPS, 120)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Check if camera accepted this resolution
        if actual_w == width and actual_h == height:
            # Measure actual FPS by reading frames
            ret, _ = cap.read()
            if ret:
                # Warm up
                for _ in range(5):
                    cap.read()
                
                # Measure actual FPS over ~1 second
                frame_count = 0
                start_time = time.perf_counter()
                measure_duration = 1.0  # seconds
                
                while time.perf_counter() - start_time < measure_duration:
                    ret, _ = cap.read()
                    if ret:
                        frame_count += 1
                    else:
                        break
                
                elapsed = time.perf_counter() - start_time
                measured_fps = frame_count / elapsed if elapsed > 0 else 0
                
                capabilities.supported_resolutions.append((width, height))
                capabilities.resolution_fps_map[(width, height)] = [measured_fps]
                
                # Track max FPS globally
                if measured_fps not in capabilities.supported_fps:
                    capabilities.supported_fps.append(measured_fps)
                
                if verbose:
                    marker = "🚀" if measured_fps >= 90 else ("⚡" if measured_fps >= 60 else "✓")
                    logger.info(f"  {marker} {width}x{height} @ {measured_fps:.1f} FPS (measured)")
            else:
                if verbose:
                    logger.warning(f"  ✗ {width}x{height} (cannot read frame)")
        else:
            if verbose:
                logger.debug(f"  ✗ {width}x{height} (got {actual_w}x{actual_h})")

        cap.release()
    
    # Sort supported_fps
    capabilities.supported_fps.sort()

    # Test FOURCC codes (only MJPG and YUYV)
    if verbose:
        logger.info("Testing FOURCC codes (MJPG, YUYV)...")
    for fourcc_str in ["MJPG", "YUYV"]:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            continue

        try:
            cap.set(cv2.CAP_PROP_FOURCC, string_to_fourcc(fourcc_str))
            actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            actual_fourcc = fourcc_to_string(actual_fourcc_int)

            if actual_fourcc == fourcc_str:
                ret, _ = cap.read()
                if ret:
                    capabilities.supported_fourcc.append(fourcc_str)
                    if verbose:
                        logger.info(f"  ✓ {fourcc_str}")
                else:
                    if verbose:
                        logger.warning(f"  ✗ {fourcc_str} (cannot read frame)")
            else:
                if verbose:
                    logger.debug(f"  ✗ {fourcc_str} (got {actual_fourcc})")
        except Exception:
            if verbose:
                logger.debug(f"  ✗ {fourcc_str} (error)")

        cap.release()

    return capabilities


def print_result(result: BenchmarkResult) -> None:
    """Print benchmark result in a formatted way."""
    logger.info("─" * 40)
    logger.info("BENCHMARK RESULTS")
    logger.info("─" * 40)

    if not result.success:
        logger.error(f"❌ FAILED: {result.error}")
        return

    fps_ratio = result.actual_fps / result.requested_fps * 100 if result.requested_fps > 0 else 0
    fps_status = "✓" if fps_ratio >= 95 else "⚠" if fps_ratio >= 80 else "✗"

    logger.info(f"Camera: {result.camera_index}")
    logger.info(
        f"Resolution: {result.actual_width}x{result.actual_height} "
        f"(requested: {result.requested_width}x{result.requested_height})"
    )
    logger.info(f"FOURCC: {result.fourcc}")
    logger.info("FPS Performance:")
    logger.info(
        f"  {fps_status} Actual FPS: {result.actual_fps:.2f} "
        f"({fps_ratio:.1f}% of requested {result.requested_fps})"
    )
    logger.info(f"  Total frames: {result.total_frames}")
    logger.info(f"  Dropped frames: {result.dropped_frames}")
    logger.info("Latency (ms):")
    logger.info(f"  Min:  {result.latency_ms_min:.2f}")
    logger.info(f"  Max:  {result.latency_ms_max:.2f}")
    logger.info(f"  Avg:  {result.latency_ms_avg:.2f} ± {result.latency_ms_std:.2f}")
    logger.info(f"  P50:  {result.latency_ms_p50:.2f}")
    logger.info(f"  P95:  {result.latency_ms_p95:.2f}")
    logger.info(f"  P99:  {result.latency_ms_p99:.2f}")
    logger.info("─" * 40)


def print_capabilities(caps: CameraCapabilities) -> None:
    """Print camera capabilities."""
    logger.info("─" * 60)
    logger.info(f"Camera {caps.camera_index} Capabilities")
    logger.info("─" * 60)
    logger.info(f"Backend: {caps.backend}")
    
    # USB connection info
    if caps.usb_speed_mbps > 0:
        usb_icon = "🚀" if caps.usb_speed_mbps >= 5000 else "⚡" if caps.usb_speed_mbps >= 480 else "🐢"
        effective_bw = caps.usb_speed_mbps * 0.8  # ~80% efficiency
        logger.info(f"USB: {usb_icon} {caps.usb_version} ({caps.usb_speed_mbps} Mbps, ~{effective_bw:.0f} Mbps effective)")
    
    # Print resolution + measured FPS
    logger.info(f"\nSupported Resolutions (measured FPS with MJPG):")
    logger.info("─" * 60)
    for w, h in caps.supported_resolutions:
        fps_list = caps.resolution_fps_map.get((w, h), [])
        if fps_list:
            max_fps = max(fps_list)
            # Highlight high FPS support
            marker = "🚀" if max_fps >= 90 else ("⚡" if max_fps >= 60 else "  ")
            logger.info(f"  {marker} {w:4d}x{h:<4d}  @  {max_fps:.1f} FPS")
        else:
            logger.info(f"     {w:4d}x{h:<4d}  @  N/A")
    
    logger.info(f"\nSupported FOURCC: {caps.supported_fourcc}")
    logger.info("─" * 60)
    
    # Summary - find best resolution for 60 FPS
    high_fps_resolutions = [
        (w, h, max(fps_list)) for (w, h), fps_list in caps.resolution_fps_map.items() 
        if fps_list and max(fps_list) >= 60
    ]
    if high_fps_resolutions:
        # Sort by resolution (pixels) descending
        high_fps_resolutions.sort(key=lambda x: x[0] * x[1], reverse=True)
        best = high_fps_resolutions[0]
        logger.info(f"\n✅ Best 60+ FPS: {best[0]}x{best[1]} @ {best[2]:.1f} FPS")
    else:
        # Find the best FPS available
        all_fps = [(w, h, max(fps_list)) for (w, h), fps_list in caps.resolution_fps_map.items() if fps_list]
        if all_fps:
            all_fps.sort(key=lambda x: x[2], reverse=True)
            best = all_fps[0]
            logger.info(f"\n⚠️ Max FPS available: {best[0]}x{best[1]} @ {best[2]:.1f} FPS (60 FPS not supported)")
    
    # USB bandwidth recommendation
    if caps.usb_speed_mbps > 0 and caps.usb_speed_mbps < 5000:
        logger.info(f"\n💡 Tip: Camera is on {caps.usb_version}. Use MJPG for high FPS.")
        logger.info(f"   For YUYV high FPS, connect to USB 3.0 (blue port).")


def find_cameras(max_index: int = 10) -> list[int]:
    """Find available camera indices."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def save_results_csv(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save benchmark results to CSV."""
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "camera_index",
                "requested_fps",
                "requested_width",
                "requested_height",
                "actual_fps",
                "actual_width",
                "actual_height",
                "fourcc",
                "duration_s",
                "total_frames",
                "dropped_frames",
                "latency_ms_min",
                "latency_ms_max",
                "latency_ms_avg",
                "latency_ms_std",
                "latency_ms_p50",
                "latency_ms_p95",
                "latency_ms_p99",
                "success",
                "error",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.camera_index,
                    r.requested_fps,
                    r.requested_width,
                    r.requested_height,
                    r.actual_fps,
                    r.actual_width,
                    r.actual_height,
                    r.fourcc,
                    r.duration_s,
                    r.total_frames,
                    r.dropped_frames,
                    r.latency_ms_min,
                    r.latency_ms_max,
                    r.latency_ms_avg,
                    r.latency_ms_std,
                    r.latency_ms_p50,
                    r.latency_ms_p95,
                    r.latency_ms_p99,
                    r.success,
                    r.error or "",
                ]
            )

    logger.info(f"Results saved to: {output_path}")


def compare_fourcc_formats(
    camera_index: int | str,
    fps: float = 60,
    width: int = 1280,
    height: int = 720,
    duration_s: float = 5.0,
    warmup_s: float = 1.0,
) -> None:
    """
    Compare MJPG vs YUYV performance (bandwidth and FPS).

    Args:
        camera_index: Camera index or path
        fps: Target FPS to test
        width: Resolution width
        height: Resolution height
        duration_s: Test duration per format
        warmup_s: Warmup time per format
    """
    import time

    logger.info("=" * 70)
    logger.info(f"FOURCC Format Comparison: MJPG vs YUYV")
    logger.info(f"Camera: {camera_index} | Resolution: {width}x{height} | Target FPS: {fps}")
    logger.info("=" * 70)

    formats_to_test = ["MJPG", "YUYV"]
    results: dict[str, dict] = {}

    for fourcc_str in formats_to_test:
        logger.info(f"\n📹 Testing {fourcc_str}...")

        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_index}")
            continue

        # Set format
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc_str)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        # Read actual settings
        actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = "".join([chr((actual_fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps_reported = cap.get(cv2.CAP_PROP_FPS)

        if actual_fourcc != fourcc_str:
            logger.warning(f"  ⚠️ Requested {fourcc_str}, got {actual_fourcc}")

        # Warmup
        warmup_frames = 0
        warmup_start = time.perf_counter()
        while time.perf_counter() - warmup_start < warmup_s:
            ret, frame = cap.read()
            if ret:
                warmup_frames += 1

        # Benchmark
        frame_count = 0
        total_bytes = 0
        latencies = []
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration_s:
            frame_start = time.perf_counter()
            ret, frame = cap.read()
            frame_end = time.perf_counter()

            if ret:
                frame_count += 1
                total_bytes += frame.nbytes
                latencies.append((frame_end - frame_start) * 1000)

        elapsed = time.perf_counter() - start_time
        cap.release()

        # Calculate metrics
        actual_fps = frame_count / elapsed if elapsed > 0 else 0
        decoded_throughput = (total_bytes / elapsed) / (1024 * 1024) if elapsed > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        # Calculate USB bandwidth in Mbps (actual data transferred over USB)
        # YUYV = 2 bytes per pixel (uncompressed)
        # MJPG = compressed, typically 1/10 to 1/20 of YUYV
        # Convert: bytes/s -> Mbps = bytes/s * 8 / 1_000_000
        if fourcc_str == "YUYV":
            usb_bandwidth = (actual_width * actual_height * 2 * actual_fps) * 8 / 1_000_000
        else:  # MJPG - estimate based on typical compression ratio
            # MJPG compression ratio varies, typically 10:1 to 20:1 for video
            usb_bandwidth = (actual_width * actual_height * 2 * actual_fps) * 8 / 1_000_000 / 10

        results[fourcc_str] = {
            "actual_fourcc": actual_fourcc,
            "actual_fps": actual_fps,
            "fps_ratio": (actual_fps / fps * 100) if fps > 0 else 0,
            "usb_bandwidth_mbps": usb_bandwidth,
            "decoded_throughput_mbps": decoded_throughput,
            "avg_latency_ms": avg_latency,
            "frame_count": frame_count,
            "resolution": f"{actual_width}x{actual_height}",
            "width": actual_width,
            "height": actual_height,
        }

        logger.info(f"  Resolution: {actual_width}x{actual_height}")
        logger.info(f"  Actual FPS: {actual_fps:.1f} ({results[fourcc_str]['fps_ratio']:.0f}% of target)")
        logger.info(f"  USB Bandwidth: ~{usb_bandwidth:.0f} Mbps {'(uncompressed)' if fourcc_str == 'YUYV' else '(compressed ~10:1)'}")
        logger.info(f"  Avg Latency: {avg_latency:.1f} ms")

    # Print comparison summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 COMPARISON SUMMARY")
    logger.info("=" * 70)

    # Table header
    logger.info(f"{'Format':<8} {'Resolution':<12} {'FPS':<12} {'USB BW':<15} {'Latency':<12} {'Status'}")
    logger.info("-" * 70)

    for fourcc_str, data in results.items():
        fps_status = "✓" if data["fps_ratio"] >= 95 else "⚠" if data["fps_ratio"] >= 50 else "✗"
        logger.info(
            f"{fourcc_str:<8} {data['resolution']:<12} "
            f"{data['actual_fps']:.1f} FPS{'':<4} "
            f"~{data['usb_bandwidth_mbps']:.0f} Mbps{'':<6} "
            f"{data['avg_latency_ms']:.1f} ms{'':<5} "
            f"{fps_status}"
        )

    # Analysis
    if "MJPG" in results and "YUYV" in results:
        mjpg = results["MJPG"]
        yuyv = results["YUYV"]

        logger.info("\n" + "-" * 70)
        logger.info("📈 ANALYSIS")
        logger.info("-" * 70)

        fps_diff = mjpg["actual_fps"] - yuyv["actual_fps"]
        fps_ratio = (mjpg["actual_fps"] / yuyv["actual_fps"]) if yuyv["actual_fps"] > 0 else 0

        if fps_diff > 0:
            logger.info(f"⚡ MJPG is {fps_ratio:.1f}x faster ({mjpg['actual_fps']:.1f} vs {yuyv['actual_fps']:.1f} FPS)")
        elif fps_diff < 0:
            fps_ratio_inv = (yuyv["actual_fps"] / mjpg["actual_fps"]) if mjpg["actual_fps"] > 0 else 0
            logger.info(f"⚡ YUYV is {fps_ratio_inv:.1f}x faster ({yuyv['actual_fps']:.1f} vs {mjpg['actual_fps']:.1f} FPS)")
        else:
            logger.info("⚡ Both formats achieve similar FPS")

        # Calculate theoretical USB bandwidth for uncompressed YUYV at MJPG's FPS (in Mbps)
        w, h = mjpg["width"], mjpg["height"]
        yuyv_theoretical_at_mjpg_fps = (w * h * 2 * mjpg["actual_fps"]) * 8 / 1_000_000
        mjpg_usb_bw = mjpg["usb_bandwidth_mbps"]
        yuyv_usb_bw = yuyv["usb_bandwidth_mbps"]

        logger.info(f"\n💾 USB Bandwidth Analysis:")
        logger.info(f"   YUYV @ {yuyv['actual_fps']:.0f} FPS: ~{yuyv_usb_bw:.0f} Mbps (uncompressed)")
        logger.info(f"   MJPG @ {mjpg['actual_fps']:.0f} FPS: ~{mjpg_usb_bw:.0f} Mbps (compressed)")
        logger.info(f"   YUYV @ {mjpg['actual_fps']:.0f} FPS would need: ~{yuyv_theoretical_at_mjpg_fps:.0f} Mbps")

        bw_savings = yuyv_theoretical_at_mjpg_fps / mjpg_usb_bw if mjpg_usb_bw > 0 else 0
        logger.info(f"   💡 MJPG saves ~{bw_savings:.0f}x USB bandwidth vs YUYV at same FPS")

        # USB bandwidth info
        logger.info("\n📡 USB Bandwidth Reference:")
        logger.info("   USB 2.0: ~280-320 Mbps effective (of 480 Mbps)")
        logger.info("   USB 3.0: ~3200 Mbps effective (of 5000 Mbps)")

        yuyv_limited_by_usb = yuyv["actual_fps"] < fps * 0.5  # Less than 50% of target
        if yuyv_limited_by_usb:
            logger.info(f"\n⚠️ YUYV is bandwidth-limited (only {yuyv['fps_ratio']:.0f}% of target FPS)")
            logger.info("✅ Recommendation: Use MJPG for high FPS on USB 2.0")

    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="OpenCV Camera Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all cameras with device/manufacturer information.",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=25,
        help="Maximum camera index to scan when using --list (default: 25).",
    )
    parser.add_argument(
        "--index",
        "-i",
        type=int,
        nargs="+",
        default=None,
        help="Camera index(es) to test. If not specified, auto-detect cameras.",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        nargs="+",
        default=None,
        help="Camera path(s) to test (e.g., /dev/video0).",
    )
    parser.add_argument(
        "--fps",
        "-f",
        type=float,
        default=30,
        help="Requested FPS (default: 30).",
    )
    parser.add_argument(
        "--width",
        "-W",
        type=int,
        default=1280,
        help="Requested width (default: 1280).",
    )
    parser.add_argument(
        "--height",
        "-H",
        type=int,
        default=720,
        help="Requested height (default: 720).",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default="MJPG",
        help="FOURCC codec (e.g., MJPG, YUYV). Default: MJPG.",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=10.0,
        help="Benchmark duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=2.0,
        help="Warmup duration in seconds for benchmark/stream (default: 2).",
    )
    parser.add_argument(
        "--scan-capabilities",
        "-s",
        action="store_true",
        help="Scan camera capabilities (resolutions, FPS, FOURCC).",
    )
    parser.add_argument(
        "--video-stream",
        "-v",
        action="store_true",
        help="Stream video with Rerun visualization (MJPG default, warmup before FPS measurement).",
    )
    parser.add_argument(
        "--use-lerobot",
        "-L",
        action="store_true",
        help="Use LeRobot's OpenCVCamera implementation (threaded, lower latency).",
    )
    parser.add_argument(
        "--sync-read",
        action="store_true",
        help="Use synchronous read() instead of async_read() when using --use-lerobot.",
    )
    parser.add_argument(
        "--cv2-display",
        action="store_true",
        help="Use cv2.imshow() for lowest latency display (instead of Rerun). Press Q to quit.",
    )
    parser.add_argument(
        "--test-all-resolutions",
        action="store_true",
        help="Test all common resolutions.",
    )
    parser.add_argument(
        "--test-all-fps",
        action="store_true",
        help="Test all common FPS values.",
    )
    parser.add_argument(
        "--compare-fourcc",
        action="store_true",
        help="Compare MJPG vs YUYV performance (bandwidth and FPS).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV file for results.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output.",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Handle --list option
    if args.list:
        cameras = list_cameras(max_index=args.max_index)
        print_camera_list(cameras)
        return

    # Determine camera indices to test
    camera_ids: list[int | str] = []
    if args.path:
        camera_ids.extend(args.path)
    if args.index:
        camera_ids.extend(args.index)
    if not camera_ids:
        # Auto-detect
        logger.info("Auto-detecting cameras...")
        camera_ids = find_cameras()
        if not camera_ids:
            logger.error("No cameras found!")
            return
        logger.info(f"Found cameras: {camera_ids}")

    # Handle --video-stream option
    if args.video_stream:
        if len(camera_ids) > 1:
            logger.warning("Video stream only supports one camera. Using first camera.")
        cam_id = camera_ids[0]
        # Default to MJPG for video streaming if not specified
        fourcc = args.fourcc if args.fourcc else "MJPG"
        duration = args.duration if args.duration != 10.0 else None  # None for infinite if default

        if args.use_lerobot:
            # Use LeRobot's OpenCVCamera implementation
            stream_video_with_lerobot(
                camera_index=cam_id,
                fps=args.fps,
                width=args.width,
                height=args.height,
                fourcc=fourcc,
                duration_s=duration,
                warmup_s=args.warmup,
                use_async=not args.sync_read,
                use_cv2_display=args.cv2_display,
            )
        else:
            # Use raw OpenCV implementation
            stream_video_with_rerun(
                camera_index=cam_id,
                fps=args.fps,
                width=args.width,
                height=args.height,
                fourcc=fourcc,
                duration_s=duration,
                warmup_s=args.warmup,
                use_cv2_display=args.cv2_display,
            )
        return

    all_results: list[BenchmarkResult] = []

    for cam_id in camera_ids:
        # Scan capabilities if requested
        if args.scan_capabilities:
            caps = scan_camera_capabilities(cam_id, verbose=verbose)
            print_capabilities(caps)

        # Test all resolutions if requested
        if args.test_all_resolutions:
            logger.info(f"Testing all resolutions for camera {cam_id}...")
            for width, height in COMMON_RESOLUTIONS:
                result = benchmark_camera(
                    cam_id,
                    fps=args.fps,
                    width=width,
                    height=height,
                    fourcc=args.fourcc,
                    duration_s=min(args.duration, 5.0),  # Shorter duration for matrix
                    warmup_s=args.warmup,
                    verbose=verbose,
                )
                all_results.append(result)

        # Test all FPS if requested
        elif args.test_all_fps:
            logger.info(f"Testing all FPS values for camera {cam_id}...")
            for fps in COMMON_FPS:
                result = benchmark_camera(
                    cam_id,
                    fps=fps,
                    width=args.width,
                    height=args.height,
                    fourcc=args.fourcc,
                    duration_s=min(args.duration, 5.0),  # Shorter duration for matrix
                    warmup_s=args.warmup,
                    verbose=verbose,
                )
                all_results.append(result)

        # Compare MJPG vs YUYV
        elif args.compare_fourcc:
            compare_fourcc_formats(
                cam_id,
                fps=args.fps,
                width=args.width,
                height=args.height,
                duration_s=min(args.duration, 5.0),
                warmup_s=args.warmup,
            )

        # Standard benchmark
        else:
            result = benchmark_camera(
                cam_id,
                fps=args.fps,
                width=args.width,
                height=args.height,
                fourcc=args.fourcc,
                duration_s=args.duration,
                warmup_s=args.warmup,
                verbose=verbose,
            )
            all_results.append(result)

    # Summary
    if len(all_results) > 1:
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        header = f"{'Camera':<10} {'Resolution':<12} {'Req FPS':<8} {'Act FPS':<10} {'Latency (ms)':<15} {'Status'}"
        logger.info(header)
        logger.info("-" * 70)
        for r in all_results:
            if r.success:
                fps_ratio = r.actual_fps / r.requested_fps * 100 if r.requested_fps > 0 else 0
                status = "✓" if fps_ratio >= 95 else "⚠" if fps_ratio >= 80 else "✗"
                res = f"{r.actual_width}x{r.actual_height}"
                latency = f"{r.latency_ms_avg:.1f}±{r.latency_ms_std:.1f}"
                logger.info(
                    f"{str(r.camera_index):<10} {res:<12} {r.requested_fps:<8.0f} "
                    f"{r.actual_fps:<10.2f} {latency:<15} {status}"
                )
            else:
                logger.info(
                    f"{str(r.camera_index):<10} {'FAILED':<12} {r.requested_fps:<8.0f} "
                    f"{'-':<10} {'-':<15} ✗ {r.error}"
                )

    # Save results
    if args.output:
        save_results_csv(all_results, args.output)


if __name__ == "__main__":
    main()
