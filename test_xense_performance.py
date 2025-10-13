#!/usr/bin/env python3
"""
Performance test script for XenseTactileCamera.

This script runs extended tests to measure latency and throughput.
"""

import sys
import time
from collections import deque

import numpy as np

from src.lerobot.cameras.xense import (
    XenseCameraConfig,
    XenseOutputType,
    XenseTactileCamera,
)


def performance_test_sync(camera, duration_seconds=30):
    """Test synchronous reading performance."""
    print(f"\n{'='*60}")
    print(f"Synchronous Reading Performance Test ({duration_seconds}s)")
    print(f"{'='*60}")

    latencies = []
    frame_count = 0
    start_time = time.time()
    last_print_time = start_time

    print("Reading frames... (Press Ctrl+C to stop early)")

    try:
        while time.time() - start_time < duration_seconds:
            read_start = time.perf_counter()
            _ = camera.read()  # We only care about timing, not the data
            read_end = time.perf_counter()

            latency_ms = (read_end - read_start) * 1000
            latencies.append(latency_ms)
            frame_count += 1

            # Print progress every 2 seconds
            current_time = time.time()
            if current_time - last_print_time >= 2.0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed
                recent_latencies = (
                    latencies[-100:] if len(latencies) > 100 else latencies
                )
                avg_latency = np.mean(recent_latencies)
                print(
                    f"  [{elapsed:.1f}s] Frames: {frame_count}, "
                    f"FPS: {fps:.1f}, "
                    f"Avg latency: {avg_latency:.2f}ms"
                )
                last_print_time = current_time

    except KeyboardInterrupt:
        print("\n  Interrupted by user")

    total_elapsed = time.time() - start_time

    # Calculate statistics
    latencies_array = np.array(latencies)
    print("\n" + "=" * 60)
    print("Synchronous Reading Statistics:")
    print("=" * 60)
    print(f"Total frames:      {frame_count}")
    print(f"Total time:        {total_elapsed:.2f}s")
    print(f"Average FPS:       {frame_count / total_elapsed:.2f}")
    print("\nLatency Statistics (ms):")
    print(f"  Mean:            {np.mean(latencies_array):.2f}")
    print(f"  Median:          {np.median(latencies_array):.2f}")
    print(f"  Std Dev:         {np.std(latencies_array):.2f}")
    print(f"  Min:             {np.min(latencies_array):.2f}")
    print(f"  Max:             {np.max(latencies_array):.2f}")
    print(f"  P95:             {np.percentile(latencies_array, 95):.2f}")
    print(f"  P99:             {np.percentile(latencies_array, 99):.2f}")

    return latencies_array


def performance_test_async(camera, duration_seconds=30):
    """Test asynchronous reading performance."""
    print(f"\n{'='*60}")
    print(f"Asynchronous Reading Performance Test ({duration_seconds}s)")
    print(f"{'='*60}")

    latencies = []
    frame_times = deque(maxlen=1000)
    frame_count = 0
    start_time = time.time()
    last_print_time = start_time

    print("Reading frames asynchronously... (Press Ctrl+C to stop early)")

    try:
        while time.time() - start_time < duration_seconds:
            read_start = time.perf_counter()
            _ = camera.async_read(
                timeout_ms=500
            )  # We only care about timing, not the data
            read_end = time.perf_counter()

            latency_ms = (read_end - read_start) * 1000
            latencies.append(latency_ms)

            current_time = time.time()
            frame_times.append(current_time)
            frame_count += 1

            # Print progress every 2 seconds
            if current_time - last_print_time >= 2.0:
                elapsed = current_time - start_time

                # Calculate FPS from recent frames
                if len(frame_times) >= 2:
                    time_span = frame_times[-1] - frame_times[0]
                    recent_fps = (
                        (len(frame_times) - 1) / time_span if time_span > 0 else 0
                    )
                else:
                    recent_fps = 0

                recent_latencies = (
                    latencies[-100:] if len(latencies) > 100 else latencies
                )
                avg_latency = np.mean(recent_latencies)

                print(
                    f"  [{elapsed:.1f}s] Frames: {frame_count}, "
                    f"Recent FPS: {recent_fps:.1f}, "
                    f"Avg latency: {avg_latency:.2f}ms"
                )
                last_print_time = current_time

    except KeyboardInterrupt:
        print("\n  Interrupted by user")

    total_elapsed = time.time() - start_time

    # Calculate statistics
    latencies_array = np.array(latencies)
    print("\n" + "=" * 60)
    print("Asynchronous Reading Statistics:")
    print("=" * 60)
    print(f"Total frames:      {frame_count}")
    print(f"Total time:        {total_elapsed:.2f}s")
    print(f"Average FPS:       {frame_count / total_elapsed:.2f}")
    print("\nLatency Statistics (ms):")
    print(f"  Mean:            {np.mean(latencies_array):.2f}")
    print(f"  Median:          {np.median(latencies_array):.2f}")
    print(f"  Std Dev:         {np.std(latencies_array):.2f}")
    print(f"  Min:             {np.min(latencies_array):.2f}")
    print(f"  Max:             {np.max(latencies_array):.2f}")
    print(f"  P95:             {np.percentile(latencies_array, 95):.2f}")
    print(f"  P99:             {np.percentile(latencies_array, 99):.2f}")

    return latencies_array


def main():
    print("=" * 60)
    print("Xense Tactile Camera Performance Test")
    print("=" * 60)

    # Find sensors
    print("\nSearching for Xense sensors...")
    found_sensors = XenseTactileCamera.find_cameras()

    if not found_sensors:
        print("❌ No Xense sensors found!")
        sys.exit(1)

    print(f"✓ Found {len(found_sensors)} sensor(s):")
    for sensor_info in found_sensors:
        print(
            f"  - Serial: {sensor_info['serial_number']}, "
            f"Cam ID: {sensor_info['cam_id']}"
        )

    # Use first sensor
    serial_number = found_sensors[0]["serial_number"]
    print(f"\nUsing sensor: {serial_number}")

    # Create camera
    config = XenseCameraConfig(
        serial_number=serial_number,
        fps=60,
        output_types=[XenseOutputType.FORCE, XenseOutputType.FORCE_RESULTANT],
        warmup_s=0.5,
    )

    camera = XenseTactileCamera(config)

    try:
        # Connect
        print("\nConnecting to sensor...")
        camera.connect(warmup=True)
        print("✓ Connected successfully!")

        # Ask for test duration
        print("\nTest configuration:")
        print("1. Quick test (10 seconds each)")
        print("2. Standard test (30 seconds each)")
        print("3. Extended test (60 seconds each)")
        print("4. Custom duration")

        choice = input("\nEnter choice (1-4, default=2): ").strip() or "2"

        if choice == "1":
            duration = 10
        elif choice == "2":
            duration = 30
        elif choice == "3":
            duration = 60
        elif choice == "4":
            duration = int(input("Enter duration in seconds: "))
        else:
            duration = 30

        # Run tests
        print(f"\nRunning tests with {duration}s duration...")

        # Test 1: Synchronous reading
        sync_latencies = performance_test_sync(camera, duration)

        # Small pause between tests
        print("\nPausing for 2 seconds...")
        time.sleep(2)

        # Test 2: Asynchronous reading
        async_latencies = performance_test_async(camera, duration)

        # Comparison
        print(f"\n{'='*60}")
        print("Comparison: Sync vs Async")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Sync (ms)':<15} {'Async (ms)':<15} {'Diff':<10}")
        print("-" * 60)

        metrics = [
            ("Mean latency", np.mean(sync_latencies), np.mean(async_latencies)),
            ("Median latency", np.median(sync_latencies), np.median(async_latencies)),
            (
                "P95 latency",
                np.percentile(sync_latencies, 95),
                np.percentile(async_latencies, 95),
            ),
            (
                "P99 latency",
                np.percentile(sync_latencies, 99),
                np.percentile(async_latencies, 99),
            ),
        ]

        for name, sync_val, async_val in metrics:
            diff = async_val - sync_val
            sign = "+" if diff > 0 else ""
            print(f"{name:<20} {sync_val:<15.2f} {async_val:<15.2f} {sign}{diff:.2f}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Disconnect
        print("\nDisconnecting sensor...")
        try:
            camera.disconnect()
            print("✓ Disconnected successfully!")
        except Exception as e:
            print(f"⚠️  Warning during disconnect: {e}")

    print("\n" + "=" * 60)
    print("Performance test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
