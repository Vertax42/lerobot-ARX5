#!/usr/bin/env python

# Copyright 2025 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TRLC Leader Teleoperator live test script.

Connects to a single TRLC Leader arm and streams joint positions
to the terminal at ~50 Hz, showing per-read latency statistics.

Usage:
    python trlc_test.py --port /dev/ttyUSB0

Press Ctrl+C to stop and print a summary report.
"""

import argparse
import statistics
import sys
import time


def clear_screen():
    print("\033[2J\033[H", end="")


def run_single(port: str, hz: float = 50.0):
    from lerobot.teleoperators.trlc_leader import TRLCLeader, TRLCLeaderConfig

    config = TRLCLeaderConfig(port=port)
    teleop = TRLCLeader(config)

    print(f"Connecting to TRLC Leader on {port} ...")
    teleop.connect()
    print("Connected. Press Ctrl+C to stop.\n")

    read_times = []
    i = 0
    period = 1.0 / hz

    try:
        while True:
            loop_start = time.perf_counter()

            t0 = time.perf_counter()
            action = teleop.get_action()
            read_ms = (time.perf_counter() - t0) * 1e3

            read_times.append(read_ms)
            if len(read_times) > 500:
                read_times.pop(0)

            if i % 10 == 0:
                clear_screen()
                print("=" * 60)
                print(f"  TRLC Leader  |  port: {port}  |  iter: {i + 1}")
                print("=" * 60)

                print("\n[Joint Positions]")
                for key, val in action.items():
                    if key == "gripper.pos":
                        print(f"  {key:<20s}  {val:6.3f}  (0=open, 1=closed)")
                    else:
                        print(f"  {key:<20s}  {val:+.4f} rad  ({val * 57.296:+7.2f} deg)")

                print("\n[Read Latency]")
                print(f"  current:  {read_ms:.2f} ms")
                if len(read_times) > 1:
                    print(f"  avg:      {statistics.mean(read_times):.2f} ms")
                    print(f"  max:      {max(read_times):.2f} ms")
                    print(f"  std:      {statistics.stdev(read_times):.2f} ms")

                loop_ms = (time.perf_counter() - loop_start) * 1e3
                print(f"\n[Loop]  {loop_ms:.1f} ms  ({1000 / max(loop_ms, 0.1):.0f} Hz)")
                print("=" * 60)
                print("  Ctrl+C to exit")
                sys.stdout.flush()

            elapsed = time.perf_counter() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            i += 1

    except KeyboardInterrupt:
        _print_summary(read_times, i)
    finally:
        teleop.disconnect()
        print("Disconnected.")


def _print_summary(read_times: list[float], iterations: int):
    print("\n\nInterrupted.")
    print("=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)
    print(f"  Total iterations: {iterations}")
    if read_times:
        print(f"  Read Latency:")
        print(f"    avg:     {statistics.mean(read_times):.2f} ms")
        print(f"    max:     {max(read_times):.2f} ms")
        if len(read_times) > 1:
            print(f"    std dev: {statistics.stdev(read_times):.2f} ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TRLC Leader live test")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port (e.g. /dev/ttyUSB0)")
    parser.add_argument("--hz", type=float, default=50.0, help="Target polling frequency (default: 50)")
    args = parser.parse_args()

    run_single(args.port, hz=args.hz)


if __name__ == "__main__":
    main()
