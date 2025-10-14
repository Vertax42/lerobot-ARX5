# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460828611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5A460826981 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)

# Import mock_teleop to make it available for CLI
try:
    from tests.mocks.mock_teleop import MockTeleopConfig  # noqa: F401
except ImportError:
    # If tests module is not available, create a minimal mock config
    @TeleoperatorConfig.register_subclass("mock_teleop")
    @dataclass
    class MockTeleopConfig(TeleoperatorConfig):
        n_motors: int = 3
        random_values: bool = True
        static_values: list[float] | None = None
        calibrated: bool = True


from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possible in teleop.py and record.py with List[Teleoperator]  # noqa: E501
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Enable detailed timing debug output
    debug_timing: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        action = teleop.get_action()
        if display_data:
            observation = robot.get_observation()
            log_rerun_data(observation, action)

        robot.send_action(action)
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start

        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        for motor, value in action.items():
            print(f"{motor:<{display_len}} | {value:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        move_cursor_up(len(action) + 5)


def bi_arx5_teleop_loop(
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    debug_timing: bool = True,
):
    """
    Specialized teleop loop for BiARX5 robot in gravity compensation mode.

    This loop is designed for manual demonstration where the robot follows human hand movements.
    It reads joint positions from the robot's observation and displays them as actions,
    but does not send commands back to the robot (manual demonstration mode).

    Args:
        robot: BiARX5 robot instance
        fps: Control frequency
        display_data: Whether to display data
        duration: Maximum duration in seconds
    """
    # display_len = max(len(key) for key in robot.action_features)  # Commented out since not used
    start = time.perf_counter()

    logging.info("Starting BiARX5 teleop loop with timing analysis")

    # Initialize timing statistics
    timing_stats = {
        "robot_obs_times": [],
        "camera_obs_times": {},
        "total_obs_times": [],
        "loop_times": [],
    }

    # Identify camera keys
    camera_keys = [
        key for key in robot.observation_features.keys() if not key.endswith(".pos")
    ]
    for cam_key in camera_keys:
        timing_stats["camera_obs_times"][cam_key] = []

    while True:
        loop_start = time.perf_counter()

        # Time the complete observation acquisition
        obs_start = time.perf_counter()

        # Get robot state (joints) timing
        robot_state_start = time.perf_counter()

        # We need to call the robot's internal method to separate timing
        # Get joint states from both arms
        if hasattr(robot, "left_arm") and hasattr(robot, "right_arm"):
            left_joint_state = robot.left_arm.get_joint_state()
            right_joint_state = robot.right_arm.get_joint_state()

        robot_obs_time = time.perf_counter() - robot_state_start
        timing_stats["robot_obs_times"].append(robot_obs_time * 1000)  # Convert to ms

        # Get camera observations timing
        camera_obs_start = time.perf_counter()
        camera_observations = {}
        camera_times = {}
        for cam_key, cam in robot.cameras.items():
            cam_start = time.perf_counter()
            camera_observations[cam_key] = cam.async_read()
            cam_time = time.perf_counter() - cam_start
            cam_time_ms = cam_time * 1000
            camera_times[cam_key] = cam_time_ms
            timing_stats["camera_obs_times"][cam_key].append(cam_time_ms)

        total_camera_time = time.perf_counter() - camera_obs_start
        total_camera_time_ms = total_camera_time * 1000

        # Build complete observation dict (similar to robot.get_observation())
        observation = {}

        # Add robot joint observations
        left_pos = left_joint_state.pos().copy()
        for i in range(6):
            observation[f"left_joint_{i+1}.pos"] = float(left_pos[i])
        observation["left_gripper.pos"] = float(left_joint_state.gripper_pos)

        right_pos = right_joint_state.pos().copy()
        for i in range(6):
            observation[f"right_joint_{i+1}.pos"] = float(right_pos[i])
        observation["right_gripper.pos"] = float(right_joint_state.gripper_pos)

        # Add camera observations
        observation.update(camera_observations)

        total_obs_time = time.perf_counter() - obs_start
        timing_stats["total_obs_times"].append(total_obs_time * 1000)  # Convert to ms

        # Extract joint positions as action
        action = {}
        for key, value in observation.items():
            if (
                key.endswith(".pos")
                and not key.startswith("head")
                and not key.startswith("left_wrist")
                and not key.startswith("right_wrist")
            ):
                action[key] = value

        # Display data if requested
        if display_data:
            log_rerun_data(observation, action)

        # Note: No send_action needed for manual demonstration

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start
        timing_stats["loop_times"].append(loop_s * 1000)  # Convert to ms

        # Display current state with specific observation values (commented out for timing focus)
        # print("\n" + "-" * (display_len + 15))
        # print(f"{'NAME':<{display_len}} | {'VALUE':>10}")
        # for motor, value in action.items():
        #     print(f"{motor:<{display_len}} | {value:>10.4f}")

        # Display additional observation info (commented out for timing focus)
        # left_joints = [
        #     f"{observation.get(f'left_joint_{i+1}.pos', 0):.3f}" for i in range(6)
        # ]
        # right_joints = [
        #     f"{observation.get(f'right_joint_{i+1}.pos', 0):.3f}" for i in range(6)
        # ]
        # print(f"\nLeft arm joints: {left_joints}")
        # print(f"Right arm joints: {right_joints}")
        # print(f"Left gripper: {observation.get('left_gripper.pos', 0):.3f}")
        # print(f"Right gripper: {observation.get('right_gripper.pos', 0):.3f}")

        # Display detailed timing analysis
        if debug_timing:
            # Clear screen and display timing info
            import os

            os.system("clear" if os.name == "posix" else "cls")

            print("🔍 TELEOP TIMING DEBUG")
            print("=" * 50)
            print(f"🤖 Robot state:     {robot_obs_time * 1000:.1f}ms")
            print(f"📷 Total cameras:   {total_camera_time_ms:.1f}ms")
            print()

            # Display individual camera timings with stability indicators
            for cam_key, cam_time_ms in camera_times.items():
                if cam_time_ms > 10:  # Slow camera warning
                    print(f"🐌 {cam_key:12}: {cam_time_ms:5.1f}ms ⚠️")
                elif cam_time_ms > 5:  # Medium speed
                    print(f"⚡ {cam_key:12}: {cam_time_ms:5.1f}ms")
                else:  # Fast camera
                    print(f"✅ {cam_key:12}: {cam_time_ms:5.1f}ms")

            print()
            print(f"📊 Total observation: {total_obs_time * 1000:.1f}ms")
            print(f"⏱️  Loop time:        {loop_s * 1000:.1f}ms")
            print(f"🎯 Target period:     {1000/fps:.1f}ms")
            print(f"📈 Loop efficiency:   {(1000/fps)/(loop_s * 1000)*100:.1f}%")

            # Camera stability warning
            if total_camera_time_ms > 20:
                print()
                print(f"⚠️  SLOW CAMERAS DETECTED! Total: {total_camera_time_ms:.1f}ms")

            print("=" * 50)
        else:
            # Simplified output - only show warnings
            if total_camera_time_ms > 20:
                print(f"⚠️  SLOW CAMERAS: {total_camera_time_ms:.1f}ms")
                for cam_key, cam_time_ms in camera_times.items():
                    if cam_time_ms > 10:
                        print(f"  🐌 {cam_key}: {cam_time_ms:.1f}ms")

            # # Calculate and display statistics every 30 loops
            # if (
            #     len(timing_stats["robot_obs_times"]) > 0
            #     and len(timing_stats["robot_obs_times"]) % 30 == 0
            # ):
            #     print("\n=== TIMING STATISTICS (last 30 loops) ===")
            #     recent_robot = timing_stats["robot_obs_times"][-30:]
            #     recent_total = timing_stats["total_obs_times"][-30:]
            #     recent_loops = timing_stats["loop_times"][-30:]

            #     print(
            #         f"Robot obs - avg: {sum(recent_robot)/len(recent_robot):.2f}ms, "
            #         f"min: {min(recent_robot):.2f}ms, max: {max(recent_robot):.2f}ms"
            #     )
            #     print(
            #         f"Total obs - avg: {sum(recent_total)/len(recent_total):.2f}ms, "
            #         f"min: {min(recent_total):.2f}ms, max: {max(recent_total):.2f}ms"
            #     )
            #     print(
            #         f"Loop time - avg: {sum(recent_loops)/len(recent_loops):.2f}ms, "
            #         f"min: {min(recent_loops):.2f}ms, max: {max(recent_loops):.2f}ms"
            #     )

            # Calculate synchronization metrics
            # robot_camera_delay = []
            # for i in range(len(recent_robot)):
            #     if i < len(recent_total):
            #         delay = (
            #             recent_total[i] - recent_robot[i]
            #         )  # Camera delay relative to robot
            #         robot_camera_delay.append(delay)

            # if robot_camera_delay:
            #     avg_delay = sum(robot_camera_delay) / len(robot_camera_delay)
            #     print(f"Avg camera delay: {avg_delay:.2f}ms")
            #     if abs(avg_delay) > 5:  # If delay > 5ms, warn user
            #         print("⚠️  WARNING: Significant timing difference detected!")

        if duration is not None and time.perf_counter() - start >= duration:
            # Print final statistics before exiting
            if len(timing_stats["robot_obs_times"]) > 10:
                print("\n=== FINAL TIMING REPORT ===")
                all_robot = timing_stats["robot_obs_times"]
                all_total = timing_stats["total_obs_times"]
                all_loops = timing_stats["loop_times"]

                print(f"Total samples: {len(all_robot)}")
                print(f"Robot obs - avg: {sum(all_robot)/len(all_robot):.2f}ms")
                print(f"Total obs - avg: {sum(all_total)/len(all_total):.2f}ms")
                print(f"Loop time - avg: {sum(all_loops)/len(all_loops):.2f}ms")

                # Final camera analysis
                for cam_key, cam_times in timing_stats["camera_obs_times"].items():
                    if cam_times:
                        avg_cam_time = sum(cam_times) / len(cam_times)
                        print(f"{cam_key} - avg: {avg_cam_time:.2f}ms")
            return

        # Adjust move_cursor_up for additional lines
        # move_cursor_up(len(action) + 15)  # Increased for timing info


@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation")

    # Check if this is BiARX5 robot
    if cfg.robot.type == "bi_arx5":
        logging.info("Detected BiARX5 robot, using specialized teleop loop")

        # Create robot instance
        robot = make_robot_from_config(cfg.robot)
        robot.connect()

        try:
            bi_arx5_teleop_loop(
                robot,
                cfg.fps,
                display_data=cfg.display_data,
                duration=cfg.teleop_time_s,
                debug_timing=cfg.debug_timing,  # Use command line parameter
            )
        except KeyboardInterrupt:
            pass
        finally:
            if cfg.display_data:
                rr.rerun_shutdown()
            robot.disconnect()
    else:
        # Standard teleoperation flow
        teleop = make_teleoperator_from_config(cfg.teleop)
        robot = make_robot_from_config(cfg.robot)

        teleop.connect()
        robot.connect()

        try:
            teleop_loop(
                teleop,
                robot,
                cfg.fps,
                display_data=cfg.display_data,
                duration=cfg.teleop_time_s,
            )
        except KeyboardInterrupt:
            pass
        finally:
            if cfg.display_data:
                rr.rerun_shutdown()
            teleop.disconnect()
            robot.disconnect()


def main():
    teleoperate()


if __name__ == "__main__":
    main()
