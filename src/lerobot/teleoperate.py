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
    robot: Robot, fps: int, display_data: bool = False, duration: float | None = None
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
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    logging.info("Starting BiARX5 teleop loop")

    while True:
        loop_start = time.perf_counter()

        # Get current observation
        observation = robot.get_observation()

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

        # Display current state with specific observation values
        print("\n" + "-" * (display_len + 15))
        print(f"{'NAME':<{display_len}} | {'VALUE':>10}")
        for motor, value in action.items():
            print(f"{motor:<{display_len}} | {value:>10.4f}")

        # Display additional observation info
        left_joints = [
            f"{observation.get(f'left_joint_{i+1}.pos', 0):.3f}" for i in range(6)
        ]
        right_joints = [
            f"{observation.get(f'right_joint_{i+1}.pos', 0):.3f}" for i in range(6)
        ]
        print(f"\nLeft arm joints: {left_joints}")
        print(f"Right arm joints: {right_joints}")
        print(f"Left gripper: {observation.get('left_gripper.pos', 0):.3f}")
        print(f"Right gripper: {observation.get('right_gripper.pos', 0):.3f}")
        print(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        move_cursor_up(len(action) + 5)


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
