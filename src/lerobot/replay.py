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
Replays the actions of an episode from a dataset on a robot.

Examples:

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=aliberts/record-test \
    --dataset.episode=2
```

Example replay with bimanual so100:
```shell
lerobot-replay \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import draccus

from lerobot.datasets.lerobot_dataset import LeRobotDataset
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

from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, log_say


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True


@draccus.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)

    # Add timing for dataset loading
    logging.info("Loading dataset...")
    start_dataset_t = time.perf_counter()
    dataset = LeRobotDataset(
        cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode]
    )
    dataset_load_time = time.perf_counter() - start_dataset_t
    logging.info(f"✓ Dataset loaded in {dataset_load_time:.2f}s")

    logging.info("Selecting action columns...")
    start_actions_t = time.perf_counter()
    actions = dataset.hf_dataset.select_columns("action")
    actions_time = time.perf_counter() - start_actions_t
    logging.info(f"✓ Actions selected in {actions_time:.2f}s")

    try:
        robot.connect()

        # Switch to normal position control if supported (for BiARX5)
        if hasattr(robot, "set_to_normal_position_control") and callable(
            getattr(robot, "set_to_normal_position_control")
        ):
            logging.info("Switching robot to normal position control mode for replay")
            robot.set_to_normal_position_control()

        log_say("Replaying episode", cfg.play_sounds, blocking=True)

        # Pre-compute action names for faster access
        action_names = dataset.features["action"]["names"]
        num_frames = dataset.num_frames

        logging.info(f"Starting replay of {num_frames} frames at {dataset.fps} FPS")

        for idx in range(num_frames):
            start_episode_t = time.perf_counter()

            action_array = actions[idx]["action"]
            action = {}
            for i, name in enumerate(action_names):
                action[name] = action_array[i]

            # Debug: Print action instead of sending to robot
            # print(f"Frame {idx}: {action}")
            robot.send_action(action)

            dt_s = time.perf_counter() - start_episode_t
            busy_wait(1 / dataset.fps - dt_s)

    except KeyboardInterrupt:
        logging.info("\nKeyboardInterrupt received. Stopping replay...")
    finally:
        if robot.is_connected:
            logging.info("Disconnecting robot...")
            robot.disconnect()
            logging.info("✓ Robot disconnected safely")


def main():
    replay()


if __name__ == "__main__":
    main()
