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

Example (SO-101):

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

Example (Flexiv Rizon4 RT + Pico4):

```shell
lerobot-teleoperate \
    --robot.type=flexiv_rizon4_rt \
    --robot.robot_ip=192.168.2.100 \
    --robot.local_ip=192.168.2.1 \
    --robot.id=right \
    --teleop.type=pico4 \
    --teleop.id=right \
    --fps=60 \
    --no_obs=true \
    --debug_timing=true
```

Example (Bimanual Flexiv Rizon4 RT + Bi-Pico4):

```shell
lerobot-teleoperate \
    --robot.type=bi_flexiv_rizon4_rt \
    --robot.left_config.robot_ip=192.168.2.100 \
    --robot.left_config.local_ip=192.168.2.1 \
    --robot.right_config.robot_ip=192.168.3.100 \
    --robot.right_config.local_ip=192.168.3.1 \
    --robot.id=bimanual \
    --teleop.type=bi_pico4 \
    --teleop.id=bimanual \
    --fps=60 \
    --no_obs=true
```

Example (XenseFlare + SpaceMouse):

```shell
lerobot-teleoperate \
    --robot.type=xense_flare \
    --robot.id=right \
    --teleop.type=spacemouse \
    --teleop.id=right \
    --fps=30 \
    --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    arx5_follower,
    bi_arx5,
    bi_flexiv_rizon4_rt,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    flexiv_rizon4,
    flexiv_rizon4_rt,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    pylibfranka_research3,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
    xense_flare as xense_flare_robot,
    xense_multisensor,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_pico4,
    bi_so_leader,
    btgamepad,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    pico4,
    reachy2_teleoperator,
    so_leader,
    spacemouse,
    unitree_g1,
    vive_tracker,
    xense_flare,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to display compressed images in Rerun
    display_compressed_images: bool = False
    # Skip robot.get_observation() each loop tick for maximum teleop frequency.
    # Disables display_data and observation-dependent features.
    no_obs: bool = False
    # Print per-step timing breakdown instead of action values.
    debug_timing: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
    no_obs: bool = False,
    debug_timing: bool = False,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
        no_obs: If True, skip robot.get_observation() each loop tick for higher frequency teleop.
                Disables display_data automatically.
        debug_timing: If True, print per-step timing breakdown instead of action table.
    """
    # no_obs mode disables display_data since there's no observation to display
    if no_obs:
        display_data = False

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation (skip if no_obs mode for maximum frequency)
        obs = None
        obs_time_ms = 0.0
        if not no_obs:
            obs_t0 = time.perf_counter()
            obs = robot.get_observation()
            obs_time_ms = (time.perf_counter() - obs_t0) * 1e3

        if robot.name == "unitree_g1":
            teleop.send_feedback(obs)

        # Get teleop action
        teleop_t0 = time.perf_counter()
        raw_action = teleop.get_action()
        teleop_time_ms = (time.perf_counter() - teleop_t0) * 1e3

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot
        send_t0 = time.perf_counter()
        _ = robot.send_action(robot_action_to_send)
        send_time_ms = (time.perf_counter() - send_t0) * 1e3

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )

            if not debug_timing:
                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'VALUE':>9}")
                for motor, value in robot_action_to_send.items():
                    print(f"{motor:<{display_len}} | {value:>9.4f}")
                move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start

        if debug_timing:
            print(
                f"\r\033[K"
                f"obs: {obs_time_ms:5.1f}ms | "
                f"teleop: {teleop_time_ms:5.1f}ms | "
                f"send: {send_time_ms:5.1f}ms | "
                f"loop: {loop_s * 1e3:5.1f}ms | "
                f"target: {1e3 / fps:.1f}ms | "
                f"eff: {(1 / fps) / loop_s * 100:5.1f}%",
                end="",
                flush=True,
            )
        elif not display_data:
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # no_obs overrides display_data
    if cfg.no_obs and cfg.display_data:
        logging.warning("no_obs=True: disabling display_data")
        cfg.display_data = False

    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
            no_obs=cfg.no_obs,
            debug_timing=cfg.debug_timing,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
