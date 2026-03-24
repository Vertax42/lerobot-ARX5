# Copyright 2026 XenseRobotics Inc. team. All rights reserved.
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
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --display_data=true
```

Example (ARX5 single-arm, teach mode data collection):

```shell
lerobot-teleoperate \
    --robot.type=arx5_follower \
    --robot.arm_port=can0 \
    --teleop.type=btgamepad \
    --fps=30 \
    --display_data=true
```

Example (Bimanual ARX5, teach mode data collection):

```shell
lerobot-teleoperate \
    --robot.type=bi_arx5 \
    --robot.left_config.arm_port=can0 \
    --robot.right_config.arm_port=can1 \
    --teleop.type=btgamepad \
    --fps=30
```

Example (Flexiv Rizon4 + Pico4):

```shell
lerobot-teleoperate \
    --robot.type=flexiv_rizon4 \
    --robot.robot_sn=Rizon4-062855 \
    --teleop.type=pico4 \
    --fps=60
```

Example (Flexiv Rizon4 + SpaceMouse):

```shell
lerobot-teleoperate \
    --robot.type=flexiv_rizon4 \
    --robot.robot_sn=Rizon4-062855 \
    --teleop.type=spacemouse \
    --fps=30 \
    --display_data=true
```

Example (Flexiv Rizon4 + Vive Tracker):

```shell
lerobot-teleoperate \
    --robot.type=flexiv_rizon4 \
    --robot.robot_sn=Rizon4-062855 \
    --teleop.type=vive_tracker \
    --fps=60
```

Example (Flexiv Rizon4 + Xense Flare teleoperator):

```shell
lerobot-teleoperate \
    --robot.type=flexiv_rizon4 \
    --robot.robot_sn=Rizon4-062855 \
    --teleop.type=xense_flare \
    --fps=60
```

Example (Flexiv Rizon4 RT + Pico4):

```shell
lerobot-teleoperate \
    --robot.type=flexiv_rizon4_rt \
    --robot.robot_sn=Rizon4-063423 \
    --teleop.type=pico4 \
    --fps=60 \
    --debug_timing=true
```

Example (Flexiv Rizon4 RT + SpaceMouse):

```shell
lerobot-teleoperate \
    --robot.type=flexiv_rizon4_rt \
    --robot.robot_sn=Rizon4-063423 \
    --teleop.type=spacemouse \
    --fps=30 \
    --display_data=true
```

Example (Bimanual Flexiv Rizon4 RT + Bi-Pico4):

```shell
lerobot-teleoperate \
    --robot.type=bi_flexiv_rizon4_rt \
    --robot.left_robot_sn=Rizon4-063423 \
    --robot.right_robot_sn=Rizon4-063424 \
    --teleop.type=bi_pico4 \
    --fps=60
```

Example (Xense Flare gripper, data collection — no teleoperator needed):

```shell
lerobot-teleoperate \
    --robot.type=xense_flare \
    --teleop.type=btgamepad \
    --fps=30 \
    --display_data=true
```

Example (Xense Multisensor, data collection — no teleoperator needed):

```shell
lerobot-teleoperate \
    --robot.type=xense_multisensor \
    --teleop.type=btgamepad \
    --fps=30 \
    --display_data=true
```

Example (Flexiv Rizon4 RT + Pico4):

```shell
lerobot-teleoperate \
    --robot.type=pylibfranka_research3 \
    --teleop.type=pico4 \
    --fps=30 \
    --display_data=true
```

Example (Flexiv Rizon4 RT + SpaceMouse):

lerobot-teleoperate \
    --robot.type=pylibfranka_research3 \
    --teleop.type=spacemouse \
    --fps=30 \
    --display_data=true

Example (Franka research3 + BtGamepad):

```shell
lerobot-teleoperate \
    --robot.type=pylibfranka_research3 \
    --teleop.type=btgamepad \
    --fps=30 \
    --display_data=true
```

"""

import time
import traceback
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

import numpy as np
import rerun as rr

from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    arx5_follower,
    bi_arx5,
    bi_flexiv_rizon4_rt,
    flexiv_rizon4,
    flexiv_rizon4_rt,
    make_robot_from_config,
    pylibfranka_research3,
    xense_flare as xense_flare_robot,
    xense_multisensor,
    mock_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_pico4,
    btgamepad,
    gamepad,
    make_teleoperator_from_config,
    mock_teleop,
    pico4,
    spacemouse,
    vive_tracker,
    xense_flare,
    trlc_leader,
)
from lerobot.utils.robot_utils import (
    get_logger,
    precise_sleep,
    rotation_6d_to_quaternion,
    busy_wait,
)
from lerobot.utils.utils import move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


logger = get_logger("Teleoperate")


def make_default_processors(*args, **kwargs):
    """Lazy wrapper — defers lerobot.processor (torch) import until first use."""
    from lerobot.processor import make_default_processors as _fn
    return _fn(*args, **kwargs)


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
    # Print per-step timing breakdown instead of action values.
    debug_timing: bool = False
    # Dryrun mode: print actions but do not send to robot
    dryrun: bool = False


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _safe_disconnect(obj, name: str) -> None:
    if obj is None:
        return
    try:
        if obj.is_connected:
            obj.disconnect()
            logger.info(f"{name} disconnected")
    except Exception as e:
        logger.error(f"Error disconnecting {name}: {e}\n{traceback.format_exc()}")


def _cleanup(robot, teleop, display_data: bool) -> None:
    if display_data:
        try:
            rr.rerun_shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down rerun: {e}")
    _safe_disconnect(teleop, teleop.__class__.__name__ if teleop else "teleop")
    _safe_disconnect(robot, robot.__class__.__name__ if robot else "robot")


def _check_cartesian_mode(robot, teleop_name: str) -> None:
    from lerobot.robots.flexiv_rizon4.config_flexiv_rizon4 import ControlMode

    if robot.config.control_mode != ControlMode.CARTESIAN_MOTION_FORCE:
        raise ValueError(
            f"{teleop_name} requires CARTESIAN_MOTION_FORCE mode, "
            f"got {robot.config.control_mode}"
        )


def _print_obs_state(obs: dict, display_len: int, status: str) -> None:
    """Print scalar observation values with a status tag (used during reset/moving)."""
    scalar_keys = [k for k, v in obs.items() if not isinstance(v, np.ndarray)]
    col = max((len(k) for k in scalar_keys), default=display_len)
    print("\n" + "-" * (col + 18))
    print(f"{'NAME':<{col}} | {'OBS':>10}  {status}")
    for k in scalar_keys:
        print(f"{k:<{col}} | {float(obs[k]):>10.4f}")
    move_cursor_up(len(scalar_keys) + 5)


# ---------------------------------------------------------------------------
# Generic teleop loop (upstream)
# ---------------------------------------------------------------------------


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: Any,
    robot_action_processor: Any,
    robot_observation_processor: Any,
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
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
        debug_timing: If True, print per-step timing breakdown instead of action table.
    """
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        obs_t0 = time.perf_counter()
        obs = robot.get_observation()
        obs_time_ms = (time.perf_counter() - obs_t0) * 1e3

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


# ---------------------------------------------------------------------------
# Specialised teleop loops
# ---------------------------------------------------------------------------
def mock_robot_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: Any,
    robot_action_processor: Any,
    robot_observation_processor: Any,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
    debug_timing: bool = False,
):
    """
    Dedicated teleoperation loop for Mock Robot.

    This loop keeps the same processor pipeline as the default path but adds:
    - Action key filtering to robot.action_features
    - Dedicated timing/terminal display for mock robot teleoperation
    """
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    robot_action_keys = set(robot.action_features.keys())
    warned_unmapped_keys = False

    while True:
        loop_start = time.perf_counter()

        obs_start = time.perf_counter()
        obs = robot.get_observation()
        obs_dt_ms = (time.perf_counter() - obs_start) * 1e3

        raw_action = teleop.get_action()
        teleop_action = teleop_action_processor((raw_action, obs))

        # Keep only keys known by mock robot action schema.
        filtered_action = {
            k: v for k, v in teleop_action.items() if k in robot_action_keys
        }
        if not filtered_action and teleop_action:
            filtered_action = teleop_action
        elif len(filtered_action) != len(teleop_action) and not warned_unmapped_keys:
            dropped = sorted(set(teleop_action) - robot_action_keys)
            logger.warn(
                f"Action keys not present in mock robot action schema, dropping: {dropped}"
            )
            warned_unmapped_keys = True

        robot_action_to_send = robot_action_processor((filtered_action, obs))

        if not dryrun:
            _ = robot.send_action(robot_action_to_send)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start

        if display_data:
            obs_transition = robot_observation_processor(obs)
            log_rerun_data(observation=obs_transition, action=teleop_action)

            ordered_keys = [
                k for k in robot.action_features if k in robot_action_to_send
            ]
            ordered_keys.extend(
                k for k in robot_action_to_send if k not in ordered_keys
            )

            panel_lines = []
            panel_lines.append("-" * (display_len + 38))
            panel_lines.append(
                f"{'NAME':<{display_len}} | {'CMD':>8} | {'OBS':>8} | {'ERR':>8}"
            )
            for motor in ordered_keys:
                cmd = float(robot_action_to_send[motor])
                obs_val = obs.get(motor, None)
                if obs_val is None or isinstance(obs_val, np.ndarray):
                    panel_lines.append(
                        f"{motor:<{display_len}} | {cmd:>8.4f} | {'-':>8} | {'-':>8}"
                    )
                    continue

                obs_num = float(obs_val)
                err = cmd - obs_num
                panel_lines.append(
                    f"{motor:<{display_len}} | {cmd:>8.4f} | {obs_num:>8.4f} | {err:>+8.4f}"
                )

            panel_lines.append(
                f"{'timing':<{display_len}} | {'loop':>8} | {loop_s * 1e3:>6.2f}ms | {obs_dt_ms:>6.2f}ms"
            )

            print("\n".join(panel_lines), flush=True)
            move_cursor_up(len(panel_lines))

        if debug_timing and not display_data:
            dryrun_tag = " | DRYRUN" if dryrun else ""
            print(
                f"\r\033[KMOCK obs: {obs_dt_ms:5.1f}ms | loop: {loop_s * 1e3:5.1f}ms ({1 / loop_s:4.0f}Hz){dryrun_tag}",
                end="",
                flush=True,
            )
        elif not display_data:
            action_summary = " ".join(
                f"{k}={float(v):+.3f}" for k, v in robot_action_to_send.items()
            )
            dryrun_tag = "[DRYRUN] " if dryrun else ""
            print(
                f"\r\033[K{dryrun_tag}{loop_s * 1e3:5.1f}ms ({1 / loop_s:4.0f}Hz) | {action_summary}",
                end="",
                flush=True,
            )

        if duration is not None and time.perf_counter() - start >= duration:
            return


def arx5_teleop_loop(
    robot: Robot,
    fps: int,
    teleop_action_processor: Any,
    robot_action_processor: Any,
    robot_observation_processor: Any,
    display_data: bool = False,
    duration: float | None = None,
    debug_timing: bool = False,
):
    """
    Teleop loop for ARX5 robots (both single-arm and bimanual).

    Supports:
    - Single arm mode (arx5_follower): robot.arm
    - Bimanual mode (bi_arx5): robot.left_arm, robot.right_arm
    """
    start = time.perf_counter()
    timing_stats = {
        "robot_obs_times": [],
        "camera_obs_times": {},
        "total_obs_times": [],
        "loop_times": [],
    }

    is_bimanual = hasattr(robot, "left_arm") and hasattr(robot, "right_arm")
    is_single_arm = hasattr(robot, "arm") and not is_bimanual

    if not is_bimanual and not is_single_arm:
        raise ValueError(
            "Robot must have either 'arm' (single) or 'left_arm'/'right_arm' (bimanual)"
        )

    camera_keys = [
        key for key in robot.observation_features.keys() if not key.endswith(".pos")
    ]
    for cam_key in camera_keys:
        timing_stats["camera_obs_times"][cam_key] = []

    while True:
        loop_start = time.perf_counter()

        obs_start = time.perf_counter()
        robot_state_start = time.perf_counter()

        if is_bimanual:
            left_joint_state = robot.left_arm.get_joint_state()
            right_joint_state = robot.right_arm.get_joint_state()
        else:
            joint_state = robot.arm.get_joint_state()

        robot_obs_time = time.perf_counter() - robot_state_start
        timing_stats["robot_obs_times"].append(robot_obs_time * 1000)

        camera_obs_start = time.perf_counter()
        camera_observations = {}
        camera_times = {}
        for cam_key, cam in robot.cameras.items():
            cam_start = time.perf_counter()
            camera_observations[cam_key] = cam.async_read()
            cam_time_ms = (time.perf_counter() - cam_start) * 1000
            camera_times[cam_key] = cam_time_ms
            timing_stats["camera_obs_times"][cam_key].append(cam_time_ms)

        total_camera_time_ms = (time.perf_counter() - camera_obs_start) * 1000

        raw_observation = {}

        if is_bimanual:
            left_pos = left_joint_state.pos().copy()
            for i in range(6):
                raw_observation[f"left_joint_{i + 1}.pos"] = float(left_pos[i])
            raw_observation["left_gripper.pos"] = float(left_joint_state.gripper_pos)

            right_pos = right_joint_state.pos().copy()
            for i in range(6):
                raw_observation[f"right_joint_{i + 1}.pos"] = float(right_pos[i])
            raw_observation["right_gripper.pos"] = float(right_joint_state.gripper_pos)
        else:
            pos = joint_state.pos().copy()
            for i in range(6):
                raw_observation[f"joint_{i + 1}.pos"] = float(pos[i])
            raw_observation["gripper.pos"] = float(joint_state.gripper_pos)

        raw_observation.update(camera_observations)

        total_obs_time = time.perf_counter() - obs_start
        timing_stats["total_obs_times"].append(total_obs_time * 1000)

        raw_action = {
            key: value
            for key, value in raw_observation.items()
            if (
                key.endswith(".pos")
                and not key.startswith("head")
                and not key.startswith("left_wrist")
                and not key.startswith("right_wrist")
            )
        }

        if display_data:
            obs_transition = robot_observation_processor(raw_observation)
            log_rerun_data(observation=obs_transition, action=raw_action)

            if not debug_timing:
                if is_bimanual:
                    left_motors = {
                        k: v for k, v in raw_action.items() if k.startswith("left_")
                    }
                    right_motors = {
                        k: v for k, v in raw_action.items() if k.startswith("right_")
                    }
                    col_width = 25
                    print("\n" + "-" * (col_width * 2 + 3))
                    print(f"{'LEFT ARM':<{col_width}} | {'RIGHT ARM':<{col_width}}")
                    print("-" * (col_width * 2 + 3))
                    max_motors = max(len(left_motors), len(right_motors))
                    left_items = list(left_motors.items())
                    right_items = list(right_motors.items())
                    for i in range(max_motors):
                        left_str = ""
                        right_str = ""
                        if i < len(left_items):
                            motor_name = left_items[i][0].replace("left_", "")
                            left_str = f"{motor_name}: {left_items[i][1]:>7.3f}"
                        if i < len(right_items):
                            motor_name = right_items[i][0].replace("right_", "")
                            right_str = f"{motor_name}: {right_items[i][1]:>7.3f}"
                        print(f"{left_str:<{col_width}} | {right_str:<{col_width}}")
                    move_cursor_up(max_motors + 4)
                else:
                    col_width = 20
                    print("\n" + "-" * (col_width + 12))
                    print(f"{'JOINT':<{col_width}} | {'VALUE':>7}")
                    print("-" * (col_width + 12))
                    motor_items = list(raw_action.items())
                    for motor, value in motor_items:
                        print(f"{motor:<{col_width}} | {value:>7.3f}")
                    move_cursor_up(len(motor_items) + 4)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0))
        loop_s = time.perf_counter() - loop_start
        timing_stats["loop_times"].append(loop_s * 1000)

        if debug_timing:
            print()
            print("TELEOP TIMING DEBUG")
            print("=" * 50)
            print(f"Robot state:     {robot_obs_time * 1000:.1f}ms")
            print(f"Total cameras:   {total_camera_time_ms:.1f}ms")
            print()
            num_cameras = len(camera_times)
            for cam_key, cam_time_ms in camera_times.items():
                speed = (
                    "SLOW"
                    if cam_time_ms > 10
                    else ("MED " if cam_time_ms > 5 else "FAST")
                )
                print(f"  {speed} {cam_key:12}: {cam_time_ms:5.1f}ms")
            print()
            print(f"Total observation: {total_obs_time * 1000:.1f}ms")
            print(f"Loop time:        {loop_s * 1000:.1f}ms")
            print(f"Target period:    {1000 / fps:.1f}ms")
            print(f"Loop efficiency:  {(1000 / fps) / (loop_s * 1000) * 100:.1f}%")
            extra_warning_lines = 0
            if total_camera_time_ms > 20:
                print()
                print(f"SLOW CAMERAS DETECTED! Total: {total_camera_time_ms:.1f}ms")
                extra_warning_lines = 2
            print("=" * 50)
            total_lines = (
                1 + 1 + 1 + 2 + 1 + num_cameras + 1 + 4 + extra_warning_lines + 1
            )
            move_cursor_up(total_lines)
        else:
            if total_camera_time_ms > 20:
                print(f"SLOW CAMERAS: {total_camera_time_ms:.1f}ms")
                for cam_key, cam_time_ms in camera_times.items():
                    if cam_time_ms > 10:
                        print(f"  SLOW {cam_key}: {cam_time_ms:.1f}ms")

        if duration is not None and time.perf_counter() - start >= duration:
            if len(timing_stats["robot_obs_times"]) > 10:
                print("\n=== FINAL TIMING REPORT ===")
                all_robot = timing_stats["robot_obs_times"]
                all_total = timing_stats["total_obs_times"]
                all_loops = timing_stats["loop_times"]
                print(f"Total samples: {len(all_robot)}")
                print(f"Robot obs - avg: {sum(all_robot) / len(all_robot):.2f}ms")
                print(f"Total obs - avg: {sum(all_total) / len(all_total):.2f}ms")
                print(f"Loop time - avg: {sum(all_loops) / len(all_loops):.2f}ms")
                for cam_key, cam_times in timing_stats["camera_obs_times"].items():
                    if cam_times:
                        print(
                            f"{cam_key} - avg: {sum(cam_times) / len(cam_times):.2f}ms"
                        )
            return


def arx5_trlc_leader_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: Any,
    robot_action_processor: Any,
    robot_observation_processor: Any,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
    debug_timing: bool = False,
):
    """
    Dedicated teleoperation loop for ARX5 + TRLC leader.

    TRLC leader outputs joint-space actions (`joint_i.pos` + `gripper.pos`), so this loop
    validates key compatibility with the robot's action schema and then sends commands.
    """
    # ARX5 trlc leader loop currently supports single-arm follower only.
    if hasattr(robot, "left_arm") and hasattr(robot, "right_arm"):
        raise ValueError(
            "TRLC leader teleoperation currently supports arx5_follower only, not bi_arx5."
        )

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    robot_action_keys = set(robot.action_features.keys())
    warned_unmapped_keys = False

    while True:
        loop_start = time.perf_counter()

        obs_start = time.perf_counter()
        obs = robot.get_observation()
        obs_dt_ms = (time.perf_counter() - obs_start) * 1e3

        raw_action = teleop.get_action()
        for k in raw_action.keys():
            if "gripper" in k:
                raw_action[k] = (1 - raw_action[k]) * 1.57
        teleop_action = teleop_action_processor((raw_action, obs))

        filtered_action = {
            k: v for k, v in teleop_action.items() if k in robot_action_keys
        }
        if len(filtered_action) != len(teleop_action) and not warned_unmapped_keys:
            dropped = sorted(set(teleop_action) - robot_action_keys)
            logger.warning(
                f"TRLC action keys not present in ARX5 action schema, dropping: {dropped}"
            )
            warned_unmapped_keys = True

        if not filtered_action:
            raise ValueError(
                "No overlapping action keys between TRLC leader output and ARX5 action schema. "
                "Check robot.control_mode (TRLC requires joint-style keys like joint_i.pos, gripper.pos)."
            )

        robot_action_to_send = robot_action_processor((filtered_action, obs))

        if not dryrun:
            _ = robot.send_action(robot_action_to_send)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start

        if display_data:
            obs_transition = robot_observation_processor(obs)
            log_rerun_data(observation=obs_transition, action=teleop_action)

            ordered_keys = [
                k for k in robot.action_features if k in robot_action_to_send
            ]
            ordered_keys.extend(
                k for k in robot_action_to_send if k not in ordered_keys
            )

            panel_lines = []
            panel_lines.append("-" * (display_len + 38))
            panel_lines.append(
                f"{'NAME':<{display_len}} | {'CMD':>8} | {'OBS':>8} | {'ERR':>8}"
            )
            for motor in ordered_keys:
                cmd = float(robot_action_to_send[motor])
                obs_val = obs.get(motor, None)
                if obs_val is None or isinstance(obs_val, np.ndarray):
                    panel_lines.append(
                        f"{motor:<{display_len}} | {cmd:>8.4f} | {'-':>8} | {'-':>8}"
                    )
                    continue

                obs_num = float(obs_val)
                err = cmd - obs_num
                panel_lines.append(
                    f"{motor:<{display_len}} | {cmd:>8.4f} | {obs_num:>8.4f} | {err:>+8.4f}"
                )

            panel_lines.append(
                f"{'timing':<{display_len}} | {'loop':>8} | {loop_s * 1e3:>6.2f}ms | {obs_dt_ms:>6.2f}ms"
            )
            print("\n".join(panel_lines), flush=True)
            move_cursor_up(len(panel_lines))
        elif debug_timing:
            dryrun_tag = " | DRYRUN" if dryrun else ""
            print(
                f"\r\033[KARX5+TRLC obs: {obs_dt_ms:5.1f}ms | loop: {loop_s * 1e3:5.1f}ms ({1 / loop_s:4.0f}Hz){dryrun_tag}",
                end="",
                flush=True,
            )
        else:
            action_summary = " ".join(
                f"{k}={float(v):+.3f}" for k, v in robot_action_to_send.items()
            )
            dryrun_tag = "[DRYRUN] " if dryrun else ""
            print(
                f"\r\033[K{dryrun_tag}{loop_s * 1e3:5.1f}ms ({1 / loop_s:4.0f}Hz) | {action_summary}",
                end="",
                flush=True,
            )

        if duration is not None and time.perf_counter() - start >= duration:
            return


def spacemouse_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: Any,
    robot_action_processor: Any,
    robot_observation_processor: Any,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
    debug_timing: bool = False,
):
    """Teleop loop for SpaceMouse teleoperator."""
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    timing_stats = {"obs_times": [], "loop_times": []}

    from lerobot.robots.flexiv_rizon4.config_flexiv_rizon4 import ControlMode

    is_flexiv_nrt = (
        robot.name == "flexiv_rizon4"
        and hasattr(robot.config, "control_mode")
        and robot.config.control_mode == ControlMode.CARTESIAN_MOTION_FORCE
    )
    is_flexiv_rt = robot.name == "flexiv_rizon4_rt"
    # pylibfranka_research3 in Cartesian impedance mode also needs special handling for reset
    is_pylibfranka_cartesian = (
        robot.name == "pylibfranka_research3"
        and hasattr(robot.config, "control_mode")
        and robot.config.control_mode.value == "cartesian_impedance"
    )
    is_flexiv = (is_flexiv_nrt or is_flexiv_rt) and teleop.name == "spacemouse"

    _prev_rt_moving = False
    _reset_display_cleared = False
    _spacemouse_both_buttons_prev = False

    while True:
        loop_start = time.perf_counter()

        obs_start = time.perf_counter()
        obs = robot.get_observation()
        obs_time = time.perf_counter() - obs_start
        timing_stats["obs_times"].append(obs_time * 1000)

        raw_action = teleop.get_action()

        if teleop.name == "spacemouse":
            button_left = teleop._spacemouse.is_left_button_pressed()
            button_right = teleop._spacemouse.is_right_button_pressed()
            both_buttons = button_left and button_right
            both_buttons_rising = both_buttons and not _spacemouse_both_buttons_prev
            _spacemouse_both_buttons_prev = both_buttons

            if both_buttons:
                is_arx5_family = getattr(robot, "name", None) in (
                    "arx5_follower",
                    "bi_arx5",
                )
                if dryrun:
                    logger.info(
                        "[DRYRUN] Reset to initial position triggered by both buttons"
                    )
                    if hasattr(teleop, "_start_pose_6d") and hasattr(
                        teleop, "_start_gripper_pos"
                    ):
                        teleop.reset_to_pose(
                            teleop._start_pose_6d, teleop._start_gripper_pos
                        )
                elif (
                    is_arx5_family
                    and hasattr(robot, "smooth_go_start")
                    and both_buttons_rising
                ):
                    try:
                        # Match connect(go_to_start=True): interpolated move, not a teleop target jump.
                        robot.smooth_go_start(duration=2.0)
                        eef = robot.get_start_eef_pose()
                        teleop.reset_to_pose(eef[:6], float(eef[6]))
                        teleop._start_pose_6d = eef[:6].copy()
                        teleop._start_gripper_pos = float(eef[6])
                        logger.info(
                            "SpaceMouse both buttons: smooth go-to-start (ARX5 / bi_arx5)"
                        )
                    except Exception as e:
                        logger.error(
                            f"Smooth go-to-start failed: {e}\n{traceback.format_exc()}"
                        )
                elif is_flexiv_rt and hasattr(robot, "reset_to_initial_position"):
                    try:
                        robot.reset_to_initial_position()
                    except Exception as e:
                        logger.error(
                            f"Failed to reset robot position: {e}\n{traceback.format_exc()}"
                        )
                elif is_flexiv_nrt and hasattr(robot, "reset_to_initial_position"):
                    try:
                        robot.reset_to_initial_position()
                        current_pose_euler = robot.get_current_tcp_pose_euler()
                        teleop.reset_to_pose(
                            current_pose_euler[:6], current_pose_euler[6]
                        )
                        teleop._start_pose_6d = current_pose_euler[:6].copy()
                        teleop._start_gripper_pos = current_pose_euler[6]
                        logger.info(
                            "Reset to initial position triggered by both buttons"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to reset robot position: {e}\n{traceback.format_exc()}"
                        )
                elif is_pylibfranka_cartesian and hasattr(
                    robot, "reset_to_initial_position"
                ):
                    try:
                        # pylibfranka_research3: blocking reset via WebSocket move_home.
                        robot.reset_to_initial_position()
                        current_pose_euler = robot.get_current_tcp_pose_euler()
                        teleop.reset_to_pose(
                            current_pose_euler[:6], current_pose_euler[6]
                        )
                        teleop._start_pose_6d = current_pose_euler[:6].copy()
                        teleop._start_gripper_pos = current_pose_euler[6]
                        logger.info(
                            "Reset to initial position triggered by both buttons (pylibfranka)"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to reset robot position: {e}\n{traceback.format_exc()}"
                        )
                elif not (
                    is_arx5_family and hasattr(robot, "smooth_go_start")
                ):
                    if hasattr(teleop, "_start_pose_6d") and hasattr(
                        teleop, "_start_gripper_pos"
                    ):
                        teleop.reset_to_pose(
                            teleop._start_pose_6d, teleop._start_gripper_pos
                        )
                        logger.info(
                            "Reset to initial position triggered by both buttons"
                        )
                if display_data and obs is not None:
                    log_rerun_data(observation=obs)
                if obs is not None:
                    if not _reset_display_cleared:
                        print("\033[2J\033[H", end="", flush=True)
                        _reset_display_cleared = True
                    _print_obs_state(obs, display_len, "RESETTING")
                continue

        if is_flexiv_rt and hasattr(robot, "rt_moving") and robot.rt_moving:
            if display_data and obs is not None:
                log_rerun_data(observation=obs)
            if obs is not None:
                if not _reset_display_cleared:
                    print("\033[2J\033[H", end="", flush=True)
                    _reset_display_cleared = True
                _print_obs_state(obs, display_len, "MOVING")
            _prev_rt_moving = True
            continue

        if _prev_rt_moving:
            _prev_rt_moving = False
            _reset_display_cleared = False
            try:
                current_pose_euler = robot.get_current_tcp_pose_euler()
                teleop.reset_to_pose(current_pose_euler[:6], current_pose_euler[6])
                teleop._start_pose_6d = current_pose_euler[:6].copy()
                teleop._start_gripper_pos = current_pose_euler[6]
                logger.info("Teleop synced to robot pose after reset complete")
            except Exception as e:
                logger.error(f"Failed to sync teleop after reset: {e}")
            continue

        teleop_action = teleop_action_processor((raw_action, obs))

        if is_flexiv:
            robot_action_to_send = teleop.convert_to_flexiv_action(teleop_action)
        else:
            robot_action_to_send = robot_action_processor((teleop_action, obs))

        if not dryrun:
            _ = robot.send_action(robot_action_to_send)

        if display_data:
            log_rerun_data(observation=obs, action=teleop_action)
            if not debug_timing:
                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                for motor, value in robot_action_to_send.items():
                    print(f"{motor:<{display_len}} | {value:>7.3f}")
                move_cursor_up(len(robot_action_to_send) + 5)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0))
        loop_s = time.perf_counter() - loop_start
        timing_stats["loop_times"].append(loop_s * 1000)

        if debug_timing:
            print(
                f"\r\033[Kobs: {obs_time * 1000:5.1f}ms | loop: {loop_s * 1000:5.1f}ms | "
                f"target: {1000 / fps:.1f}ms | eff: {(1 / fps) / loop_s * 100:5.1f}%",
                end="",
                flush=True,
            )
        elif not display_data:
            pos_x = robot_action_to_send.get("tcp.x", teleop_action.get("x", 0))
            pos_y = robot_action_to_send.get("tcp.y", teleop_action.get("y", 0))
            pos_z = robot_action_to_send.get("tcp.z", teleop_action.get("z", 0))
            gripper = robot_action_to_send.get(
                "gripper.pos", teleop_action.get("gripper_pos", 0)
            )
            roll = teleop_action.get("roll", 0)
            pitch = teleop_action.get("pitch", 0)
            yaw = teleop_action.get("yaw", 0)
            pos_str = f"pos=[{pos_x:+.3f}, {pos_y:+.3f}, {pos_z:+.3f}]"
            ori_str = f"rpy=[{roll:+.3f}, {pitch:+.3f}, {yaw:+.3f}]"
            grip_str = f"grip={gripper:.2f}"
            flag_str = "[DRYRUN] " if dryrun else ""
            print(
                f"\r\033[K{loop_s * 1e3:5.1f}ms ({1 / loop_s:3.0f}Hz) | {flag_str}{pos_str} | {ori_str} | {grip_str}",
                end="",
                flush=True,
            )

        if duration is not None and time.perf_counter() - start >= duration:
            return


def btgamepad_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: Any,
    robot_action_processor: Any,
    robot_observation_processor: Any,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
):
    """
    Teleop loop for BT Gamepad with Pylibfranka Research3 robot.

    Control scheme:
    - Left stick: XY position control
    - Right stick Y: Z position control
    - Right stick X: Rotation around Z axis
    - LB/RB: Open/close gripper
    - BACK button: Reset to initial position
    """
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        obs = robot.get_observation()
        raw_action = teleop.get_action()

        reset_button = teleop.get_reset_button()
        if reset_button:
            try:
                if dryrun:
                    logger.info(
                        "[DRYRUN] Reset to initial position (BACK pressed) - robot movement skipped"
                    )
                    current_pose_quat = robot.get_current_tcp_pose_quat()
                    teleop.reset_to_pose(current_pose_quat[:7], current_pose_quat[7])
                else:
                    if hasattr(robot, "reset_to_initial_position"):
                        robot.reset_to_initial_position()
                    logger.info("Reset to initial position (BACK pressed)")
                    current_pose_quat = robot.get_current_tcp_pose_quat()
                    teleop.reset_to_pose(current_pose_quat[:7], current_pose_quat[7])
            except Exception as e:
                logger.error(
                    f"Failed to reset robot position: {e}\n{traceback.format_exc()}"
                )
            if display_data:
                log_rerun_data(observation=obs)
            _print_obs_state(obs, display_len, "RESETTING")
            continue

        teleop_action = teleop_action_processor((raw_action, obs))
        robot_action_to_send = teleop_action

        if not dryrun:
            _ = robot.send_action(robot_action_to_send)

        if display_data:
            log_rerun_data(observation=obs, action=teleop_action)
            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.4f}")
            move_cursor_up(len(robot_action_to_send) + 5)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0))
        loop_s = time.perf_counter() - loop_start

        if not display_data:
            gripper_pos_str = (
                f"gripper={robot_action_to_send.get('gripper.pos', 0.0):.2f}"
            )
            dryrun_str = "[DRYRUN] | " if dryrun else ""
            print(
                f"\r\033[Ktime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | {dryrun_str}{gripper_pos_str}",
                end="",
                flush=True,
            )

        if duration is not None and time.perf_counter() - start >= duration:
            return


def pico4_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: Any,
    robot_action_processor: Any,
    robot_observation_processor: Any,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
):
    """
    Teleop loop for Pico4 VR controller with Flexiv Rizon4 robot.

    Control scheme:
    - Grip: Enable control (must be held to move robot)
    - Trigger: Controls gripper position (0=closed, 1=open)
    - A button: Reset to initial position
    """
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    is_flexiv_rt = robot.name == "flexiv_rizon4_rt"
    _prev_rt_moving = False
    _reset_display_cleared = False

    while True:
        loop_start = time.perf_counter()

        obs = robot.get_observation()
        raw_action = teleop.get_action()

        reset_button = teleop.get_reset_button()
        if reset_button:
            try:
                if dryrun:
                    logger.info(
                        "[DRYRUN] Reset to initial position (A button pressed) - robot movement skipped"
                    )
                    current_pose_quat = robot.get_current_tcp_pose_quat()
                    teleop.reset_to_pose(current_pose_quat[:7], current_pose_quat[7])
                elif is_flexiv_rt and hasattr(robot, "reset_to_initial_position"):
                    robot.reset_to_initial_position()
                    logger.info(
                        "Reset to initial position (A button pressed) — RT non-blocking"
                    )
                else:
                    if hasattr(robot, "reset_to_initial_position"):
                        robot.reset_to_initial_position()
                    logger.info("Reset to initial position (A button pressed)")
                    current_pose_quat = robot.get_current_tcp_pose_quat()
                    teleop.reset_to_pose(current_pose_quat[:7], current_pose_quat[7])
            except Exception as e:
                logger.error(
                    f"Failed to reset robot position: {e}\n{traceback.format_exc()}"
                )
            if display_data:
                log_rerun_data(observation=obs)
            if not _reset_display_cleared:
                print("\033[2J\033[H", end="", flush=True)
                _reset_display_cleared = True
            _print_obs_state(obs, display_len, "RESETTING")
            continue

        if is_flexiv_rt and hasattr(robot, "rt_moving") and robot.rt_moving:
            if display_data:
                log_rerun_data(observation=obs)
            if not _reset_display_cleared:
                print("\033[2J\033[H", end="", flush=True)
                _reset_display_cleared = True
            _print_obs_state(obs, display_len, "MOVING")
            _prev_rt_moving = True
            continue

        if _prev_rt_moving:
            _prev_rt_moving = False
            _reset_display_cleared = False
            try:
                current_pose_quat = robot.get_current_tcp_pose_quat()
                teleop.reset_to_pose(current_pose_quat[:7], current_pose_quat[7])
                logger.info("Teleop synced to robot pose after reset complete")
            except Exception as e:
                logger.error(f"Failed to sync teleop after reset: {e}")
            continue

        teleop_action = teleop_action_processor((raw_action, obs))
        robot_action_to_send = teleop_action

        if not dryrun:
            _ = robot.send_action(robot_action_to_send)

        if display_data:
            log_rerun_data(observation=obs, action=teleop_action)
            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.4f}")
            move_cursor_up(len(robot_action_to_send) + 5)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0))
        loop_s = time.perf_counter() - loop_start

        if not display_data:
            enable_str = "ENABLED" if teleop._enabled else "DISABLED"
            ori_str = "ORI:ON" if teleop._orientation_control_active else "ORI:OFF"
            grip_str = f"grip={teleop._last_grip:.2f}"
            gripper_pos_str = (
                f"gripper={robot_action_to_send.get('gripper.pos', 0.0):.2f}"
            )
            dryrun_str = "[DRYRUN] | " if dryrun else ""
            print(
                f"\r\033[Ktime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | {dryrun_str}{enable_str} | {grip_str} | {gripper_pos_str} | {ori_str}",
                end="",
                flush=True,
            )

        if duration is not None and time.perf_counter() - start >= duration:
            return


def bi_pico4_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
    debug_timing: bool = False,
):
    """
    Teleop loop for BiPico4 VR controllers with BiFlexivRizon4RT bimanual robot.

    Control scheme:
    - Left grip:  Enable left arm control
    - Right grip: Enable right arm control
    - Left/Right trigger: Control respective gripper position
    - A button (right controller): Reset both arms to initial position (non-blocking)
    """
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    _prev_rt_moving = False
    _reset_display_cleared = False

    while True:
        loop_start = time.perf_counter()

        obs = robot.get_observation()
        t_obs = time.perf_counter()

        raw_action = teleop.get_action()
        t_action = time.perf_counter()

        reset_button = teleop.get_reset_button()
        if reset_button:
            try:
                if dryrun:
                    logger.info(
                        "[DRYRUN] Reset to initial position (A button) - robot movement skipped"
                    )
                    left_pose, right_pose = robot.get_current_tcp_pose_quat()
                    teleop.reset_to_pose(
                        left_pose[:7], right_pose[:7], left_pose[7], right_pose[7]
                    )
                elif hasattr(robot, "reset_to_initial_position"):
                    robot.reset_to_initial_position()
                    logger.info(
                        "Reset to initial position (A button) — RT non-blocking"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to reset robot position: {e}\n{traceback.format_exc()}"
                )

            if display_data:
                log_rerun_data(observation=obs)
            if not _reset_display_cleared:
                print("\033[2J\033[H", end="", flush=True)
                _reset_display_cleared = True
            _print_obs_state(obs, display_len, "RESETTING")
            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0))
            continue

        if hasattr(robot, "rt_moving") and robot.rt_moving:
            if display_data:
                log_rerun_data(observation=obs)
            if not _reset_display_cleared:
                print("\033[2J\033[H", end="", flush=True)
                _reset_display_cleared = True
            _print_obs_state(obs, display_len, "MOVING")
            _prev_rt_moving = True
            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0))
            continue

        if _prev_rt_moving:
            _prev_rt_moving = False
            _reset_display_cleared = False
            try:
                left_pose, right_pose = robot.get_current_tcp_pose_quat()
                teleop.reset_to_pose(
                    left_pose[:7], right_pose[:7], left_pose[7], right_pose[7]
                )
                logger.info("BiPico4 synced to robot poses after reset complete")
            except Exception as e:
                logger.error(f"Failed to sync teleop after reset: {e}")
            continue

        if not dryrun:
            robot.send_action(raw_action)
        t_send = time.perf_counter()

        if display_data:
            log_rerun_data(observation=obs, action=raw_action)
            t_rerun = time.perf_counter()
            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in raw_action.items():
                print(f"{motor:<{display_len}} | {value:>7.4f}")
            move_cursor_up(len(raw_action) + 5)
        else:
            t_rerun = t_send

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0))
        loop_s = time.perf_counter() - loop_start

        if debug_timing:
            obs_ms    = (t_obs    - loop_start) * 1e3
            action_ms = (t_action - t_obs)      * 1e3
            send_ms   = (t_send   - t_action)   * 1e3
            rerun_ms  = (t_rerun  - t_send)     * 1e3
            sleep_ms  = loop_s * 1e3 - dt_s * 1e3

            lines = [
                f"obs={obs_ms:5.1f}ms  action={action_ms:4.1f}ms  send={send_ms:4.1f}ms  "
                f"rerun={rerun_ms:4.1f}ms  sleep={sleep_ms:5.1f}ms  "
                f"| total={loop_s*1e3:5.1f}ms ({1/loop_s:.0f}Hz)",
            ]
            if hasattr(robot, "_last_obs_timing"):
                t = robot._last_obs_timing
                lines.append(
                    f"  l_arm={t['left_arm_ms']:.1f}  r_arm={t['right_arm_ms']:.1f}"
                    f"  l_grip={t['left_grip_ms']:.1f}  r_grip={t['right_grip_ms']:.1f}"
                    f"  cams={t['cameras_ms']:.1f}"
                )
                cam_parts = "  ".join(
                    f"{k[4:-3]}={v:.1f}" for k, v in t.items() if k.startswith("cam[")
                )
                if cam_parts:
                    lines.append(f"  [{cam_parts}]")

            for line in lines:
                print(f"\033[K{line}", flush=True)
            move_cursor_up(len(lines))
        elif not display_data:
            left_enabled = "ON " if teleop._left_pico4._enabled else "OFF"
            right_enabled = "ON " if teleop._right_pico4._enabled else "OFF"
            left_grip = f"{teleop._left_pico4._last_grip:.2f}"
            right_grip = f"{teleop._right_pico4._last_grip:.2f}"
            left_gripper = raw_action.get("left_gripper.pos", 0.0)
            right_gripper = raw_action.get("right_gripper.pos", 0.0)
            dryrun_str = "[DRYRUN] " if dryrun else ""
            print(
                f"\r\033[K{loop_s * 1e3:.1f}ms ({1 / loop_s:.0f}Hz) | {dryrun_str}"
                f"L:{left_enabled} grip={left_grip} gripper={left_gripper:.2f} | "
                f"R:{right_enabled} grip={right_grip} gripper={right_gripper:.2f}",
                end="",
                flush=True,
            )

        if duration is not None and time.perf_counter() - start >= duration:
            return


def vive_tracker_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: Any,
    robot_action_processor: Any,
    robot_observation_processor: Any,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
):
    """
    Teleop loop for Vive Tracker with Flexiv Rizon4 robot.

    Control scheme:
    - Vive Tracker provides absolute 6-DoF pose tracking
    - No enable/disable control (always active after connect)
    """
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        obs = robot.get_observation()

        try:
            raw_action = teleop.get_action()
        except Exception as e:
            logger.error(f"Error getting Vive Tracker action: {e}")
            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0))
            continue

        teleop_action = teleop_action_processor((raw_action, obs))
        robot_action_to_send = teleop_action

        if not dryrun:
            try:
                _ = robot.send_action(robot_action_to_send)
            except Exception as e:
                logger.error(f"Error sending action to robot: {e}")

        if display_data:
            obs_transition = robot_observation_processor(obs)
            log_rerun_data(observation=obs_transition, action=teleop_action)
            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.4f}")
            move_cursor_up(len(robot_action_to_send) + 5)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0))
        loop_s = time.perf_counter() - loop_start

        action_str = ", ".join(
            [f"{k}={v:.4f}" for k, v in robot_action_to_send.items()]
        )
        dryrun_str = "[DRYRUN] | " if dryrun else ""
        print(
            f"\rtime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | {dryrun_str}{action_str}",
            end="",
            flush=True,
        )

        if duration is not None and time.perf_counter() - start >= duration:
            return


def xense_flare_flexiv_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: Any,
    robot_action_processor: Any,
    robot_observation_processor: Any,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
    debug_timing: bool = False,
):
    """
    Teleop loop for Xense Flare teleoperator with Flexiv Rizon4 robot.

    Xense Flare outputs actions in Flexiv format:
    - tcp.x, tcp.y, tcp.z: absolute TCP position (meters)
    - tcp.r1-r6: absolute TCP orientation (6D rotation representation)
    - gripper.pos: gripper position from encoder (0=closed, 1=open)
    """
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        obs_start = time.perf_counter()
        obs = robot.get_observation()
        obs_time = time.perf_counter() - obs_start

        teleop_start = time.perf_counter()
        try:
            raw_action = teleop.get_action()
        except Exception as e:
            logger.error(f"Error getting Xense Flare action: {e}")
            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0))
            continue
        teleop_time = time.perf_counter() - teleop_start

        teleop_action = raw_action
        robot_action_to_send = teleop_action

        send_time = 0.0
        if not dryrun:
            send_start = time.perf_counter()
            try:
                _ = robot.send_action(teleop_action)
            except Exception as e:
                logger.error(f"Error sending action to robot: {e}")
            send_time = time.perf_counter() - send_start

        if display_data:
            obs_transition = robot_observation_processor(obs)
            log_rerun_data(observation=obs_transition, action=teleop_action)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0))
        loop_s = time.perf_counter() - loop_start

        gripper_str = f"grip={robot_action_to_send.get('gripper.pos', 0.0):.2f}"
        pos_str = (
            f"pos=[{robot_action_to_send.get('tcp.x', 0):.3f}, "
            f"{robot_action_to_send.get('tcp.y', 0):.3f}, "
            f"{robot_action_to_send.get('tcp.z', 0):.3f}]"
        )
        dryrun_str = "[DRY] " if dryrun else ""

        if debug_timing:
            timing_str = f"obs:{obs_time * 1e3:.1f} teleop:{teleop_time * 1e3:.1f} send:{send_time * 1e3:.1f}"
            print(
                f"\r\033[K{loop_s * 1e3:.1f}ms ({1 / loop_s:.0f}Hz) | {timing_str} | {dryrun_str}{pos_str} | {gripper_str}",
                end="",
                flush=True,
            )
        else:
            print(
                f"\r\033[Ktime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | {dryrun_str}{pos_str} | {gripper_str}",
                end="",
                flush=True,
            )

        if duration is not None and time.perf_counter() - start >= duration:
            return


def xense_flare_teleop_loop(
    robot: Robot,
    fps: int,
    robot_observation_processor: Any,
    display_data: bool = False,
    duration: float | None = None,
    debug_timing: bool = False,
):
    """
    Data collection loop for Xense Flare gripper.

    Xense Flare is a pure observation device (similar to teach mode).
    This loop continuously reads multi-modal sensor data:
    - Vive Tracker: 6DoF pose (tcp.x/y/z/qw/qx/qy/qz)
    - Wrist Camera: RGB image
    - Tactile Sensors: Tactile images
    - Gripper: Position

    No actions are sent to the robot - it is manually operated.
    """
    import warnings

    warnings.filterwarnings("ignore", message=".*RotationQuatBatch.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="rerun")

    start = time.perf_counter()
    timing_stats = {
        "vive_times": [],
        "gripper_times": [],
        "camera_times": [],
        "sensor_times": [],
        "total_obs_times": [],
        "loop_times": [],
    }

    trajectory_points: list[np.ndarray] = []
    max_trajectory_points = 500
    trajectory_line_radius = 0.005

    while True:
        loop_start = time.perf_counter()

        obs_start = time.perf_counter()
        try:
            obs = robot.get_observation()
        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0))
            continue

        total_obs_time = time.perf_counter() - obs_start
        timing_stats["total_obs_times"].append(total_obs_time * 1000)

        raw_action = {}
        raw_action = robot.get_action()

        if display_data:
            log_rerun_data(observation=obs, action=raw_action)

            if "tcp.x" in obs and "tcp.y" in obs and "tcp.z" in obs:
                pos = np.array([obs["tcp.x"], obs["tcp.y"], obs["tcp.z"]])

                if "tcp.r1" in obs:
                    r6d = np.array(
                        [
                            obs["tcp.r1"],
                            obs["tcp.r2"],
                            obs["tcp.r3"],
                            obs["tcp.r4"],
                            obs["tcp.r5"],
                            obs["tcp.r6"],
                        ]
                    )
                    quat_wxyz = rotation_6d_to_quaternion(r6d)
                    rot_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
                else:
                    rot_xyzw = [
                        obs.get("tcp.qx", 0.0),
                        obs.get("tcp.qy", 0.0),
                        obs.get("tcp.qz", 0.0),
                        obs.get("tcp.qw", 1.0),
                    ]

                rr.log(
                    "tracker/pose",
                    rr.Transform3D(
                        translation=pos.tolist(),
                        rotation=rr.Quaternion(xyzw=rot_xyzw),
                    ),
                )

                rr.log(
                    "tracker/point",
                    rr.Points3D([pos], radii=[0.015], colors=[[255, 50, 50]]),
                )

                qx, qy, qz, qw = rot_xyzw[0], rot_xyzw[1], rot_xyzw[2], rot_xyzw[3]
                R = np.array(
                    [
                        [
                            1 - 2 * (qy**2 + qz**2),
                            2 * (qx * qy - qz * qw),
                            2 * (qx * qz + qy * qw),
                        ],
                        [
                            2 * (qx * qy + qz * qw),
                            1 - 2 * (qx**2 + qz**2),
                            2 * (qy * qz - qx * qw),
                        ],
                        [
                            2 * (qx * qz - qy * qw),
                            2 * (qy * qz + qx * qw),
                            1 - 2 * (qx**2 + qy**2),
                        ],
                    ]
                )
                axis_length = 0.08
                x_axis = R @ np.array([axis_length, 0, 0])
                y_axis = R @ np.array([0, axis_length, 0])
                z_axis = R @ np.array([0, 0, axis_length])

                rr.log(
                    "tracker/axes",
                    rr.Arrows3D(
                        origins=np.array([pos, pos, pos]),
                        vectors=np.array([x_axis, y_axis, z_axis]),
                        colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
                        radii=0.003,
                    ),
                )

                trajectory_points.append(pos.copy())
                if len(trajectory_points) > max_trajectory_points:
                    trajectory_points.pop(0)
                if len(trajectory_points) >= 2:
                    points_array = np.array(trajectory_points)
                    rr.log(
                        "tracker/trajectory",
                        rr.LineStrips3D([points_array], radii=[trajectory_line_radius]),
                    )

            vive_tracker = robot.get_vive_tracker()
            if vive_tracker is not None:
                try:
                    all_poses = vive_tracker.get_pose()
                    if all_poses:
                        for device_name, pose_data in all_poses.items():
                            if device_name.startswith("LH") and pose_data is not None:
                                lh_pos = list(pose_data.position)
                                lh_rot = pose_data.rotation  # [qw, qx, qy, qz]
                                lh_rot_xyzw = [
                                    lh_rot[1],
                                    lh_rot[2],
                                    lh_rot[3],
                                    lh_rot[0],
                                ]

                                if device_name == "LH0":
                                    color = [0, 255, 100]
                                elif device_name == "LH1":
                                    color = [100, 180, 255]
                                else:
                                    color = [255, 200, 100]

                                base_path = f"lighthouse/{device_name}"

                                rr.log(
                                    f"{base_path}/pose",
                                    rr.Transform3D(
                                        translation=lh_pos,
                                        rotation=rr.Quaternion(xyzw=lh_rot_xyzw),
                                    ),
                                )

                                rr.log(
                                    f"{base_path}/point",
                                    rr.Points3D(
                                        [lh_pos],
                                        radii=[0.05],
                                        colors=[color],
                                        labels=[device_name],
                                    ),
                                )

                                qx, qy, qz, qw = (
                                    lh_rot_xyzw[0],
                                    lh_rot_xyzw[1],
                                    lh_rot_xyzw[2],
                                    lh_rot_xyzw[3],
                                )
                                R = np.array(
                                    [
                                        [
                                            1 - 2 * (qy**2 + qz**2),
                                            2 * (qx * qy - qz * qw),
                                            2 * (qx * qz + qy * qw),
                                        ],
                                        [
                                            2 * (qx * qy + qz * qw),
                                            1 - 2 * (qx**2 + qz**2),
                                            2 * (qy * qz - qx * qw),
                                        ],
                                        [
                                            2 * (qx * qz - qy * qw),
                                            2 * (qy * qz + qx * qw),
                                            1 - 2 * (qx**2 + qy**2),
                                        ],
                                    ]
                                )
                                lh_axis_length = 0.15
                                lh_x_axis = R @ np.array([lh_axis_length, 0, 0])
                                lh_y_axis = R @ np.array([0, lh_axis_length, 0])
                                lh_z_axis = R @ np.array([0, 0, lh_axis_length])

                                rr.log(
                                    f"{base_path}/axes",
                                    rr.Arrows3D(
                                        origins=np.array([lh_pos, lh_pos, lh_pos]),
                                        vectors=np.array(
                                            [lh_x_axis, lh_y_axis, lh_z_axis]
                                        ),
                                        colors=np.array(
                                            [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
                                        ),
                                        radii=0.005,
                                    ),
                                )
                except Exception:
                    pass  # Lighthouse visualization is optional

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0))
        loop_s = time.perf_counter() - loop_start
        timing_stats["loop_times"].append(loop_s * 1000)

        if debug_timing:
            print(
                f"\r\033[Kobs: {total_obs_time * 1000:5.1f}ms | loop: {loop_s * 1000:5.1f}ms | "
                f"target: {1000 / fps:.1f}ms | eff: {(1 / fps) / loop_s * 100:5.1f}%",
                end="",
                flush=True,
            )
        else:
            pose_str = ""
            if "tcp.x" in obs and "tcp.y" in obs and "tcp.z" in obs:
                pose_str = (
                    f"pos=[{obs['tcp.x']:.3f}, {obs['tcp.y']:.3f}, {obs['tcp.z']:.3f}]"
                )
            gripper_str = f"grip={obs.get('gripper.pos', 0.0):.2f}"
            print(
                f"\r\033[Ktime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | {pose_str} | {gripper_str}",
                end="",
                flush=True,
            )

        if duration is not None and time.perf_counter() - start >= duration:
            if len(timing_stats["total_obs_times"]) > 10:
                print("\n=== FINAL TIMING REPORT ===")
                all_total = timing_stats["total_obs_times"]
                all_loops = timing_stats["loop_times"]
                print(f"Total samples: {len(all_total)}")
                print(f"Total obs - avg: {sum(all_total) / len(all_total):.2f}ms")
                print(f"Loop time - avg: {sum(all_loops) / len(all_loops):.2f}ms")
            return


def xense_multisensor_teleop_loop(
    robot: Robot,
    fps: int,
    robot_observation_processor: Any,
    display_data: bool = False,
    duration: float | None = None,
    debug_timing: bool = False,
):
    """
    Data collection loop for Xense Multisensor robot.

    Xense Multisensor is a pure observation device (similar to teach mode).
    No actions are sent to the robot - it is a data collection device.
    """
    start = time.perf_counter()
    timing_stats = {
        "camera_times": {},
        "total_obs_times": [],
        "loop_times": [],
    }

    camera_keys = list(robot.observation_features.keys())
    for cam_key in camera_keys:
        timing_stats["camera_times"][cam_key] = []

    while True:
        loop_start = time.perf_counter()

        obs_start = time.perf_counter()
        try:
            obs = robot.get_observation()
        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt_s, 0))
            continue

        total_obs_time = time.perf_counter() - obs_start
        timing_stats["total_obs_times"].append(total_obs_time * 1000)

        if robot_observation_processor is not None:
            obs = robot_observation_processor(obs)

        if display_data:
            log_rerun_data(observation=obs, action={})

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0))
        loop_s = time.perf_counter() - loop_start
        timing_stats["loop_times"].append(loop_s * 1000)

        if debug_timing:
            print(
                f"\r\033[Kobs: {total_obs_time * 1000:5.1f}ms | loop: {loop_s * 1000:5.1f}ms | "
                f"target: {1000 / fps:.1f}ms | eff: {(1 / fps) / loop_s * 100:5.1f}%",
                end="",
                flush=True,
            )
        else:
            camera_count = len(
                [k for k in obs.keys() if isinstance(obs.get(k), np.ndarray)]
            )
            print(
                f"\r\033[Ktime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | cameras: {camera_count}",
                end="",
                flush=True,
            )

        if duration is not None and time.perf_counter() - start >= duration:
            if len(timing_stats["total_obs_times"]) > 10:
                print("\n=== FINAL TIMING REPORT ===")
                all_total = timing_stats["total_obs_times"]
                all_loops = timing_stats["loop_times"]
                print(f"Total samples: {len(all_total)}")
                print(f"Total obs - avg: {sum(all_total) / len(all_total):.2f}ms")
                print(f"Loop time - avg: {sum(all_loops) / len(all_loops):.2f}ms")
            return


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    logger.info(pformat(asdict(cfg)))
    if cfg.dryrun:
        logger.warn(
            "DRYRUN MODE ENABLED - Actions will be printed but NOT sent to robot"
        )

    if cfg.display_data:
        teleop_name = cfg.teleop.type if cfg.teleop else "none"
        session_name = f"teleop_{cfg.robot.type}_{teleop_name}"
        init_rerun(session_name=session_name, ip=cfg.display_ip, port=cfg.display_port)

    display_compressed_images = (
        True
        if (
            cfg.display_data
            and cfg.display_ip is not None
            and cfg.display_port is not None
        )
        else cfg.display_compressed_images
    )

    robot = None
    teleop = None

    try:
        # --- xense_flare (robot-only, data collection) ---
        if cfg.robot.type == "xense_flare":
            logger.info("Detected Xense Flare data collection gripper")
            robot = make_robot_from_config(cfg.robot)
            robot.connect()
            logger.info(f"Xense Flare connected — MAC: {robot.config.mac_addr}")

            _, _, robot_observation_processor = make_default_processors()
            try:
                xense_flare_teleop_loop(
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    robot_observation_processor=robot_observation_processor,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                logger.info("Data collection interrupted by user")

        # --- xense_multisensor (robot-only, data collection) ---
        elif cfg.robot.type == "xense_multisensor":
            logger.info("Detected Xense Multisensor data collection device")
            robot = make_robot_from_config(cfg.robot)
            robot.connect()
            logger.info(
                f"Xense Multisensor connected — cameras: {list(robot.cameras.keys())}"
            )

            _, _, robot_observation_processor = make_default_processors()
            try:
                xense_multisensor_teleop_loop(
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    robot_observation_processor=robot_observation_processor,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                logger.info("Data collection interrupted by user")

        # --- arx5_follower / bi_arx5 + spacemouse ---
        elif (
            cfg.robot.type in ("bi_arx5", "arx5_follower")
            and cfg.teleop.type == "spacemouse"
        ):
            mode = "bimanual" if cfg.robot.type == "bi_arx5" else "single-arm"
            logger.info(f"Detected ARX5 ({mode}) + SpaceMouse")
            robot = make_robot_from_config(cfg.robot)
            robot.connect()
            teleop = make_teleoperator_from_config(cfg.teleop)
            logger.info(
                f"Current TCP pose (euler+gripper): {robot.get_current_tcp_pose_euler()}"
            )
            teleop.connect(current_tcp_pose_euler=robot.get_current_tcp_pose_euler())
            (
                teleop_action_processor,
                robot_action_processor,
                robot_observation_processor,
            ) = make_default_processors()
            try:
                spacemouse_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dryrun=cfg.dryrun,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                pass

        # --- arx5_follower + trlc_leader ---
        elif cfg.robot.type == "arx5_follower" and cfg.teleop.type == "trlc_leader":
            logger.info("Detected ARX5 Follower + TRLC Leader")
            robot = make_robot_from_config(cfg.robot)
            robot.connect()
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect()
            (
                teleop_action_processor,
                robot_action_processor,
                robot_observation_processor,
            ) = make_default_processors()
            try:
                arx5_trlc_leader_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    debug_timing=cfg.debug_timing,
                    dryrun=cfg.dryrun,
                )
            except KeyboardInterrupt:
                pass

        # --- arx5_follower / bi_arx5 (other teleops) ---
        elif cfg.robot.type in ("bi_arx5", "arx5_follower"):
            mode = "bimanual" if cfg.robot.type == "bi_arx5" else "single-arm"
            logger.info(f"Detected ARX5 ({mode}), using ARX5 teleop loop")
            robot = make_robot_from_config(cfg.robot)
            robot.connect()
            (
                teleop_action_processor,
                robot_action_processor,
                robot_observation_processor,
            ) = make_default_processors()
            try:
                arx5_teleop_loop(
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                pass

        # --- flexiv_rizon4 + pico4 ---
        elif cfg.robot.type == "flexiv_rizon4" and cfg.teleop.type == "pico4":
            logger.info("Detected Flexiv Rizon4 + Pico4")
            robot = make_robot_from_config(cfg.robot)
            _check_cartesian_mode(robot, "Pico4")
            robot.connect(go_to_start=True)
            logger.info(f"Start EEF pose: {robot.get_current_tcp_pose_quat()}")
            (
                teleop_action_processor,
                robot_action_processor,
                robot_observation_processor,
            ) = make_default_processors()
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect(current_tcp_pose_quat=robot.get_current_tcp_pose_quat())
            try:
                pico4_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dryrun=cfg.dryrun,
                )
            except KeyboardInterrupt:
                logger.info("Teleoperation interrupted by user")

        # --- flexiv_rizon4 + spacemouse ---
        elif cfg.robot.type == "flexiv_rizon4" and cfg.teleop.type == "spacemouse":
            logger.info("Detected Flexiv Rizon4 + SpaceMouse")
            robot = make_robot_from_config(cfg.robot)
            _check_cartesian_mode(robot, "SpaceMouse")
            robot.connect(go_to_start=True)
            start_obs = robot.get_observation()
            tcp_keys = [k for k in start_obs if k.startswith("tcp.")]
            logger.info(
                "Start pose: " + ", ".join(f"{k}={start_obs[k]:.6f}" for k in tcp_keys)
            )
            (
                teleop_action_processor,
                robot_action_processor,
                robot_observation_processor,
            ) = make_default_processors()
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect(current_tcp_pose_euler=robot.get_current_tcp_pose_euler())
            try:
                spacemouse_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dryrun=cfg.dryrun,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                logger.info("Teleoperation interrupted by user")

        # --- flexiv_rizon4 + vive_tracker ---
        elif cfg.robot.type == "flexiv_rizon4" and cfg.teleop.type == "vive_tracker":
            logger.info("Detected Flexiv Rizon4 + Vive Tracker")
            robot = make_robot_from_config(cfg.robot)
            _check_cartesian_mode(robot, "Vive Tracker")
            robot.connect(go_to_start=False)
            logger.info(f"Start TCP pose (quat): {robot.get_current_tcp_pose_quat()}")
            (
                teleop_action_processor,
                robot_action_processor,
                robot_observation_processor,
            ) = make_default_processors()
            teleop = make_teleoperator_from_config(cfg.teleop)
            current_tcp_pose = robot.get_current_tcp_pose_quat()[:7]
            teleop.connect(current_tcp_pose_quat=current_tcp_pose)
            try:
                vive_tracker_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dryrun=cfg.dryrun,
                )
            except KeyboardInterrupt:
                logger.info("Teleoperation interrupted by user")

        # --- flexiv_rizon4 + xense_flare (teleop) ---
        elif cfg.robot.type == "flexiv_rizon4" and cfg.teleop.type == "xense_flare":
            logger.info("Detected Flexiv Rizon4 + Xense Flare teleoperator")
            robot = make_robot_from_config(cfg.robot)
            _check_cartesian_mode(robot, "Xense Flare")
            robot.connect(go_to_start=False)
            logger.info(f"Start TCP pose (quat): {robot.get_current_tcp_pose_quat()}")
            (
                teleop_action_processor,
                robot_action_processor,
                robot_observation_processor,
            ) = make_default_processors()
            teleop = make_teleoperator_from_config(cfg.teleop)
            current_tcp_pose = robot.get_current_tcp_pose_quat()[:7]
            teleop.connect(current_tcp_pose_quat=current_tcp_pose)
            try:
                xense_flare_flexiv_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dryrun=cfg.dryrun,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                logger.info("Teleoperation interrupted by user")

        # --- flexiv_rizon4_rt + spacemouse ---
        elif cfg.robot.type == "flexiv_rizon4_rt" and cfg.teleop.type == "spacemouse":
            logger.info("Detected Flexiv Rizon4 RT + SpaceMouse")
            robot = make_robot_from_config(cfg.robot)
            robot.connect(go_to_start=True)
            start_obs = robot.get_observation()
            tcp_keys = [k for k in start_obs if k.startswith("tcp.")]
            logger.info(
                "Start pose: " + ", ".join(f"{k}={start_obs[k]:.6f}" for k in tcp_keys)
            )
            (
                teleop_action_processor,
                robot_action_processor,
                robot_observation_processor,
            ) = make_default_processors()
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect(current_tcp_pose_euler=robot.get_current_tcp_pose_euler())
            try:
                spacemouse_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dryrun=cfg.dryrun,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                logger.info("Teleoperation interrupted by user")

        # --- flexiv_rizon4_rt + pico4 ---
        elif cfg.robot.type == "flexiv_rizon4_rt" and cfg.teleop.type == "pico4":
            logger.info("Detected Flexiv Rizon4 RT + Pico4")
            robot = make_robot_from_config(cfg.robot)
            robot.connect(go_to_start=True)
            start_obs = robot.get_observation()
            tcp_keys = [k for k in start_obs if k.startswith("tcp.")]
            logger.info(
                "Start pose: " + ", ".join(f"{k}={start_obs[k]:.6f}" for k in tcp_keys)
            )
            (
                teleop_action_processor,
                robot_action_processor,
                robot_observation_processor,
            ) = make_default_processors()
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect(current_tcp_pose_quat=robot.get_current_tcp_pose_quat())
            try:
                pico4_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dryrun=cfg.dryrun,
                )
            except KeyboardInterrupt:
                logger.info("Teleoperation interrupted by user")

        # --- bi_flexiv_rizon4_rt + bi_pico4 ---
        elif cfg.robot.type == "bi_flexiv_rizon4_rt" and cfg.teleop.type == "bi_pico4":
            logger.info("Detected BiFlexivRizon4RT + BiPico4")
            robot = make_robot_from_config(cfg.robot)
            teleop = make_teleoperator_from_config(cfg.teleop)

            # Pre-initialize the VR SDK in background while the robot connects
            # (robot.connect() takes ~20-40s; VR SDK init takes ~3s → free overlap)
            from concurrent.futures import ThreadPoolExecutor as _TPE
            try:
                with _TPE(max_workers=2) as _ex:
                    _robot_fut = _ex.submit(robot.connect, go_to_start=True)
                    _teleop_fut = _ex.submit(teleop.pre_init)
                    _teleop_fut.result()   # raise immediately if VR SDK fails
                    _robot_fut.result()    # raise immediately if robot fails
            except KeyboardInterrupt:
                logger.info("Startup interrupted by user")
                raise

            left_pose, right_pose = robot.get_current_tcp_pose_quat()
            logger.info(f"Left start pose:  {left_pose}")
            logger.info(f"Right start pose: {right_pose}")
            teleop.connect(left_tcp_pose_quat=left_pose, right_tcp_pose_quat=right_pose)
            try:
                bi_pico4_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    dryrun=cfg.dryrun,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                logger.info("Teleoperation interrupted by user")

        # ======================== Pylibfranka Research3 ========================
        # Check if this is Pylibfranka Research3 robot with pico4
        elif cfg.robot.type == "pylibfranka_research3" and cfg.teleop.type == "pico4":
            logger.info(
                "Detected Pylibfranka Research3 robot with Pico4, using specialized teleop loop"
            )

            robot = None
            teleop = None

            try:
                # Create robot instance
                robot = make_robot_from_config(cfg.robot)

                # Ensure robot is in CARTESIAN_IMPEDANCE mode for pico4 teleop
                from lerobot.robots.pylibfranka_research3.config_pylibfranka_research3 import (
                    ControlMode as FR3ControlMode,
                )

                if robot.config.control_mode != FR3ControlMode.CARTESIAN_IMPEDANCE:
                    raise ValueError(
                        f"Pico4 teleoperation requires CARTESIAN_IMPEDANCE mode, "
                        f"but robot is configured with {robot.config.control_mode}"
                    )

                # Connect to robot (launches ws_teleop_server, moves to home)
                try:
                    robot.connect(go_to_start=True)
                    logger.info(f"Start EEF pose: {robot.get_current_tcp_pose_quat()}")
                except Exception as e:
                    logger.error(
                        f"Failed to connect to robot: {e}\n{traceback.format_exc()}"
                    )
                    raise

                (
                    teleop_action_processor,
                    robot_action_processor,
                    robot_observation_processor,
                ) = make_default_processors()

                # Connect to teleoperator with robot's current TCP pose
                try:
                    teleop = make_teleoperator_from_config(cfg.teleop)
                    teleop.connect(
                        current_tcp_pose_quat=robot.get_current_tcp_pose_quat()
                    )
                    logger.info("Connected to Pico4")
                except Exception as e:
                    logger.error(
                        f"Failed to connect to Pico4: {e}\n{traceback.format_exc()}"
                    )
                    raise

                # Run teleoperation loop (reuse pico4_teleop_loop — same action format)
                try:
                    pico4_teleop_loop(
                        teleop=teleop,
                        robot=robot,
                        fps=cfg.fps,
                        display_data=cfg.display_data,
                        duration=cfg.teleop_time_s,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        dryrun=cfg.dryrun,
                    )
                except KeyboardInterrupt:
                    logger.info("Teleoperation interrupted by user")
                except Exception as e:
                    logger.error(
                        f"Error during teleoperation loop: {e}\n{traceback.format_exc()}"
                    )
                    raise

            except Exception as e:
                logger.error(
                    f"Error in teleoperation setup or execution: {e}\n{traceback.format_exc()}"
                )
            finally:
                if cfg.display_data:
                    try:
                        rr.rerun_shutdown()
                    except Exception as e:
                        logger.warn(f"Error shutting down rerun: {e}")

                if teleop is not None:
                    try:
                        if teleop.is_connected:
                            teleop.disconnect()
                            logger.info("Pico4 disconnected")
                    except Exception as e:
                        logger.error(
                            f"Error disconnecting Pico4: {e}\n{traceback.format_exc()}"
                        )

                if robot is not None:
                    try:
                        if robot.is_connected:
                            robot.disconnect()
                            logger.info("Robot safely disconnected")
                    except Exception as e:
                        logger.error(
                            f"Error disconnecting robot: {e}\n{traceback.format_exc()}"
                        )

        # Check if this is Pylibfranka Research3 robot with spacemouse
        elif (
            cfg.robot.type == "pylibfranka_research3"
            and cfg.teleop.type == "spacemouse"
        ):
            logger.info(
                "Detected Pylibfranka Research3 robot with Spacemouse, using specialized teleop loop"
            )

            robot = None
            teleop = None

            try:
                # Create robot instance
                robot = make_robot_from_config(cfg.robot)

                # Ensure robot is in CARTESIAN_IMPEDANCE mode for spacemouse teleop
                from lerobot.robots.pylibfranka_research3.config_pylibfranka_research3 import (
                    ControlMode as FR3ControlMode,
                )

                if robot.config.control_mode != FR3ControlMode.CARTESIAN_IMPEDANCE:
                    raise ValueError(
                        f"Spacemouse teleoperation requires CARTESIAN_IMPEDANCE mode, "
                        f"but robot is configured with {robot.config.control_mode}"
                    )

                # Connect to robot (launches ws_teleop_server, moves to home)
                try:
                    robot.connect(go_to_start=True)
                    start_obs = robot.get_observation()
                    tcp_keys = [k for k in start_obs if k.startswith("tcp.")]
                    logger.info(
                        "Start pose: "
                        + ", ".join(f"{k}={start_obs[k]:.6f}" for k in tcp_keys)
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to connect to robot: {e}\n{traceback.format_exc()}"
                    )
                    raise

                (
                    teleop_action_processor,
                    robot_action_processor,
                    robot_observation_processor,
                ) = make_default_processors()

                # Connect to teleoperator with robot's current TCP pose (Euler format for spacemouse)
                try:
                    teleop = make_teleoperator_from_config(cfg.teleop)
                    teleop.connect(
                        current_tcp_pose_euler=robot.get_current_tcp_pose_euler()
                    )
                    logger.info("Connected to Spacemouse")
                except Exception as e:
                    logger.error(
                        f"Failed to connect to Spacemouse: {e}\n{traceback.format_exc()}"
                    )
                    raise

                # Run teleoperation loop (reuse spacemouse_teleop_loop — handles Euler→6D conversion)
                try:
                    spacemouse_teleop_loop(
                        teleop=teleop,
                        robot=robot,
                        fps=cfg.fps,
                        display_data=cfg.display_data,
                        duration=cfg.teleop_time_s,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        dryrun=cfg.dryrun,
                        debug_timing=cfg.debug_timing,
                        no_obs=cfg.no_obs,
                    )
                except KeyboardInterrupt:
                    logger.info("Teleoperation interrupted by user")
                except Exception as e:
                    logger.error(
                        f"Error during teleoperation loop: {e}\n{traceback.format_exc()}"
                    )
                    raise

            except Exception as e:
                logger.error(
                    f"Error in teleoperation setup or execution: {e}\n{traceback.format_exc()}"
                )
            finally:
                if cfg.display_data:
                    try:
                        rr.rerun_shutdown()
                    except Exception as e:
                        logger.warn(f"Error shutting down rerun: {e}")

                if teleop is not None:
                    try:
                        if teleop.is_connected:
                            teleop.disconnect()
                            logger.info("Spacemouse disconnected")
                    except Exception as e:
                        logger.error(
                            f"Error disconnecting Spacemouse: {e}\n{traceback.format_exc()}"
                        )

                if robot is not None:
                    try:
                        if robot.is_connected:
                            robot.disconnect()
                            logger.info("Robot safely disconnected")
                    except Exception as e:
                        logger.error(
                            f"Error disconnecting robot: {e}\n{traceback.format_exc()}"
                        )

        # Check if this is Pylibfranka Research3 robot with btgamepad
        elif (
            cfg.robot.type == "pylibfranka_research3" and cfg.teleop.type == "btgamepad"
        ):
            logger.info(
                "Detected Pylibfranka Research3 robot with BT Gamepad, using specialized teleop loop"
            )

            robot = None
            teleop = None

            try:
                # Create robot instance
                robot = make_robot_from_config(cfg.robot)

                # Ensure robot is in CARTESIAN_IMPEDANCE mode for btgamepad teleop
                from lerobot.robots.pylibfranka_research3.config_pylibfranka_research3 import (
                    ControlMode as FR3ControlMode,
                )

                if robot.config.control_mode != FR3ControlMode.CARTESIAN_IMPEDANCE:
                    raise ValueError(
                        f"BT Gamepad teleoperation requires CARTESIAN_IMPEDANCE mode, "
                        f"but robot is configured with {robot.config.control_mode}"
                    )

                # Connect to robot (launches ws_teleop_server, moves to home)
                try:
                    robot.connect(go_to_start=True)
                    logger.info(f"Start EEF pose: {robot.get_current_tcp_pose_quat()}")
                except Exception as e:
                    logger.error(
                        f"Failed to connect to robot: {e}\n{traceback.format_exc()}"
                    )
                    raise

                (
                    teleop_action_processor,
                    robot_action_processor,
                    robot_observation_processor,
                ) = make_default_processors()

                # Connect to teleoperator with robot's current TCP pose (quat format for btgamepad)
                try:
                    teleop = make_teleoperator_from_config(cfg.teleop)
                    teleop.connect(
                        current_tcp_pose_quat=robot.get_current_tcp_pose_quat()
                    )
                    logger.info("Connected to BT Gamepad")
                except Exception as e:
                    logger.error(
                        f"Failed to connect to BT Gamepad: {e}\n{traceback.format_exc()}"
                    )
                    raise

                # Run teleoperation loop
                try:
                    btgamepad_teleop_loop(
                        teleop=teleop,
                        robot=robot,
                        fps=cfg.fps,
                        display_data=cfg.display_data,
                        duration=cfg.teleop_time_s,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        dryrun=cfg.dryrun,
                    )
                except KeyboardInterrupt:
                    logger.info("Teleoperation interrupted by user")
                except Exception as e:
                    logger.error(
                        f"Error during teleoperation loop: {e}\n{traceback.format_exc()}"
                    )
                    raise

            except Exception as e:
                logger.error(
                    f"Error in teleoperation setup or execution: {e}\n{traceback.format_exc()}"
                )
            finally:
                if cfg.display_data:
                    try:
                        rr.rerun_shutdown()
                    except Exception as e:
                        logger.warn(f"Error shutting down rerun: {e}")

                if teleop is not None:
                    try:
                        if teleop.is_connected:
                            teleop.disconnect()
                            logger.info("BT Gamepad disconnected")
                    except Exception as e:
                        logger.error(
                            f"Error disconnecting BT Gamepad: {e}\n{traceback.format_exc()}"
                        )

                if robot is not None:
                    try:
                        if robot.is_connected:
                            robot.disconnect()
                            logger.info("Robot safely disconnected")
                    except Exception as e:
                        logger.error(
                            f"Error disconnecting robot: {e}\n{traceback.format_exc()}"
                        )

        # ======================== Mock Robot ========================
        elif cfg.robot.type == "mock_robot":
            logger.info("Detected mock robot, using mock teleop loop")
            robot = make_robot_from_config(cfg.robot)
            robot.connect()
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect()
            (
                teleop_action_processor,
                robot_action_processor,
                robot_observation_processor,
            ) = make_default_processors()
            try:
                mock_robot_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dryrun=cfg.dryrun,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                pass

        # --- generic fallback ---
        else:
            teleop = make_teleoperator_from_config(cfg.teleop)
            robot = make_robot_from_config(cfg.robot)
            (
                teleop_action_processor,
                robot_action_processor,
                robot_observation_processor,
            ) = make_default_processors()
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
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                pass

    except Exception as e:
        logger.error(f"Error in teleoperation: {e}\n{traceback.format_exc()}")
    finally:
        _cleanup(robot, teleop, cfg.display_data)


def main():
    teleoperate()


if __name__ == "__main__":
    main()
