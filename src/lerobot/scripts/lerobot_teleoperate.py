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

Example (mock robot + keyboard):

```shell
lerobot-teleoperate \
    --robot.type=mock_robot \
    --teleop.type=keyboard \
    --fps=30
```

Example (mock robot + gamepad):

```shell
lerobot-teleoperate \
    --robot.type=mock_robot \
    --teleop.type=gamepad \
    --fps=30
```
"""

import time
import traceback
from dataclasses import asdict, dataclass
from pprint import pformat

import numpy as np
import rerun as rr

from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    mock_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    gamepad,
    make_teleoperator_from_config,
    mock_teleop,
)
from lerobot.utils.robot_utils import (
    busy_wait,
    get_logger,
)
from lerobot.utils.utils import move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


logger = get_logger("Teleoperate")


@dataclass
class TeleoperateConfig:
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
    display_compressed_images: bool = True
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
# Shared timing helpers
# ---------------------------------------------------------------------------


def _format_slow_frame_obs_suffix(robot: Robot | None) -> str:
    if robot is None:
        return ""

    timing = getattr(robot, "_last_obs_timing", None)
    if not isinstance(timing, dict):
        return ""

    parts: list[str] = []
    total_ms = timing.get("total_ms")
    if isinstance(total_ms, (int, float)):
        parts.append(f"obs={float(total_ms):.1f}ms")

    arm_items = [
        (key[:-3], float(value))
        for key, value in timing.items()
        if key.endswith("_arm_ms") and isinstance(value, (int, float))
    ]
    if arm_items:
        parts.append(f"arms={sum(value for _, value in arm_items):.1f}ms")

    grip_items = [
        (key[:-3], float(value))
        for key, value in timing.items()
        if key.endswith("_grip_ms") and isinstance(value, (int, float))
    ]
    if grip_items:
        parts.append(f"grips={sum(value for _, value in grip_items):.1f}ms")

    cameras_ms = timing.get("cameras_ms")
    if isinstance(cameras_ms, (int, float)):
        parts.append(f"cams={float(cameras_ms):.1f}ms")

    cam_items = [
        (key[4:-4], float(value))
        for key, value in timing.items()
        if (
            key.startswith("cam[")
            and key.endswith("]_ms")
            and isinstance(value, (int, float))
        )
    ]
    cam_items.sort(key=lambda item: item[1], reverse=True)

    obs_part_items = arm_items + grip_items + cam_items
    obs_part_items.sort(key=lambda item: item[1], reverse=True)
    if obs_part_items:
        visible_obs_items = [item for item in obs_part_items if item[1] >= 0.1]
        if not visible_obs_items:
            visible_obs_items = obs_part_items
        top_parts = ", ".join(
            f"{name}={value:.1f}ms" for name, value in visible_obs_items[:4]
        )
        parts.append(f"top_obs={top_parts}")

    return f" | {' '.join(parts)}" if parts else ""


def _teleop_loop_sleep(
    start_loop_t: float,
    fps: int,
    session_start_t: float,
    robot: Robot | None = None,
) -> None:
    """Sleep for the remaining frame budget; log a warning if the loop overran."""
    if fps <= 0:
        return

    budget_s = 1.0 / fps
    dt_s = time.perf_counter() - start_loop_t
    remaining_s = budget_s - dt_s
    if remaining_s > 0:
        busy_wait(remaining_s)
        return

    session_t_s = time.perf_counter() - session_start_t
    robot_name = (
        getattr(robot, "name", None) or getattr(type(robot), "__name__", "teleop")
        if robot is not None
        else "teleop"
    )
    logger.warn(
        f"[slow_frame] robot={robot_name} t={session_t_s:.3f}s "
        f"loop={dt_s * 1e3:.1f}ms budget={budget_s * 1e3:.1f}ms "
        f"overrun={(-remaining_s) * 1e3:.1f}ms"
        f"{_format_slow_frame_obs_suffix(robot)}"
    )


# ---------------------------------------------------------------------------
# Generic teleop loop
# ---------------------------------------------------------------------------


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = True,
    debug_timing: bool = False,
):
    """
    Continuously reads actions from a teleoperation device, sends them to a robot, and
    optionally displays the robot's state. The loop runs at a specified frequency until
    a set duration is reached or it is manually interrupted.
    """
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        obs_t0 = time.perf_counter()
        obs = robot.get_observation()
        obs_time_ms = (time.perf_counter() - obs_t0) * 1e3

        # Get teleop action
        teleop_t0 = time.perf_counter()
        action = teleop.get_action()
        teleop_time_ms = (time.perf_counter() - teleop_t0) * 1e3

        # Send action to robot
        send_t0 = time.perf_counter()
        _ = robot.send_action(action)
        send_time_ms = (time.perf_counter() - send_t0) * 1e3

        if display_data:
            log_rerun_data(
                observation=obs,
                action=action,
                compress_images=display_compressed_images,
            )

        _teleop_loop_sleep(loop_start, fps, start, robot)
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
# Mock robot teleop loop
# ---------------------------------------------------------------------------


def mock_robot_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
    debug_timing: bool = False,
):
    """
    Dedicated teleoperation loop for Mock Robot.

    Filters teleop actions to keys the mock robot's action schema knows about, and
    formats a per-step CMD/OBS/ERR panel when display_data is enabled.
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

        # Keep only keys known by mock robot action schema.
        filtered_action = {
            k: v for k, v in raw_action.items() if k in robot_action_keys
        }
        if not filtered_action and raw_action:
            filtered_action = raw_action
        elif len(filtered_action) != len(raw_action) and not warned_unmapped_keys:
            dropped = sorted(set(raw_action) - robot_action_keys)
            logger.warn(
                f"Action keys not present in mock robot action schema, dropping: {dropped}"
            )
            warned_unmapped_keys = True

        if not dryrun:
            _ = robot.send_action(filtered_action)

        _teleop_loop_sleep(loop_start, fps, start, robot)
        loop_s = time.perf_counter() - loop_start

        if display_data:
            log_rerun_data(observation=obs, action=filtered_action)

            ordered_keys = [k for k in robot.action_features if k in filtered_action]
            ordered_keys.extend(k for k in filtered_action if k not in ordered_keys)

            panel_lines = []
            panel_lines.append("-" * (display_len + 38))
            panel_lines.append(
                f"{'NAME':<{display_len}} | {'CMD':>8} | {'OBS':>8} | {'ERR':>8}"
            )
            for motor in ordered_keys:
                cmd = float(filtered_action[motor])
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
                f"{k}={float(v):+.3f}" for k, v in filtered_action.items()
            )
            dryrun_tag = "[DRYRUN] " if dryrun else ""
            print(
                f"\r\033[K{dryrun_tag}{loop_s * 1e3:5.1f}ms ({1 / loop_s:4.0f}Hz) | {action_summary}",
                end="",
                flush=True,
            )

        if duration is not None and time.perf_counter() - start >= duration:
            return


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
        robot = make_robot_from_config(cfg.robot)
        teleop = make_teleoperator_from_config(cfg.teleop)
        robot.connect()
        teleop.connect()

        if cfg.robot.type == "mock_robot":
            logger.info("Detected mock robot, using mock teleop loop")
            try:
                mock_robot_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    dryrun=cfg.dryrun,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                pass
        else:
            try:
                teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
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
