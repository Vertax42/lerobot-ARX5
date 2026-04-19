# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 XenseRobotics Inc. All rights reserved.
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
Records a dataset using a teleoperator to drive a generic Robot.
"""

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import (
    RealSenseCameraConfig,
)  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    build_dataset_frame,
    combine_feature_dicts,
    hw_to_dataset_features,
)
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    gamepad,
    make_teleoperator_from_config,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop  # noqa: F401

# Import mock_teleop to register its config with draccus ChoiceRegistry
try:
    from lerobot.teleoperators.mock_teleop import MockTeleopConfig  # noqa: F401
except ImportError:
    # If tests are not available, mock_teleop won't be available
    pass
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    refresh_listener_events,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait, get_logger
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = get_logger("lerobot_record")


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


def extract_joint_positions(obs):
    """Extract joint positions from an observation dict, skipping camera keys."""
    joint_positions = {}
    for key, value in obs.items():
        if (
            key.endswith(".pos")
            and not key.startswith("head")
            and not key.startswith("left_wrist")
            and not key.startswith("right_wrist")
        ):
            joint_positions[key] = value
    return joint_positions


def apply_velocity_limits(
    current_action: dict, prev_action: dict, dt: float, robot=None
) -> dict:
    """Apply velocity limits to an action dict to clip unsafe jumps.

    This is intended for bimanual arms that expose 6 joints per side plus a
    gripper; callers without that schema can safely skip this helper.
    """
    if prev_action is None:
        return current_action

    if robot is not None and hasattr(robot, "robot_configs"):
        left_config = robot.robot_configs["left_config"]
        joint_vel_limits = left_config.joint_vel_max.tolist()
        gripper_vel_limit = left_config.gripper_vel_max
    else:
        joint_vel_limits = [20.0, 20.0, 20.5, 20.5, 20.0, 20.0]
        gripper_vel_limit = 0.3

    limited_action = current_action.copy()
    clip_count = 0

    for i in range(6):
        left_key = f"left_joint_{i+1}.pos"
        right_key = f"right_joint_{i+1}.pos"
        for key in [left_key, right_key]:
            if key in current_action and key in prev_action:
                current_pos = current_action[key]
                prev_pos = prev_action[key]
                delta_pos = current_pos - prev_pos
                max_delta = joint_vel_limits[i] * dt
                if abs(delta_pos) > max_delta:
                    sign = 1 if delta_pos > 0 else -1
                    limited_action[key] = prev_pos + sign * max_delta
                    clip_count += 1

    for gripper_key in ["left_gripper.pos", "right_gripper.pos"]:
        if gripper_key in current_action and gripper_key in prev_action:
            current_pos = current_action[gripper_key]
            prev_pos = prev_action[gripper_key]
            delta_pos = current_pos - prev_pos
            max_delta = gripper_vel_limit * dt
            if abs(delta_pos) > max_delta:
                sign = 1 if delta_pos > 0 else -1
                limited_action[gripper_key] = prev_pos + sign * max_delta
                clip_count += 1

    if clip_count > 0:
        logger.debug(f"Applied velocity limits: {clip_count} joints clipped")

    return limited_action


def _record_loop_sleep(
    start_loop_t: float,
    fps: int,
    start_episode_t: float,
    robot: Robot | None = None,
) -> None:
    if fps <= 0:
        return

    budget_s = 1.0 / fps
    dt_s = time.perf_counter() - start_loop_t
    remaining_s = budget_s - dt_s
    if remaining_s > 0:
        busy_wait(remaining_s)
        return

    episode_t_s = time.perf_counter() - start_episode_t
    robot_name = (
        getattr(robot, "name", None) or getattr(type(robot), "__name__", "record")
        if robot is not None
        else "record"
    )
    logger.warn(
        f"[slow_frame] robot={robot_name} t={episode_t_s:.3f}s "
        f"loop={dt_s * 1e3:.1f}ms budget={budget_s * 1e3:.1f}ms "
        f"overrun={(-remaining_s) * 1e3:.1f}ms"
        f"{_format_slow_frame_obs_suffix(robot)}"
    )


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording.
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 1200
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    num_image_writer_threads_per_camera: int = 4
    # Number of episodes to record before batch encoding videos
    video_encoding_batch_size: int = 1
    # Video codec to use for encoding.
    vcodec: str = "auto"
    # Encode frames in real-time while recording (streaming encoding).
    streaming_encoding: bool = False
    # Maximum number of frames to buffer per camera when streaming encoding is active.
    encoder_queue_maxsize: int = 30
    # Number of threads per encoder process (None = codec default).
    encoder_threads: int | None = None
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Whether to control the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    def __post_init__(self):
        if self.teleop is None:
            raise ValueError("A teleoperator is required to control the robot.")


def _build_dataset_features(robot: Robot, use_videos: bool) -> dict[str, dict]:
    return combine_feature_dicts(
        hw_to_dataset_features(robot.action_features, ACTION, use_videos),
        hw_to_dataset_features(robot.observation_features, OBS_STR, use_videos),
    )


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    teleop: Teleoperator,
    events: dict,
    fps: int,
    dataset: LeRobotDataset | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    """Generic record loop: observation from robot, action from teleop, dataset write + rerun."""
    if dataset is not None and dataset.fps != fps:
        raise ValueError(
            f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps})."
        )

    timestamp = 0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        refresh_listener_events(events)

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Get teleop action
        action = teleop.get_action()

        # Send action to robot
        sent_action = robot.send_action(action)

        # Write to dataset
        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs, action=sent_action)

        _record_loop_sleep(
            start_loop_t=start_loop_t,
            fps=fps,
            start_episode_t=start_episode_t,
            robot=robot,
        )

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    logger.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    dataset_features = _build_dataset_features(robot, cfg.dataset.video)

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(
            dataset, robot, cfg.dataset.fps, dataset_features
        )
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.dataset.repo_id, None)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
            * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
        )

    if not cfg.dataset.streaming_encoding:
        logger.info(
            "Streaming encoding is disabled. For faster episode saving, consider enabling: "
            "--dataset.streaming_encoding=true --dataset.encoder_threads=2"
        )

    robot.connect()
    teleop.connect()

    listener, events = init_keyboard_listener(teleop=teleop)

    try:
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while (
                recorded_episodes < cfg.dataset.num_episodes
                and not events["stop_recording"]
            ):
                if dataset is not None:
                    dataset.prepare_episode_recording()

                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)

                record_loop(
                    robot=robot,
                    teleop=teleop,
                    events=events,
                    fps=cfg.dataset.fps,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                )

                # Execute a few seconds without recording to give time to manually reset the environment
                # Skip reset for the last episode to be recorded
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1)
                    or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        teleop=teleop,
                        events=events,
                        fps=cfg.dataset.fps,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
                events["exit_early"] = False  # clear so the reset loop runs normally

        log_say("Stop recording", cfg.play_sounds, blocking=True)
    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt received. Stopping recording...")
    except Exception as e:
        import traceback

        logger.error(f"Error during recording: {e}\n{traceback.format_exc()}")
    finally:
        try:
            if robot.is_connected:
                logger.info("Disconnecting robot...")
                robot.disconnect()
                logger.info("Robot disconnected safely")
        except Exception as e:
            logger.error(f"Error during robot disconnect: {e}")

        try:
            if (
                teleop is not None
                and hasattr(teleop, "is_connected")
                and teleop.is_connected
            ):
                logger.info("Disconnecting teleop...")
                teleop.disconnect()
                logger.info("Teleop disconnected safely")
        except Exception as e:
            logger.error(f"Error during teleop disconnect: {e}")

        try:
            if not is_headless() and listener is not None:
                listener.stop()
        except Exception as e:
            logger.error(f"Error stopping listener: {e}")

        try:
            if cfg.dataset.push_to_hub:
                dataset.push_to_hub(
                    tags=cfg.dataset.tags,
                    private=cfg.dataset.private,
                    upload_large_folder=True,
                )
        except Exception as e:
            logger.error(f"Error pushing to hub: {e}")

    log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_devices()
    record()


if __name__ == "__main__":
    main()
