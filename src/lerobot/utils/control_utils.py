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

########################################################################################
# Utilities
########################################################################################


import logging
import traceback
from functools import cache
from typing import Any

from deepdiff import DeepDiff

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.robots import Robot


@cache
def is_headless():
    """
    Detects if the Python script is running in a headless environment (e.g., without a display).

    This function attempts to import `pynput`, a library that requires a graphical environment.
    If the import fails, it assumes the environment is headless. The result is cached to avoid
    re-running the check.

    Returns:
        True if the environment is determined to be headless, False otherwise.
    """
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def init_keyboard_listener(teleop: Any | None = None):
    """
    Initializes a non-blocking keyboard listener for real-time user interaction.

    This function sets up a listener for specific keys (right arrow, left arrow, escape, space)
    to control the program flow during execution. When `teleop` is provided and exposes
    ``poll_buttons()`` + ``get_reset_button()``, the teleop's reset button is mapped to the same
    ``go_start`` event as the Space key. Headless environments skip the keyboard listener but
    still surface the event dict so recording loops keep working.

    Args:
        teleop: Optional teleoperator whose button events should be mapped into the same
            `events` dictionary as keyboard shortcuts.

    Returns:
        A tuple containing:
        - The `pynput.keyboard.Listener` instance, or `None` if in a headless environment.
        - A dictionary of event flags (e.g., `exit_early`) that are set by key presses.
    """
    events: dict[str, Any] = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    events["go_start"] = False

    def refresh_events_from_teleop() -> None:
        if teleop is None:
            return
        try:
            if hasattr(teleop, "poll_buttons"):
                teleop.poll_buttons()
            # Only A button (reset) is mapped — B/X/Y caused accidental triggers.
            if hasattr(teleop, "get_reset_button") and teleop.get_reset_button():
                events["go_start"] = True
        except Exception as e:
            logging.debug(f"Error refreshing teleop control events: {e}")

    events["_refresh_events"] = refresh_events_from_teleop

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.space:
                print("Space key pressed. Robot will go to start pose while recording continues...")
                events["go_start"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def refresh_listener_events(events: dict[str, Any]) -> None:
    """Polls teleop-derived control events attached by init_keyboard_listener()."""
    refresh = events.get("_refresh_events")
    if callable(refresh):
        refresh()


def sanity_check_dataset_name(repo_id):
    """Rejects dataset names starting with 'eval_', which are reserved for policy evaluation runs
    (unsupported in this minimal build)."""
    _, dataset_name = repo_id.split("/")
    if dataset_name.startswith("eval_"):
        raise ValueError(
            f"Dataset name '{dataset_name}' begins with 'eval_', which is reserved for policy "
            "evaluation — policy inference is not supported in this build."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    Checks if a dataset's metadata is compatible with the current robot and recording setup.

    This function compares key metadata fields (`robot_type`, `fps`, and `features`) from the
    dataset against the current configuration to ensure that appended data will be consistent.

    Args:
        dataset: The `LeRobotDataset` instance to check.
        robot: The `Robot` instance representing the current hardware setup.
        fps: The current recording frequency (frames per second).
        features: The dictionary of features for the current recording session.

    Raises:
        ValueError: If any of the checked metadata fields do not match.
    """
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
