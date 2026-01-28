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
"""

# Import mock_teleop FIRST to register its config with draccus ChoiceRegistry
# This must happen before any other imports that might use TeleoperatorConfig
import time
import traceback
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    xense_multisensor,  # noqa: F401
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    mock_teleop,
    pico4,
)
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait, get_logger, rotation_6d_to_quaternion
from lerobot.utils.utils import move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Create global logger for teleoperate script
logger = get_logger("Teleoperate")


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi)
    # its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 100
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    debug_timing: bool = False
    # Dryrun mode: print actions without sending to robot
    dryrun: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
):
    """
    This function continuously reads actions from a teleoperation device, sends them to a robot,
    and optionally displays the robot's state. The loop runs at a specified frequency until a
    set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        obs = robot.get_observation()

        # Get teleop action
        action = teleop.get_action()

        # Send action to robot
        robot.send_action(action)

        if display_data:
            rr.set_time("timeline", sequence=int(frame_id))
            log_rerun_data(observation=obs, action=action)
            frame_id += 1

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in action.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(action) + 5)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return


# Note: spacemouse_teleop_loop and vive_tracker_teleop_loop removed as spacemouse and vive_tracker teleoperators have been deleted


def pico4_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
):
    """
    Teleop loop for Pico4 VR controller with Flexiv Rizon4 robot.

    Pico4 outputs actions directly in Flexiv format:
    - tcp.x, tcp.y, tcp.z: absolute TCP position (meters)
    - tcp.r1, tcp.r2, tcp.r3, tcp.r4, tcp.r5, tcp.r6: absolute TCP orientation (6D rotation)
    - gripper.pos: absolute gripper position (meters)

    Control scheme:
    - Grip: Enable control (must be held to move robot)
    - Trigger: Controls gripper position (0=closed, 1=open)
    - A button: Reset to initial position
    """
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    frame_id = 0
    while True:
        loop_start = time.perf_counter()

        # Get robot observation (for visualization)
        obs = robot.get_observation()

        # Get teleop action first (this also caches A button state for get_reset_button)
        raw_action = teleop.get_action()

        # Check for reset button (uses cached A button state from get_action)
        reset_button = teleop.get_reset_button()
        if reset_button:
            try:
                if not dryrun:
                    # Reset robot to initial position
                    if hasattr(robot, "reset_to_initial_position"):
                        robot.reset_to_initial_position()
                    logger.info("Reset to initial position (A button pressed)")
                else:
                    logger.info(
                        "[DRYRUN] Reset to initial position (A button pressed) - robot movement skipped"
                    )

                # Always reset teleop state (both dryrun and normal mode)
                current_pose_quat = robot.get_current_tcp_pose_quat()
                teleop.reset_to_pose(current_pose_quat[:7], current_pose_quat[7])
            except Exception as e:
                logger.error(
                    f"Failed to reset robot position: {e}\n{traceback.format_exc()}"
                )
            # Skip this loop iteration (don't send action after reset)
            continue

        # For Pico4 + Flexiv, action is already in correct format
        # No conversion needed (unlike Spacemouse which needs Euler->Quaternion)
        action = raw_action

        # Send action to robot
        if not dryrun:
            robot.send_action(action)

        if display_data:
            # Log raw observation directly (including images from FlareGripper)
            rr.set_time("timeline", sequence=int(frame_id))
            log_rerun_data(
                observation=obs,  # Use raw obs to ensure images are included
                action=action,
            )
            frame_id += 1

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in action.items():
                print(f"{motor:<{display_len}} | {value:>7.4f}")
            move_cursor_up(len(action) + 5)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start

        # Print status line with enable state and grip value for debugging
        # Only print if not using display_data (to avoid conflicting with Rerun terminal output)
        if not display_data:
            enable_str = "ENABLED" if teleop._enabled else "DISABLED"
            ori_str = "ORI:ON" if teleop._orientation_control_active else "ORI:OFF"
            grip_str = f"grip={teleop._last_grip:.2f}"
            gripper_pos_str = (
                f"gripper={action.get('gripper.pos', 0.0):.2f}"
            )
            if dryrun:
                print(
                    f"\r\033[Ktime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | [DRYRUN] | {enable_str} | {grip_str} | {gripper_pos_str} | {ori_str}",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\r\033[Ktime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | {enable_str} | {grip_str} | {gripper_pos_str} | {ori_str}",
                    end="",
                    flush=True,
                )

        if duration is not None and time.perf_counter() - start >= duration:
            return


# Note: xense_flare_flexiv_teleop_loop and xense_flare_teleop_loop removed as xense_flare robot and teleoperator have been deleted


def xense_multisensor_teleop_loop(
    robot: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None,
    debug_timing: bool = False,
):
    """
    Data collection loop for Xense Multisensor robot.

    Xense Multisensor is a pure observation device (similar to teach mode).
    This loop continuously reads multi-modal sensor data from multiple cameras:
    - RealSense cameras: RGB images
    - Xense tactile sensors: Tactile images

    No actions are sent to the robot - it is a data collection device.
    """
    import numpy as np

    start = time.perf_counter()
    timing_stats = {
        "camera_times": {},
        "total_obs_times": [],
        "loop_times": [],
    }

    # Identify camera keys
    camera_keys = list(robot.observation_features.keys())
    for cam_key in camera_keys:
        timing_stats["camera_times"][cam_key] = []

    frame_id = 0
    while True:
        loop_start = time.perf_counter()

        # Time the complete observation acquisition
        obs_start = time.perf_counter()

        try:
            # Get all observations from the robot
            obs = robot.get_observation()
        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            dt_s = time.perf_counter() - loop_start
            busy_wait(1 / fps - dt_s)
            continue

        total_obs_time = time.perf_counter() - obs_start
        timing_stats["total_obs_times"].append(total_obs_time * 1000)

        if display_data:
            # Log all camera data to Rerun
            # rr.set_time("timeline", sequence=int(frame_id))
            log_rerun_data(
                observation=obs,
                action={},  # No actions for data collection device
            )
            frame_id += 1

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        timing_stats["loop_times"].append(loop_s * 1000)

        if debug_timing:
            # Display timing info (single line, clear before print)
            print(
                f"\r\033[K🔍 obs: {total_obs_time * 1000:5.1f}ms | loop: {loop_s * 1000:5.1f}ms | target: {1000 / fps:.1f}ms | eff: {(1 / fps) / loop_s * 100:5.1f}%",
                end="",
                flush=True,
            )
        else:
            # Simple status line (single line with clear)
            camera_count = len(
                [k for k in obs.keys() if isinstance(obs.get(k), np.ndarray)]
            )
            print(
                f"\r\033[Ktime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | cameras: {camera_count}",
                end="",
                flush=True,
            )

        if duration is not None and time.perf_counter() - start >= duration:
            # Print final statistics before exiting
            if len(timing_stats["total_obs_times"]) > 10:
                print("\n=== FINAL TIMING REPORT ===")
                all_total = timing_stats["total_obs_times"]
                all_loops = timing_stats["loop_times"]

                print(f"Total samples: {len(all_total)}")
                print(f"Total obs - avg: {sum(all_total) / len(all_total):.2f}ms")
                print(f"Loop time - avg: {sum(all_loops) / len(all_loops):.2f}ms")
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    logger.info(pformat(asdict(cfg)))
    if cfg.dryrun:
        logger.warn(
            "⚠️  DRYRUN MODE ENABLED - Actions will be printed but NOT sent to robot"
        )
    # Note: xense_flare robot support removed
    # Check if this is Xense Multisensor (data collection device - no teleoperator needed)
    if cfg.robot.type == "xense_multisensor":
        logger.info("Detected Xense Multisensor data collection device")

        if cfg.display_data:
            # Use robot name in session name
            session_name = f"teleop_{cfg.robot.type}"
            init_rerun(session_name=session_name)

        robot = None

        try:
            # Create robot instance
            robot = make_robot_from_config(cfg.robot)

            # Connect to robot
            try:
                robot.connect()
                logger.info("✅ Xense Multisensor connected")
                logger.info(f"   Cameras: {list(robot.cameras.keys())}")
            except Exception as e:
                logger.error(
                    f"Failed to connect to Xense Multisensor: {e}\n{traceback.format_exc()}"
                )
                raise

            # Run data collection loop
            try:
                xense_multisensor_teleop_loop(
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                logger.info("Data collection interrupted by user")
            except Exception as e:
                logger.error(
                    f"Error during data collection: {e}\n{traceback.format_exc()}"
                )
                raise

        except Exception as e:
            logger.error(
                f"Error in Xense Multisensor setup: {e}\n{traceback.format_exc()}"
            )
        finally:
            # Safe disconnect
            if cfg.display_data:
                try:
                    rr.rerun_shutdown()
                except Exception as e:
                    logger.warn(f"Error shutting down rerun: {e}")

            if robot is not None:
                try:
                    if robot.is_connected:
                        robot.disconnect()
                        logger.info("✅ Xense Multisensor disconnected")
                except Exception as e:
                    logger.error(
                        f"Error disconnecting Xense Multisensor: {e}\n{traceback.format_exc()}"
                    )

    # Check if this is Flexiv Rizon4 robot with pico4
    elif cfg.robot.type == "flexiv_rizon4" and cfg.teleop.type == "pico4":
        logger.info(
            "Detected Flexiv Rizon4 robot with Pico4, using specialized teleop loop"
        )

        robot = None
        teleop = None

        try:
            # Create robot instance
            robot = make_robot_from_config(cfg.robot)

            # Ensure robot is in CARTESIAN_MOTION_FORCE mode for pico4 teleop
            from lerobot.robots.flexiv_rizon4.config_flexiv_rizon4 import ControlMode

            if robot.config.control_mode != ControlMode.CARTESIAN_MOTION_FORCE:
                raise ValueError(
                    f"Pico4 teleoperation requires CARTESIAN_MOTION_FORCE mode, "
                    f"but robot is configured with {robot.config.control_mode}"
                )

            # Connect to robot with error handling
            try:
                robot.connect(go_to_start=False)
                logger.info(f"Start EEF pose: {robot.get_current_tcp_pose_quat()}")
            except Exception as e:
                logger.error(
                    f"Failed to connect to robot: {e}\n{traceback.format_exc()}"
                )
                raise

            # Connect to teleoperator with error handling
            try:
                teleop = make_teleoperator_from_config(cfg.teleop)
                teleop.connect(current_tcp_pose_quat=robot.get_current_tcp_pose_quat())
                logger.info("Connected to Pico4")
            except Exception as e:
                logger.error(
                    f"Failed to connect to Pico4: {e}\n{traceback.format_exc()}"
                )
                raise

            # Run teleoperation loop
            try:
                pico4_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
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
            logger.error(f"Teleoperation failed\n{traceback.format_exc()}")
        finally:
            # Safe disconnect - ensure both robot and teleop are disconnected
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
                    # Force cleanup even if disconnect fails
                    try:
                        if hasattr(robot, "_robot") and robot._robot is not None:
                            robot._robot.Stop()
                    except Exception:
                        pass
    # Note: spacemouse, vive_tracker, and xense_flare teleoperator support removed
    else:
        teleop = make_teleoperator_from_config(cfg.teleop)
        robot = make_robot_from_config(cfg.robot)

        teleop.connect()
        robot.connect()

        try:
            teleop_loop(
                teleop=teleop,
                robot=robot,
                fps=cfg.fps,
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
    # Mock teleop is now available as a regular teleoperator
    register_third_party_devices()
    teleoperate()


if __name__ == "__main__":
    main()
