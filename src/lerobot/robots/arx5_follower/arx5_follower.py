#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import math
import time
from functools import cached_property
from typing import Any, Sequence

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_arx5_follower import ARX5FollowerConfig

# Import ARX5 SDK
import os
import sys

# Add ARX5 SDK path to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
arx5_sdk_path = os.path.join(current_dir, "..", "bi_arx5", "ARX5_SDK", "python")
if arx5_sdk_path not in sys.path:
    sys.path.insert(0, arx5_sdk_path)

try:
    import arx5_interface as arx5
except ImportError as e:
    if "LogLevel" in str(e) and "already registered" in str(e):
        # LogLevel already registered, try to get the existing module
        if "arx5_interface" in sys.modules:
            arx5 = sys.modules["arx5_interface"]
        else:
            raise e
    else:
        raise e


logger = logging.getLogger(__name__)


class ARX5Follower(Robot):
    """
    Single ARX5 Follower Arm using CAN bus control
    """

    config_class = ARX5FollowerConfig
    name = "arx5_follower"

    def __init__(self, config: ARX5FollowerConfig):
        super().__init__(config)
        self.config = config

        # Initialize arm when connect
        self.arm = None
        self._is_connected = False

        # Control mode state variables
        self._is_gravity_compensation_mode = False
        self._is_position_control_mode = False

        # Use configurable preview time for inference mode
        self.default_preview_time = (
            self.config.preview_time if self.config.inference_mode else 0.0
        )

        # RPC timeout
        self.rpc_timeout: float = getattr(config, "rpc_timeout", 5.0)

        # Pre-compute action keys for faster lookup
        self._joint_keys = [f"joint_{i+1}.pos" for i in range(6)]

        # Pre-allocate JointState command buffer
        self._cmd_buffer = None

        # Define home and start positions
        self._home_position = self.config.home_position
        self._start_position = self.config.start_position

        # Robot and controller configs
        self.robot_config = arx5.RobotConfigFactory.get_instance().get_config(
            config.arm_model
        )
        self.robot_config.gripper_open_readout = config.gripper_open_readout

        self.controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", self.robot_config.joint_dof
        )

        self.controller_config.controller_dt = config.controller_dt
        self.controller_config.default_preview_time = self.default_preview_time

        # Use multithreading by default
        if config.use_multithreading:
            self.controller_config.background_send_recv = True
        else:
            self.controller_config.background_send_recv = False

        self.cameras = make_cameras_from_configs(config.cameras)
        np.set_printoptions(precision=3, suppress=True)

    @property
    def _motors_ft(self) -> dict[str, type]:
        # ARX5 has 6 joints + 1 gripper
        joint_names = [f"joint_{i}" for i in range(1, 7)] + ["gripper"]
        return {f"{joint}.pos": float for joint in joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self.arm is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def is_gravity_compensation_mode(self) -> bool:
        """Check if robot is currently in gravity compensation mode"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._is_gravity_compensation_mode

    def is_position_control_mode(self) -> bool:
        """Check if robot is currently in position control mode"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._is_position_control_mode

    def connect(self, calibrate: bool = True, go_to_start: bool = False) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} already connected, do not run `robot.connect()` twice."
            )

        try:
            logger.info("Creating ARX5 arm controller...")
            self.arm = arx5.Arx5JointController(
                self.robot_config,
                self.controller_config,
                self.config.arm_port,
            )
            time.sleep(0.5)
            logger.info("✓ Arm controller created successfully")
            logger.info(
                f"preview_time: {self.controller_config.default_preview_time}"
            )
        except Exception as e:
            logger.error(f"Failed to create robot controller: {e}")
            self.arm = None
            raise e

        # Set log level
        self.set_log_level(self.config.log_level)

        # Reset to home using SDK method
        self.reset_to_home()

        # Default to normal position control mode
        # self.set_to_normal_position_control()

        # Connect cameras with delay between Xense cameras to avoid V4L2 timeout
        xense_camera_count = 0
        for cam_name, cam in self.cameras.items():
            if "tactile" in cam_name and xense_camera_count > 0:
                logger.info(f"Waiting 2s before connecting {cam_name} to avoid V4L2 timeout...")
                time.sleep(2.0)

            cam.connect()

            if "tactile" in cam_name:
                xense_camera_count += 1

        # Initialize command buffer
        self._cmd_buffer = arx5.JointState(self.robot_config.joint_dof)

        self._is_connected = True

        # Set default control mode to gravity compensation after connection
        self._is_gravity_compensation_mode = False
        self._is_position_control_mode = True

        # Run calibration if needed
        if not self.is_calibrated and calibrate:
            logger.info(
                "Calibration not found or mismatch, running calibration..."
            )
            self.calibrate()
            # After calibration, switch back to normal position control
            self.set_to_normal_position_control()

        # Go to start position (default: False for teleoperation)
        logger.info("ARX5 Follower Robot connected.")
        if go_to_start:
            self.smooth_go_start(duration=2.0)
            logger.info(
                "✓ Robot moved to start position, arm is now in gravity compensation mode"
            )
        else:
            logger.info(
                "Robot at current position, arm is now in gravity compensation mode"
            )

        gain = self.arm.get_gain()
        logger.info(
            f"Arm gain: {gain.kp()}, {gain.kd()}, {gain.gripper_kp}, {gain.gripper_kd}"
        )

        if self.config.inference_mode:
            self.set_to_normal_position_control()
            logger.info("✓ Robot is now in normal position control mode for inference")

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration exists and is valid"""
        return self.calibration is not None and len(self.calibration) > 0

    def calibrate(self) -> None:
        """
        Calibrate ARX5 follower by recording joint ranges.
        This is needed for proper action normalization when used as a teleoperation leader.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using existing calibration file associated with the id {self.id}")
                return

        logger.info(f"\nRunning calibration of {self}")
        logger.info("This will record the joint ranges for proper action normalization")

        # Switch to gravity compensation mode for manual movement
        logger.info("Switching to gravity compensation mode for calibration...")
        self.set_to_gravity_compensation_mode()

        # Get current joint positions as reference
        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        state = self.arm.get_joint_state()
        mid_positions = state.pos().copy()
        mid_gripper = state.gripper_pos

        logger.info(f"Middle position recorded: joints={mid_positions}, gripper={mid_gripper}")

        # Record range of motion
        print(
            "Move all joints sequentially through their entire ranges of motion.\n"
            "Recording positions. Press ENTER to stop..."
        )

        # Initialize min/max with current position
        range_mins = {f"joint_{i+1}": float(mid_positions[i]) for i in range(6)}
        range_maxes = {f"joint_{i+1}": float(mid_positions[i]) for i in range(6)}
        range_mins["gripper"] = mid_gripper
        range_maxes["gripper"] = mid_gripper

        # Record ranges
        import threading
        recording = threading.Event()
        recording.set()

        def record_loop():
            while recording.is_set():
                state = self.arm.get_joint_state()
                pos = state.pos().copy()
                gripper_pos = state.gripper_pos

                for i in range(6):
                    joint_name = f"joint_{i+1}"
                    range_mins[joint_name] = min(range_mins[joint_name], float(pos[i]))
                    range_maxes[joint_name] = max(range_maxes[joint_name], float(pos[i]))

                range_mins["gripper"] = min(range_mins["gripper"], gripper_pos)
                range_maxes["gripper"] = max(range_maxes["gripper"], gripper_pos)

                # Print current ranges (clear previous lines and reprint all)
                print("\r\033[K", end="")  # Clear current line
                for i in range(3):  # Clear 3 lines
                    print("\033[F\033[K", end="")

                # Print all joint ranges in a readable format
                print(
                    f"J1[{range_mins['joint_1']:6.3f}, {range_maxes['joint_1']:6.3f}]  "
                    f"J2[{range_mins['joint_2']:6.3f}, {range_maxes['joint_2']:6.3f}]  "
                    f"J3[{range_mins['joint_3']:6.3f}, {range_maxes['joint_3']:6.3f}]"
                )
                print(
                    f"J4[{range_mins['joint_4']:6.3f}, {range_maxes['joint_4']:6.3f}]  "
                    f"J5[{range_mins['joint_5']:6.3f}, {range_maxes['joint_5']:6.3f}]  "
                    f"J6[{range_mins['joint_6']:6.3f}, {range_maxes['joint_6']:6.3f}]"
                )
                print(
                    f"Gripper[{range_mins['gripper']:6.3f}, {range_maxes['gripper']:6.3f}]",
                    end="",
                    flush=True,
                )

                time.sleep(0.05)

        record_thread = threading.Thread(target=record_loop, daemon=True)
        record_thread.start()

        input()  # Wait for user to press ENTER
        recording.clear()
        record_thread.join()
        print()  # New line after recording

        # Store calibration
        self.calibration = {}
        for i in range(6):
            joint_name = f"joint_{i+1}"
            self.calibration[joint_name] = {
                "range_min": range_mins[joint_name],
                "range_max": range_maxes[joint_name],
            }
        self.calibration["gripper"] = {
            "range_min": range_mins["gripper"],
            "range_max": range_maxes["gripper"],
        }

        # Save calibration to file
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")
        logger.info("Recorded ranges:")
        for joint_name, calib in self.calibration.items():
            logger.info(
                f"  {joint_name}: [{calib['range_min']:.3f}, {calib['range_max']:.3f}]"
            )

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        """ARX5 motors use pre-configured IDs, no runtime setup needed"""
        logger.info(
            f"{self} ARX5 motors use pre-configured IDs, no runtime setup needed"
        )
        logger.info("Motor IDs are defined in the robot configuration:")
        logger.info("  - Joint motors: [1, 2, 4, 5, 6, 7]")
        logger.info("  - Gripper motor: 8")
        logger.info("Make sure your hardware matches these ID configurations")
        return

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Get joint state from arm
        joint_state = self.arm.get_joint_state()
        pos = joint_state.pos().copy()  # numpy array of joint positions (in radians)

        # Return raw radian values (no normalization)
        for i in range(6):  # 6 joints
            obs_dict[f"joint_{i+1}.pos"] = float(pos[i])
        obs_dict["gripper.pos"] = float(joint_state.gripper_pos)

        # Add camera observations
        camera_times = {}

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            image = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            obs_dict[cam_key] = image
            camera_times[cam_key] = dt_ms

        # Store camera timing info for debugging
        self.last_camera_times = camera_times

        return obs_dict

    def send_action(self, action: dict[str, Any], normalize: bool = True) -> dict[str, Any]:
        """Send action to the robot arm.

        Args:
            action: Dictionary containing joint positions and gripper position.
            normalize: If True, unnormalize action values from [-100, 100] to radians.
                      If False, assume action values are already in radians.

        Returns:
            The input action dictionary.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Use pre-allocated JointState object
        cmd = self._cmd_buffer
        cmd_pos = cmd.pos()

        # Process joint actions
        for i, key in enumerate(self._joint_keys):
            if key in action:
                val = action[key]
                if normalize and self.calibration:
                    # Unnormalize: [-100, 100] -> [range_min, range_max] radians
                    joint_name = f"joint_{i+1}"
                    calib = self.calibration[joint_name]
                    # Convert from [-100, 100] to [0, 1] then to [min, max]
                    ratio = (val + 100) / 200.0
                    cmd_pos[i] = calib.range_min + ratio * (calib.range_max - calib.range_min)
                else:
                    # No normalization, assume value is already in radians
                    cmd_pos[i] = val

        # Process gripper action
        if "gripper.pos" in action:
            val = action["gripper.pos"]
            if normalize and self.calibration:
                # Gripper uses RANGE_0_100, so convert [0, 100] to [min, max]
                calib = self.calibration["gripper"]
                ratio = val / 100.0
                cmd.gripper_pos = calib.range_min + ratio * (calib.range_max - calib.range_min)
            else:
                # No normalization, assume value is already in radians
                cmd.gripper_pos = val

        print(f"cmd: {cmd.pos()}, {cmd.gripper_pos}")
        print(f"action: {action['joint_1.pos']}, {action['joint_2.pos']}, {action['joint_3.pos']}, {action['joint_4.pos']}, {action['joint_5.pos']}, {action['joint_6.pos']}, {action['gripper.pos']}")
        # self.arm.set_joint_cmd(cmd)

        # Simply return the input action
        return action

    @staticmethod
    def _ease_in_out_quad(t: float) -> float:
        """Smooth easing function used for joint interpolation."""
        tt = t * 2.0
        if tt < 1.0:
            return (tt * tt) / 2.0
        tt -= 1.0
        return -(tt * (tt - 2.0) - 1.0) / 2.0

    def move_joint_trajectory(
        self,
        target_joint_poses: Sequence[float] | Sequence[Sequence[float]],
        durations: float | Sequence[float],
        *,
        easing: str = "ease_in_out_quad",
        steps_per_segment: int | None = None,
    ) -> None:
        """Move arm smoothly towards the provided joint targets.

        Args:
            target_joint_poses: A sequence of 6 or 7 joint values (including gripper)
                or a sequence of such sequences to execute multiple segments.
            durations: Duration in seconds for the corresponding target poses.
            easing: Easing profile to apply ("ease_in_out_quad" or "linear").
            steps_per_segment: Optional fixed number of interpolation steps per
                segment. When omitted the controller's ``controller_dt`` is used
                to compute the number of steps from the duration.

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
            ValueError: If inputs are malformed.
        """

        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Normalize inputs to lists
        if isinstance(target_joint_poses[0], (int, float)):
            trajectory = [target_joint_poses]
        else:
            trajectory = list(target_joint_poses)

        if isinstance(durations, (int, float)):
            segment_durations = [float(durations)]
        else:
            segment_durations = [float(d) for d in durations]

        if len(trajectory) != len(segment_durations):
            raise ValueError(
                "target_joint_poses and durations must have the same length"
            )

        # Determine controller timestep
        controller_dt = getattr(self.config, "interpolation_controller_dt", 0.01)

        # Fetch current joint positions as starting state
        def _get_current_state() -> np.ndarray:
            state = self.arm.get_joint_state()
            return np.concatenate(
                (state.pos().copy(), np.array([state.gripper_pos]))
            )

        current = _get_current_state()

        def _parse_target(segment: Sequence[float], default: np.ndarray) -> np.ndarray:
            arr = np.asarray(segment, dtype=float)
            if arr.shape[0] not in (6, 7):
                raise ValueError(
                    "Target must provide 6 joint values (+ optional gripper)"
                )
            if arr.shape[0] == 6:
                arr = np.concatenate((arr, np.array([default[-1]])))
            return arr

        def _apply_easing(alpha: float) -> float:
            alpha = max(0.0, min(1.0, alpha))
            if easing == "ease_in_out_quad":
                return self._ease_in_out_quad(alpha)
            if easing == "linear":
                return alpha
            raise ValueError(f"Unsupported easing profile: {easing}")

        try:
            for segment, duration in zip(trajectory, segment_durations, strict=True):
                target = _parse_target(segment, current)

                if duration <= 0:
                    action = {}
                    for i in range(6):
                        action[f"joint_{i+1}.pos"] = float(target[i])
                    action["gripper.pos"] = float(target[6])
                    self.send_action(action)
                    current = target
                    continue

                steps = (
                    steps_per_segment
                    if steps_per_segment is not None
                    else max(1, int(math.ceil(duration / controller_dt)))
                )

                for step in range(1, steps + 1):
                    progress = step / steps
                    ratio = _apply_easing(progress)
                    interp = current + (target - current) * ratio

                    action = {}
                    for i in range(6):
                        action[f"joint_{i+1}.pos"] = float(interp[i])
                    action["gripper.pos"] = float(interp[6])

                    self.send_action(action)
                    time.sleep(duration / steps if steps_per_segment else controller_dt)

                current = target
        except KeyboardInterrupt:
            logger.warning(
                "Joint trajectory interrupted by user. Holding current pose."
            )

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Reset to home and set to damping mode
        try:
            logger.info("Disconnecting arm...")
            self.arm.reset_to_home()
            self.arm.set_to_damping()
            logger.info("✓ Arm disconnected successfully")
        except Exception as e:
            logger.warning(f"Arm disconnect failed: {e}")

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        # Destroy arm object
        self.arm = None

        self._is_connected = False

        logger.info(f"{self} disconnected.")

    def set_log_level(self, level: str):
        """Set robot log level

        Args:
            level: Log level string, supports: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL, OFF
        """
        log_level_map = {
            "TRACE": arx5.LogLevel.TRACE,
            "DEBUG": arx5.LogLevel.DEBUG,
            "INFO": arx5.LogLevel.INFO,
            "WARNING": arx5.LogLevel.WARNING,
            "ERROR": arx5.LogLevel.ERROR,
            "CRITICAL": arx5.LogLevel.CRITICAL,
            "OFF": arx5.LogLevel.OFF,
        }

        if level.upper() not in log_level_map:
            raise ValueError(
                f"Invalid log level: {level}. Supported levels: {list(log_level_map.keys())}"
            )

        log_level = log_level_map[level.upper()]

        if self.arm is not None:
            self.arm.set_log_level(log_level)

    def reset_to_home(self):
        """Reset arm to home position"""
        if self.arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.arm.reset_to_home()
        logger.info("Arm reset to home position.")

    def set_to_gravity_compensation_mode(self):
        """Switch from normal position control to gravity compensation mode"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Switching to gravity compensation mode...")

        zero_grav_gain = arx5.Gain(self.robot_config.joint_dof)
        zero_grav_gain.kp()[:] = 0.0
        zero_grav_gain.kd()[:] = self.controller_config.default_kd * 0.1
        zero_grav_gain.gripper_kp = 0.0
        zero_grav_gain.gripper_kd = self.controller_config.default_gripper_kd * 0.1

        self.arm.set_gain(zero_grav_gain)

        # Update control mode state
        self._is_gravity_compensation_mode = True
        self._is_position_control_mode = False

        logger.info("✓ Arm is now in gravity compensation mode")

    def set_to_normal_position_control(self):
        """Switch from gravity compensation to normal position control mode"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Switching to normal position control mode...")

        default_gain = self.arm.get_gain()
        default_gain.kp()[:] = self.controller_config.default_kp * 0.4
        default_gain.kd()[:] = self.controller_config.default_kd * 1.2

        if self.config.inference_mode:
            default_gain.kp()[-3:] *= 1.5
            default_gain.kd()[-3:] *= 0.5

        default_gain.gripper_kp = self.controller_config.default_gripper_kp
        default_gain.gripper_kd = self.controller_config.default_gripper_kd

        self.arm.set_gain(default_gain)

        # Update control mode state
        self._is_gravity_compensation_mode = False
        self._is_position_control_mode = True

        logger.info("✓ Arm is now in normal position control mode")

    def smooth_go_start(
        self, duration: float = 2.0, easing: str = "ease_in_out_quad"
    ) -> None:
        """
        Smoothly move arm to the start position using trajectory interpolation.

        Args:
            duration: Duration in seconds for the movement (default: 2.0)
            easing: Easing profile to apply ("ease_in_out_quad" or "linear")

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info(f"Smoothly going to start position over {duration:.1f} seconds...")

        # First, set current position as target to avoid large position error
        state = self.arm.get_joint_state()

        current_cmd = arx5.JointState(self.robot_config.joint_dof)
        current_cmd.pos()[:] = state.pos()
        current_cmd.gripper_pos = state.gripper_pos

        self.arm.set_joint_cmd(current_cmd)

        # Now safe to switch to normal position control
        self.set_to_normal_position_control()

        # Execute smooth trajectory to start position
        self.move_joint_trajectory(
            target_joint_poses=self._start_position.copy(),
            durations=duration,
            easing=easing
        )

        # Switch back to gravity compensation mode
        self.set_to_gravity_compensation_mode()

        logger.info(
            "✓ Successfully moved to start position and switched to gravity compensation mode"
        )

    def smooth_go_home(
        self, duration: float = 2.0, easing: str = "ease_in_out_quad"
    ) -> None:
        """
        Smoothly move arm to the home position using trajectory interpolation.

        Args:
            duration: Duration in seconds for the movement (default: 2.0)
            easing: Easing profile to apply ("ease_in_out_quad" or "linear")

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info(
            f"Smoothly returning to home position over {duration:.1f} seconds..."
        )

        # First, set current position as target
        state = self.arm.get_joint_state()

        current_cmd = arx5.JointState(self.robot_config.joint_dof)
        current_cmd.pos()[:] = state.pos()
        current_cmd.gripper_pos = state.gripper_pos

        self.arm.set_joint_cmd(current_cmd)

        # Switch to normal position control
        self.set_to_normal_position_control()

        # Execute smooth trajectory to home position
        self.move_joint_trajectory(
            target_joint_poses=self._home_position.copy(),
            durations=duration,
            easing=easing
        )

        # Switch back to gravity compensation mode
        self.set_to_gravity_compensation_mode()

        logger.info(
            "✓ Successfully returned to home position and switched to gravity compensation mode"
        )
