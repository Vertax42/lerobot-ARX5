#!/usr/bin/env python

# Copyright 2025 The XenseRobotics Inc. team. All rights reserved.
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
3D Spacemouse teleoperator for end-effector control.

This teleoperator provides 6-DoF absolute pose control (translation + rotation)
and gripper control via buttons. It outputs accumulated target_pose_6d that can
be directly sent to a Cartesian controller (e.g., Arx5CartesianController).

The output format matches the legacy ARX5 SDK spacemouse example:
- target_pose_6d: [x, y, z, roll, pitch, yaw] - absolute EEF pose
- gripper_pos: absolute gripper position in meters

This implementation uses the modern `pyspacemouse` backend.

On Linux, if opening the device fails, see ``README.md`` in this package for ``hidraw``/udev setup.
"""

import time
from queue import Queue
from typing import Any

import numpy as np

from lerobot.teleoperators.spacemouse.config_spacemouse import SpacemouseConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import (
    euler_to_quaternion,
    get_logger,
    normalize_quaternion,
    quaternion_to_rotation_6d,
)

# Appended to errors when pyspacemouse/easyhid cannot open the HID device (common on Linux).
_SPACEMOUSE_HID_OPEN_HINTS = """
Linux HID troubleshooting (Failed to open device):
  • Quit other programs using the SpaceMouse: 3DxWare / 3Dconnexion driver, spacenavd,
    Blender/other apps with SpaceMouse support, or a second lerobot-teleoperate instance.
  • If you see "UniversalReceiver" in the log, a Logitech receiver stack may be involved;
    try unplugging other Logitech dongles or disabling their daemon temporarily.
  • Permissions: install udev rules for your device (3Dconnexion / SpaceMouse VID:PID),
    add your user to plugdev (or the group owning /dev/hidraw*), run
    `sudo udevadm control --reload-rules && sudo udevadm trigger`, then replug USB.
  • Quick checks: `ls -l /dev/hidraw*` (readable?) and `sudo fuser -v /dev/hidraw*`
    to see which process holds the device.
"""


class SpacemouseTeleop(Teleoperator):
    """
    3D Spacemouse teleoperator for end-effector control.

    This teleoperator reads 6-DoF motion data from a 3Dconnexion SpaceMouse
    and maintains an accumulated target_pose_6d suitable for Cartesian control
    of robotic arms.

    The spacemouse provides:
    - 6-DoF motion: (dx, dy, dz, drx, dry, drz) - delta position and rotation
    - 2 buttons: typically used for gripper open/close or control events

    Output action format (matches ARX5 SDK spacemouse_teleop.py):
    - x, y, z, roll, pitch, yaw: absolute EEF target pose (accumulated)
    - gripper_pos: absolute gripper position in meters

    Usage:
    1. Call set_target_pose() to initialize with robot's current EEF pose
    2. Call get_action() to get updated target_pose_6d based on spacemouse input
    """

    config_class = SpacemouseConfig
    name = "spacemouse"

    def __init__(self, config: SpacemouseConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger("SpacemouseTeleop")
        self._is_connected = False
        self._spacemouse = None
        self._start_pose_6d: np.ndarray = np.zeros(6, dtype=np.float32)
        self._start_gripper_pos: float = 0.0
        # Smoothing filter queue (moving average)
        self._motion_queue: Queue = Queue(self.config.filter_window_size)

        # State tracking
        self._enabled: bool = False

        # Event tracking for teleop events
        self._both_buttons_pressed_time: float | None = None
        self._reset_triggered: bool = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Spacemouse doesn't require calibration."""
        return True

    @property
    def action_features(self) -> dict:
        """
        Return action features matching ARX5 SDK's target_pose_6d format.

        Returns a dictionary with dtype, shape, and names for the action space:
        - x, y, z: absolute EEF position (meters)
        - roll, pitch, yaw: absolute EEF orientation (radians)
        - gripper_pos: absolute gripper position (meters)
        
        For dual-hand mode, the output combines both devices according to their enabled_axes configuration.
        """
        if self.config.multi_device_mode:
            # In dual-hand mode, we still output the same unified format
            # but internally combine inputs from left (position) and right (orientation) devices
            return {
                "dtype": "float32",
                "shape": (7,),
                "names": {
                    "x": 0,        # from left device (if enabled)
                    "y": 1,        # from left device (if enabled) 
                    "z": 2,        # from left device (if enabled)
                    "roll": 3,     # from right device (if enabled)
                    "pitch": 4,    # from right device (if enabled)
                    "yaw": 5,      # from right device (if enabled)
                    "gripper_pos": 6,  # from either device (buttons combined)
                },
                "_mode": "dual_hand",
                "_left_axes": self.config.left_device.enabled_axes,
                "_right_axes": self.config.right_device.enabled_axes,
            }
        else:
            return {
                "dtype": "float32",
                "shape": (7,),
                "names": {
                    "x": 0,
                    "y": 1,
                    "z": 2,
                    "roll": 3,
                    "pitch": 4,
                    "yaw": 5,
                    "gripper_pos": 6,
                },
                "_mode": "single_device",
            }

    @property
    def feedback_features(self) -> dict[str, type]:
        """Spacemouse doesn't support feedback."""
        return {}

    def connect(
        self, calibrate: bool = True, current_tcp_pose_euler: np.ndarray = np.zeros(7, dtype=np.float32)
    ) -> None:
        """Connect to the 3D Spacemouse."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.logger.info("Connecting to 3D Spacemouse...")

        # Lazy import to avoid requiring pyspacemouse when module is loaded but not used
        try:
            from lerobot.teleoperators.spacemouse.peripherals import Spacemouse
        except ImportError as e:
            raise ImportError(
                "pyspacemouse is required for Spacemouse teleoperator. "
                "Install it with: pip install pyspacemouse"
            ) from e

        try:
            # Pass multi-device configuration if enabled
            if self.config.multi_device_mode:
                self._spacemouse = Spacemouse(
                    multi_device_mode=True,
                    left_device_config=self.config.left_device,
                    right_device_config=self.config.right_device,
                )
            else:
                self._spacemouse = Spacemouse()

            self._spacemouse.connect()
            self._is_connected = True

            # Set target pose on connect and save initial pose for reset
            self._target_pose_6d = current_tcp_pose_euler[:6].copy()
            self._target_gripper_pos = current_tcp_pose_euler[6]
            # Save initial pose for reset functionality
            self._start_pose_6d = current_tcp_pose_euler[:6].copy()
            self._start_gripper_pos = current_tcp_pose_euler[6]

            self.logger.info("✅ 3D Spacemouse connected successfully.")

        except Exception as e:
            err = str(e).lower()
            hint = (
                _SPACEMOUSE_HID_OPEN_HINTS
                if ("open device" in err or "hid" in err or "easyhid" in err)
                else ""
            )
            raise RuntimeError(f"❌ Failed to connect to 3D Spacemouse: {e}{hint}") from e

    def calibrate(self) -> None:
        """No calibration needed for spacemouse."""
        pass

    def configure(self) -> None:
        """No additional configuration needed."""
        pass

    def reset_to_pose(self, pose_6d: np.ndarray, gripper_pos: float = 0.0) -> None:
        """
        Reset target pose to a specific pose (e.g., home pose).

        Args:
            pose_6d: 6D EEF pose [x, y, z, roll, pitch, yaw]
            gripper_pos: Gripper position in meters
        """
        self._target_pose_6d = np.asarray(pose_6d, dtype=np.float32).copy()
        self._target_gripper_pos = gripper_pos
        self.logger.info(f"✅ Reset target pose to: {pose_6d}, gripper: {gripper_pos}")

    def _get_filtered_state(self) -> np.ndarray:
        """Get filtered spacemouse state with moving average.
        
        Note: This uses cached data from the last poll() call.
        """
        raw_state = self._spacemouse.get_motion_state_transformed()

        # Apply additional deadzone filtering after transformation
        positive_idx = raw_state >= self.config.deadzone
        negative_idx = raw_state <= -self.config.deadzone
        filtered_state = np.zeros_like(raw_state)
        filtered_state[positive_idx] = (raw_state[positive_idx] - self.config.deadzone) / (
            1 - self.config.deadzone
        )
        filtered_state[negative_idx] = (raw_state[negative_idx] + self.config.deadzone) / (
            1 - self.config.deadzone
        )

        # Apply axis inversion
        invert = np.where(self.config.invert_axes, -1.0, 1.0)
        filtered_state *= invert

        # Moving average filter Use public method of queue to avoid race condition
        if self._motion_queue.full():
            self._motion_queue.get()
        self._motion_queue.put(filtered_state)

        return np.mean(np.array(list(self._motion_queue.queue)), axis=0)

    def get_action(self) -> dict[str, Any]:
        """
        Get the current target pose from the Spacemouse.

        Returns a dictionary with absolute EEF pose (matching ARX5 SDK format):
        - x, y, z: absolute EEF position (meters)
        - roll, pitch, yaw: absolute EEF orientation (radians)
        - gripper_pos: absolute gripper position (meters)
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Poll device data once per cycle (ensures motion & button data are synchronized)
        self._spacemouse.poll()

        # Use fixed control_dt for consistent velocity scaling
        # This should match the external control loop period (e.g., 1/fps)
        dt = self.config.control_dt

        # Get filtered motion state (uses cached data from poll())
        state = self._get_filtered_state()  # (6,) normalized [-1, 1]

        # Get button states (uses cached data from poll())
        # Use device-aware methods that handle different button layouts
        button_left = self._spacemouse.is_left_button_pressed()
        button_right = self._spacemouse.is_right_button_pressed()

        # Compute gripper command based on buttons
        if self.config.swap_gripper_buttons:
            button_open, button_close = button_left, button_right
        else:
            button_open, button_close = button_right, button_left

        if button_open and not button_close:
            gripper_cmd = 1  # Open
        elif button_close and not button_open:
            gripper_cmd = -1  # Close
        else:
            gripper_cmd = 0  # Stay

        # Update target pose with increments, using device-specific sensitivities in multi-device mode
        if self.config.multi_device_mode:
            # Apply sensitivities based on which device controls each axis
            for i in range(3):  # Position axes (x, y, z)
                if self.config.left_device.enabled_axes[i]:
                    self._target_pose_6d[i] += state[i] * self.config.left_device.pos_sensitivity * dt
                elif self.config.right_device.enabled_axes[i]:
                    self._target_pose_6d[i] += state[i] * self.config.right_device.pos_sensitivity * dt
            
            for i in range(3, 6):  # Orientation axes (roll, pitch, yaw)
                if self.config.left_device.enabled_axes[i]:
                    self._target_pose_6d[i] += state[i] * self.config.left_device.ori_sensitivity * dt
                elif self.config.right_device.enabled_axes[i]:
                    self._target_pose_6d[i] += state[i] * self.config.right_device.ori_sensitivity * dt
        else:
            # Single device mode (original behavior)
            self._target_pose_6d[:3] += state[:3] * self.config.pos_sensitivity * dt
            self._target_pose_6d[3:] += state[3:] * self.config.ori_sensitivity * dt

        # Update gripper position with clamping
        if self.config.multi_device_mode:
            # Use gripper speed from whichever device has buttons pressed (or default)
            gripper_speed = max(self.config.left_device.gripper_speed, self.config.right_device.gripper_speed)
        else:
            gripper_speed = self.config.gripper_speed
            
        self._target_gripper_pos += gripper_cmd * gripper_speed * dt
        self._target_gripper_pos = np.clip(self._target_gripper_pos, 0, self.config.gripper_width)

        # Check if any input is active
        motion_active = np.any(np.abs(state) > 0.01)
        self._enabled = motion_active or button_left or button_right

        # Return absolute pose dict
        return {
            "x": self._target_pose_6d[0],
            "y": self._target_pose_6d[1],
            "z": self._target_pose_6d[2],
            "roll": self._target_pose_6d[3],
            "pitch": self._target_pose_6d[4],
            "yaw": self._target_pose_6d[5],
            "gripper_pos": self._target_gripper_pos,
        }

    def get_target_pose_array(self) -> tuple[np.ndarray, float]:
        """
        Get the current target pose as numpy array (for direct use with ARX5 SDK).

        Returns:
            Tuple of (pose_6d, gripper_pos) where pose_6d is [x, y, z, roll, pitch, yaw]
        """
        return self._target_pose_6d.copy(), self._target_gripper_pos

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the spacemouse such as intervention status,
        episode termination, success indicators, etc. Mainly used for HIL-SERL integration.

        Spacemouse button mappings:
        - Any motion or button pressed = intervention active
        - Both buttons pressed together for 1s = reset/rerecord episode
        - Left button only = close gripper (normal operation)
        - Right button only = open gripper (normal operation)

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        
        Note: This uses cached button state from the last poll() call.
              Make sure to call get_action() first to ensure data is fresh.
        """
        if not self._is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Get current button states (uses cached data from last poll())
        # Use device-aware methods that handle different button layouts
        button_left = self._spacemouse.is_left_button_pressed()
        button_right = self._spacemouse.is_right_button_pressed()

        # Check if both buttons are pressed (for reset)
        both_pressed = button_left and button_right
        current_time = time.monotonic()

        terminate_episode = False
        rerecord_episode = False

        if both_pressed:
            if self._both_buttons_pressed_time is None:
                self._both_buttons_pressed_time = current_time
            elif (current_time - self._both_buttons_pressed_time) > 1.0 and not self._reset_triggered:
                # Both buttons held for 1 second - trigger reset
                terminate_episode = True
                rerecord_episode = True
                self._reset_triggered = True
                self.logger.info("Both buttons held - triggering episode reset")
        else:
            self._both_buttons_pressed_time = None
            self._reset_triggered = False

        return {
            TeleopEvents.IS_INTERVENTION: self._enabled,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: False,  # No success signal from spacemouse
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Spacemouse doesn't support feedback."""
        raise NotImplementedError("Feedback is not supported for Spacemouse teleoperator.")

    def disconnect(self) -> None:
        """Disconnect from the Spacemouse."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info("Disconnecting from Spacemouse...")

        if self._spacemouse is not None:
            self._spacemouse.disconnect()
            self._spacemouse = None

        self._is_connected = False
        self.logger.info(f"{self} disconnected.")

    def convert_to_flexiv_action(self, spacemouse_action: dict[str, Any]) -> dict[str, Any]:
        """Convert spacemouse action (Euler angles) to Flexiv Rizon4 action (6D rotation).

        This matches the behavior of spacemouse_teleop.py example:
        - Spacemouse maintains absolute pose in Euler angles [x, y, z, roll, pitch, yaw]
        - Convert to 6D rotation format [x, y, z, r1-r6] for Flexiv robot

        Args:
            spacemouse_action: Dictionary with keys {x, y, z, roll, pitch, yaw, gripper_pos}
        Returns:
            Dictionary with keys {tcp.x, tcp.y, tcp.z, tcp.r1-r6, gripper.pos}
        """
        # Convert Euler angles to quaternion first
        quat = euler_to_quaternion(
            spacemouse_action["roll"],
            spacemouse_action["pitch"],
            spacemouse_action["yaw"],
        )  # Returns np.ndarray [qw, qx, qy, qz]

        # Normalize quaternion to ensure unit length
        quat = normalize_quaternion(quat, input_format="wxyz")

        # Convert quaternion to 6D rotation representation
        r6d = quaternion_to_rotation_6d(quat[0], quat[1], quat[2], quat[3])

        # Map to Flexiv action format with 6D rotation
        return {
            "tcp.x": spacemouse_action["x"],
            "tcp.y": spacemouse_action["y"],
            "tcp.z": spacemouse_action["z"],
            "tcp.r1": r6d[0],
            "tcp.r2": r6d[1],
            "tcp.r3": r6d[2],
            "tcp.r4": r6d[3],
            "tcp.r5": r6d[4],
            "tcp.r6": r6d[5],
            "gripper.pos": spacemouse_action["gripper_pos"],
        }

    def __del__(self):
        """Cleanup on deletion."""
        if self._is_connected:
            try:
                self.disconnect()
            except Exception as e:
                self.logger.error(f"Failed to disconnect from Spacemouse: {e}")
