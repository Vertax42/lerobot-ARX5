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

"""PySpaceMouse wrapper for lerobot integration.

This module provides a simple wrapper around the PySpaceMouse library,
with proper state caching to ensure motion and button data are synchronized.

Key Design:
- Call poll() once per control cycle to read all device data
- Then use get_motion_state() and get_button_state() to access cached data
- This ensures motion and button states come from the same HID report
"""

import numpy as np

try:
    import pyspacemouse
except ImportError:
    pyspacemouse = None


class Spacemouse:
    """Wrapper for PySpaceMouse device(s).

    This class provides a synchronized interface for reading SpaceMouse data,
    ensuring that motion and button states are always from the same HID read.

    Usage:
        with Spacemouse() as sm:
            while True:
                sm.poll()  # Read all data once per cycle
                motion = sm.get_motion_state_transformed()
                buttons = sm.get_button_state()

    Args:
        device_name: Device name to open (e.g., "SpaceNavigator"). None for auto-detect.
        device_index: Which device instance to open if multiple connected (0-based).
        dtype: Data type for returned arrays.
        multi_device_mode: If True, use dual SpaceMouse setup.
        left_device_config: Configuration for left hand device.
        right_device_config: Configuration for right hand device.
        invert_axes: List of axis names to invert (e.g., ["x", "roll"]).
    """

    def __init__(
        self,
        device_name: str = None,
        device_index: int = 0,
        dtype=np.float32,
        multi_device_mode: bool = False,
        left_device_config=None,
        right_device_config=None,
        invert_axes: list = None,
    ):
        """Initialize SpaceMouse wrapper."""
        if pyspacemouse is None:
            raise ImportError(
                "pyspacemouse is required. Install it with: pip install pyspacemouse"
            )

        self.device_name = device_name
        self.device_index = device_index
        self.dtype = dtype
        self.multi_device_mode = multi_device_mode
        self.left_device_config = left_device_config
        self.right_device_config = right_device_config
        self.invert_axes = invert_axes or []

        # Device instances (SpaceMouseDevice from pyspacemouse)
        self._device = None
        self._left_device = None
        self._right_device = None
        self._is_connected = False

        # Cached state from last poll() - ensures motion & buttons are synced
        self._state: pyspacemouse.SpaceMouseState = None
        self._left_state: pyspacemouse.SpaceMouseState = None
        self._right_state: pyspacemouse.SpaceMouseState = None

        # Custom device spec (if invert_axes is specified)
        self._device_spec = None

        # Button index mapping (populated on connect)
        # Maps logical names to actual button indices for the connected device
        self._left_button_idx: int = 0
        self._right_button_idx: int = 1

        # Position transformation matrix (SpaceMouse -> Robot coordinates)
        # Pure axis remapping WITHOUT sign inversion (inversion handled by config.invert_axes)
        # SpaceMouse axes: Y=forward/back, X=left/right, Z=up/down
        # Robot axes: X=forward/back, Y=left/right, Z=up/down
        # 
        # Mapping: new_x=old_y, new_y=old_x, new_z=old_z
        self.tx_pos = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=dtype)
        
        # Rotation transformation matrix (SpaceMouse -> Robot coordinates)
        # Pure axis mapping: roll->roll, pitch->pitch, yaw->yaw (no remapping needed)
        # Sign inversion handled by config.invert_axes
        self.tx_rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def connect(self):
        """Connect to SpaceMouse device(s)."""
        if self._is_connected:
            return

        # Prepare custom device spec if axis inversion is needed
        if self.invert_axes:
            self._prepare_device_spec()

        if self.multi_device_mode:
            self._connect_dual()
        else:
            self._connect_single()

        self._is_connected = True

    def _prepare_device_spec(self):
        """Prepare custom device spec with axis inversions."""
        # First, detect what device we're dealing with
        connected = pyspacemouse.get_connected_devices()
        if not connected:
            raise RuntimeError("No SpaceMouse devices found")

        device_name = self.device_name or connected[0]
        specs = pyspacemouse.get_device_specs()

        if device_name not in specs:
            raise ValueError(f"Unknown device: {device_name}. Available: {list(specs.keys())}")

        base_spec = specs[device_name]
        self._device_spec = pyspacemouse.modify_device_info(
            base_spec,
            invert_axes=self.invert_axes,
        )

    def _connect_single(self):
        """Connect to single SpaceMouse device."""
        open_kwargs = {
            "device_index": self.device_index,
            "nonblocking": True,
        }

        if self.device_name is not None:
            open_kwargs["device"] = self.device_name

        if self._device_spec is not None:
            open_kwargs["device_spec"] = self._device_spec

        self._device = pyspacemouse.open(**open_kwargs)
        print(f"Connected to SpaceMouse: {self._device.name}")
        
        # Detect correct button indices for this device
        self._detect_button_indices(self._device)

    def _connect_dual(self):
        """Connect to dual SpaceMouse devices."""
        if self.left_device_config is None or self.right_device_config is None:
            raise ValueError("Dual device mode requires left_device_config and right_device_config")

        # Get common device name
        connected = pyspacemouse.get_connected_devices()
        if len(connected) < 2:
            raise RuntimeError(
                f"Dual device mode requires 2 SpaceMouse devices, found {len(connected)}: {connected}"
            )

        device_name = self.device_name or connected[0]

        # Open left device
        left_kwargs = {
            "device": device_name,
            "device_index": self.left_device_config.device_index,
            "nonblocking": True,
        }
        if self._device_spec is not None:
            left_kwargs["device_spec"] = self._device_spec

        self._left_device = pyspacemouse.open(**left_kwargs)

        # Open right device
        right_kwargs = {
            "device": device_name,
            "device_index": self.right_device_config.device_index,
            "nonblocking": True,
        }
        if self._device_spec is not None:
            right_kwargs["device_spec"] = self._device_spec

        self._right_device = pyspacemouse.open(**right_kwargs)

        print(f"Connected to dual SpaceMouse devices:")
        print(f"  Left (index {self.left_device_config.device_index}):  {self._left_device.name}")
        print(f"  Right (index {self.right_device_config.device_index}): {self._right_device.name}")
        
        # Detect correct button indices (use left device as reference)
        self._detect_button_indices(self._left_device)

    def _detect_button_indices(self, device):
        """Detect the correct button indices for physical left/right buttons.
        
        Different SpaceMouse devices have different button layouts:
        - SpaceNavigator/Compact/Wireless: LEFT=0, RIGHT=1
        - Pro/Enterprise/UniversalReceiver: MENU=0 (physical left), FIT=last (physical right)
        
        This method finds the correct indices based on button names.
        """
        button_names = device.info.button_names
        
        if not button_names:
            # No buttons on this device
            self._left_button_idx = -1
            self._right_button_idx = -1
            print("  No buttons detected on device")
            return
        
        # Try to find LEFT/RIGHT buttons first (SpaceNavigator style)
        left_idx = None
        right_idx = None
        
        for i, name in enumerate(button_names):
            name_upper = name.upper()
            if name_upper == "LEFT":
                left_idx = i
            elif name_upper == "RIGHT":
                right_idx = i
        
        if left_idx is not None and right_idx is not None:
            self._left_button_idx = left_idx
            self._right_button_idx = right_idx
            print(f"  Button mapping: LEFT={left_idx}, RIGHT={right_idx}")
            return
        
        # For Pro-style devices, MENU is physical left, FIT is physical right
        menu_idx = None
        fit_idx = None
        
        for i, name in enumerate(button_names):
            name_upper = name.upper()
            if name_upper == "MENU":
                menu_idx = i
            elif name_upper == "FIT":
                fit_idx = i
        
        if menu_idx is not None and fit_idx is not None:
            self._left_button_idx = menu_idx
            self._right_button_idx = fit_idx
            print(f"  Button mapping: MENU(left)={menu_idx}, FIT(right)={fit_idx}")
            return
        
        # Fallback: use first and second buttons
        self._left_button_idx = 0
        self._right_button_idx = 1 if len(button_names) > 1 else 0
        print(f"  Button mapping (fallback): idx 0 and 1")

    def disconnect(self):
        """Disconnect from SpaceMouse device(s)."""
        if not self._is_connected:
            return

        if self._device is not None:
            self._device.close()
            self._device = None

        if self._left_device is not None:
            self._left_device.close()
            self._left_device = None

        if self._right_device is not None:
            self._right_device.close()
            self._right_device = None

        self._is_connected = False
        print("SpaceMouse disconnected")

    # =========================================================================
    # Core API: poll() then get_*() for synchronized data access
    # =========================================================================

    def poll(self):
        """Read device data once per control cycle.

        This method should be called ONCE at the beginning of each control loop.
        All subsequent get_motion_state() and get_button_state() calls will use
        this cached data, ensuring synchronization.
        
        Note: SpaceMouse sends different HID reports for axes (channel 1/2) and 
        buttons (channel 3). In nonblocking mode, each read() only gets one report.
        We need to drain the buffer to ensure we have the latest state for all channels.
        """
        if not self._is_connected:
            raise RuntimeError("SpaceMouse not connected")

        if self.multi_device_mode:
            # Drain HID buffer for left device
            self._left_state = self._drain_device_buffer(self._left_device)
            # Drain HID buffer for right device  
            self._right_state = self._drain_device_buffer(self._right_device)
        else:
            # Drain HID buffer for single device
            self._state = self._drain_device_buffer(self._device)

    def _drain_device_buffer(self, device) -> "pyspacemouse.SpaceMouseState":
        """Read all pending HID reports from device buffer.
        
        SpaceMouse sends separate HID reports for:
        - Channel 1/2: Axis data (motion)
        - Channel 3: Button data
        
        In nonblocking mode, read() returns immediately with one report or None.
        We need to read repeatedly until buffer is empty to get the latest
        state for ALL channels (motion + buttons).
        
        Args:
            device: SpaceMouseDevice instance
            
        Returns:
            The device's accumulated state after processing all pending reports
        """
        # Keep reading until no more data in buffer
        # This ensures we process all pending HID reports (both motion and button)
        max_reads = 64  # Safety limit to prevent infinite loop
        
        for _ in range(max_reads):
            # Read raw HID data directly (nonblocking returns None if empty)
            # Note: We access internal _device because pyspacemouse doesn't expose
            # a "drain buffer" API. device.info is the public DeviceInfo.
            raw_data = device._device.read(device.info.bytes_to_read)
            if not raw_data:
                # Buffer is empty, we have the latest state
                break
            # Process this HID report (updates device._state internally)
            device._process(raw_data)
        
        return device._state

    def get_motion_state(self):
        """Get normalized motion state from cached data.

        Returns:
            np.ndarray: 6-DoF motion state [x, y, z, roll, pitch, yaw] in range [-1, 1]

        Note: Call poll() first to update the cached state.
        """
        if self.multi_device_mode:
            return self._get_motion_state_dual()
        else:
            return self._get_motion_state_single()

    def _get_motion_state_single(self):
        """Get motion state from single device's cached state."""
        if self._state is None:
            return np.zeros(6, dtype=self.dtype)

        return np.array(
            [
                self._state.x,
                self._state.y,
                self._state.z,
                self._state.roll,
                self._state.pitch,
                self._state.yaw,
            ],
            dtype=self.dtype,
        )

    def _get_motion_state_dual(self):
        """Get combined motion state from dual devices' cached states."""
        if self._left_state is None or self._right_state is None:
            return np.zeros(6, dtype=self.dtype)

        state = np.zeros(6, dtype=self.dtype)

        # Left device values
        left_values = [
            self._left_state.x,
            self._left_state.y,
            self._left_state.z,
            self._left_state.roll,
            self._left_state.pitch,
            self._left_state.yaw,
        ]

        # Right device values
        right_values = [
            self._right_state.x,
            self._right_state.y,
            self._right_state.z,
            self._right_state.roll,
            self._right_state.pitch,
            self._right_state.yaw,
        ]

        # Combine based on enabled axes
        # Priority: left_device > right_device (if both enable same axis)
        # For additive combination (e.g., fine+coarse), both should be enabled
        for i in range(6):
            left_enabled = self.left_device_config.enabled_axes[i]
            right_enabled = self.right_device_config.enabled_axes[i]

            if left_enabled and right_enabled:
                # Both enabled: add values (useful for fine+coarse control)
                state[i] = left_values[i] + right_values[i]
            elif left_enabled:
                state[i] = left_values[i]
            elif right_enabled:
                state[i] = right_values[i]
            # else: state[i] = 0 (already initialized)

        return state

    def get_motion_state_transformed(self):
        """Get motion state with coordinate transformation.

        Transforms from SpaceMouse coordinate system to robot coordinate system:

        Robot coordinates (z-up, looking from above):
            y (right)
            ^
            |   _
            |  (O) spacemouse
            *------>x (forward)
            
        Position: SpaceMouse forward -> Robot +X, right -> +Y, up -> +Z
        Rotation: roll/pitch/yaw kept aligned (pitch negated to match forward)

        Returns:
            np.ndarray: Transformed 6-DoF state [x, y, z, roll, pitch, yaw]
        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_pos @ state[:3]  # Transform position
        tf_state[3:] = self.tx_rot @ state[3:]  # Transform rotation
        return tf_state

    def get_button_state(self):
        """Get button state from cached data.

        Returns:
            np.ndarray: Boolean array of button states

        Note: Call poll() first to update the cached state.
        """
        if self.multi_device_mode:
            return self._get_button_state_dual()
        else:
            return self._get_button_state_single()

    def _get_button_state_single(self):
        """Get button state from single device's cached state."""
        if self._state is None or not self._state.buttons:
            return np.array([False, False], dtype=bool)

        return np.array(self._state.buttons, dtype=bool)

    def _get_button_state_dual(self):
        """Get combined button state from dual devices' cached states."""
        if self._left_state is None or self._right_state is None:
            return np.array([False, False], dtype=bool)

        left_buttons = self._left_state.buttons if self._left_state.buttons else []
        right_buttons = self._right_state.buttons if self._right_state.buttons else []

        # Combine buttons using OR logic (either device can trigger)
        max_len = max(len(left_buttons), len(right_buttons), 2)
        button_state = np.zeros(max_len, dtype=bool)

        for i in range(max_len):
            left_pressed = left_buttons[i] if i < len(left_buttons) else False
            right_pressed = right_buttons[i] if i < len(right_buttons) else False
            button_state[i] = bool(left_pressed) or bool(right_pressed)

        return button_state

    def is_button_pressed(self, button_id: int) -> bool:
        """Check if specific button is pressed by raw index.

        Args:
            button_id: Button index to check (0-based, raw device index)

        Returns:
            bool: True if button is pressed

        Note: Call poll() first to update the cached state.
        Note: For physical left/right buttons, use is_left_button_pressed() 
              and is_right_button_pressed() which handle device differences.
        """
        button_state = self.get_button_state()
        if 0 <= button_id < len(button_state):
            return bool(button_state[button_id])
        return False

    def is_left_button_pressed(self) -> bool:
        """Check if physical LEFT button is pressed.
        
        This handles device differences automatically:
        - SpaceNavigator: LEFT button
        - Pro/Enterprise: MENU button (physical left position)
        
        Returns:
            bool: True if left button is pressed
        """
        if self._left_button_idx < 0:
            return False
        return self.is_button_pressed(self._left_button_idx)

    def is_right_button_pressed(self) -> bool:
        """Check if physical RIGHT button is pressed.
        
        This handles device differences automatically:
        - SpaceNavigator: RIGHT button  
        - Pro/Enterprise: FIT button (physical right position)
        
        Returns:
            bool: True if right button is pressed
        """
        if self._right_button_idx < 0:
            return False
        return self.is_button_pressed(self._right_button_idx)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self._is_connected

    @property
    def device_name_actual(self) -> str:
        """Get the actual connected device name."""
        if self._device is not None:
            return self._device.name
        if self._left_device is not None:
            return self._left_device.name
        return "Not connected"


def test():
    """Test function to verify SpaceMouse functionality."""
    import time

    print("Testing PySpaceMouse wrapper...")
    print()

    # List connected devices
    connected = pyspacemouse.get_connected_devices()
    print(f"Connected devices: {connected}")

    if not connected:
        print("No SpaceMouse devices found!")
        return

    with Spacemouse() as sm:
        print(f"✅ Connected to: {sm.device_name_actual}")
        print("\nMove the SpaceMouse to see values (Ctrl+C to exit)")
        print("Press buttons to test button detection\n")

        last_button_state = [False, False]
        
        try:
            for i in range(200000):
                # Poll once per cycle (drains HID buffer)
                sm.poll()

                # Get synchronized data
                state = sm.get_motion_state_transformed()
                
                # Use device-aware button methods
                button_left = sm.is_left_button_pressed()
                button_right = sm.is_right_button_pressed()

                # Print button change events
                if button_left != last_button_state[0]:
                    print(f"\n🔘 LEFT button {'PRESSED' if button_left else 'RELEASED'}")
                    last_button_state[0] = button_left
                if button_right != last_button_state[1]:
                    print(f"\n🔘 RIGHT button {'PRESSED' if button_right else 'RELEASED'}")
                    last_button_state[1] = button_right

                print(
                    f"\rIter {i:5d}: "
                    f"x={state[0]:+.2f} y={state[1]:+.2f} z={state[2]:+.2f} "
                    f"r={state[3]:+.2f} p={state[4]:+.2f} yaw={state[5]:+.2f} "
                    f"btn=[L:{int(button_left)},R:{int(button_right)}]  ",
                    end="",
                    flush=True,
                )
                time.sleep(1 / 100)
        except KeyboardInterrupt:
            print("\n\n⏹️  Test stopped")

    print("✅ Test complete")


if __name__ == "__main__":
    test()
