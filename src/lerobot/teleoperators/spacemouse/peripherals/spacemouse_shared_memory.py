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

import multiprocessing as mp
import time
import threading
from multiprocessing.managers import SharedMemoryManager
from typing import Any

import numpy as np

try:
    import pyspacemouse
except ImportError:
    pyspacemouse = None

from ..shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class Spacemouse(mp.Process):
    def __init__(
        self,
        shm_manager,
        get_max_k=30,
        frequency=200,
        max_value=500,
        deadzone=(0, 0, 0, 0, 0, 0),
        dtype=np.float32,
        n_buttons=2,
        multi_device_mode=False,
        left_device_config=None,
        right_device_config=None,
    ):
        """
        Continuously listen to 3D connection space navigator events
        using PySpaceMouse library and update the latest state.

        Supports both single device and multi-device (dual-hand) modes.

        Args:
            multi_device_mode: If True, use dual SpaceMouse setup
            left_device_config: Configuration for left hand device
            right_device_config: Configuration for right hand device
            max_value: {300, 500} 300 for wired version and 500 for wireless
            deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0

        front
        z
        ^   _
        |  (O) space mouse
        |
        *----->x right
        y
        """
        super().__init__()
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        # copied variables
        self.frequency = frequency
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.n_buttons = n_buttons

        # Multi-device configuration
        self.multi_device_mode = multi_device_mode
        self.left_device_config = left_device_config
        self.right_device_config = right_device_config

        # Coordinate transformation matrix (same as before to maintain compatibility)
        self.tx_zup_spnav = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]], dtype=dtype)

        example = {
            # 3 translation, 3 rotation, 1 period
            "motion_event": np.zeros((7,), dtype=np.int64),
            # left and right button
            "button_state": np.zeros((n_buttons,), dtype=bool),
            "receive_timestamp": time.monotonic(),
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        # shared variables
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer

        # PySpaceMouse device instances
        self._left_device = None
        self._right_device = None
        self._single_device = None
        self._running = False

    # ======= get state APIs ==========

    def get_motion_state(self):
        """Get normalized motion state from ring buffer."""
        state = self.ring_buffer.get()
        motion_event = state["motion_event"]
        assert isinstance(motion_event, np.ndarray)
        state = np.array(motion_event[:6], dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state

    def get_motion_state_transformed(self):
        """
        Return in right-handed coordinate
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x
        back

        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]
        return tf_state

    def get_button_state(self):
        """Get button state from ring buffer."""
        state = self.ring_buffer.get()
        return state["button_state"]

    def is_button_pressed(self, button_id):
        """Check if specific button is pressed."""
        button_state = self.get_button_state()
        assert isinstance(button_state, np.ndarray)
        return button_state[button_id]

    # ========== start stop API ===========

    def start(self, wait=True):
        """Start the spacemouse process."""
        super().start()
        if wait:
            self.ready_event.wait()

    def stop(self, wait=True):
        """Stop the spacemouse process."""
        self.stop_event.set()
        if wait:
            self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= main loop ==========
    def run(self):
        """Main process loop using PySpaceMouse."""
        if pyspacemouse is None:
            raise ImportError(
                "pyspacemouse is required for Spacemouse teleoperator. "
                "Install it with: pip install pyspacemouse"
            )

        if self.multi_device_mode:
            self._run_multi_device()
        else:
            self._run_single_device()

    def _run_single_device(self):
        """Single device mode (original behavior)."""
        try:
            with pyspacemouse.open() as device:
                if device is None:
                    raise RuntimeError("Failed to open SpaceMouse device")

                self._single_device = device
                print(f"Connected to: {device.name}")

                motion_event = np.zeros((7,), dtype=np.int64)
                button_state = np.zeros((self.n_buttons,), dtype=bool)

                self.ring_buffer.put(
                    {
                        "motion_event": motion_event,
                        "button_state": button_state,
                        "receive_timestamp": time.monotonic(),
                    }
                )
                self.ready_event.set()
                self._running = True

                while not self.stop_event.is_set() and self._running:
                    try:
                        state = device.read()
                        receive_timestamp = time.monotonic()

                        if state is not None:
                            motion_event[0] = int(state.x * self.max_value)
                            motion_event[1] = int(state.y * self.max_value)
                            motion_event[2] = int(state.z * self.max_value)
                            motion_event[3] = int(state.roll * self.max_value)
                            motion_event[4] = int(state.pitch * self.max_value)
                            motion_event[5] = int(state.yaw * self.max_value)
                            motion_event[6] = 0

                            button_state.fill(False)
                            if hasattr(state, "buttons") and state.buttons is not None:
                                for i, button_pressed in enumerate(state.buttons):
                                    if i < self.n_buttons:
                                        button_state[i] = bool(button_pressed)

                            self.ring_buffer.put(
                                {
                                    "motion_event": motion_event.copy(),
                                    "button_state": button_state.copy(),
                                    "receive_timestamp": receive_timestamp,
                                }
                            )

                        time.sleep(1 / self.frequency)

                    except Exception as e:
                        print(f"Error reading SpaceMouse: {e}")
                        time.sleep(1 / self.frequency)

        except Exception as e:
            print(f"Failed to initialize SpaceMouse: {e}")
            self.ready_event.set()

        finally:
            self._running = False
            self._single_device = None

    def _run_multi_device(self):
        """Dual device mode for left/right hand control."""
        try:
            connected_devices = pyspacemouse.get_connected_devices()
            if len(connected_devices) < 2:
                raise RuntimeError(f"Need at least 2 SpaceMouse devices, found {len(connected_devices)}")

            device_name = connected_devices[0]
            print(f"Opening dual devices: {device_name}")

            with pyspacemouse.open(device=device_name, device_index=self.left_device_config.device_index) as left_device, \
                 pyspacemouse.open(device=device_name, device_index=self.right_device_config.device_index) as right_device:

                if left_device is None or right_device is None:
                    raise RuntimeError("Failed to open one or both SpaceMouse devices")

                self._left_device = left_device
                self._right_device = right_device
                print(f"Connected - Left: {left_device.name}, Right: {right_device.name}")

                motion_event = np.zeros((7,), dtype=np.int64)
                button_state = np.zeros((self.n_buttons,), dtype=bool)

                self.ring_buffer.put(
                    {
                        "motion_event": motion_event,
                        "button_state": button_state,
                        "receive_timestamp": time.monotonic(),
                    }
                )
                self.ready_event.set()
                self._running = True

                while not self.stop_event.is_set() and self._running:
                    try:
                        left_state = left_device.read()
                        right_state = right_device.read()
                        receive_timestamp = time.monotonic()

                        self._combine_device_states(left_state, right_state, motion_event, button_state)

                        self.ring_buffer.put(
                            {
                                "motion_event": motion_event.copy(),
                                "button_state": button_state.copy(),
                                "receive_timestamp": receive_timestamp,
                            }
                        )

                        time.sleep(1 / self.frequency)

                    except Exception as e:
                        print(f"Error reading SpaceMouse devices: {e}")
                        time.sleep(1 / self.frequency)

        except Exception as e:
            print(f"Failed to initialize dual SpaceMouse devices: {e}")
            self.ready_event.set()

        finally:
            self._running = False
            self._left_device = None
            self._right_device = None

    def _combine_device_states(self, left_state, right_state, motion_event, button_state):
        """Combine states from left and right devices based on enabled axes."""
        motion_event.fill(0)
        button_state.fill(False)

        # Process left device (typically position control)
        if left_state is not None and self.left_device_config is not None:
            left_values = [left_state.x, left_state.y, left_state.z,
                          left_state.roll, left_state.pitch, left_state.yaw]

            for i, (value, enabled) in enumerate(zip(left_values, self.left_device_config.enabled_axes)):
                if enabled:
                    motion_event[i] = int(value * self.max_value)

            if hasattr(left_state, 'buttons') and left_state.buttons is not None:
                for i, button_pressed in enumerate(left_state.buttons):
                    if i < self.n_buttons:
                        button_state[i] = button_state[i] or bool(button_pressed)

        # Process right device (typically orientation control)
        if right_state is not None and self.right_device_config is not None:
            right_values = [right_state.x, right_state.y, right_state.z,
                           right_state.roll, right_state.pitch, right_state.yaw]

            for i, (value, enabled) in enumerate(zip(right_values, self.right_device_config.enabled_axes)):
                if enabled:
                    motion_event[i] = int(value * self.max_value)

            if hasattr(right_state, 'buttons') and right_state.buttons is not None:
                for i, button_pressed in enumerate(right_state.buttons):
                    if i < self.n_buttons:
                        button_state[i] = button_state[i] or bool(button_pressed)


def test():
    """Test function to verify SpaceMouse functionality."""
    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager, deadzone=0.3, max_value=500) as sm:
            for i in range(2000):
                # print(sm.get_motion_state())
                print(sm.get_motion_state_transformed())
                print(sm.is_button_pressed(0))
                time.sleep(1 / 100)


if __name__ == "__main__":
    test()
