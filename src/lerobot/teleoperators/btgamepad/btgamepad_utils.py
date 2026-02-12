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

from enum import Enum
import logging
import numpy as np
from ..utils import TeleopEvents


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        """
        Initialize the controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, "success", or "failure"
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        """Start the controller and initialize resources."""
        pass

    def stop(self):
        """Stop the controller and release resources."""
        pass

    def get_deltas(self):
        """Get the current movement deltas (dx, dy, dz) in meters."""
        return 0.0, 0.0, 0.0

    def update(self):
        """Update controller state - call this once per frame."""
        pass

    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    def gripper_command(self):
        """Return the current gripper command."""
        if self.open_gripper_command == self.close_gripper_command:
            return "stay"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"
    # def gripper_command(self):
    #     if self.open_gripper_command:
    #         if self.close_gripper_command:
    #             return "close"
    #         else:
    #             return "open"
    #     else:
    #         return "stay"


class KeyboardController(InputController):
    """Generate motion deltas from keyboard input."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "quit": False,
            "success": False,
            "failure": False,
        }
        self.listener = None

    def start(self):
        """Start the keyboard listener."""
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == keyboard.Key.esc:
                    self.key_states["quit"] = True
                    self.running = False
                    return False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = TeleopEvents.SUCCESS
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = True
                    self.episode_end_status = TeleopEvents.FAILURE
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = False
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  Enter: End episode with SUCCESS")
        print("  Backspace: End episode with FAILURE")
        print("  ESC: Exit")

    def stop(self):
        """Stop the keyboard listener."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
        """Get the current movement deltas from keyboard state."""
        delta_x = delta_y = delta_z = 0.0

        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size

        return delta_x, delta_y, delta_z


class BtgamepadController(InputController):
    """Generate motion deltas from gamepad input."""

    class Button(Enum):
        """手柄按钮定义"""
        A,B,X,Y,LB,RB,BACK,START,HOME,LS,RS,test1,test2,test3,test4,test5 = 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 # 北通键位1，Xbox键位
        # A,B,tt1,X,Y,tt2,LB,RB,tt3,tt4,BACK,START,HOME,LS,RS,tt5 = 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 # 北通键位2

    class Axis(Enum):
        """手柄轴定义 (Xbox 标准)"""
        # # 若按下shift键，则左右摇杆改为十字方向键
        # LX = 0  # 左摇杆左右, -1 为左,1 为右, 
        # LY = 1  # 左摇杆上下, -1 为上,1 为下, 
        # RX = 2  # 右摇杆左右, -1 为左,1 为右,
        # RY = 3  # 右摇杆上下, -1 为上,1 为下, 
        # RT = 4  # RT，按下为1，松开为-1
        # LT = 5  # LT，按下为1，松开为-1
        # LX,LY,RX,RY,RT,LT = 0,1,2,3,4,5 # 北通键位1
        # LX,LY,RX,RY,RT,LT = 0,1,2,3,4,5 # 北通键位2
        LX,LY,LT,RX,RY,RT = 0,1,2,3,4,5 # Xbox键位

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0, deadzone=0.1):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.intervention_flag = False

    def start(self):
        """Initialize pygame and the gamepad."""
        import pygame

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            logging.error("No gamepad detected. Please connect a gamepad and try again.")
            self.running = False
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        logging.info(f"Initialized gamepad: {self.joystick.get_name()}")

    def stop(self):
        """Clean up pygame resources."""
        import pygame

        if pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        """Process pygame events to get fresh gamepad readings."""
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                # Y
                if event.button == self.Button.Y.value:
                    self.episode_end_status = TeleopEvents.SUCCESS
                # X
                elif event.button == self.Button.X.value:
                    self.episode_end_status = TeleopEvents.FAILURE
                # B
                elif event.button == self.Button.B.value:
                    self.episode_end_status = TeleopEvents.RERECORD_EPISODE
                # BACK
                elif event.button == self.Button.BACK.value:
                    self.episode_end_status = TeleopEvents.BACK_HOME
                # RB
                elif event.button == self.Button.RB.value:
                    self.close_gripper_command = True
                
                elif event.button == self.Button.LB.value:
                    self.open_gripper_command = True

            # Reset episode status on button release
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in (self.Button.X.value, self.Button.B.value, self.Button.Y.value, self.Button.BACK.value):
                    self.episode_end_status = None

                elif event.button == self.Button.RB.value:
                    self.close_gripper_command = False
                
                elif event.button == self.Button.LB.value:
                    self.open_gripper_command = False

            # Check for RB button (typically button 5) for intervention flag
            if self.joystick.get_button(self.Button.X.value):
                self.intervention_flag = True
            else:
                self.intervention_flag = False

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        import pygame

        try:
            # Read joystick axes
            # Left stick X and Y (typically axes 0 and 1)
            x_input = self.joystick.get_axis(self.Axis.LY.value)  # Up/Down (often inverted)
            y_input = self.joystick.get_axis(self.Axis.LX.value)  # Left/Right

            # Right stick Y (typically axis 3 or 4)
            z_input = self.joystick.get_axis(self.Axis.RY.value)  # Up/Down for Z

            # Apply deadzone to avoid drift
            x_input = 0 if abs(x_input) < self.deadzone else x_input
            y_input = 0 if abs(y_input) < self.deadzone else y_input
            z_input = 0 if abs(z_input) < self.deadzone else z_input

            # Calculate deltas (note: may need to invert axes depending on controller)
            delta_x = x_input * self.x_step_size  # Forward/backward
            delta_y = y_input * self.y_step_size  # Left/right
            delta_z = -z_input * self.z_step_size  # Up/down

            delta_rx = 0
            # ry_input = np.pi # flexiv
            delta_ry = 0 # franka
            delta_rz = self.joystick.get_axis(self.Axis.RX.value)  # Rotation around Z
            # delta_rz = 0

            return delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz

        except pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")
            # return 0.0, 0.0, 0.0, 0.0, np.pi , 0.0 # flexiv
            return 0.0, 0.0, 0.0, 0.0, 0.0 , 0.0 # franka
