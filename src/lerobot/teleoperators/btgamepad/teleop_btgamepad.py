# !/usr/bin/env python

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

from enum import IntEnum
from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_btgamepad import BtgamepadTeleopConfig
from lerobot.utils.robot_utils import (
    matrix_to_pose7d,
    quaternion_to_euler,
    rotation_6d_to_quaternion,
    xyz_rpy_to_matrix,
    normalize_quaternion,
    quaternion_to_matrix,
)
from scipy.spatial.transform import Rotation as R

class GripperAction(IntEnum):
    CLOSE = 1
    STAY = 0
    OPEN = 0


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


class BtgamepadTeleop(Teleoperator):
    """
    Teleop class to use gamepad inputs for control.
    """

    config_class = BtgamepadTeleopConfig
    name = "btgamepad"

    def __init__(self, config: BtgamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.gamepad = None

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (10,),
                "names": {"tcp.x": 0, "tcp.y": 1, "tcp.z": 2, "tcp.r1": 3,"tcp.r2": 4,"tcp.r3": 5,"tcp.r4": 6,"tcp.r5": 7,"tcp.r6": 8,"gripper.pos": 9},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (9,),
                "names": {"tcp.x": 0, "tcp.y": 1, "tcp.z": 2, "tcp.r1": 3,"tcp.r2": 4,"tcp.r3": 5,"tcp.r4": 6,"tcp.r5": 7,"tcp.r6": 8},
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self, current_tcp_pose_quat: np.ndarray = np.zeros(8, dtype=np.float32)) -> None:
        from .btgamepad_utils import BtgamepadController as Gamepad

        self.gamepad = Gamepad()
        self.gamepad.start()

        # Set target pose on connect
        # Input format: [x, y, z, qw, qx, qy, qz, gripper_pos] (Flexiv wxyz format)
        self._target_pos = current_tcp_pose_quat[:3].copy()  # [x, y, z]
        self._target_quat = normalize_quaternion(current_tcp_pose_quat[3:7], input_format="wxyz")
        self._target_gripper_pos = current_tcp_pose_quat[7]

        # Initialize start pose (will be updated when grip is pressed)
        self._start_pos = current_tcp_pose_quat[:3].copy()
        self._start_quat = self._target_quat.copy()

    def get_action(self) -> dict[str, Any]:
        # Update the controller to get fresh inputs
        self.gamepad.update()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz = self.gamepad.get_deltas()
        rel_pos = np.array([delta_x, delta_y, delta_z], dtype=np.float32)
        scaled_rel_pos = rel_pos * self.config.pos_sensitivity
        target_pos = self._start_pos + scaled_rel_pos
        self._start_pos = target_pos

        # quaternion update
        rotation_delta = np.array([delta_rx, delta_ry, delta_rz]) * self.config.rot_sensitivity
        rotation_delta = R.from_euler('xyz', rotation_delta).as_matrix()
        self._start_matrix = quaternion_to_matrix(
            np.array([0,0,0,self._start_quat[0],self._start_quat[1],self._start_quat[2],self._start_quat[3]]),
            input_format="wxyz"
            )

        current_ee_matrix = rotation_delta @ self._start_matrix[:3, :3] #3x3
        current_ee_quat9d = np.array(current_ee_matrix).flatten() # 9d
        self._start_quat = rotation_6d_to_quaternion(current_ee_quat9d[:6])

        action_dict = {
            "tcp.x": target_pos[0],
            "tcp.y": target_pos[1],
            "tcp.z": target_pos[2],
            "tcp.r1": current_ee_matrix[0, 0],
            "tcp.r2": current_ee_matrix[1, 0],
            "tcp.r3": current_ee_matrix[2, 0],
            "tcp.r4": current_ee_matrix[0, 1],
            "tcp.r5": current_ee_matrix[1, 1],
            "tcp.r6": current_ee_matrix[2, 1],
        }

        # Default gripper action is to stay
        gripper_action = GripperAction.STAY.value
        if self.config.use_gripper:
            gripper_command = self.gamepad.gripper_command()
            gripper_action = gripper_action_map[gripper_command]
            action_dict["gripper.pos"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the gamepad such as intervention status,
        episode termination, success indicators, etc.

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if self.gamepad is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Update gamepad state to get fresh inputs
        self.gamepad.update()

        # Check if intervention is active
        is_intervention = self.gamepad.should_intervene()

        # Get episode end status
        episode_end_status = self.gamepad.get_episode_end_status()
        terminate_episode = episode_end_status in [
            TeleopEvents.RERECORD_EPISODE,
            TeleopEvents.FAILURE,
        ]
        success = episode_end_status == TeleopEvents.SUCCESS
        rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE
        back_home = episode_end_status == TeleopEvents.BACK_HOME
        failure = episode_end_status == TeleopEvents.FAILURE

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
            TeleopEvents.BACK_HOME: back_home,
            TeleopEvents.FAILURE: failure,
        }

    def disconnect(self) -> None:
        """Disconnect from the gamepad."""
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

    def is_connected(self) -> bool:
        """Check if gamepad is connected."""
        return self.gamepad is not None

    def calibrate(self) -> None:
        """Calibrate the gamepad."""
        # No calibration needed for gamepad
        pass

    def is_calibrated(self) -> bool:
        """Check if gamepad is calibrated."""
        # Gamepad doesn't require calibration
        return True

    def configure(self) -> None:
        """Configure the gamepad."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the gamepad."""
        # Gamepad doesn't support feedback
        pass
