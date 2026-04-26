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

from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.bi_flexiv_rizon4_rt.serial_gripper import SerialGripper
from lerobot.robots.bi_xense_flare_grippers.config_bi_xense_flare_grippers import (
    BiXenseFlareGrippersConfig,
)
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import get_logger


class BiXenseFlareGrippers(Robot):
    """Dual Xense flare-gripper rig with cameras but no robot arms."""

    config_class = BiXenseFlareGrippersConfig
    name = "bi_xense_flare_grippers"

    def __init__(self, config: BiXenseFlareGrippersConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger("BiXenseFlareGrippers")
        self._is_connected = False

        self._left_gripper = (
            SerialGripper(config.left_gripper) if config.left_gripper is not None else None
        )
        self._right_gripper = (
            SerialGripper(config.right_gripper) if config.right_gripper is not None else None
        )
        self._left_gripper_key = "left_gripper.pos"
        self._right_gripper_key = "right_gripper.pos"
        self.cameras = make_cameras_from_configs(config.cameras)
        np.set_printoptions(precision=3, suppress=True)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features = {**self._cameras_ft}
        if self._left_gripper is not None:
            features[self._left_gripper_key] = float
        if self._right_gripper is not None:
            features[self._right_gripper_key] = float
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features = {}
        if self._left_gripper is not None:
            features[self._left_gripper_key] = float
        if self._right_gripper is not None:
            features[self._right_gripper_key] = float
        return features

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and all(cam.is_connected for cam in self.cameras.values())
            and all(
                gripper is None or getattr(gripper, "_is_connected", False)
                for gripper in [self._left_gripper, self._right_gripper]
            )
        )

    def connect(self, calibrate: bool = False, go_to_start: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} already connected, do not run `robot.connect()` twice."
            )

        try:
            grippers = {
                key: gripper
                for key, gripper in [("left", self._left_gripper), ("right", self._right_gripper)]
                if gripper is not None
            }
            if grippers:
                self.logger.info(
                    f"Connecting {len(grippers)} gripper(s): {', '.join(grippers.keys())}..."
                )
                with ThreadPoolExecutor(max_workers=len(grippers)) as ex:
                    gripper_futures = {
                        key: ex.submit(gripper.connect) for key, gripper in grippers.items()
                    }
                    for key, future in gripper_futures.items():
                        try:
                            future.result()
                        except Exception as e:
                            raise RuntimeError(f"Failed to connect {key} gripper: {e}") from e

            if not self.cameras:
                self.logger.info("No cameras configured; skipping camera connect.")
            else:
                self.logger.info(
                    f"Connecting {len(self.cameras)} camera(s): {', '.join(self.cameras.keys())}..."
                )
            with ThreadPoolExecutor(max_workers=len(self.cameras) or 1) as ex:
                cam_futures = {
                    cam_key: ex.submit(cam.connect) for cam_key, cam in self.cameras.items()
                }
                for cam_key, future in cam_futures.items():
                    try:
                        future.result()
                    except Exception as e:
                        raise RuntimeError(f"Failed to connect camera {cam_key}: {e}") from e

            self._is_connected = True
            self.logger.info("Bi Xense Flare Grippers connected.")
        except Exception as e:
            self.logger.error(f"Failed to connect Bi Xense Flare Grippers: {e}")
            for gripper in [self._left_gripper, self._right_gripper]:
                if gripper is not None:
                    try:
                        gripper.disconnect()
                    except Exception:
                        pass
            for cam in self.cameras.values():
                try:
                    if cam.is_connected:
                        cam.disconnect()
                except Exception:
                    pass
            raise

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        self.logger.info("Bi Xense Flare Grippers does not require runtime calibration.")

    def configure(self) -> None:
        self.logger.info("Bi Xense Flare Grippers does not require runtime configure.")

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs = {}
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        if self._left_gripper is not None:
            obs[self._left_gripper_key] = self._left_gripper.get_gripper_position()
        if self._right_gripper is not None:
            obs[self._right_gripper_key] = self._right_gripper.get_gripper_position()
        return obs

    def send_action(self, action: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action = {} if action is None else action
        sent_action = {}

        if self._left_gripper is not None and self._left_gripper_key in action:
            value = float(action[self._left_gripper_key])
            self._left_gripper.set_gripper_position(value)
            sent_action[self._left_gripper_key] = value

        if self._right_gripper is not None and self._right_gripper_key in action:
            value = float(action[self._right_gripper_key])
            self._right_gripper.set_gripper_position(value)
            sent_action[self._right_gripper_key] = value

        return sent_action

    def disconnect(self) -> None:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        self.logger.info("Disconnecting from Bi Xense Flare Grippers...")

        def _disconnect_camera(cam_key: str, cam) -> None:
            if cam.is_connected:
                cam.disconnect()
                self.logger.info(f"Disconnected camera: {cam_key}")

        def _disconnect_gripper(gripper_key: str, gripper: SerialGripper) -> None:
            gripper.disconnect()
            self.logger.info(f"Disconnected gripper: {gripper_key}")

        with ThreadPoolExecutor(max_workers=len(self.cameras) or 1) as ex:
            cam_futures = {
                cam_key: ex.submit(_disconnect_camera, cam_key, cam)
                for cam_key, cam in self.cameras.items()
            }
            for cam_key, future in cam_futures.items():
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error disconnecting camera {cam_key}: {e}")

        grippers = {
            key: gripper
            for key, gripper in [("left", self._left_gripper), ("right", self._right_gripper)]
            if gripper is not None
        }
        with ThreadPoolExecutor(max_workers=len(grippers) or 1) as ex:
            gripper_futures = {
                key: ex.submit(_disconnect_gripper, key, gripper)
                for key, gripper in grippers.items()
            }
            for key, future in gripper_futures.items():
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error disconnecting gripper {key}: {e}")

        self._is_connected = False
        self.logger.info("Bi Xense Flare Grippers disconnected.")
