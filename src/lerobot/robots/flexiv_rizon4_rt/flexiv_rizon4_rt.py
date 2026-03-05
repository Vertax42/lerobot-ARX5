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

"""Flexiv Rizon4 RT robot implementation for LeRobot.

This module provides real-time (RT) integration with Flexiv Rizon4 7-DOF collaborative robot
using flexiv_rt (libpyflexiv), which spawns a C++ RT thread at 1 kHz via
rdk::Scheduler with SCHED_FIFO priority.

Control Architecture:
    LeRobot send_action(30-100 Hz)
        -> cc.set_target_pose(pose, wrench)  [writes to shared memory via mutex]
        -> C++ RT thread reads SHM every 1 ms
        -> StreamCartesianMotionForce(pose, wrench, vel, acc)
        -> Robot

    Observations are read from SHM via cc.get_state() (CartesianState snapshot).

Control Mode: RT_CARTESIAN_MOTION_FORCE only (for now)
    - Pure motion (use_force=False):
      Action:  TCP pose (9D: xyz + 6D rotation) + gripper (1D) = 10D
      Observation: joint states (21D) or TCP pose (9D) + gripper (1D)

    - Motion + force (use_force=True):
      Action:  TCP pose (9D) + wrench (6D) + gripper (1D) = 16D
      Observation: joint states (21D) or TCP pose (9D) + wrench (6D) + gripper (1D)

Key Differences from NRT Driver (flexiv_rizon4):
    - Uses flexiv_rt.Robot instead of flexivrdk.Robot
    - RT mode: RT_CARTESIAN_MOTION_FORCE (not NRT_CARTESIAN_MOTION_FORCE)
    - connect() starts cc = robot.start_cartesian_control() -> spawns C++ RT thread
    - send_action() -> cc.set_target_pose(pose, wrench) (writes to SHM)
    - get_observation() -> cc.get_state() (reads from SHM)
    - disconnect() -> cc.stop() (blocks until RT thread joins)
    - Uses busy() polling instead of primitive_states() for MoveJ wait
    - Reuses FlareGripper (independent of arm control backend)

6D Rotation Representation:
    - r1, r2, r3: First column of rotation matrix
    - r4, r5, r6: Second column of rotation matrix
    - Reference: "On the Continuity of Rotation Representations in Neural Networks"

Reference: https://rdk.flexiv.com/api/
"""

import time
from functools import cached_property
from typing import Any

import flexiv_rt as frt
import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.flexiv_rizon4.flare_gripper import FlareGripper
from lerobot.robots.flexiv_rizon4.xense_gripper import Gripper
from lerobot.robots.flexiv_rizon4_rt.config_flexiv_rizon4_rt import FlexivRizon4RTConfig
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import (
    get_logger,
    quaternion_to_euler,
    quaternion_to_rotation_6d,
    rotation_6d_to_quaternion,
)

# Constants
CART_DOF = 6  # Cartesian degrees of freedom
JOINT_DOF = 7  # Flexiv Rizon4 robot joint DOF
POSE_SIZE_QUAT = 7  # Pose size with quaternion [x,y,z,qw,qx,qy,qz]
POSE_SIZE_6D = 9  # Pose size with 6D rotation [x,y,z,r1..r6]


class FlexivRizon4RT(Robot):
    """Flexiv Rizon4 7-DOF collaborative robot with real-time control.

    Uses flexiv_rt (libpyflexiv) to achieve 1 kHz deterministic control
    via a C++ RT thread with SCHED_FIFO scheduling.

    Python-side send_action() writes target poses to shared memory at 30-100 Hz.
    The C++ RT thread reads these at 1 kHz and streams them to the robot via
    StreamCartesianMotionForce. Safety features (NaN check, jump clamp, 500 ms
    timeout) are handled entirely in the RT thread.

    Example:
        >>> from lerobot.robots.flexiv_rizon4_rt import FlexivRizon4RT, FlexivRizon4RTConfig
        >>> config = FlexivRizon4RTConfig(
        ...     robot_sn="Rizon4-063423",
        ...     use_force=False,
        ... )
        >>> robot = FlexivRizon4RT(config)
        >>> robot.connect()
        >>> obs = robot.get_observation()
        >>> robot.send_action({"tcp.x": 0.5, "tcp.y": 0.0, "tcp.z": 0.3,
        ...     "tcp.r1": 1.0, "tcp.r2": 0.0, "tcp.r3": 0.0,
        ...     "tcp.r4": 0.0, "tcp.r5": 1.0, "tcp.r6": 0.0,
        ...     "gripper.pos": 0.8})
        >>> robot.disconnect()
    """

    config_class = FlexivRizon4RTConfig
    name = "flexiv_rizon4_rt"

    def __init__(self, config: FlexivRizon4RTConfig):
        super().__init__(config)
        self.config = config

        # Logger
        self.logger = get_logger("FlexivRizon4RT", loglevel=config.log_level)

        # Robot interface (initialized on connect)
        self._robot: frt.Robot | None = None
        self._cc: frt.CartesianMotionForceControl | None = None  # RT control handle
        self._is_connected = False

        # Gripper (independent of arm control backend)
        self._gripper: FlareGripper | Gripper | None = None
        if config.use_gripper and config.gripper_type == "flare_gripper":
            self._gripper = FlareGripper(config.gripper)
        elif config.use_gripper and config.gripper_type == "xense_gripper":
            self._gripper = Gripper(config.gripper)
        else:
            self.logger.info("No gripper configured, proceeding without gripper.")

        # Home TCP pose - stored after moving to home position
        # Format: [x, y, z, qw, qx, qy, qz] (7D) - SDK format
        self._home_tcp_pose: np.ndarray | None = None

        # Start TCP pose - cached after initial MoveJ for RT reset
        self._start_tcp_pose: list[float] | None = None

        # Gripper key
        self._gripper_key = "gripper.pos"

        # Initialize observation/action keys for Cartesian mode
        self._init_cartesian_mode()

        # External cameras
        self.cameras = make_cameras_from_configs(config.cameras)
        np.set_printoptions(precision=6, suppress=True)

    # =========================================================================
    # Key initialization
    # =========================================================================

    def _init_cartesian_mode(self) -> None:
        """Initialize keys and buffers for RT_CARTESIAN_MOTION_FORCE control mode.

        Uses 6D rotation representation (r1-r6) for continuity and better learning.
        """
        # TCP pose observation/action keys: tcp.{x, y, z, r1..r6}
        self._tcp_pose_keys = (
            "tcp.x",
            "tcp.y",
            "tcp.z",
            "tcp.r1",
            "tcp.r2",
            "tcp.r3",
            "tcp.r4",
            "tcp.r5",
            "tcp.r6",
        )

        # TCP velocity observation keys
        self._tcp_vel_keys = (
            "tcp.vx",
            "tcp.vy",
            "tcp.vz",
            "tcp.wx",
            "tcp.wy",
            "tcp.wz",
        )

        # Joint state keys (for use_joint_observation)
        self._joint_pos_keys = tuple(f"joint_{i}.pos" for i in range(1, JOINT_DOF + 1))
        self._joint_vel_keys = tuple(f"joint_{i}.vel" for i in range(1, JOINT_DOF + 1))
        self._joint_effort_keys = tuple(f"joint_{i}.effort" for i in range(1, JOINT_DOF + 1))

        # Action TCP pose keys
        self._action_tcp_pose_keys = self._tcp_pose_keys

        # Force-related keys
        self._max_contact_wrench = self.config.max_contact_wrench
        if self.config.use_force:
            self._wrench_keys = tuple(f"tcp.{axis}" for axis in ["fx", "fy", "fz", "mx", "my", "mz"])
            self._action_wrench_keys = self._wrench_keys
            self._force_control_axis = tuple(self.config.force_control_axis)

    # =========================================================================
    # Feature descriptors
    # =========================================================================

    @property
    def _action_ft(self) -> dict[str, type]:
        """Return action features for RT Cartesian motion-force mode.

        Action space (always includes gripper):
        - use_force=False: TCP pose (9D: xyz + 6D rotation) + gripper (1D) = 10D
        - use_force=True:  TCP pose (9D) + wrench (6D) + gripper (1D) = 16D
        """
        features = {}

        # TCP pose (9D: xyz + 6D rotation)
        features.update(dict.fromkeys(self._action_tcp_pose_keys, float))
        if self.config.use_force:
            # + target wrench (6D)
            features.update(dict.fromkeys(self._action_wrench_keys, float))

        # Always include gripper (1D)
        features[self._gripper_key] = float
        return features

    @property
    def _proprioception_ft(self) -> dict[str, type]:
        """Return observation features for RT Cartesian mode.

        Observation space (always includes gripper):
        - use_joint_observation=True: joint pos (7D) + vel (7D) + effort (7D) + gripper (1D) = 22D
        - use_joint_observation=False + use_force=False: TCP pose (9D) + gripper (1D) = 10D
        - use_joint_observation=False + use_force=True: TCP pose (9D) + wrench (6D) + gripper (1D) = 16D
        """
        features = {}

        if self.config.use_joint_observation:
            features.update(dict.fromkeys(self._joint_pos_keys, float))
            features.update(dict.fromkeys(self._joint_vel_keys, float))
            features.update(dict.fromkeys(self._joint_effort_keys, float))
        else:
            features.update(dict.fromkeys(self._tcp_pose_keys, float))
            if self.config.use_force:
                features.update(dict.fromkeys(self._wrench_keys, float))

        features[self._gripper_key] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Return camera/image features from gripper and external cameras."""
        features = {}

        if self._gripper and self.config.use_gripper:
            if self.config.gripper_type == "flare_gripper":
                features["wrist_cam"] = (
                    self._gripper._config.cam_size[1],
                    self._gripper._config.cam_size[0],
                    3,
                )
            if self._gripper._config.enable_sensor:
                features["left_tactile"] = (
                    self._gripper._config.rectify_size[1],
                    self._gripper._config.rectify_size[0],
                    3,
                )
                features["right_tactile"] = (
                    self._gripper._config.rectify_size[1],
                    self._gripper._config.rectify_size[0],
                    3,
                )

        for cam in self.cameras:
            features[cam] = (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)

        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Return observation features (robot states + cameras)."""
        return {**self._proprioception_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Return action features for RT Cartesian mode."""
        return self._action_ft

    # =========================================================================
    # Connection state
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self._robot is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        """Flexiv robots are factory calibrated."""
        return self.is_connected

    def calibrate(self) -> None:
        """Flexiv robots are factory calibrated, no runtime calibration needed."""
        self.logger.info("Flexiv Rizon4 is factory calibrated, no runtime calibration needed.")

    # =========================================================================
    # Connect / Disconnect
    # =========================================================================

    def connect(self, calibrate: bool = False, go_to_start: bool = True) -> None:
        """Connect to the Flexiv robot and start RT Cartesian control thread.

        Steps:
        1. Create flexiv_rt.Robot (with retry logic)
        2. Clear faults + enable robot + wait for operational
        3. Connect Flare Gripper and external cameras
        4. Move to start position via MoveJ (using busy() polling)
        5. Zero FT sensor (if configured)
        6. Switch to RT_CARTESIAN_MOTION_FORCE + configure impedance/force
        7. Start C++ RT thread via robot.start_cartesian_control()

        Args:
            calibrate: Ignored (Flexiv robots are factory calibrated)
            go_to_start: If True, move to start position before entering RT mode
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} already connected, do not run `robot.connect()` twice."
            )

        try:
            self.logger.info(f"Connecting to Flexiv robot (RT): {self.config.robot_sn}")

            # --- 1. Create robot interface ---
            self._robot = frt.Robot(
                self.config.robot_sn,
                connect_retries=self.config.connect_retries,
                retry_interval_sec=self.config.retry_interval_sec,
            )

            # --- 2. Clear faults + enable ---
            if self._robot.fault():
                self.logger.warn("Fault occurred on the connected robot, trying to clear ...")
                if not self._robot.ClearFault():
                    raise RuntimeError("Failed to clear robot fault. Check the robot status.")
                self.logger.info("Fault on the connected robot is cleared")

            self.logger.info("Enabling robot...")
            self._robot.Enable()

            timeout = 30  # seconds
            start_time = time.time()
            while not self._robot.operational():
                if time.time() - start_time > timeout:
                    raise RuntimeError(f"Robot did not become operational within {timeout} seconds")
                time.sleep(0.1)

            self.logger.info("Robot is now operational.")

            # --- 3. Connect Flare Gripper + cameras ---
            if self._gripper and self.config.use_gripper:
                self.logger.info("Connecting Flare Gripper...")
                self._gripper.connect()

            for cam in self.cameras.values():
                cam.connect()

            self._is_connected = True

            # --- 3.5 Pre-create + pre-start RT Scheduler in background ---
            # Scheduler() constructor ~2s + Start() ~2s = ~4s total.
            # Launched here so it overlaps with MoveJ (~3s) + ZeroFTSensor (~2s).
            # By the time start_cartesian_control() is called, the Scheduler
            # is already running and activation is near-instant.
            self._robot.precreate_scheduler()

            # --- 4. Move to start position ---
            self.config.go_to_start = go_to_start if go_to_start is not None else self.config.go_to_start
            if self.config.go_to_start:
                self._go_to_start()
                # Cache start TCP pose for RT-mode reset (avoids stopping RT thread)
                self._start_tcp_pose = list(self._robot.states().tcp_pose)

            # --- 5. Zero FT sensor (if configured) ---
            if self.config.zero_ft_sensor_on_connect:
                self._zero_ft_sensor()

            # --- 6. Switch to RT mode + configure ---
            self._switch_to_rt_mode()
            self.configure()

            # --- 7. Start C++ RT thread ---
            # The Flexiv Scheduler's mlockall() is intercepted at link time
            # by __wrap_mlockall (no-op), so no OOM risk regardless of
            # Python process size.  See CMakeLists.txt --wrap=mlockall.
            self._cc = self._robot.start_cartesian_control()
            self.logger.info("C++ RT thread started (1 kHz CartesianMotionForceControl)")

            # Seed the RT thread with the current TCP pose so it doesn't jump
            init_pose = list(self._robot.states().tcp_pose)
            self._cc.set_target_pose(init_pose)
            time.sleep(0.1)  # Let RT thread stabilize

            mode_desc = "RT_CARTESIAN_MOTION_FORCE"
            mode_desc += " (force enabled)" if self.config.use_force else " (motion only)"

            if self._gripper and self.config.use_gripper:
                if self.config.gripper_type == "flare_gripper":
                    gripper_devices = ["gripper", "wrist_cam"]
                    if self._gripper._config.enable_sensor:
                        gripper_devices.append("tactile")
                    gripper_status = f"with FlareGripper ({' + '.join(gripper_devices)})"
                elif self.config.gripper_type == "xense_gripper":
                    gripper_devices = ["gripper"]
                    if self._gripper._config.enable_sensor:
                        gripper_devices.append("tactile")
                    gripper_status = f"with XenseGripper ({' + '.join(gripper_devices)})"
            else:
                gripper_status = "no gripper"
            self.logger.info(f"Flexiv Rizon4 RT connected and ready in {mode_desc} mode ({gripper_status}).")

        except Exception as e:
            self.logger.error(f"Failed to connect to Flexiv robot (RT): {e}")
            # Cleanup on failure
            if self._cc is not None:
                try:
                    self._cc.stop()
                except Exception:
                    pass
                self._cc = None
            self._robot = None
            self._is_connected = False
            raise

    def disconnect(self) -> None:
        """Disconnect from the robot safely.

        Steps:
        1. Stop RT thread (cc.stop() blocks until RT thread joins)
        2. Move to home position
        3. Stop robot
        4. Disconnect gripper and cameras
        5. Clean up resources
        """
        if not self._is_connected:
            self.logger.warn(f"{self} is not connected, skipping disconnect.")
            return

        try:
            self.logger.info("Disconnecting from Flexiv robot (RT)...")

            # 1. Stop RT thread first (blocks until thread joins)
            if self._cc is not None:
                self.logger.info("Stopping RT thread...")
                try:
                    self._cc.stop()
                    self.logger.info("RT thread stopped.")
                except Exception as e:
                    self.logger.warn(f"Error stopping RT thread: {e}")
                self._cc = None

            # 2. Move to home position
            try:
                self._go_to_home()
            except Exception as e:
                self.logger.warn(f"Failed to move to home before disconnect: {e}")

            # 3. Stop any ongoing motion
            if self._robot is not None:
                try:
                    self._robot.Stop()
                except Exception as e:
                    self.logger.warn(f"Error calling robot.Stop(): {e}")

            # 4. Disconnect gripper + cameras
            if self._gripper and self.config.use_gripper:
                try:
                    self._gripper.disconnect()
                except Exception as e:
                    self.logger.warn(f"Error disconnecting gripper: {e}")

            for cam in self.cameras.values():
                try:
                    cam.disconnect()
                except Exception as e:
                    self.logger.warn(f"Error disconnecting camera: {e}")

        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
        finally:
            if self._robot is not None:
                try:
                    self._robot.close()
                except Exception:
                    pass
            self._robot = None
            self._cc = None
            self._gripper = None
            self._is_connected = False
            self.logger.info("Flexiv Rizon4 RT disconnected.")

    # =========================================================================
    # Configuration
    # =========================================================================

    def configure(self) -> None:
        """Configure the robot for RT Cartesian motion-force control.

        Sets Cartesian impedance and force control parameters before the RT thread
        is started. Must be called after switching to RT_CARTESIAN_MOTION_FORCE mode.
        """
        if not self._is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info("Configuring RT Cartesian mode...")

        robot_info = self._robot.info()

        # Cartesian impedance
        if self.config.stiffness_ratio != 1.0:
            K_x_nom = robot_info.K_x_nom
            new_kx = list(np.multiply(K_x_nom, self.config.stiffness_ratio))
            self._robot.SetCartesianImpedance(new_kx, self.config.damping_ratio)
            self.logger.info(
                f"Cartesian stiffness (ratio={self.config.stiffness_ratio}): {new_kx}"
            )
        else:
            self.logger.info(f"Using nominal Cartesian stiffness: {robot_info.K_x_nom}")

        # Force control configuration
        if self.config.use_force:
            # Set force control frame
            self._robot.SetForceControlFrame(self.config.force_control_frame)
            self.logger.info(f"Force control frame: {self.config.force_control_frame}")

            # Set which axes use force control
            self._robot.SetForceControlAxis(list(self.config.force_control_axis))
            self.logger.info(f"Force control axis: {self.config.force_control_axis}")

            # Disable max contact wrench regulation (force-controlled axes are explicit)
            self._robot.SetMaxContactWrench([float("inf")] * 6)
            self.logger.info("Max contact wrench regulation disabled (force control active)")
        else:
            # Pure motion control: set safety wrench limits
            self._robot.SetMaxContactWrench(self._max_contact_wrench)
            self.logger.info(f"Max contact wrench: {self._max_contact_wrench}")

    # =========================================================================
    # Internal: mode switching and primitives
    # =========================================================================

    def _switch_to_rt_mode(self) -> None:
        """Switch to RT_CARTESIAN_MOTION_FORCE mode.

        This is the only RT mode supported by this driver.
        """
        if not self._is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        target_mode = frt.Mode.RT_CARTESIAN_MOTION_FORCE
        current_mode = self._robot.mode()

        if current_mode == target_mode:
            self.logger.info("Already in RT_CARTESIAN_MOTION_FORCE mode.")
            return

        self.logger.info(f"Switching from {current_mode} to RT_CARTESIAN_MOTION_FORCE...")

        if self._robot.fault():
            self.logger.warn("Robot has fault, attempting to clear...")
            self._robot.ClearFault()
            time.sleep(0.5)
            if self._robot.fault():
                raise RuntimeError("Failed to clear robot fault before mode switch")

        self._robot.SwitchMode(target_mode)

        # Wait for mode switch
        max_wait = 2.0  # seconds
        wait_start = time.perf_counter()
        while time.perf_counter() - wait_start < max_wait:
            actual_mode = self._robot.mode()
            if actual_mode == target_mode:
                self.logger.info("Now in RT_CARTESIAN_MOTION_FORCE mode.")
                return
            time.sleep(0.1)

        actual_mode = self._robot.mode()
        raise RuntimeError(
            f"Mode switch failed: expected {target_mode}, got {actual_mode}. "
            f"Robot fault: {self._robot.fault()}"
        )

    def _go_to_home(self) -> None:
        """Move robot to factory home position using PLAN-Home.

        Uses NRT_PLAN_EXECUTION + ExecutePlan("PLAN-Home") + busy() polling,
        which is the recommended pattern for flexiv_rt (no primitive_states).
        """
        if not self._is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info("Moving to home position...")

        self._robot.Stop()

        self._robot.SwitchMode(frt.Mode.NRT_PLAN_EXECUTION)
        self._robot.ExecutePlan("PLAN-Home")

        # Initialize gripper position during move
        if self._gripper is not None:
            if self.config.gripper_init_open:
                self._gripper._gripper.set_position_sync(
                    self.config.gripper_max_pos,
                    vmax=self.config.gripper_v_max / 2,
                    fmax=self.config.gripper_f_max / 2,
                )
            else:
                self._gripper._gripper.set_position_sync(
                    0.0,
                    vmax=self.config.gripper_v_max / 2,
                    fmax=self.config.gripper_f_max / 2,
                )

        # Wait for plan to finish
        timeout = 30.0
        start_time = time.time()
        while self._robot.busy():
            if time.time() - start_time > timeout:
                self.logger.error(
                    f"PLAN-Home timeout after {timeout}s, "
                    f"fault={self._robot.fault()}, mode={self._robot.mode()}"
                )
                raise RuntimeError(f"PLAN-Home did not complete within {timeout}s")
            time.sleep(0.1)

        self._home_tcp_pose = np.array(self._robot.states().tcp_pose)
        self.logger.info(f"Home TCP pose: {self._home_tcp_pose}")
        self.logger.info("Robot at home position.")

    def _go_to_start(self) -> None:
        """Move robot to configured start position using MoveJ primitive.

        Uses busy() polling to wait for completion, consistent with
        _go_to_home() pattern.
        """
        if not self._is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info("Moving to start position...")

        # Stop any stale motion from a previous (possibly Ctrl-C'd) session
        self._robot.Stop()

        self._robot.SwitchMode(frt.Mode.NRT_PRIMITIVE_EXECUTION)

        self._robot.ExecutePrimitive(
            "MoveJ",
            {
                "target": self.config.start_position_degree,
                "jntVelScale": self.config.start_vel_scale,
            },
        )
        self.logger.info("MoveJ command sent, waiting for completion...")

        # Initialize gripper position during move
        if self._gripper is not None:
            if self.config.gripper_init_open:
                self._gripper._gripper.set_position_sync(
                    self.config.gripper_max_pos,
                    vmax=self.config.gripper_v_max / 2,
                    fmax=self.config.gripper_f_max / 2,
                )
            else:
                self._gripper._gripper.set_position_sync(
                    0.0,
                    vmax=self.config.gripper_v_max / 2,
                    fmax=self.config.gripper_f_max / 2,
                )

        # Wait for MoveJ to complete.
        # NOTE: Flexiv SDK docs warn that most primitives won't cause busy()
        # to become False — must check primitive_states()["reachedTarget"].
        timeout = 30.0  # seconds
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                self.logger.error(
                    f"MoveJ timeout after {timeout}s, "
                    f"fault={self._robot.fault()}, mode={self._robot.mode()}"
                )
                raise RuntimeError(f"MoveJ did not complete within {timeout}s")
            try:
                pt_states = self._robot.primitive_states()
                if pt_states.get("reachedTarget", 0) == 1:
                    break
            except Exception:
                pass
            time.sleep(0.1)

        self.logger.info("Robot at start position.")
        if self._gripper is not None:
            self.logger.info(f"Gripper position: {self._gripper.get_gripper_position()}")

    def _zero_ft_sensor(self) -> None:
        """Zero force-torque sensor offset.

        IMPORTANT: Robot must not contact anything during zeroing.
        Uses busy() polling to wait for completion (~1-1.5s typical).
        """
        if not self._is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.warn("Zeroing force-torque sensors, make sure nothing is in contact with the robot")

        # Skip redundant mode switch if already in NRT_PRIMITIVE_EXECUTION (e.g. after MoveJ)
        if self._robot.mode() != frt.Mode.NRT_PRIMITIVE_EXECUTION:
            self._robot.SwitchMode(frt.Mode.NRT_PRIMITIVE_EXECUTION)
        self._robot.ExecutePrimitive("ZeroFTSensor", {})

        # Poll primitive_states()["terminated"] for completion (per Flexiv SDK example)
        timeout = 10.0
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                self.logger.error(f"ZeroFTSensor timeout after {timeout}s")
                break
            try:
                pt_states = self._robot.primitive_states()
                if pt_states.get("terminated", 0) == 1:
                    break
            except Exception:
                pass
            time.sleep(0.1)

        self.logger.info("Force-torque sensor zeroed")

    # =========================================================================
    # Observation
    # =========================================================================

    def get_observation(self) -> dict[str, Any]:
        """Get current robot observation from the RT thread's shared memory.

        Reads CartesianState via cc.get_state() which is protected by mutex and
        returns a snapshot of the latest state written by the RT thread.

        Returns a dictionary with observation data:
        - use_joint_observation=True: joint_1-7.{pos,vel,effort} (21D) + gripper (1D) = 22D
        - use_joint_observation=False + use_force=False: tcp.{x,y,z,r1-r6} (9D) + gripper (1D) = 10D
        - use_joint_observation=False + use_force=True: tcp pose (9D) + wrench (6D) + gripper (1D) = 16D

        Also includes camera images if configured.
        """
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}

        if self._cc is not None and self._cc.is_running():
            # RT mode: read from shared memory via CartesianState
            state = self._cc.get_state()

            if self.config.use_joint_observation:
                # Joint positions (7D)
                for i, key in enumerate(self._joint_pos_keys):
                    obs_dict[key] = state.q[i]

                # Joint velocities: not directly in CartesianState, read from robot states
                robot_states = self._robot.states()
                for i, key in enumerate(self._joint_vel_keys):
                    obs_dict[key] = robot_states.dq[i]

                # Joint efforts/torques: use tau_ext from CartesianState
                for i, key in enumerate(self._joint_effort_keys):
                    obs_dict[key] = state.tau_ext[i]

            else:
                # TCP pose from CartesianState: [x, y, z, qw, qx, qy, qz]
                tcp_pose = state.tcp_pose

                obs_dict["tcp.x"] = tcp_pose[0]
                obs_dict["tcp.y"] = tcp_pose[1]
                obs_dict["tcp.z"] = tcp_pose[2]

                # Convert quaternion to 6D rotation
                r6d = quaternion_to_rotation_6d(tcp_pose[3], tcp_pose[4], tcp_pose[5], tcp_pose[6])
                obs_dict["tcp.r1"] = r6d[0]
                obs_dict["tcp.r2"] = r6d[1]
                obs_dict["tcp.r3"] = r6d[2]
                obs_dict["tcp.r4"] = r6d[3]
                obs_dict["tcp.r5"] = r6d[4]
                obs_dict["tcp.r6"] = r6d[5]

                if self.config.use_force:
                    ext_wrench = state.ext_wrench_in_tcp
                    for i, key in enumerate(self._wrench_keys):
                        obs_dict[key] = ext_wrench[i]

        else:
            # Fallback: not in RT mode, read from robot.states() directly
            states = self._robot.states()

            if self.config.use_joint_observation:
                for i, key in enumerate(self._joint_pos_keys):
                    obs_dict[key] = states.q[i]
                for i, key in enumerate(self._joint_vel_keys):
                    obs_dict[key] = states.dq[i]
                for i, key in enumerate(self._joint_effort_keys):
                    obs_dict[key] = states.tau[i]
            else:
                tcp_pose = states.tcp_pose
                obs_dict["tcp.x"] = tcp_pose[0]
                obs_dict["tcp.y"] = tcp_pose[1]
                obs_dict["tcp.z"] = tcp_pose[2]

                r6d = quaternion_to_rotation_6d(tcp_pose[3], tcp_pose[4], tcp_pose[5], tcp_pose[6])
                obs_dict["tcp.r1"] = r6d[0]
                obs_dict["tcp.r2"] = r6d[1]
                obs_dict["tcp.r3"] = r6d[2]
                obs_dict["tcp.r4"] = r6d[3]
                obs_dict["tcp.r5"] = r6d[4]
                obs_dict["tcp.r6"] = r6d[5]

                if self.config.use_force:
                    ext_wrench = states.ext_wrench_in_tcp
                    for i, key in enumerate(self._wrench_keys):
                        obs_dict[key] = ext_wrench[i]

        # --- Gripper data (gripper + wrist_cam + tactile) ---
        if self._gripper is not None and self.config.use_gripper:
            if self._gripper._enable_sensor:
                sensor_data = self._gripper.get_sensor_data()
                for key, data in sensor_data.items():
                    obs_dict[key] = data

            if self.config.gripper_type == "flare_gripper":
                camera_frame = self._gripper.get_camera_frame()
                if camera_frame is not None:
                    obs_dict["wrist_cam"] = camera_frame
                else:
                    h, w = self._gripper._config.cam_size[1], self._gripper._config.cam_size[0]
                    obs_dict["wrist_cam"] = np.zeros((h, w, 3), dtype=np.uint8)

            obs_dict[self._gripper_key] = self._gripper.get_gripper_position()

        # --- External cameras ---
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    # =========================================================================
    # Action
    # =========================================================================

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action command via the RT thread's shared memory.

        Writes target pose (and wrench) to SHM via cc.set_target_pose().
        The C++ RT thread will pick this up within 1 ms and stream it to the robot.

        If an RT trajectory is in progress (is_moving), external commands are
        skipped to avoid interfering with the trajectory. Gripper commands are
        still sent.

        The action uses 6D rotation representation which is converted to quaternion
        for the Flexiv SDK.

        Action format:
        - use_force=False: {tcp.x, y, z, r1-r6} (9D) + gripper.pos (1D) = 10D
        - use_force=True: tcp pose (9D) + {tcp.fx, fy, fz, mx, my, mz} (6D) + gripper.pos (1D) = 16D

        Args:
            action: Dictionary of action values

        Returns:
            The action that was actually sent
        """
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._cc is None or not self._cc.is_running():
            raise RuntimeError("RT thread is not running. Call connect() first.")

        # Skip arm commands while RT trajectory is in progress (e.g. reset)
        # Gripper is independent and still gets commands.
        if self._cc.is_moving():
            self._send_gripper_action(action)
            return action

        # --- Build target pose [x, y, z, qw, qx, qy, qz] ---
        x, y, z = action["tcp.x"], action["tcp.y"], action["tcp.z"]

        r6d = np.array([
            action["tcp.r1"],
            action["tcp.r2"],
            action["tcp.r3"],
            action["tcp.r4"],
            action["tcp.r5"],
            action["tcp.r6"],
        ])
        quat = rotation_6d_to_quaternion(r6d)  # [qw, qx, qy, qz]
        target_pose = [x, y, z, float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]

        # --- Build target wrench (if force control) ---
        if self.config.use_force:
            target_wrench = [action[key] for key in self._action_wrench_keys]
            self._cc.set_target_pose(target_pose, target_wrench)
        else:
            self._cc.set_target_pose(target_pose)

        # --- Gripper ---
        self._send_gripper_action(action)

        return action

    def _send_gripper_action(self, action: dict[str, Any]) -> None:
        """Send gripper command.

        Action key: gripper.pos (normalized 0-1)
        """
        if not self._gripper or not self.config.use_gripper:
            return

        if self._gripper_key not in action:
            return

        self._gripper.set_gripper_position(action[self._gripper_key])

    # =========================================================================
    # RT thread management
    # =========================================================================

    def trigger_estop(self) -> None:
        """Trigger emergency stop on the RT thread.

        Uses lockless atomic flag - takes effect within the next RT cycle (1 ms).
        The RT thread will hold the last commanded pose.
        """
        if self._cc is not None:
            self._cc.trigger_estop()
            self.logger.warn("E-stop triggered on RT thread")

    @property
    def rt_running(self) -> bool:
        """Whether the C++ RT thread is actively running."""
        return self._cc is not None and self._cc.is_running()

    @property
    def rt_moving(self) -> bool:
        """Whether the RT thread is executing a trajectory (e.g. from reset)."""
        return self._cc is not None and self._cc.is_moving()

    # =========================================================================
    # Utility methods
    # =========================================================================

    def zero_ft_sensor(self) -> None:
        """Zero force-torque sensor (public API).

        If RT thread is running, stops it first, zeros sensors, then restarts.
        """
        was_running = self.rt_running

        if was_running and self._cc is not None:
            self._cc.stop()
            self._cc = None
            self.logger.info("Stopped RT thread for FT sensor zeroing")

        self._zero_ft_sensor()

        if was_running and self._robot is not None:
            self._switch_to_rt_mode()
            self._cc = self._robot.start_cartesian_control()
            init_pose = list(self._robot.states().tcp_pose)
            self._cc.set_target_pose(init_pose)
            time.sleep(0.1)
            self.logger.info("Restarted RT thread after FT sensor zeroing")

    def clear_fault(self) -> bool:
        """Attempt to clear robot fault.

        Returns:
            True if fault was cleared, False otherwise
        """
        if self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self._robot.fault():
            self.logger.info("No fault to clear.")
            return True

        self.logger.info("Attempting to clear fault...")
        result = self._robot.ClearFault()
        if result:
            self.logger.info("Fault cleared successfully.")
        else:
            self.logger.error("Failed to clear fault.")
        return result

    def reset_to_initial_position(self) -> None:
        """Reset robot to initial position using RT-mode trajectory (non-blocking).

        Starts a min-jerk trajectory in the RT thread toward the cached start pose
        and returns immediately. While the trajectory is in progress:
        - get_observation() keeps working (display updates normally)
        - send_action() skips external commands (trajectory has priority)
        - is_moving property returns True

        Idempotent: if a trajectory is already running, this is a no-op.

        Falls back to the legacy blocking approach (stop RT → NRT MoveJ → restart RT)
        if the fast path is not available.
        """
        if not self._is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Fast path: non-blocking RT-mode trajectory to cached start pose
        if (
            self.config.go_to_start
            and self._start_tcp_pose is not None
            and self._cc is not None
            and self._cc.is_running()
        ):
            # Idempotent: skip if trajectory already running
            if self._cc.is_moving():
                return

            self.logger.info("Resetting to start position via RT trajectory (non-blocking)")
            self._cc.move_to_pose(self._start_tcp_pose, duration_sec=3.0)
            return

        # Fallback: legacy blocking approach (stop RT → NRT move → restart RT)
        self.logger.info("Using legacy reset (stop RT → NRT move → restart RT)")

        # Stop RT thread
        if self._cc is not None:
            self._cc.stop()
            self._cc = None

        # Move to position
        if self.config.go_to_start:
            self.logger.info("Resetting to start position")
            self._go_to_start()
        else:
            self.logger.info("Resetting to home position")
            self._go_to_home()

        # Restart RT mode
        self._switch_to_rt_mode()
        self.configure()
        self._cc = self._robot.start_cartesian_control()
        init_pose = list(self._robot.states().tcp_pose)
        self._cc.set_target_pose(init_pose)
        time.sleep(0.1)
        self.logger.info("RT thread restarted after position reset")

    def get_current_tcp_pose_euler(self) -> np.ndarray:
        """Get current TCP pose in Euler angles format [x, y, z, roll, pitch, yaw, gripper_pos].

        Reads from RT shared memory if available, otherwise from robot.states().

        Returns:
            numpy array of shape (7,) with [x, y, z, roll, pitch, yaw, gripper_pos]
        """
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Get TCP pose from RT state or robot states
        if self._cc is not None and self._cc.is_running():
            tcp_pose = self._cc.get_state().tcp_pose
        else:
            tcp_pose = self._robot.states().tcp_pose

        euler = quaternion_to_euler(tcp_pose[3], tcp_pose[4], tcp_pose[5], tcp_pose[6])

        gripper_pos = 0.0
        if self._gripper and self.config.use_gripper:
            gripper_pos = self._gripper.get_gripper_position()

        return np.array(
            [tcp_pose[0], tcp_pose[1], tcp_pose[2], euler[0], euler[1], euler[2], gripper_pos],
            dtype=np.float32,
        )

    def get_current_tcp_pose_quat(self) -> np.ndarray:
        """Get current TCP pose in quaternion format [x, y, z, qw, qx, qy, qz, gripper_pos].

        Reads from RT shared memory if available, otherwise from robot.states().

        Returns:
            numpy array of shape (8,) with [x, y, z, qw, qx, qy, qz, gripper_pos]
        """
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._cc is not None and self._cc.is_running():
            tcp_pose = self._cc.get_state().tcp_pose
        else:
            tcp_pose = self._robot.states().tcp_pose

        gripper_pos = 0.0
        if self._gripper and self.config.use_gripper:
            gripper_pos = self._gripper.get_gripper_position()

        return np.array(
            [*tcp_pose, gripper_pos],
            dtype=np.float32,
        )
