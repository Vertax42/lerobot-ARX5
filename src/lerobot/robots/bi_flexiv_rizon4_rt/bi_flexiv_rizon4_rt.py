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

"""BiFlexivRizon4RT: Bimanual Flexiv Rizon4 robot with real-time control.

Each arm runs an independent flexiv_rt.Robot instance with a 1 kHz C++ RT thread
(SCHED_FIFO). Python-side send_action() writes target poses to shared memory at
30-100 Hz; the RT thread streams them to each robot via StreamCartesianMotionForce.

Control Architecture (per arm):
    send_action(30-100 Hz)
        -> cc.set_target_pose(pose, wrench)  [writes to SHM via mutex]
        -> C++ RT thread reads SHM every 1 ms
        -> StreamCartesianMotionForce(pose, wrench, vel, acc)
        -> Robot

Action / Observation key prefixes:
    left_tcp.{x, y, z, r1-r6}   left_gripper.pos
    right_tcp.{x, y, z, r1-r6}  right_gripper.pos

6D Rotation Representation:
    r1, r2, r3: First column of rotation matrix
    r4, r5, r6: Second column of rotation matrix
    Reference: "On the Continuity of Rotation Representations in Neural Networks"
"""

import time
from concurrent.futures import ThreadPoolExecutor, wait
from functools import cached_property
from typing import Any

import flexiv_rt as frt
import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import BiFlexivRizon4RTConfig
from lerobot.robots.bi_flexiv_rizon4_rt.serial_gripper import SerialGripper
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import (
    get_logger,
    quaternion_to_rotation_6d,
    rotation_6d_to_quaternion,
)

# Constants (per arm)
JOINT_DOF = 7  # Flexiv Rizon4 robot joint DOF
POSE_SIZE_QUAT = 7  # [x, y, z, qw, qx, qy, qz]
POSE_SIZE_6D = 9  # [x, y, z, r1..r6]


class BiFlexivRizon4RT(Robot):
    """Bimanual Flexiv Rizon4 robot with real-time Cartesian control.

    Controls two Flexiv Rizon4 arms independently via flexiv_rt. Each arm runs
    its own 1 kHz C++ RT thread. Python-side send_action() writes target poses
    to each arm's shared memory at 30-100 Hz.

    Example:
        >>> config = BiFlexivRizon4RTConfig(
        ...     left_robot_sn="Rizon4-063423",
        ...     right_robot_sn="Rizon4-063424",
        ... )
        >>> robot = BiFlexivRizon4RT(config)
        >>> robot.connect()
        >>> obs = robot.get_observation()
        >>> robot.send_action(
        ...     {
        ...         "left_tcp.x": 0.5,
        ...         "left_tcp.y": 0.1,
        ...         "left_tcp.z": 0.3,
        ...         "left_tcp.r1": 1.0,
        ...         "left_tcp.r2": 0.0,
        ...         "left_tcp.r3": 0.0,
        ...         "left_tcp.r4": 0.0,
        ...         "left_tcp.r5": 1.0,
        ...         "left_tcp.r6": 0.0,
        ...         "left_gripper.pos": 0.8,
        ...         "right_tcp.x": 0.5,
        ...         "right_tcp.y": -0.1,
        ...         "right_tcp.z": 0.3,
        ...         "right_tcp.r1": 1.0,
        ...         "right_tcp.r2": 0.0,
        ...         "right_tcp.r3": 0.0,
        ...         "right_tcp.r4": 0.0,
        ...         "right_tcp.r5": 1.0,
        ...         "right_tcp.r6": 0.0,
        ...         "right_gripper.pos": 0.8,
        ...     }
        ... )
        >>> robot.disconnect()
    """

    config_class = BiFlexivRizon4RTConfig
    name = "bi_flexiv_rizon4_rt"

    def __init__(self, config: BiFlexivRizon4RTConfig):
        super().__init__(config)
        self.config = config

        self.logger = get_logger("BiFlexivRizon4RT", loglevel=config.log_level)

        # Robot interfaces (one per arm, initialized on connect)
        self._left_robot: frt.Robot | None = None
        self._right_robot: frt.Robot | None = None
        self._left_cc: frt.CartesianMotionForceControl | None = None
        self._right_cc: frt.CartesianMotionForceControl | None = None
        self._is_connected = False

        # Grippers
        self._left_gripper: SerialGripper | None = None
        if config.left_use_gripper:
            self._left_gripper = SerialGripper(config.left_gripper)
        else:
            self.logger.info("Left arm: no gripper configured.")

        self._right_gripper: SerialGripper | None = None
        if config.right_use_gripper:
            self._right_gripper = SerialGripper(config.right_gripper)
        else:
            self.logger.info("Right arm: no gripper configured.")

        # Cached start TCP poses for RT-mode reset
        self._left_start_tcp_pose: list[float] | None = None
        self._right_start_tcp_pose: list[float] | None = None

        # Initialize key tuples
        self._init_keys()

        # External cameras
        self.cameras = make_cameras_from_configs(config.cameras)
        np.set_printoptions(precision=6, suppress=True)

    # =========================================================================
    # Key initialization
    # =========================================================================

    def _init_keys(self) -> None:
        """Initialize observation/action key tuples for both arms."""
        self._left_tcp_pose_keys = (
            "left_tcp.x",
            "left_tcp.y",
            "left_tcp.z",
            "left_tcp.r1",
            "left_tcp.r2",
            "left_tcp.r3",
            "left_tcp.r4",
            "left_tcp.r5",
            "left_tcp.r6",
        )
        self._right_tcp_pose_keys = (
            "right_tcp.x",
            "right_tcp.y",
            "right_tcp.z",
            "right_tcp.r1",
            "right_tcp.r2",
            "right_tcp.r3",
            "right_tcp.r4",
            "right_tcp.r5",
            "right_tcp.r6",
        )

        self._left_gripper_key = "left_gripper.pos"
        self._right_gripper_key = "right_gripper.pos"

        self._max_contact_wrench = self.config.max_contact_wrench

        if self.config.use_force:
            self._left_wrench_keys = tuple(
                f"left_tcp.{axis}" for axis in ["fx", "fy", "fz", "mx", "my", "mz"]
            )
            self._right_wrench_keys = tuple(
                f"right_tcp.{axis}" for axis in ["fx", "fy", "fz", "mx", "my", "mz"]
            )

    # =========================================================================
    # Feature descriptors
    # =========================================================================

    @property
    def _action_ft(self) -> dict[str, type]:
        """Action features for both arms.

        use_force=False: left+right TCP pose (9D each) + gripper (1D each) = 20D
        use_force=True: TCP pose (9D) + wrench (6D) + gripper (1D) per arm = 32D
        """
        features = {}
        features.update(dict.fromkeys(self._left_tcp_pose_keys, float))
        features.update(dict.fromkeys(self._right_tcp_pose_keys, float))
        if self.config.use_force:
            features.update(dict.fromkeys(self._left_wrench_keys, float))
            features.update(dict.fromkeys(self._right_wrench_keys, float))
        features[self._left_gripper_key] = float
        features[self._right_gripper_key] = float
        return features

    @property
    def _proprioception_ft(self) -> dict[str, type]:
        """Observation proprioception features for both arms.

        Per arm: TCP pose (9D) + wrench (6D, if use_force) + gripper (1D).
        """
        features = {}
        features.update(dict.fromkeys(self._left_tcp_pose_keys, float))
        features.update(dict.fromkeys(self._right_tcp_pose_keys, float))
        if self.config.use_force:
            features.update(dict.fromkeys(self._left_wrench_keys, float))
            features.update(dict.fromkeys(self._right_wrench_keys, float))
        features[self._left_gripper_key] = float
        features[self._right_gripper_key] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera/image features from external cameras."""
        features = {}

        for cam_name in self.cameras:
            features[cam_name] = (
                self.config.cameras[cam_name].height,
                self.config.cameras[cam_name].width,
                3,
            )

        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._proprioception_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._action_ft

    # =========================================================================
    # Connection state
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self._left_robot is not None
            and self._right_robot is not None
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
        """Connect to both Flexiv arms and start their RT Cartesian control threads.

        Steps (same for both arms, run sequentially):
        1. Create flexiv_rt.Robot for each arm (with retry)
        2. Clear faults + enable + wait for operational
        3. Connect grippers and external cameras
        4. Move each arm to its start position via MoveJ
        5. Zero FT sensors (if configured)
        6. Switch both to RT_CARTESIAN_MOTION_FORCE + configure impedance/force
        7. Start C++ RT threads via robot.start_cartesian_control()

        Args:
            calibrate: Ignored (factory calibrated)
            go_to_start: If True, move both arms to start positions before entering RT mode
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} already connected, do not run `robot.connect()` twice."
            )

        try:
            # --- 1. Create robot interfaces (both arms in parallel) ---
            self.logger.info(
                f"Connecting to both arms in parallel: "
                f"left={self.config.left_robot_sn}, right={self.config.right_robot_sn}"
            )

            def _connect_arm(sn):
                return frt.Robot(
                    sn,
                    connect_retries=self.config.connect_retries,
                    retry_interval_sec=self.config.retry_interval_sec,
                )

            with ThreadPoolExecutor(max_workers=2) as ex:
                left_fut = ex.submit(_connect_arm, self.config.left_robot_sn)
                right_fut = ex.submit(_connect_arm, self.config.right_robot_sn)
                self._left_robot = left_fut.result()
                self._right_robot = right_fut.result()

            # --- 2. Clear faults + enable both arms ---
            for side, robot in [("left", self._left_robot), ("right", self._right_robot)]:
                if robot.fault():
                    self.logger.warn(f"{side} arm: fault detected, attempting to clear...")
                    if not robot.ClearFault():
                        raise RuntimeError(f"Failed to clear {side} arm fault.")
                    self.logger.info(f"{side} arm: fault cleared.")

                self.logger.info(f"Enabling {side} arm...")
                robot.Enable()

            # Wait for both arms to become operational (no logs here used to look like a hang;
            # grippers/cameras run only after this step).
            timeout = 30
            start_time = time.time()
            last_log = start_time
            self.logger.info(
                f"Waiting for both arms to report operational (timeout {timeout}s)..."
            )
            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise RuntimeError(f"Arms did not become operational within {timeout}s")
                left_ready = self._left_robot.operational()
                right_ready = self._right_robot.operational()
                if left_ready and right_ready:
                    break
                now = time.time()
                if now - last_log >= 2.0:
                    self.logger.info(
                        f"Still waiting for operational: left={left_ready}, right={right_ready} "
                        f"({elapsed:.1f}s / {timeout}s)"
                    )
                    last_log = now
                time.sleep(0.1)

            self.logger.info("Both arms are operational.")

            # --- 3. Connect grippers + cameras (grippers in parallel) ---
            def _connect_gripper(gripper, use_gripper, side):
                if gripper and use_gripper:
                    self.logger.info(f"Connecting {side} gripper...")
                    gripper.connect()

            with ThreadPoolExecutor(max_workers=2) as ex:
                lg = ex.submit(_connect_gripper, self._left_gripper, self.config.left_use_gripper, "left")
                rg = ex.submit(_connect_gripper, self._right_gripper, self.config.right_use_gripper, "right")
                lg.result()
                rg.result()

            if not self.cameras:
                self.logger.info("No cameras configured; skipping camera connect.")
            else:
                self.logger.info(
                    f"Connecting {len(self.cameras)} camera(s): {', '.join(self.cameras.keys())}..."
                )
            with ThreadPoolExecutor(max_workers=len(self.cameras) or 1) as ex:
                cam_futs = [ex.submit(cam.connect) for cam in self.cameras.values()]
                for f in cam_futs:
                    f.result()
            if self.cameras:
                self.logger.info("All cameras connected.")

            self._is_connected = True

            # --- 3.5 Pre-create RT Schedulers in background (overlaps with MoveJ) ---
            self._left_robot.precreate_scheduler(self.config.left_rt_cpu_affinity, task_name="CartesianRT_L")
            self._right_robot.precreate_scheduler(self.config.right_rt_cpu_affinity, task_name="CartesianRT_R")

            # --- 4. Move to start positions ---
            self.config.go_to_start = go_to_start if go_to_start is not None else self.config.go_to_start
            if self.config.go_to_start:
                self._go_to_start()
                self._left_start_tcp_pose = list(self._left_robot.states().tcp_pose)
                self._right_start_tcp_pose = list(self._right_robot.states().tcp_pose)

            # --- 5. Zero FT sensors (both arms in parallel) ---
            if self.config.zero_ft_sensor_on_connect:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    lf = ex.submit(self._zero_ft_sensor_left)
                    rf = ex.submit(self._zero_ft_sensor_right)
                    lf.result()
                    rf.result()

            # --- 6. Switch to RT mode + configure (both arms in parallel) ---
            with ThreadPoolExecutor(max_workers=2) as ex:
                ls = ex.submit(self._switch_to_rt_mode, self._left_robot, "left")
                rs = ex.submit(self._switch_to_rt_mode, self._right_robot, "right")
                ls.result()
                rs.result()
            self._configure_arm(self._left_robot, "left")
            self._configure_arm(self._right_robot, "right")

            # --- 7. Start RT threads ---
            self._left_cc = self._start_cartesian_control(self._left_robot, "CartesianRT_L", "left")

            # Seed the RT thread with current TCP pose to avoid jump
            left_init_pose = list(self._left_robot.states().tcp_pose)
            self._left_cc.set_target_pose(left_init_pose)

            self._right_cc = self._start_cartesian_control(self._right_robot, "CartesianRT_R", "right")

            right_init_pose = list(self._right_robot.states().tcp_pose)
            self._right_cc.set_target_pose(right_init_pose)

            time.sleep(0.1)  # Let RT threads stabilize

            self.logger.info("BiFlexivRizon4RT connected and ready.")

        except Exception as e:
            self.logger.error(f"Failed to connect BiFlexivRizon4RT: {e}")
            self._cleanup_on_failure()
            raise

    def disconnect(self) -> None:
        """Disconnect both arms safely.

        Steps:
        1. Stop RT threads
        2. Send gripper open command (non-blocking, runs in parallel with arm home)
        3. Move both arms to home position (blocking)
        4. Stop robots
        5. Disconnect grippers and cameras
        """
        if not self._is_connected:
            self.logger.warn(f"{self} is not connected, skipping disconnect.")
            return

        try:
            self.logger.info("Disconnecting BiFlexivRizon4RT...")

            # 1. Stop both RT threads in parallel.
            #    cc.stop() joins the RT thread AND calls robot_.Stop() at the
            #    C++ level (no timeliness gap).
            def _stop_rt(cc, side):
                if cc is not None:
                    cc.stop()
                    self.logger.info(f"{side} arm: RT thread stopped.")

            with ThreadPoolExecutor(max_workers=2) as executor:
                stop_futures = {
                    "left": executor.submit(_stop_rt, self._left_cc, "left"),
                    "right": executor.submit(_stop_rt, self._right_cc, "right"),
                }
                for side, fut in stop_futures.items():
                    try:
                        fut.result()
                    except Exception as e:
                        self.logger.warn(f"{side} arm: error stopping RT: {e}")
            self._left_cc = None
            self._right_cc = None

            # 2. Trigger gripper open *before* the blocking arm home movement so
            #    the gripper opens in parallel while the arms are returning home.
            for side, gripper, use_gripper in [
                ("left", self._left_gripper, self.config.left_use_gripper),
                ("right", self._right_gripper, self.config.right_use_gripper),
            ]:
                if gripper and use_gripper:
                    try:
                        gripper.set_gripper_position(1.0)
                        self.logger.info(f"{side} gripper: open command sent.")
                    except Exception as e:
                        self.logger.warn(f"{side} gripper open command failed (non-fatal): {e}")

            # 3. Move both arms to home in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                home_futures = {
                    "left": executor.submit(self._go_to_home_arm, self._left_robot, "left"),
                    "right": executor.submit(self._go_to_home_arm, self._right_robot, "right"),
                }
                for side, fut in home_futures.items():
                    try:
                        fut.result()
                    except Exception as e:
                        self.logger.warn(f"Failed to move {side} arm to home: {e}")

            # 4. Stop robots
            for side, robot in [("left", self._left_robot), ("right", self._right_robot)]:
                if robot is not None:
                    try:
                        robot.Stop()
                    except Exception as e:
                        self.logger.warn(f"{side} arm: error calling Stop(): {e}")

            # 5. Disconnect grippers + cameras
            for side, gripper, use_gripper in [
                ("left", self._left_gripper, self.config.left_use_gripper),
                ("right", self._right_gripper, self.config.right_use_gripper),
            ]:
                if gripper and use_gripper:
                    try:
                        gripper.disconnect()
                    except Exception as e:
                        self.logger.warn(f"{side} gripper disconnect error: {e}")

            for cam in self.cameras.values():
                try:
                    cam.disconnect()
                except Exception as e:
                    self.logger.warn(f"Camera disconnect error: {e}")

        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
        finally:
            self._cleanup_resources()
            self.logger.info("BiFlexivRizon4RT disconnected.")

    def _cleanup_on_failure(self) -> None:
        """Clean up resources after a connection failure."""
        for cc in [self._left_cc, self._right_cc]:
            if cc is not None:
                try:
                    cc.stop()
                except Exception:
                    pass

        # Disconnect any grippers that were successfully connected during connect()
        for gripper, use_gripper in [
            (self._left_gripper, self.config.left_use_gripper),
            (self._right_gripper, self.config.right_use_gripper),
        ]:
            if gripper and use_gripper:
                try:
                    gripper.disconnect()
                except Exception:
                    pass

        self._cleanup_resources()

    def _cleanup_resources(self) -> None:
        """Release all robot resources.

        Explicit destruction order matters: CC objects must be destroyed before
        Robot objects, and gc.collect() ensures the C++ destructors run NOW
        (while the DDS transport is still alive) rather than at interpreter
        shutdown where they would trigger 'terminate called without an active
        exception'.
        """
        self._left_cc = None
        self._right_cc = None

        left_robot = self._left_robot
        right_robot = self._right_robot
        self._left_robot = None
        self._right_robot = None

        for robot in [left_robot, right_robot]:
            if robot is not None:
                try:
                    robot.close()
                except Exception:
                    pass
        del left_robot, right_robot

        import gc
        gc.collect()

        self._left_gripper = None
        self._right_gripper = None
        self._is_connected = False

    # =========================================================================
    # Configuration
    # =========================================================================

    def configure(self) -> None:
        """Configure both arms for RT Cartesian motion-force control."""
        self._configure_arm(self._left_robot, "left")
        self._configure_arm(self._right_robot, "right")

    def _configure_arm(self, robot: frt.Robot, side: str) -> None:
        """Configure a single arm for RT Cartesian motion-force control."""
        if robot is None:
            raise DeviceNotConnectedError(f"{side} arm robot is not connected.")

        self.logger.info(f"Configuring {side} arm RT Cartesian mode...")
        robot_info = robot.info()

        # Cartesian impedance
        if self.config.stiffness_ratio != 1.0:
            K_x_nom = robot_info.K_x_nom
            new_kx = list(np.multiply(K_x_nom, self.config.stiffness_ratio))
            robot.SetCartesianImpedance(new_kx, self.config.damping_ratio)
            self.logger.info(
                f"{side} arm: Cartesian stiffness (ratio={self.config.stiffness_ratio}): {new_kx}"
            )
        else:
            self.logger.info(f"{side} arm: Using nominal Cartesian stiffness: {robot_info.K_x_nom}")

        # Force control configuration
        if self.config.use_force:
            robot.SetForceControlFrame(self.config.force_control_frame)
            robot.SetForceControlAxis(list(self.config.force_control_axis))
            robot.SetMaxContactWrench([float("inf")] * 6)
            self.logger.info(
                f"{side} arm: Force control enabled (frame={self.config.force_control_frame}, "
                f"axes={self.config.force_control_axis})"
            )
        else:
            robot.SetMaxContactWrench(self._max_contact_wrench)
            self.logger.info(f"{side} arm: Max contact wrench: {self._max_contact_wrench}")

    # =========================================================================
    # Internal: mode switching and motion primitives
    # =========================================================================

    def _switch_to_rt_mode(self, robot: frt.Robot, side: str) -> None:
        """Switch a robot arm to RT_CARTESIAN_MOTION_FORCE mode."""
        target_mode = frt.Mode.RT_CARTESIAN_MOTION_FORCE
        current_mode = robot.mode()

        if current_mode == target_mode:
            self.logger.info(f"{side} arm: already in RT_CARTESIAN_MOTION_FORCE mode.")
            return

        self.logger.info(f"{side} arm: switching to RT_CARTESIAN_MOTION_FORCE...")

        if robot.fault():
            self.logger.warn(f"{side} arm: fault detected, clearing before mode switch...")
            robot.ClearFault()
            time.sleep(0.5)
            if robot.fault():
                raise RuntimeError(f"{side} arm: failed to clear fault before mode switch")

        robot.SwitchMode(target_mode)

        max_wait = 2.0
        wait_start = time.perf_counter()
        while time.perf_counter() - wait_start < max_wait:
            if robot.mode() == target_mode:
                self.logger.info(f"{side} arm: now in RT_CARTESIAN_MOTION_FORCE mode.")
                return
            time.sleep(0.1)

        raise RuntimeError(
            f"{side} arm: mode switch failed: expected {target_mode}, got {robot.mode()}. "
            f"Fault: {robot.fault()}"
        )

    def _go_to_start(self) -> None:
        """Move both arms to their start positions simultaneously using threads."""
        self.logger.info("Moving both arms to start position simultaneously...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(
                self._move_arm_to_position,
                self._left_robot,
                self._left_gripper,
                "left",
                self.config.left_start_position_degree,
                self.config.start_vel_scale,
                self.config.left_use_gripper,
                self.config.left_gripper_init_open,
                self.config.left_gripper_max_pos,
                self.config.left_gripper_v_max,
                self.config.left_gripper_f_max,
            )
            right_future = executor.submit(
                self._move_arm_to_position,
                self._right_robot,
                self._right_gripper,
                "right",
                self.config.right_start_position_degree,
                self.config.start_vel_scale,
                self.config.right_use_gripper,
                self.config.right_gripper_init_open,
                self.config.right_gripper_max_pos,
                self.config.right_gripper_v_max,
                self.config.right_gripper_f_max,
            )
            done, _ = wait([left_future, right_future])

        # Re-raise any exception from either thread
        for future in [left_future, right_future]:
            future.result()

        self.logger.info("Both arms at start position.")

    def _go_to_home_arm(self, robot: frt.Robot, side: str) -> None:
        """Move a single arm to home position via MoveJ."""
        home_deg = (
            self.config.left_home_position_degree
            if side == "left"
            else self.config.right_home_position_degree
        )
        self._move_arm_to_position(robot, None, side, home_deg, self.config.home_vel_scale)

    def _move_arm_to_position(
        self,
        robot: frt.Robot,
        gripper: SerialGripper | None,
        side: str,
        target_position_degree: list[float],
        vel_scale: int,
        use_gripper: bool = False,
        gripper_init_open: bool = True,
        gripper_max_pos: float = 85.0,
        gripper_v_max: float = 80.0,
        gripper_f_max: float = 20.0,
    ) -> None:
        """Move a single arm to a target joint position via MoveJ (blocking)."""
        if robot is None:
            return
        self.logger.info(f"{side} arm: moving to position {target_position_degree}...")
        robot.Stop()
        robot.SwitchMode(frt.Mode.NRT_PRIMITIVE_EXECUTION)
        robot.ExecutePrimitive(
            "MoveJ",
            {
                "target": target_position_degree,
                "jntVelScale": vel_scale,
            },
        )
        self.logger.info(f"{side} arm: MoveJ sent, waiting for completion...")

        # Initialize gripper during move (non-blocking — MCU does not
        # respond to status queries while idle, so sync would timeout).
        if gripper is not None and use_gripper:
            target_normalized = 1.0 if gripper_init_open else 0.0
            gripper.set_gripper_position(target_normalized)

        timeout = 30.0
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise RuntimeError(f"{side} arm: MoveJ did not complete within {timeout}s")
            try:
                pt_states = robot.primitive_states()
                if pt_states.get("reachedTarget", 0) == 1:
                    break
            except Exception:
                pass
            time.sleep(0.1)

        self.logger.info(f"{side} arm: at position.")

    def _zero_ft_sensor_left(self) -> None:
        self._zero_ft_sensor_arm(self._left_robot, "left")

    def _zero_ft_sensor_right(self) -> None:
        self._zero_ft_sensor_arm(self._right_robot, "right")

    def _zero_ft_sensor_arm(self, robot: frt.Robot, side: str) -> None:
        """Zero FT sensor for a single arm."""
        if robot is None:
            return
        self.logger.warn(f"{side} arm: zeroing FT sensor, make sure nothing is in contact...")

        if robot.mode() != frt.Mode.NRT_PRIMITIVE_EXECUTION:
            robot.SwitchMode(frt.Mode.NRT_PRIMITIVE_EXECUTION)
        robot.ExecutePrimitive("ZeroFTSensor", {})

        timeout = 10.0
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                self.logger.error(f"{side} arm: ZeroFTSensor timeout after {timeout}s")
                break
            try:
                pt_states = robot.primitive_states()
                if pt_states.get("terminated", 0) == 1:
                    break
            except Exception:
                pass
            time.sleep(0.1)

        self.logger.info(f"{side} arm: FT sensor zeroed.")

    # =========================================================================
    # Observation
    # =========================================================================

    def get_observation(self) -> dict[str, Any]:
        """Get current robot observation from both arms' RT shared memory.

        Returns:
            Dictionary with prefixed keys for both arms and camera images.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}
        _t = time.perf_counter

        # --- Left arm state ---
        t0 = _t()
        self._read_arm_state(
            cc=self._left_cc,
            robot=self._left_robot,
            tcp_pose_keys=self._left_tcp_pose_keys,
            wrench_keys=self._left_wrench_keys if self.config.use_force else None,
            obs_dict=obs_dict,
        )
        t1 = _t()

        # --- Right arm state ---
        self._read_arm_state(
            cc=self._right_cc,
            robot=self._right_robot,
            tcp_pose_keys=self._right_tcp_pose_keys,
            wrench_keys=self._right_wrench_keys if self.config.use_force else None,
            obs_dict=obs_dict,
        )
        t2 = _t()

        # --- Left gripper ---
        self._read_gripper_state(
            gripper=self._left_gripper,
            use_gripper=self.config.left_use_gripper,
            gripper_key=self._left_gripper_key,
            obs_dict=obs_dict,
        )
        t3 = _t()

        # --- Right gripper ---
        self._read_gripper_state(
            gripper=self._right_gripper,
            use_gripper=self.config.right_use_gripper,
            gripper_key=self._right_gripper_key,
            obs_dict=obs_dict,
        )
        t4 = _t()

        # --- External cameras ---
        cam_timings: dict[str, float] = {}
        for cam_key, cam in self.cameras.items():
            tc0 = _t()
            obs_dict[cam_key] = cam.async_read()
            cam_timings[cam_key] = (_t() - tc0) * 1e3

        t5 = _t()

        self._last_obs_timing = {
            "left_arm_ms":    (t1 - t0) * 1e3,
            "right_arm_ms":   (t2 - t1) * 1e3,
            "left_grip_ms":   (t3 - t2) * 1e3,
            "right_grip_ms":  (t4 - t3) * 1e3,
            "cameras_ms":     (t5 - t4) * 1e3,
            "total_ms":       (t5 - t0) * 1e3,
            **{f"cam[{k}]_ms": v for k, v in cam_timings.items()},
        }

        return obs_dict

    def _read_arm_state(
        self,
        cc: frt.CartesianMotionForceControl | None,
        robot: frt.Robot,
        tcp_pose_keys: tuple,
        wrench_keys: tuple | None,
        obs_dict: dict,
    ) -> None:
        """Read state from a single arm and populate obs_dict."""
        if cc is not None and cc.is_running():
            state = cc.get_state()
            tcp_pose = state.tcp_pose
            obs_dict[tcp_pose_keys[0]] = tcp_pose[0]
            obs_dict[tcp_pose_keys[1]] = tcp_pose[1]
            obs_dict[tcp_pose_keys[2]] = tcp_pose[2]
            r6d = quaternion_to_rotation_6d(tcp_pose[3], tcp_pose[4], tcp_pose[5], tcp_pose[6])
            for i in range(6):
                obs_dict[tcp_pose_keys[3 + i]] = r6d[i]
            if wrench_keys is not None:
                ext_wrench = state.ext_wrench_in_tcp
                for i, key in enumerate(wrench_keys):
                    obs_dict[key] = ext_wrench[i]
        else:
            states = robot.states()
            tcp_pose = states.tcp_pose
            obs_dict[tcp_pose_keys[0]] = tcp_pose[0]
            obs_dict[tcp_pose_keys[1]] = tcp_pose[1]
            obs_dict[tcp_pose_keys[2]] = tcp_pose[2]
            r6d = quaternion_to_rotation_6d(tcp_pose[3], tcp_pose[4], tcp_pose[5], tcp_pose[6])
            for i in range(6):
                obs_dict[tcp_pose_keys[3 + i]] = r6d[i]
            if wrench_keys is not None:
                ext_wrench = states.ext_wrench_in_tcp
                for i, key in enumerate(wrench_keys):
                    obs_dict[key] = ext_wrench[i]

    def _read_gripper_state(
        self,
        gripper: SerialGripper | None,
        use_gripper: bool,
        gripper_key: str,
        obs_dict: dict,
    ) -> None:
        """Read gripper position into obs_dict."""
        if gripper is None or not use_gripper:
            return
        obs_dict[gripper_key] = gripper.get_gripper_position()

    # =========================================================================
    # Action
    # =========================================================================

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action to both arms via their RT shared memory.

        Converts 6D rotation representation to quaternion for the Flexiv SDK.
        If an arm's RT thread is executing a trajectory (is_moving), arm commands
        are skipped for that arm. Gripper commands are always sent.

        Action format (use_force=False):
            left_tcp.{x,y,z,r1-r6} (9D) + left_gripper.pos (1D)
            right_tcp.{x,y,z,r1-r6} (9D) + right_gripper.pos (1D)

        Args:
            action: Dictionary of action values

        Returns:
            The action dict that was sent
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._left_cc is None or not self._left_cc.is_running():
            raise RuntimeError("Left RT thread is not running. Call connect() first.")
        if self._right_cc is None or not self._right_cc.is_running():
            raise RuntimeError("Right RT thread is not running. Call connect() first.")

        # --- Left arm ---
        if not self._left_cc.is_moving():
            self._send_arm_action(
                action, self._left_cc, self._left_tcp_pose_keys,
                wrench_keys=self._left_wrench_keys if self.config.use_force else None,
            )
        self._send_gripper_action(
            action, self._left_gripper, self.config.left_use_gripper, self._left_gripper_key
        )

        # --- Right arm ---
        if not self._right_cc.is_moving():
            self._send_arm_action(
                action, self._right_cc, self._right_tcp_pose_keys,
                wrench_keys=self._right_wrench_keys if self.config.use_force else None,
            )
        self._send_gripper_action(
            action, self._right_gripper, self.config.right_use_gripper, self._right_gripper_key
        )

        return action

    def _send_arm_action(
        self,
        action: dict[str, Any],
        cc: frt.CartesianMotionForceControl,
        tcp_pose_keys: tuple,
        wrench_keys: tuple | None,
    ) -> None:
        """Build target pose from action dict and send to one arm's RT thread."""
        x = action[tcp_pose_keys[0]]
        y = action[tcp_pose_keys[1]]
        z = action[tcp_pose_keys[2]]

        r6d = np.array(
            [
                action[tcp_pose_keys[3]],
                action[tcp_pose_keys[4]],
                action[tcp_pose_keys[5]],
                action[tcp_pose_keys[6]],
                action[tcp_pose_keys[7]],
                action[tcp_pose_keys[8]],
            ]
        )
        quat = rotation_6d_to_quaternion(r6d)  # [qw, qx, qy, qz]
        target_pose = [x, y, z, float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]

        if wrench_keys is not None:
            target_wrench = [action[k] for k in wrench_keys]
            cc.set_target_pose(target_pose, target_wrench)
        else:
            cc.set_target_pose(target_pose)

    def _send_gripper_action(
        self,
        action: dict[str, Any],
        gripper: SerialGripper | None,
        use_gripper: bool,
        gripper_key: str,
    ) -> None:
        """Send gripper command if the key is present in action."""
        if not gripper or not use_gripper:
            return
        if gripper_key not in action:
            return
        gripper.set_gripper_position(action[gripper_key])

    # =========================================================================
    # RT thread management
    # =========================================================================

    def trigger_estop(self) -> None:
        """Trigger emergency stop on both RT threads (lockless atomic flag)."""
        if self._left_cc is not None:
            self._left_cc.trigger_estop()
            self.logger.warn("E-stop triggered on left arm RT thread")
        if self._right_cc is not None:
            self._right_cc.trigger_estop()
            self.logger.warn("E-stop triggered on right arm RT thread")

    @property
    def left_rt_running(self) -> bool:
        return self._left_cc is not None and self._left_cc.is_running()

    @property
    def right_rt_running(self) -> bool:
        return self._right_cc is not None and self._right_cc.is_running()

    @property
    def rt_moving(self) -> bool:
        """Whether either arm's RT thread is executing a trajectory (e.g. from reset)."""
        left_moving = self._left_cc is not None and self._left_cc.is_moving()
        right_moving = self._right_cc is not None and self._right_cc.is_moving()
        return left_moving or right_moving

    def get_current_tcp_pose_quat(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current TCP poses for both arms in quaternion format.

        Reads from RT shared memory if available, otherwise from robot.states().

        Returns:
            Tuple of (left_pose, right_pose), each np.ndarray of shape (8,)
            with [x, y, z, qw, qx, qy, qz, gripper_pos].
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        def _read_arm_pose(cc, robot, gripper, use_gripper) -> np.ndarray:
            if cc is not None and cc.is_running():
                tcp_pose = cc.get_state().tcp_pose
            else:
                tcp_pose = robot.states().tcp_pose
            gripper_pos = 0.0
            if gripper is not None and use_gripper:
                gripper_pos = gripper.get_gripper_position()
            return np.array([*tcp_pose, gripper_pos], dtype=np.float32)

        left_pose = _read_arm_pose(
            self._left_cc, self._left_robot,
            self._left_gripper, self.config.left_use_gripper,
        )
        right_pose = _read_arm_pose(
            self._right_cc, self._right_robot,
            self._right_gripper, self.config.right_use_gripper,
        )
        return left_pose, right_pose

    # =========================================================================
    # Utility methods
    # =========================================================================

    def _start_cartesian_control(
        self,
        robot: frt.Robot,
        task_name: str,
        side: str,
    ) -> frt.CartesianMotionForceControl:
        """Start one arm's flexiv_rt Cartesian RT thread using configured SHM consumption params."""
        ctrl = robot.start_cartesian_control(
            task_name=task_name,
            inner_control_hz=self.config.inner_control_hz,
            interpolate_cmds=self.config.interpolate_cmds,
        )
        self.logger.info(
            f"{side} arm: C++ RT thread started "
            f"(inner_control_hz={self.config.inner_control_hz}, "
            f"interpolate_cmds={self.config.interpolate_cmds})"
        )
        return ctrl

    def reset_to_initial_position(self) -> None:
        """Reset both arms to their initial start positions via RT trajectory.

        Uses non-blocking RT-mode trajectory if available (fast path), otherwise
        falls back to stop RT -> NRT MoveJ -> restart RT.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for side, cc, robot, start_pose in [
            ("left", self._left_cc, self._left_robot, self._left_start_tcp_pose),
            ("right", self._right_cc, self._right_robot, self._right_start_tcp_pose),
        ]:
            if self.config.go_to_start and start_pose is not None and cc is not None and cc.is_running():
                if cc.is_moving():
                    continue
                self.logger.info(f"{side} arm: resetting to start via RT trajectory (non-blocking)")
                cc.move_to_pose(start_pose, duration_sec=3.0)
            else:
                self.logger.info(f"{side} arm: using legacy reset (stop RT -> NRT move -> restart RT)")
                if cc is not None:
                    cc.stop()
                    if side == "left":
                        self._left_cc = None
                    else:
                        self._right_cc = None

                if self.config.go_to_start:
                    start_deg = (
                        self.config.left_start_position_degree
                        if side == "left"
                        else self.config.right_start_position_degree
                    )
                    gripper = self._left_gripper if side == "left" else self._right_gripper
                    use_gripper = (
                        self.config.left_use_gripper if side == "left" else self.config.right_use_gripper
                    )
                    init_open = (
                        self.config.left_gripper_init_open
                        if side == "left"
                        else self.config.right_gripper_init_open
                    )
                    max_pos = (
                        self.config.left_gripper_max_pos
                        if side == "left"
                        else self.config.right_gripper_max_pos
                    )
                    v_max = (
                        self.config.left_gripper_v_max if side == "left" else self.config.right_gripper_v_max
                    )
                    f_max = (
                        self.config.left_gripper_f_max if side == "left" else self.config.right_gripper_f_max
                    )
                    self._move_arm_to_position(
                        robot, gripper, side, start_deg, self.config.start_vel_scale,
                        use_gripper, init_open, max_pos, v_max, f_max
                    )
                else:
                    self._go_to_home_arm(robot, side)

                self._switch_to_rt_mode(robot, side)
                self._configure_arm(robot, side)
                new_cc = self._start_cartesian_control(
                    robot,
                    task_name=f"CartesianRT_{side[0].upper()}",
                    side=side,
                )
                init_pose = list(robot.states().tcp_pose)
                new_cc.set_target_pose(init_pose)
                time.sleep(0.1)

                if side == "left":
                    self._left_cc = new_cc
                else:
                    self._right_cc = new_cc

                self.logger.info(f"{side} arm: RT thread restarted after position reset")

    def clear_fault(self, side: str = "both") -> dict[str, bool]:
        """Attempt to clear robot fault.

        Args:
            side: "left", "right", or "both"

        Returns:
            Dict with results for each requested side
        """
        results = {}
        targets = []
        if side in ("left", "both"):
            targets.append(("left", self._left_robot))
        if side in ("right", "both"):
            targets.append(("right", self._right_robot))

        for arm_side, robot in targets:
            if robot is None:
                raise DeviceNotConnectedError(f"{arm_side} arm is not connected.")
            if not robot.fault():
                self.logger.info(f"{arm_side} arm: no fault to clear.")
                results[arm_side] = True
                continue
            self.logger.info(f"{arm_side} arm: attempting to clear fault...")
            result = robot.ClearFault()
            results[arm_side] = result
            if result:
                self.logger.info(f"{arm_side} arm: fault cleared.")
            else:
                self.logger.error(f"{arm_side} arm: failed to clear fault.")

        return results
