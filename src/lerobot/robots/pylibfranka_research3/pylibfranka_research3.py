#!/usr/bin/env python
"""
PylibfrankaResearch3 — Franka Research3 robot driver via WebSocket

通过 WebSocket 客户端-服务端架构控制 Franka 机械臂。
服务端 (ws_teleop_server.py) 运行在机器人侧，直接管理 Franka 控制器。
客户端 (本模块) 通过 WebSocket 发送指令并接收状态反馈。

指令协议:
  - get_state           : 获取机器人状态
  - cartesian_absolute  : 设置绝对末端位姿 (4x4 矩阵)
  - joint_absolute      : 设置绝对关节位置 (7 float)
  - move_home           : 移动到指定关节位置
  - configure           : 配置控制器参数
  - reset               : 重置到初始位置
  - stop                : 停止
"""

import atexit
import json
import logging
import subprocess
import sys
import time
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
from websockets.sync.client import connect

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import (
    matrix_to_pose7d,
    quaternion_to_euler,
    rotation_6d_to_quaternion,
    quaternion_to_matrix,
)

from ..robot import Robot
from .config_pylibfranka_research3 import ControlMode, PylibfrankaResearch3Config
from scipy.spatial.transform import Rotation as R, Slerp

logger = logging.getLogger(__name__)

JOINT_DOF = 7  # Franka Research3 robot joint DOF

SERVER_SCRIPT = Path(__file__).parent / "ws_teleop_server.py"


class PylibfrankaResearch3(Robot):
    config_class = PylibfrankaResearch3Config
    name = "pylibfranka_research3"

    def __init__(self, config: PylibfrankaResearch3Config):
        super().__init__(config)
        self.config = config

        # WebSocket / Server
        self._ws = None                 # WebSocket connection (sync)
        self._server_proc = None        # Server subprocess
        self._last_state = None         # Cached robot state from server
        self._server_uri = f"ws://localhost:{config.port}"

        # Gripper
        if config.use_gripper:
            self.gripper_server_ip = config.gripper_server_ip
            self.gripper_server_port = config.gripper_server_port
            self.gripper_url = f"http://{self.gripper_server_ip}:{self.gripper_server_port}"
        self._gripper_key = "gripper.pos"

        # Initialize keys and buffers based on control mode
        if config.control_mode == ControlMode.JOINT_IMPEDANCE:
            self._init_joint_mode()
        elif config.control_mode == ControlMode.CARTESIAN_IMPEDANCE:
            self._init_cartesian_mode()
        else:
            raise ValueError(f"Unsupported control_mode: {config.control_mode}")

        self.cameras = make_cameras_from_configs(config.cameras)

        self._is_connected = False
        self._robot_connected = False
        self._gripper_connected = False
        self._is_resetting = False  # Flag to block actions during reset

        logger.info(f"Initialized {self.name}")
        logger.info(f"  Robot: Franka Follower at {config.fci_ip}")
        if config.use_gripper:
            logger.info(f"  Gripper: Xense Hand at {config.gripper_server_ip}:{config.gripper_server_port}")
        logger.info(f"  Cameras: {len(self.cameras)} camera(s)")
        logger.info(f"  Control mode: {config.control_mode}")

    # ======================== Server Management ========================

    def _launch_server(self, robot_ip: str, port: int) -> subprocess.Popen:
        """在本机后台启动 ws_teleop_server.py，返回子进程对象。

        服务端负责：
        1. 连接 Franka 机器人
        2. 移动到初始位姿
        3. 切换到 OSC 控制模式
        4. 启动 WebSocket 监听
        """
        # Map ControlMode enum to server CLI argument
        mode_map = {
            ControlMode.CARTESIAN_IMPEDANCE: "cartesian",
            ControlMode.JOINT_IMPEDANCE: "joint",
        }
        control_mode_str = mode_map.get(self.config.control_mode, "cartesian")

        cmd = [
            sys.executable, "-u", str(SERVER_SCRIPT),
            "--robot-ip", robot_ip,
            "--port", str(port),
            "--control-mode", control_mode_str,
            "--freq", "30",
            "--home",
        ] + [str(x) for x in self.config.robot_home_position]

        print(f"[Client] Launching server: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )

        # 退出时自动清理服务端
        def cleanup():
            if proc.poll() is None:
                print("[Client] Shutting down server ...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        atexit.register(cleanup)

        # 等待服务端就绪（服务端输出 "listening" 或 "waiting" 表示就绪）
        print("[Client] Waiting for server to be ready ...")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if line:
                print(f"  [Server] {line.rstrip()}")
                if "listening" in line.lower():
                    print("[Client] Server is ready!")
                    return proc
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited unexpectedly (code={proc.returncode})")
        raise TimeoutError("Server did not become ready within 120s")

    # ======================== WebSocket Communication ========================

    def _ws_send_recv(self, cmd: dict, timeout: float = 5.0) -> dict:
        """通过 WebSocket 发送指令并接收服务端返回的机器人状态。

        Args:
            cmd: 指令字典，必须包含 "type" 字段
            timeout: 接收超时（秒）

        Returns:
            服务端返回的状态字典，包含:
            - O_T_EE: 末端位姿 (16 float, 列主序)
            - q: 关节位置 (7 float)
            - dq: 关节速度 (7 float)
            - tau: 关节力矩 (7 float)
            - ext_wrench: 外部力/力矩 (6 float)
            - ee_desired: 期望末端位姿 (4x4 嵌套列表)
            - timestamp: 时间戳
        """
        if self._ws is None:
            raise DeviceNotConnectedError(f"{self} WebSocket 未连接")

        self._ws.send(json.dumps(cmd))
        response = self._ws.recv(timeout=timeout)
        state_data = json.loads(response)

        if "error" in state_data:
            logger.warning(f"Server error: {state_data['error']}")

        self._last_state = state_data
        return state_data

    def _parse_ee_matrix(self, state_data: dict) -> np.ndarray:
        """从服务端状态中解析末端位姿为 4x4 行主序矩阵。

        服务端返回 ee_desired 为 16 元素列主序数组（Franka 标准格式），
        需要 reshape(4,4).T 转换为行主序：
          列主序: [r11, r21, r31, 0, r12, r22, r32, 0, r13, r23, r33, 0, px, py, pz, 1]
          行主序: [[r11, r12, r13, px],
                   [r21, r22, r23, py],
                   [r31, r32, r33, pz],
                   [0,   0,   0,   1]]
        """
        ee_desired = np.array(state_data.get("ee_desired", []), dtype=np.float64)
        # print("Received ee_desired from server:", ee_desired)
        return ee_desired

    # ======================== Key Initialization ========================

    def _init_joint_mode(self) -> None:
        """Initialize keys and buffers for JOINT_POSITION control mode."""
        # Joint state observation keys: joint_{1-7}.{pos, vel, effort}
        self._joint_pos_keys = tuple(f"joint_{i}.pos" for i in range(1, JOINT_DOF + 1))
        self._joint_vel_keys = tuple(f"joint_{i}.vel" for i in range(1, JOINT_DOF + 1))
        self._joint_effort_keys = tuple(f"joint_{i}.effort" for i in range(1, JOINT_DOF + 1))

        # Joint action keys: joint_{1-7}.pos
        self._action_joint_keys = tuple(f"joint_{i}.pos" for i in range(1, JOINT_DOF + 1))

    def _init_cartesian_mode(self) -> None:
        """Initialize keys and buffers for CARTESIAN_POSITION control mode.

        Uses 6D rotation representation (r1-r6) instead of quaternion for:
        - Continuity: No discontinuities like Euler angles (gimbal lock)
        - No double-cover: Unlike quaternions where q and -q represent same rotation
        - Better for neural networks: Continuous representation is easier to learn

        Reference: "On the Continuity of Rotation Representations in Neural Networks"
        """
        # TCP pose observation/action keys: tcp.{x, y, z, r1, r2, r3, r4, r5, r6}
        # 6D rotation: r1-r3 = first column, r4-r6 = second column of rotation matrix
        self._tcp_pose_keys = (
            "tcp.x", "tcp.y", "tcp.z",
            "tcp.r1", "tcp.r2", "tcp.r3",
            "tcp.r4", "tcp.r5", "tcp.r6",
        )

        # TCP velocity observation keys: tcp.{vx, vy, vz, wx, wy, wz}
        self._tcp_vel_keys = (
            "tcp.vx", "tcp.vy", "tcp.vz",
            "tcp.wx", "tcp.wy", "tcp.wz",
        )

        # TCP pose action keys (same as observation keys for 6D rotation)
        self._action_tcp_pose_keys = self._tcp_pose_keys

        # Initialize force-related keys if use_force is enabled
        if self.config.use_force:
            self._wrench_keys = (
                "tcp.fx", "tcp.fy", "tcp.fz",
                "tcp.mx", "tcp.my", "tcp.mz",
            )
            self._action_wrench_keys = self._wrench_keys

    # ======================== Features ========================

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features = {}

        if self.config.control_mode == ControlMode.JOINT_IMPEDANCE:
            # Robot: 7 joint positions
            features.update(dict.fromkeys(self._joint_pos_keys, float))
            # Joint velocities (7 joints)
            features.update(dict.fromkeys(self._joint_vel_keys, float))
            # Joint efforts/torques (7D)
            features.update(dict.fromkeys(self._joint_effort_keys, float))
        elif self.config.control_mode == ControlMode.CARTESIAN_IMPEDANCE:
            # TCP pose (9D: xyz + 6D rotation)
            features.update(dict.fromkeys(self._action_tcp_pose_keys, float))
            if self.config.use_force:
                # + external wrench (6D)
                features.update(dict.fromkeys(self._action_wrench_keys, float))
        else:
            raise ValueError(f"Unsupported control mode: {self.config.control_mode}")

        # Gripper: position (0.0=open, 1.0=closed)
        features[self._gripper_key] = float

        # Cameras
        for cam_name, cam_config in self.config.cameras.items():
            features[cam_name] = (cam_config.height, cam_config.width, 3)

        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        action_dict = {}

        if self.config.control_mode == ControlMode.JOINT_IMPEDANCE:
            # 7 joint position commands
            action_dict.update(dict.fromkeys(self._action_joint_keys, float))
        elif self.config.control_mode == ControlMode.CARTESIAN_IMPEDANCE:
            # Cartesian position commands (x, y, z, r1, r2, r3, r4, r5, r6)
            action_dict.update(dict.fromkeys(self._action_tcp_pose_keys, float))
            if self.config.use_force:
                # + target wrench (6D)
                action_dict.update(dict.fromkeys(self._action_wrench_keys, float))
        else:
            raise ValueError(f"Unsupported control mode: {self.config.control_mode}")

        # Gripper position
        if self.config.use_gripper:
            action_dict[self._gripper_key] = float
        return action_dict

    # ======================== Robot State ========================

    def _get_robot_state(self) -> Optional[dict]:
        """Get full robot state via WebSocket."""
        if not self.is_connected:
            return None
        try:
            state_data = self._ws_send_recv({"type": "get_state"})
            ee_matrix = self._parse_ee_matrix(state_data)
            return {
                "joint_positions": np.array(state_data.get("q", []), dtype=np.float32),
                "joint_velocities": np.array(state_data.get("dq", []), dtype=np.float32),
                "ee_pose": ee_matrix.flatten().astype(np.float32),
                "joint_torques": np.array(state_data.get("tau", []), dtype=np.float32),
                "ext_wrench": np.array(state_data.get("ext_wrench", []), dtype=np.float32),
                "ee_desired": np.array(state_data.get("ee_desired", np.eye(4).tolist()), dtype=np.float64),
            }
        except Exception as e:
            logger.error(f"Failed to read robot state: {e}")
            return None

    # ======================== Gripper ========================

    def _gripper_health_check(self) -> bool:
        """Check if hand server is reachable."""
        try:
            response = requests.get(f"{self.gripper_url}/get_pos", timeout=2.0)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Hand health check failed: {e}")
            return False

    def _get_gripper_position(self) -> float:
        """Get normalized gripper position [0.0=open, 1.0=closed]."""
        try:
            response = requests.get(f"{self.gripper_url}/get_pos", timeout=self.config.gripper_timeout)
            if response.status_code != 200:
                logger.warning(f"获取夹爪位置失败: HTTP {response.status_code}")
                return self.config.gripper_home_position
            data = response.json()
            min_w = float(self.config.gripper_min_width_mm)
            max_w = float(self.config.gripper_max_width_mm)
            width = data.get("position", 0.0)
            width = max(min_w, min(max_w, width))
            # 映射: width (mm) -> closed_ratio (0-1)
            closed_ratio = 1.0 - (width - min_w) / (max_w - min_w)
            position = float(np.clip(closed_ratio, self.config.gripper_min_position, self.config.gripper_max_position))
            logger.debug(f"获取夹爪位置: width_mm={width:.1f} -> closed_ratio={position:.3f}")
            return position
        except Exception as e:
            logger.error(f"获取夹爪位置错误: {e}")
            return self.config.gripper_home_position

    def _send_gripper_position_command(self, position: float) -> bool:
        """position: Normalized target position [0.0, 1.0] 0.0 = open, 1.0 = closed"""
        try:
            min_w = float(self.config.gripper_min_width_mm)
            max_w = float(self.config.gripper_max_width_mm)

            # 映射: closed_ratio -> width
            target_width = min_w + (1.0 - position) * (max_w - min_w)
            target_width = float(np.clip(target_width, min_w, max_w))

            logger.debug(f"转换夹爪位置: closed_ratio={position:.3f} -> width_mm={target_width:.1f}")

            response = requests.post(
                f"{self.gripper_url}/move",
                json={
                    "pos": float(target_width),
                    "vmax": self.config.gripper_default_velocity,
                    "fmax": self.config.gripper_default_force,
                },
                timeout=self.config.gripper_timeout,
            )

            if response.status_code == 200:
                return True
            else:
                logger.warning(f"发送夹爪命令失败: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"发送夹爪位置命令错误: {e}")
            return False

    # ======================== Connection Management ========================

    @property
    def is_connected(self) -> bool:
        """Check if robot (WebSocket) and gripper are connected."""
        if self.config.use_gripper:
            return self._is_connected and self._robot_connected and self._gripper_connected
        else:
            return self._is_connected and self._robot_connected

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return self.is_connected  # Franka gripper doesn't require calibration

    def calibrate(self) -> None:
        """Calibrate robot (gripper doesn't require calibration)."""
        pass

    def configure(self) -> None:
        """Configure robot controller via WebSocket."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        logger.debug(f"{self} configured (mode={self.config.control_mode})")

    def connect(self, calibrate: bool = True, go_to_start: bool = True) -> None:
        """Connect to Franka robot via WebSocket server and gripper via REST.

        流程:
        1. 启动服务端子进程（初始化机器人、移动到初始位姿、切换 OSC 模式）
        2. 建立 WebSocket 连接
        3. 发送 get_state 验证连接
        4. 连接夹爪（REST）
        5. 连接相机
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            # 1. Launch server subprocess (initializes robot, moves to home)
            logger.info(f"Launching robot server for {self.config.fci_ip}...")
            self._server_proc = self._launch_server(self.config.fci_ip, self.config.port)

            # 2. Open WebSocket connection (with retry)
            logger.info(f"Connecting to server: {self._server_uri}")
            self._ws = None
            max_retries = 10
            for attempt in range(1, max_retries + 1):
                try:
                    self._ws = connect(self._server_uri)
                    break
                except ConnectionRefusedError:
                    if attempt == max_retries:
                        raise ConnectionRefusedError(
                            f"Cannot connect to server at {self._server_uri} after {max_retries} attempts"
                        )
                    logger.info(f"Connection attempt {attempt}/{max_retries} refused, retrying in 1s...")
                    time.sleep(1.0)

            # 3. Verify connection by getting state
            state = self._ws_send_recv({"type": "get_state"}, timeout=10.0)
            q = state.get("q", [])
            logger.info(f"Robot connected, joint positions: {np.round(q, 4).tolist()}")
            self._robot_connected = True

            # 4. Connect gripper
            if self.config.use_gripper:
                logger.info(f"Connecting to gripper: {self.gripper_server_ip}:{self.gripper_server_port}")
                if not self._gripper_health_check():
                    raise ConnectionError(
                        f"Cannot reach gripper at {self.gripper_url}"
                    )
                self._gripper_connected = True
                logger.info("Gripper connection successful")

            # 5. Connect cameras
            logger.info("Connecting cameras...")
            for cam in self.cameras.values():
                cam.connect()

            self._is_connected = True
            logger.info(f"✅ {self} connected successfully")

            # 6. Configure
            self.configure()

        except Exception as e:
            # Cleanup on partial failure
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None

            for cam in self.cameras.values():
                try:
                    if cam.is_connected:
                        cam.disconnect()
                except Exception:
                    pass

            if self._server_proc is not None and self._server_proc.poll() is None:
                self._server_proc.terminate()
                self._server_proc = None

            self._robot_connected = False
            self._gripper_connected = False
            self._is_connected = False
            logger.error(f"Failed to connect: {e}")
            raise

    def _go_to_home(self) -> None:
        """Move robot to home position (delegates to _go_to_start)."""
        self._go_to_start()

    def _go_to_start(self) -> None:
        """Move robot to home position via WebSocket move_home command.

        服务端接收 move_home 指令后会：
        1. 调用 controller.move() 移动到目标关节位置
        2. 重新切换回 OSC 控制模式
        """
        if not self._is_connected or self._ws is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Moving to home position via WebSocket...")

        if self.config.control_mode == ControlMode.JOINT_IMPEDANCE:
            target_joint = self.config.robot_home_position
            logger.info(f"Target position: {target_joint}")
            cmd = {"type": "move_home", "positions": target_joint}
            self._ws_send_recv(cmd, timeout=30.0)


        elif self.config.control_mode == ControlMode.CARTESIAN_IMPEDANCE:
            now_pose = self.get_current_tcp_pose_quat()[:7]
            target_pose = self.config.robot_tcp_home_position  # TCP  (x, y, z, w, x, y, z)
            vel = 0.2  # m/s
            delta = np.linalg.norm(np.array(target_pose[:3]) - np.array(now_pose[:3]))
            timeout = delta / vel
            hz = 50.0
            """Move the robot to the goal position with linear interpolation."""
            steps = int(timeout * hz)
            print(f"Moving to start position with {steps} steps over {timeout:.2f} seconds...")
            # positional linear interpolation from now_pose to target_pose
            pos_path = np.linspace(now_pose[:3], target_pose[:3], steps)
            # SLERP rotation from now_pose to target_pose
            r0 = R.from_quat(now_pose[3:7])
            r1 = R.from_quat(target_pose[3:7])

            key_times = [0, 1]
            key_rots = R.concatenate([r0, r1])
            slerp = Slerp(key_times, key_rots)

            interp_times = np.linspace(0, 1, steps)
            rot_path = slerp(interp_times)

            # Combine into complete path
            path = np.zeros((steps, 7))
            path[:, :3] = pos_path
            path[:, 3:7] = rot_path.as_quat()

            for p in path:
                # Send absolute pose as cartesian_absolute command
                ee_matrix = quaternion_to_matrix(p, input_format="wxyz")
                cmd = {"type": "cartesian_absolute", "pose": ee_matrix.tolist()}
                self._ws_send_recv(cmd)
                time.sleep(1 / hz)
        

    def reset_to_initial_position(self) -> None:
        """Reset robot to initial position based on config.go_to_start.

        在 reset 期间设置 _is_resetting 标志，send_action 会忽略所有指令，
        防止手柄残留的旧位姿在 move_home 后瞬间发到机械臂导致抽动。
        """
        if not self._is_connected or self._ws is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._is_resetting = True
        try:
            if self.config.go_to_start:
                logger.info("Resetting to start position (config.go_to_start=True)")
                self._go_to_start()
            else:
                logger.info("Resetting to home position (config.go_to_start=False)")
                self._go_to_home()

            # Switch back to control mode after reset
            self._switch_to_control_mode()
        finally:
            self._is_resetting = False

    def _switch_to_control_mode(self) -> None:
        """切换控制模式（通过 WebSocket configure 命令）。

        服务端默认使用 OSC 模式。如需切换，可发送 configure 命令:
          self._ws_send_recv({"type": "configure", "switch_mode": "osc"})
        """
        if not self._is_connected or self._ws is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        logger.debug(f"Control mode: {self.config.control_mode}")

    # ======================== Observation ========================

    def get_observation(self) -> dict[str, Any]:
        """Get synchronized observation from robot, gripper, and cameras.

        通过 WebSocket 获取机器人状态，结合夹爪位置和相机图像构建完整观测。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        obs_dict = {}

        # Get robot state via WebSocket
        state_data = self._ws_send_recv({"type": "get_state"})

        if self.config.control_mode == ControlMode.JOINT_IMPEDANCE:
            q = state_data.get("q", [0.0] * JOINT_DOF)
            dq = state_data.get("dq", [0.0] * JOINT_DOF)
            tau = state_data.get("tau", [0.0] * JOINT_DOF)
            # Joint positions (7D)
            for i, key in enumerate(self._joint_pos_keys):
                obs_dict[key] = float(q[i]) if i < len(q) else 0.0
            # Joint velocities (7D)
            for i, key in enumerate(self._joint_vel_keys):
                obs_dict[key] = float(dq[i]) if i < len(dq) else 0.0
            # Joint efforts/torques (7D)
            for i, key in enumerate(self._joint_effort_keys):
                obs_dict[key] = float(tau[i]) if i < len(tau) else 0.0

        elif self.config.control_mode == ControlMode.CARTESIAN_IMPEDANCE:
            # Convert column-major O_T_EE to row-major 4x4 matrix
            ee_matrix = self._parse_ee_matrix(state_data)
            rot = ee_matrix[:3, :3]

            # Position (3D)
            obs_dict["tcp.x"] = float(ee_matrix[0, 3])
            obs_dict["tcp.y"] = float(ee_matrix[1, 3])
            obs_dict["tcp.z"] = float(ee_matrix[2, 3])

            # 6D Rotation: first two columns of rotation matrix
            # r1-r3 = first column, r4-r6 = second column
            obs_dict["tcp.r1"] = float(rot[0, 0])
            obs_dict["tcp.r2"] = float(rot[1, 0])
            obs_dict["tcp.r3"] = float(rot[2, 0])
            obs_dict["tcp.r4"] = float(rot[0, 1])
            obs_dict["tcp.r5"] = float(rot[1, 1])
            obs_dict["tcp.r6"] = float(rot[2, 1])

            if self.config.use_force:
                ext_wrench = state_data.get("ext_wrench", [0.0] * 6)
                for i, key in enumerate(self._wrench_keys):
                    obs_dict[key] = float(ext_wrench[i]) if i < len(ext_wrench) else 0.0

        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        # Gripper
        if self.config.use_gripper:
            gripper_position = self._get_gripper_position()
            obs_dict[self._gripper_key] = gripper_position

        # Cameras
        for cam_name, cam in self.cameras.items():
            obs_dict[cam_name] = cam.async_read()

        return obs_dict

    def get_current_tcp_pose_euler(self) -> np.ndarray:
        """Get current TCP pose in Euler angles format [x, y, z, roll, pitch, yaw, gripper_pos].

        This method can be used for initializing teleoperators (e.g., spacemouse)
        with the robot's current TCP pose. Only available in CARTESIAN_IMPEDANCE mode.

        Returns:
            numpy array of shape (7,) with [x, y, z, roll, pitch, yaw, gripper_pos]
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.control_mode != ControlMode.CARTESIAN_IMPEDANCE:
            raise ValueError("get_current_tcp_pose_euler requires CARTESIAN_IMPEDANCE mode")

        state_data = self._ws_send_recv({"type": "get_state"})
        ee_matrix = self._parse_ee_matrix(state_data)
        tcp_pose = matrix_to_pose7d(ee_matrix, output_format="wxyz")
        # Convert quaternion to Euler angles
        euler = quaternion_to_euler(tcp_pose[3], tcp_pose[4], tcp_pose[5], tcp_pose[6])
        roll, pitch, yaw = euler[0], euler[1], euler[2]

        # Get gripper position
        gripper_pos = 0.0
        if self.config.use_gripper:
            gripper_pos = self._get_gripper_position()

        return np.array(
            [tcp_pose[0], tcp_pose[1], tcp_pose[2], roll, pitch, yaw, gripper_pos],
            dtype=np.float32,
        )

    def get_current_tcp_pose_quat(self) -> np.ndarray:
        """Get current TCP pose in quaternion format [x, y, z, qw, qx, qy, qz, gripper_pos].

        This method can be used for initializing teleoperators (e.g., pico4)
        with the robot's current TCP pose. Only available in CARTESIAN_IMPEDANCE mode.

        Returns:
            numpy array of shape (8,) with [x, y, z, qw, qx, qy, qz, gripper_pos]
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.control_mode != ControlMode.CARTESIAN_IMPEDANCE:
            raise ValueError("get_current_tcp_pose_quat requires CARTESIAN_IMPEDANCE mode")

        state_data = self._ws_send_recv({"type": "get_state"})
        ee_matrix = self._parse_ee_matrix(state_data)
        tcp_pose = matrix_to_pose7d(ee_matrix, output_format="wxyz")

        # Get gripper position
        gripper_pos = 0.0
        if self.config.use_gripper:
            gripper_pos = self._get_gripper_position()

        return np.array([*tcp_pose, gripper_pos], dtype=np.float32)

    # ======================== Action ========================

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send synchronized action to robot and gripper.

        Args:
            action: Dictionary containing pose/joint commands and optional gripper command.
                For CARTESIAN_IMPEDANCE: tcp.{x,y,z,r1,r2,r3,r4,r5,r6}
                For JOINT_IMPEDANCE: joint_{1-7}.pos
                Optional: gripper.pos (0.0=open, 1.0=closed)

        Returns:
            The action that was sent.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # Block actions during reset to prevent stale teleop values from causing jerks
        if self._is_resetting:
            logger.debug("Skipping action during reset")
            return action

        # Send robot arm action
        if self.config.control_mode == ControlMode.JOINT_IMPEDANCE:
            result = self._send_joint_position_action(action)

        elif self.config.control_mode == ControlMode.CARTESIAN_IMPEDANCE:
            if self.config.use_force:
                result = self._send_cartesian_motion_force_action(action)
            else:
                result = self._send_cartesian_pure_motion_action(action)

        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        # Send gripper action
        self._send_gripper_action(action)

        return result

    def _send_joint_position_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send absolute joint position action via WebSocket.

        Sends joint_absolute command with 7 joint positions to the server.
        Server will clip joint positions to joint limits before applying.
        """
        try:
            joint_pos = [float(action[key]) for key in self._action_joint_keys]
            cmd = {"type": "joint_absolute", "q_desired": joint_pos}
            self._ws_send_recv(cmd)
            return action
        except Exception as e:
            logger.warning(f"Error sending joint action: {e}")
            return action

    def _send_cartesian_pure_motion_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send Cartesian absolute pose command via WebSocket.

        Converts 6D rotation representation to quaternion, then to 4x4 matrix,
        and sends as cartesian_absolute command to the server.

        Action keys: tcp.{x,y,z,r1,r2,r3,r4,r5,r6}
        """
        try:
            # Extract position
            x = float(action["tcp.x"])
            y = float(action["tcp.y"])
            z = float(action["tcp.z"])

            # Extract 6D rotation and convert to quaternion → 4x4 matrix
            r6d = np.array([
                action["tcp.r1"], action["tcp.r2"], action["tcp.r3"],
                action["tcp.r4"], action["tcp.r5"], action["tcp.r6"],
            ], dtype=np.float64)
            quat = rotation_6d_to_quaternion(r6d)  # Returns [qw, qx, qy, qz]

            # Build full 4x4 pose matrix from position + quaternion
            pos7 = np.array([x, y, z, quat[0], quat[1], quat[2], quat[3]])
            ee_matrix = quaternion_to_matrix(pos7, input_format="wxyz")
            # print("Sending Cartesian action, pose matrix:\n", ee_matrix)
            
            # Send absolute pose as cartesian_absolute command
            cmd = {"type": "cartesian_absolute", "pose": ee_matrix.tolist()}
            self._ws_send_recv(cmd)
            return action

        except Exception as e:
            logger.warning(f"Error sending cartesian action: {e}")
            return action

    def _send_cartesian_motion_force_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send Cartesian motion + force command.

        TODO: Implement force control via WebSocket.
        """
        logger.warning("Cartesian motion+force action not yet implemented")
        return action

    def _send_gripper_action(self, action: dict[str, Any]) -> None:
        """Send gripper action if gripper is enabled and action contains gripper command."""
        if not self.config.use_gripper:
            return
        try:
            if "gripper.pos" in action:
                target = float(action["gripper.pos"])
                target = float(np.clip(
                    target,
                    self.config.gripper_min_position,
                    self.config.gripper_max_position,
                ))
                success = self._send_gripper_position_command(target)
                if not success:
                    logger.warning("Failed to send action to gripper")
        except Exception as e:
            logger.warning(f"Error sending gripper action: {e}")

    # ======================== Disconnect ========================

    def disconnect(self) -> None:
        """Disconnect from robot server, gripper, and cameras."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # Close WebSocket
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            self._ws = None

        # Terminate server subprocess
        if self._server_proc is not None:
            try:
                self._server_proc.terminate()
                self._server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self._server_proc.kill()
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Error terminating server: {e}")
            self._server_proc = None

        # Disconnect cameras
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception as e:
                logger.error(f"Failed to disconnect camera: {e}")

        self._gripper_connected = False
        self._robot_connected = False
        self._is_connected = False
        self._last_state = None
        logger.info(f"{self} disconnected successfully")

    def __repr__(self) -> str:
        return (
            f"FrankaResearch3("
            f"fci_ip={self.config.fci_ip}, "
            f"gripper_port={self.config.gripper_server_port if self.config.use_gripper else 'N/A'}, "
            f"connected={self.is_connected})"
        )
