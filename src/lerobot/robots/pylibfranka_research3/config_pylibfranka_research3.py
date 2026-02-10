from dataclasses import dataclass, field
from typing import Dict
from enum import Enum

from lerobot.cameras.utils import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.cameras.configs import ColorMode

class ControlMode(str, Enum):
    """Control mode for Flexiv Rizon4.

    JOINT_IMPEDANCE:
        Joint impedance control (maps to NRT_JOINT_IMPEDANCE).
        Uses impedance control with configurable stiffness via stiffness_ratio.
        - Action: joint positions (7D) + gripper (1D) = 8D
        - Observation: joint positions (7D) + velocities (7D) + efforts (7D) + gripper (1D) = 22D

    CARTESIAN_IMPEDANCE:
        Cartesian motion control (maps to NRT_CARTESIAN_MOTION_FORCE).
        When use_force=False: pure motion control
        When use_force=True: motion + force control
        - Action: TCP pose (7D) + gripper (1D) = 8D, or pose + wrench (13D) + gripper (1D) = 14D
        - Observation: TCP pose (7D) + gripper (1D) = 8D, or pose + wrench (13D) + gripper (1D) = 14D
    """

    JOINT_IMPEDANCE = "joint_impedance"
    CARTESIAN_IMPEDANCE = "cartesian_impedance"

@RobotConfig.register_subclass("pylibfranka_research3")
@dataclass
class PylibfrankaResearch3Config(RobotConfig):

    # ======================== Franka Follower Arm Configuration ========================
    
    # FCI (Fast Communication Interface) IP for Franka control
    fci_ip: str = "192.168.99.111"
    
    # control_mode: str = "joint_impedance"  # Options: joint_impedance, cartesian_impedance
    control_mode: ControlMode = ControlMode.CARTESIAN_IMPEDANCE

    # use_force: Enable force control (only applies to start_torque_control mode)
    use_force: bool = False # joint torque (7D)
    
    # Connection behavior
    go_to_start: bool = (
        True  # If True, move robot to start position after connecting. If False, stay at current position.
    )

    # Joint motion constraints (from examples: MAX_VEL = [2.0] * DoF, MAX_ACC = [3.0] * DoF)
    # velmax [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]
    joint_max_vel: list[float] = field(
        default_factory=lambda: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]  # rad/s
    )
    # accmax [10, 10, 10, 10, 10, 10, 10]
    joint_max_acc: list[float] = field(
        default_factory=lambda: [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]  # rad/s^2
    )

    # Home position for robot (7 joint angles in radians)
    # robot_home_position: list = field(default_factory=lambda: [-0.030264, -0.523095, -0.091621, -2.812467, -0.089465,  2.25039,   0.709976])
    # robot_home_position: list = field(default_factory=lambda: [-0.08211147, -0.6067168,  -0.03138583, -2.7927575,  -0.02443479,  2.211011, 0.67388374])
    robot_home_position: list = field(default_factory=lambda: [-0.82917076, 0.04384267, -0.03789382, -2.4083676, -0.01327341, 2.4243426, -0.10018315])
    
    # ======================== Xense Gripper Configuration ========================

    # Whether to use the gripper
    use_gripper: bool = True
    # Hand server communication
    gripper_server_ip: str = "127.0.0.1"
    gripper_server_port: int = 7001
    
    # Gripper hardware identification
    gripper_id: str = "7ec0c7f50ea6"  # USB device ID
    
    # Gripper motion parameters
    gripper_default_velocity: float = 100.0   # vel
    gripper_default_force: float = 30.0        # force
    
    # Gripper position limits (0.0=open, 1.0=closed)
    gripper_min_position: float = 0.0
    gripper_max_position: float = 1.0
    
    # Gripper home position
    gripper_home_position: float = 0.0  # Middle position
    
    # Physical width mapping (mm)
    gripper_min_width_mm: float = 0.0    # When open
    gripper_max_width_mm: float = 85.0   # When closed
    
    # Gripper communication timeout (seconds)
    gripper_timeout: float = 2.0
    
    # ======================== Camera Configuration ========================
    
    # RealSense cameras (2 cameras recommended: main + wrist)
    cameras: Dict[str, RealSenseCameraConfig] = field(default_factory=lambda: {
        # # Main external camera
        # "image": RealSenseCameraConfig(
        #     serial_number_or_name="135522074323",
        #     fps=30,
        #     width=640,
        #     height=480,
        #     color_mode=ColorMode.RGB
        # ),
        # # Wrist-mounted camera
        # "wrist_image": RealSenseCameraConfig(
        #     serial_number_or_name="249322063436",
        #     fps=30,
        #     width=640,
        #     height=480,
        #     color_mode=ColorMode.RGB
        # )
    })
    
    # ======================== Action Synchronization ========================
    
    # Send arm and hand actions simultaneously
    synchronize_actions: bool = True
    
    # Timeout for synchronized actions (seconds)
    action_timeout: float = 0.1
    
