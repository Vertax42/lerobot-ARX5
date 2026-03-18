from dataclasses import dataclass, field
from typing import Dict, Union
from enum import Enum

from lerobot.cameras.utils import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.cameras.configs import ColorMode

from lerobot.robots.pylibfranka_research3.config_franka_gripper import FrankaGripperConfig
from lerobot.robots.pylibfranka_research3.config_xense_gripper import XenseGripperConfig, SensorOutputType

class ControlMode(str, Enum):
    """Control mode for Flexiv Rizon4.

    JOINT_IMPEDANCE:
        Joint impedance control (maps to NRT_JOINT_IMPEDANCE).
        Uses impedance control with configurable stiffness via stiffness_ratio.
        - Action: joint positions (7D) + gripper (1D) = 8D
        - Observation: joint positions (7D) + velocities (7D) + efforts (7D) + gripper (1D) = 22D

    CARTESIAN_IMPEDANCE:
        Cartesian motion control (maps to
          NRT_CARTESIAN_MOTION_FORCE).
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
    port: int = 8765
    # fci_ip: str = None
    
    # control_mode: str = "joint_impedance"  # Options: joint_impedance, cartesian_impedance
    control_mode: ControlMode = ControlMode.CARTESIAN_IMPEDANCE

    use_joint_observation: bool = False # whether to use joint positions/velocities/efforts in observation (only applies to cartesian_impedance mode)

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
    # robot_home_position: list = field(default_factory=lambda: [-0.82917076, 0.04384267, -0.03789382, -2.4083676, -0.01327341, 2.4243426, -0.10018315])
    robot_home_position: list = field(default_factory=lambda: [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853])
    
    robot_tcp_home_position: list = field(default_factory=lambda: [0.5592, -0.0073, 0.5123, 0.0, 1.0, 0.0, 0.0]) # x,y,z,w,x,y,z (wxyz)

    gripper_type: str = "xense_gripper"  # Options: "franka_gripper", "xense_gripper"

    # Whether to use the gripper
    use_gripper: bool = True

    # ======================== Franka Gripper Parameters ========================
    gripper_ip: str = "192.168.99.111"  # Franka gripper IP (shares FCI connection)
    gripper_speed: float = 0.5  # m/s
    gripper_force: float = 60.0  # N
    gripper_min_pos: float = 0.0  # m (fully closed)
    gripper_max_pos: float = 0.08  # m (fully open, ~80mm for Franka Hand)
    gripper_init_open: bool = True

    # ======================== Xense Gripper Parameters ========================
    gripper_mac_addr: str = "7e0b26fa1cbe"
    gripper_enable_sensor: bool = True
    gripper_rectify_size: tuple[int, int] = (96, 160)
    gripper_sensor_output_type: SensorOutputType = SensorOutputType.RECTIFY
    gripper_sensor_keys: dict[str, str] = field(
        default_factory=lambda: {
            "OG000651": "left_tactile",
            "OG000652": "right_tactile",
        }
    )
    gripper_xense_min_pos: float = 0.0  # mm (fully closed)
    gripper_xense_max_pos: float = 85.0  # mm (fully open)
    gripper_xense_v_max: float = 100.0  # Maximum velocity mm/s
    gripper_xense_f_max: float = 30.0  # Maximum force N
    gripper_xense_init_open: bool = True

    # Auto-created in __post_init__ from gripper_* parameters (do not set directly)
    gripper: Union[FrankaGripperConfig, XenseGripperConfig] | None = field(default=None, init=False)
    
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

    def __post_init__(self):
        super().__post_init__()

        # Create gripper config from exposed parameters
        if self.use_gripper and self.gripper_type == "franka_gripper":
            self.gripper = FrankaGripperConfig(
                gripper_ip=self.gripper_ip,
                gripper_speed=self.gripper_speed,
                gripper_force=self.gripper_force,
                gripper_min_pos=self.gripper_min_pos,
                gripper_max_pos=self.gripper_max_pos,
                init_open=self.gripper_init_open,
            )
        elif self.use_gripper and self.gripper_type == "xense_gripper":
            self.gripper = XenseGripperConfig(
                mac_addr=self.gripper_mac_addr,
                enable_sensor=self.gripper_enable_sensor,
                rectify_size=self.gripper_rectify_size,
                sensor_output_type=self.gripper_sensor_output_type,
                sensor_keys=self.gripper_sensor_keys,
                gripper_min_pos=self.gripper_xense_min_pos,
                gripper_max_pos=self.gripper_xense_max_pos,
                gripper_v_max=self.gripper_xense_v_max,
                gripper_f_max=self.gripper_xense_f_max,
                init_open=self.gripper_xense_init_open,
            )
        else:
            self.gripper = None
