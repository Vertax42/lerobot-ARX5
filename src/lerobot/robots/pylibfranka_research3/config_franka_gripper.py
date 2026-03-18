from dataclasses import dataclass


@dataclass
class FrankaGripperConfig:
    """Configuration for Franka Gripper (via pylibfranka).

    Attributes:
        gripper_ip: IP address of the Franka robot (gripper shares FCI connection)
        gripper_speed: Gripper movement speed [m/s]
        gripper_force: Maximum gripper force [N]
        gripper_min_pos: Minimum gripper width [m] (fully closed)
        gripper_max_pos: Maximum gripper width [m] (fully open, ~0.08m for Franka Hand)
        init_open: Whether to open the gripper on connect
    """

    gripper_ip: str = "192.168.99.111"

    gripper_speed: float = 0.5  # m/s
    gripper_force: float = 60.0  # N (0-70 N for Franka Hand)

    # Physical width range [m]
    gripper_min_pos: float = 0.0  # fully closed
    gripper_max_pos: float = 0.08  # fully open (~80mm for Franka Hand)

    # Initialize gripper to fully open on connect
    init_open: bool = True

    def __post_init__(self):
        if not self.gripper_ip:
            raise ValueError("gripper_ip is required for FrankaGripper")
