from dataclasses import dataclass


@dataclass
class XenseGripperConfig:
    """Configuration for XenseGripper via HTTP server (gripper_server_xense.py).

    Attributes:
        gripper_server_ip: IP address of the xense gripper HTTP server
        gripper_server_port: Port of the xense gripper HTTP server
        gripper_id: USB device ID for the xense gripper hardware
        gripper_default_velocity: Default gripper velocity [mm/s]
        gripper_default_force: Default gripper force [N]
        gripper_min_width_mm: Minimum gripper width [mm] (fully closed)
        gripper_max_width_mm: Maximum gripper width [mm] (fully open)
        gripper_timeout: HTTP request timeout [seconds]
    """

    gripper_server_ip: str = "127.0.0.1"
    gripper_server_port: int = 7001

    gripper_id: str = "7ec0c7f50ea6"  # USB device ID

    gripper_default_velocity: float = 100.0  # mm/s
    gripper_default_force: float = 30.0  # N

    # Physical width range [mm]
    gripper_min_width_mm: float = 0.0  # fully closed
    gripper_max_width_mm: float = 85.0  # fully open

    gripper_timeout: float = 2.0  # seconds

    def __post_init__(self):
        if not self.gripper_server_ip:
            raise ValueError("gripper_server_ip is required for XenseGripper")
