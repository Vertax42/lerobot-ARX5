from dataclasses import dataclass, field
from typing import Dict

from enum import Enum

class SensorOutputType(Enum):
    """Output type for tactile sensors."""

    RECTIFY = "rectify"
    DIFFERENCE = "difference"


@dataclass
class GripperConfig:
    """Configuration for XenseGripper
    
    Attributes:
        mac_addr: Serial number of the robot (e.g., "e2b26adbb104")
        enable_sensor: Whether to enable tactile sensors
        cameras: Dictionary of camera configurations
        sensor_keys: Mapping from sensor SN to feature key name
        gripper_v_max: Maximum velocity mm/s
        gripper_f_max: Maximum force N
        gripper_min_pos: float, the minimum position of the gripper
        gripper_max_pos: float, the maximum position of the gripper
        init_open: bool, whether to open the gripper on connect
    """
    
    # Gripper identification
    mac_addr: str = "bef1504b5391"  # Gripper serial number
    
    enable_sensor: bool = True
    # Sensor settings
    rectify_size: tuple[int, int] = (96, 160)
    sensor_output_type: SensorOutputType = SensorOutputType.RECTIFY
    sensor_keys: dict[str, str] = field(default_factory=dict)
    
    # Example: MIN_OPEN=0 mm, MAX_OPEN=85 mm
    gripper_min_pos: float = 0.0
    gripper_max_pos: float = 85.0

    # Gripper control parameters for set_position()
    gripper_v_max: float = 100.0  # Maximum velocity (0-350 mm/s)
    gripper_f_max: float = 30.0  # Maximum force (0-60 N)

    # Initialize gripper to fully open on connect
    init_open: bool = True

    def __post_init__(self):
        if not self.mac_addr:
            raise ValueError("mac_addr is required for XenseGripper")

        # Set default sensor_keys if not provided
        if not self.sensor_keys:
            self.sensor_keys = {
                "left_tactile": "OG000619",
                "right_tactile": "OG000628",
            }