# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from enum import Enum

from ..configs import CameraConfig


class XenseOutputType(Enum):
    """Xense sensor output types matching SDK's Sensor.OutputType."""

    # Image outputs
    RECTIFY = "rectify"  # Rectified image, shape=(700, 400, 3), RGB
    DIFFERENCE = "difference"  # Difference image, shape=(700, 400, 3), RGB
    DEPTH = "depth"  # Depth map, shape=(700, 400), unit: mm

    # 2D/3D marker and force outputs
    MARKER_2D = "marker_2d"  # Tangential displacement, shape=(35, 20, 2)
    FORCE = "force"  # 3D force distribution, shape=(35, 20, 3)
    FORCE_NORM = "force_norm"  # Normal force component, shape=(35, 20, 3)
    FORCE_RESULTANT = "force_resultant"  # 6D force resultant, shape=(6,)

    # 3D mesh outputs
    MESH_3D = "mesh_3d"  # Current frame 3D mesh, shape=(35, 20, 3)
    MESH_3D_INIT = "mesh_3d_init"  # Initial 3D mesh, shape=(35, 20, 3)
    MESH_3D_FLOW = "mesh_3d_flow"  # Mesh deformation vector, shape=(35, 20, 3)


@CameraConfig.register_subclass("xense")
@dataclass
class XenseCameraConfig(CameraConfig):
    """Configuration class for Xense tactile sensor devices.

    This class provides configuration options for Xense tactile sensors,
    supporting various output types including force distribution, depth maps,
    and 2D marker tracking.

    Example configurations:
    ```python
    # Basic force sensing configuration
    XenseCameraConfig(
        serial_number="OG000344",
        fps=60,
        output_types=[XenseOutputType.FORCE, XenseOutputType.FORCE_RESULTANT]
    )

    # Multi-modal configuration with depth
    XenseCameraConfig(
        serial_number="OG000352",
        fps=30,
        output_types=[XenseOutputType.FORCE, XenseOutputType.DEPTH]
    )
    ```

    Attributes:
        serial_number: Xense sensor serial number (e.g., "OG000344")
        fps: Requested frames per second for data acquisition (default: 60)
        width: Frame width in pixels (auto-set based on output_types: 400 for images, 20 for force)
        height: Frame height in pixels (auto-set based on output_types: 700 for images, 35 for force)
        output_types: List of output types to read from the sensor
        warmup_s: Time to wait before returning from connect (in seconds)

    Note:
        - Image outputs (DIFFERENCE, RECTIFY) have shape (700, 400, 3)
        - Depth output has shape (700, 400)
        - Force distribution output has shape (35, 20, 3)
        - Force resultant output has shape (6,) representing 6D force/torque
        - Width and height are automatically set based on the first output type
    """

    serial_number: str
    output_types: list[XenseOutputType] = None
    warmup_s: float = 0.5

    def __post_init__(self):
        # Set default output types if not provided
        if self.output_types is None:
            self.output_types = [XenseOutputType.FORCE, XenseOutputType.FORCE_RESULTANT]

        # Validate output types
        for output_type in self.output_types:
            if not isinstance(output_type, XenseOutputType):
                raise ValueError(
                    f"Invalid output_type: {output_type}. Must be a XenseOutputType enum."
                )

        # Set default FPS if not provided
        if self.fps is None:
            self.fps = 30

        # Set width and height based on the primary output type
        # DIFFERENCE/RECTIFY/DEPTH images have shape (700, 400, 3) or (700, 400)
        # Force/mesh data have shape (35, 20, 3)
        if self.width is None or self.height is None:
            # Check if using image outputs (DIFFERENCE, RECTIFY, or DEPTH)
            image_outputs = {
                XenseOutputType.DIFFERENCE,
                XenseOutputType.RECTIFY,
                XenseOutputType.DEPTH,
            }
            if any(ot in image_outputs for ot in self.output_types):
                # Image outputs: height=700, width=400
                if self.width is None:
                    self.width = 400
                if self.height is None:
                    self.height = 700
            else:
                # Force/mesh outputs: height=35, width=20
                if self.width is None:
                    self.width = 20
                if self.height is None:
                    self.height = 35
