"""XenseGripper — HTTP-based xense gripper control for pylibfranka_research3.

Communicates with gripper_server_xense.py (FastAPI server) via HTTP requests.

Position convention: 0.0 = open, 1.0 = closed.
"""

import logging

import numpy as np
import requests

from .config_xense_gripper import XenseGripperConfig


class XenseGripper:
    """XenseGripper via HTTP server with normalized position control.

    Provides the same interface as FrankaGripper:
    connect/disconnect/get_gripper_position/set_gripper_position.
    """

    config_class = XenseGripperConfig

    def __init__(self, config: XenseGripperConfig):
        self._config = config
        self._server_ip = config.gripper_server_ip
        self._server_port = config.gripper_server_port
        self._default_velocity = config.gripper_default_velocity
        self._default_force = config.gripper_default_force
        self._min_width_mm = config.gripper_min_width_mm
        self._max_width_mm = config.gripper_max_width_mm
        self._timeout = config.gripper_timeout

        self._base_url = f"http://{self._server_ip}:{self._server_port}"
        self._is_connected = False
        self._logger = logging.getLogger(f"XenseGripper-{self._server_ip}:{self._server_port}")

    def connect(self) -> None:
        """Connect to the xense gripper HTTP server (health check)."""
        if self._is_connected:
            raise RuntimeError("XenseGripper already connected")

        self._logger.info(f"Connecting to XenseGripper server at {self._base_url}...")
        if not self._health_check():
            raise ConnectionError(f"Cannot reach xense gripper server at {self._base_url}")

        self._is_connected = True
        self._logger.info("XenseGripper connected.")

    def disconnect(self) -> None:
        """Disconnect from the xense gripper HTTP server."""
        if not self._is_connected:
            return
        self._is_connected = False
        self._logger.info("XenseGripper disconnected.")

    def get_gripper_position(self) -> float:
        """Get normalized gripper position [0.0=open, 1.0=closed]."""
        if not self._is_connected:
            return 0.0
        try:
            response = requests.get(f"{self._base_url}/get_pos", timeout=self._timeout)
            if response.status_code != 200:
                self._logger.warning(f"Failed to get gripper position: HTTP {response.status_code}")
                return 0.0
            data = response.json()
            width = data.get("position", 0.0)
            width = max(self._min_width_mm, min(self._max_width_mm, width))

            width_range = self._max_width_mm - self._min_width_mm
            if width_range <= 0:
                return 0.0
            # 0.0=open (max_width), 1.0=closed (min_width)
            closed_ratio = 1.0 - (width - self._min_width_mm) / width_range
            return float(np.clip(closed_ratio, 0.0, 1.0))
        except Exception as e:
            self._logger.error(f"Failed to get gripper position: {e}")
            return 0.0

    def set_gripper_position(self, normalized_pos: float) -> None:
        """Set gripper position.

        Args:
            normalized_pos: Target position [0.0=open, 1.0=closed]
        """
        if not self._is_connected:
            self._logger.warning("Gripper not connected, cannot set position")
            return

        normalized_pos = max(0.0, min(1.0, normalized_pos))
        # Convert: 0.0=open (max_width), 1.0=closed (min_width)
        target_width = self._min_width_mm + (1.0 - normalized_pos) * (
            self._max_width_mm - self._min_width_mm
        )
        target_width = float(np.clip(target_width, self._min_width_mm, self._max_width_mm))

        try:
            response = requests.post(
                f"{self._base_url}/move",
                json={
                    "pos": target_width,
                    "vmax": self._default_velocity,
                    "fmax": self._default_force,
                },
                timeout=self._timeout,
            )
            if response.status_code != 200:
                self._logger.warning(f"Failed to send gripper command: HTTP {response.status_code}")
        except Exception as e:
            self._logger.error(f"Failed to send gripper position command: {e}")

    def _health_check(self) -> bool:
        """Check if the xense gripper HTTP server is reachable."""
        try:
            response = requests.get(f"{self._base_url}/get_pos", timeout=self._timeout)
            return response.status_code == 200
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False
