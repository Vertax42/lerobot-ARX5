#!/usr/bin/env python

# Copyright 2026 The XenseRobotics Inc. team. All rights reserved.
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

"""Turn a gripper auto-discovery sweep into entries in an arm's ``config.cameras``.

Every arm that mounts a TacCap or serial gripper wires the same cameras the same
way: one wrist camera per side, plus two tactile sensors when they are enabled.
That mapping is a property of the *gripper*, not of the arm holding it, so it
lives here beside the discovery sweeps that feed it rather than being copied into
each robot package (which is what it was before).

Camera settings are deliberately frozen to the values the recipes used to spell
out by hand — the tuned tactile exposure is now the ``XenseTactileCameraConfig``
default — so turning discovery on does not change a single recorded pixel.

Import this module directly, not via ``lerobot.grippers``: it pulls in camera
configs, and the package root stays importable on a host with no camera stack.
"""

from collections.abc import Sequence
from logging import Logger

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.xense import XenseOutputType, XenseTactileCameraConfig
from lerobot.utils.errors import DeviceNotConnectedError

# Wrist camera: MJPG so the USB bus carries compressed frames.
WRIST_FOURCC = "MJPG"
WRIST_WIDTH = 640
WRIST_HEIGHT = 480
WRIST_FPS = 30
WRIST_WARMUP_S = 1.0

# GSPS tactile sensors: two per gripper, one per jaw.
TACTILE_FPS = 30
TACTILE_WARMUP_S = 0.05
TACTILE_PER_SIDE = 2


def _wrist_camera_config(wrist_camera_name: str) -> OpenCVCameraConfig:
    """The wrist camera config for a discovered V4L2 device name."""
    return OpenCVCameraConfig(
        index_or_path=wrist_camera_name,
        fourcc=WRIST_FOURCC,
        width=WRIST_WIDTH,
        height=WRIST_HEIGHT,
        fps=WRIST_FPS,
        warmup_s=WRIST_WARMUP_S,
    )


def _tactile_camera_config(serial_number: str) -> XenseTactileCameraConfig:
    """The tactile camera config for one discovered GSPS serial number."""
    return XenseTactileCameraConfig(
        serial_number=serial_number,
        fps=TACTILE_FPS,
        output_types=[XenseOutputType.RECTIFY],
        warmup_s=TACTILE_WARMUP_S,
    )


def _add_side_cameras(
    cameras: dict,
    side: str,
    wrist_camera_name: str,
    tactile_sns: Sequence[str],
    *,
    enable_tactile: bool,
    too_few_tactile_msg: str,
) -> None:
    """Add one side's wrist (and optionally tactile) cameras under the shared key
    scheme ``<side>_wrist`` / ``<side>_tactile_<i>``.

    The keys match the station-driven wiring, so datasets recorded with
    discovery on stay compatible with ones recorded from a hand-written recipe.

    Raises:
        DeviceNotConnectedError: If tactile sensors are enabled but the sweep
            found fewer than ``TACTILE_PER_SIDE`` of them on this side.
    """
    cameras[f"{side}_wrist"] = _wrist_camera_config(wrist_camera_name)

    if not enable_tactile:
        return

    if len(tactile_sns) < TACTILE_PER_SIDE:
        raise DeviceNotConnectedError(too_few_tactile_msg)

    for i, sn in enumerate(tactile_sns):
        cameras[f"{side}_tactile_{i}"] = _tactile_camera_config(sn)


def inject_taccap_cameras(
    cameras: dict,
    *,
    sides: Sequence[str],
    enable_tactile: bool,
    logger: Logger,
) -> dict[str, str]:
    """Discover TacCap wrist + GSPS tactile devices and add them to ``cameras``.

    Args:
        cameras: The arm's ``config.cameras`` dict, mutated in place.
        sides: Sides that must be found; a missing one is an error.
        enable_tactile: Whether to also wire the GSPS tactile sensors.
        logger: The arm's logger, so messages carry the arm's name.

    Returns:
        ``{side: mcu_device}`` for every requested side. The sweep already
        resolved each gripper's MCU path, so hand it to the follower driver via
        :func:`adopt_taccap_mcu_device` and its ``connect()`` skips a second
        find_left/find_right scan of the same bus.

    Raises:
        DeviceNotConnectedError: If a requested side has no gripper, or has too
            few tactile sensors.
    """
    from lerobot.grippers.taccap.discovery import discover_taccap_sides

    found = discover_taccap_sides()
    mcu_devices: dict[str, str] = {}

    for side in sides:
        dev = found.get(side)
        if dev is None:
            raise DeviceNotConnectedError(
                f"taccap auto-discover: no {side} gripper/wrist camera found."
            )
        _add_side_cameras(
            cameras,
            side,
            dev.wrist_camera_name,
            dev.tactile_sns,
            enable_tactile=enable_tactile,
            too_few_tactile_msg=(
                f"taccap auto-discover: expected {TACTILE_PER_SIDE} GSPS tactile "
                f"sensors for {side}, found {list(dev.tactile_sns)}."
            ),
        )
        mcu_devices[side] = dev.mcu_device

    logger.info(f"taccap auto-discovered cameras: {sorted(cameras)}")
    return mcu_devices


def inject_serial_gripper_cameras(
    cameras: dict,
    *,
    sides: Sequence[str],
    enable_tactile: bool,
    logger: Logger,
) -> None:
    """Same as :func:`inject_taccap_cameras`, for serial (parallel-jaw) grippers.

    Each arm's gripper board, wrist camera and two tactile sensors share one USB
    hub, so the gripper — already side-resolved by board-SN parity — identifies
    the hub, and the cameras on it follow.

    Args:
        cameras: The arm's ``config.cameras`` dict, mutated in place.
        sides: Only the sides that actually have a gripper. A bench running one
            arm without one must not fail discovery for the other, so callers
            filter this list rather than passing both sides unconditionally.
        enable_tactile: Whether to also wire the tactile sensors.
        logger: The arm's logger, so messages carry the arm's name.

    Raises:
        DeviceNotConnectedError: If a requested side has no gripper, or has too
            few tactile sensors.
    """
    from lerobot.grippers.serial.discovery import discover_serial_gripper_cameras

    if not sides:
        return

    found = discover_serial_gripper_cameras(sides=tuple(sides))

    for side in sides:
        dev = found.get(side)
        if dev is None:
            raise DeviceNotConnectedError(
                f"serial gripper auto-discover: no {side} gripper found, so its "
                "wrist and tactile cameras could not be resolved."
            )
        _add_side_cameras(
            cameras,
            side,
            dev.wrist_camera_name,
            dev.tactile_sns,
            enable_tactile=enable_tactile,
            too_few_tactile_msg=(
                f"serial gripper auto-discover: expected {TACTILE_PER_SIDE} tactile "
                f"sensors on the {side} gripper's hub {dev.usb_hub}, "
                f"found {list(dev.tactile_sns)}."
            ),
        )

    logger.info(f"serial gripper auto-discovered cameras: {sorted(cameras)}")


def adopt_taccap_mcu_device(gripper, side: str, mcu_device: str, logger: Logger) -> None:
    """Pin an already-discovered MCU path onto that side's follower driver.

    No-op when the side has no gripper, or when the operator pinned
    ``mcu_device`` explicitly in config — an explicit value always wins.
    """
    if gripper is None or getattr(gripper, "_mcu_device", None):
        return
    gripper._mcu_device = mcu_device
    logger.debug(f"[{side}] taccap follower pinned to discovered {mcu_device}")
