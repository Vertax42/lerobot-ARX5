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

"""Auto-discovery of the per-side devices that ship with a TacCap follower gripper.

A TacCap gripper is a self-contained USB hub carrying the gripper MCU (a serial
device), one wrist UVC camera, and two GSPS visuotactile sensors. This module
sniffs the two grippers via the TacCap SDK and resolves, for each side (left /
right), the identifiers the existing LeRobot camera stack needs:

  - wrist camera : a V4L2 device *name* (e.g. "XCA25Z0011s"), derived from the
                   gripper firmware SN (product prefix "TCGU01" → "XC"). Consumed
                   by OpenCVCameraConfig, which resolves names via v4l2-ctl.
  - tactile      : the two GSPS sensor serial numbers (e.g. "GSPS01A27Z0094"),
                   enumerated by xensesdk and matched to the side by USB topology
                   (same USB hub as that side's gripper MCU). GSPS serials do not
                   encode left/right, so topology is the only reliable mapping.

Side comes from the firmware-burned gripper SN via the TacCap SDK, never guessed.
"""

import os
import re
from dataclasses import dataclass, field

from lerobot.utils.robot_utils import get_logger

logger = get_logger("TaccapDiscovery")

# sysfs path like ".../usb1/1-3/1-3.1/1-3.1:1.2/tty/ttyACM1" — the top hub token
# ("1-3") groups every device physically behind one gripper's USB hub, and the
# full port token ("1-3.1") orders devices within that hub.
_HUB_RE = re.compile(r"/usb\d+/(\d+-\d+)/")
_PORT_RE = re.compile(r"/(\d+-\d+(?:\.\d+)+)/")

_WRIST_NAME_PREFIX = "XC"  # wrist V4L2 name = "XC" + gripper SN minus product code


@dataclass
class TaccapSideDevices:
    side: str                      # "left" | "right"
    firmware_sn: str               # gripper SN, e.g. "TCGU01A25Z0011s"
    mcu_device: str                # e.g. "/dev/serial/by-id/usb-1a86_...-if02"
    usb_hub: str                   # e.g. "1-3"
    wrist_camera_name: str         # e.g. "XCA25Z0011s" (V4L2 device name)
    tactile_sns: list[str] = field(default_factory=list)  # ordered GSPS serials


def _import_sdks():
    """Import the TacCap SDK and xensesdk, guarded with actionable errors."""
    try:
        import xense.taccap as taccap  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "xense.taccap is not importable — the TacCap SDK native extension is "
            "missing or stale. Rebuild it with `bash setup_env.sh --install` (or "
            f"`pip install third_party/taccap-gripper`). Original error: {e!r}"
        ) from e
    try:
        from xensesdk import Sensor  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "xensesdk is not importable — needed to enumerate GSPS visuotactile "
            f"sensors. Install the xensesdk wheel. Original error: {e!r}"
        ) from e
    return taccap, Sensor


def _sysfs_device_dir(dev_class: str, node: str) -> str:
    """Realpath of /sys/class/<dev_class>/<node>/device (follows symlinks)."""
    return os.path.realpath(f"/sys/class/{dev_class}/{os.path.basename(node)}/device")


def _usb_hub_and_port(sysfs_dir: str) -> tuple[str | None, str | None]:
    """Extract the USB hub token (e.g. '1-3') and full port token (e.g. '1-3.2')."""
    padded = sysfs_dir + "/"
    hub = _HUB_RE.search(padded)
    port = _PORT_RE.search(padded)
    return (hub.group(1) if hub else None, port.group(1) if port else None)


def _gripper_hub(mcu_device: str) -> str | None:
    """Resolve a gripper serial device path to its USB hub token."""
    tty = os.path.realpath(mcu_device)                       # by-id → /dev/ttyACMx
    hub, _ = _usb_hub_and_port(_sysfs_device_dir("tty", tty))
    return hub


def _video_node(cam) -> str:
    """xensesdk scanSerialNumber() maps SN → cam id (int) or a /dev path."""
    if isinstance(cam, str) and cam.startswith("/dev/"):
        return cam
    return f"/dev/video{int(cam)}"


def discover_taccap_sides() -> dict[str, TaccapSideDevices]:
    """Discover per-side (left/right) TacCap wrist camera + GSPS tactile SNs.

    Returns:
        {"left": TaccapSideDevices, "right": TaccapSideDevices} — only the sides
        that are actually connected are included.

    Raises:
        RuntimeError: If the SDKs are unavailable, no gripper is found, or a side's
            device topology can't be resolved.
    """
    taccap, Sensor = _import_sdks()

    # --- 1. Sniff grippers per side, derive wrist name, resolve USB hub ---
    sides: dict[str, TaccapSideDevices] = {}
    for finder, side in ((taccap.find_left, "left"), (taccap.find_right, "right")):
        try:
            eps = finder()
        except Exception as e:
            logger.info(f"No {side} TacCap gripper found: {e}")
            continue
        parsed = taccap.parse_serial(eps.firmware_sn)
        product = parsed.product or "TCGU01"
        wrist_name = _WRIST_NAME_PREFIX + eps.firmware_sn[len(product):]
        hub = _gripper_hub(eps.mcu_device)
        if hub is None:
            raise RuntimeError(
                f"Could not resolve USB hub for {side} gripper {eps.firmware_sn} "
                f"({eps.mcu_device}); cannot map its tactile sensors."
            )
        sides[side] = TaccapSideDevices(
            side=side,
            firmware_sn=eps.firmware_sn,
            mcu_device=eps.mcu_device,
            usb_hub=hub,
            wrist_camera_name=wrist_name,
        )

    if not sides:
        raise RuntimeError(
            "No TacCap grippers found via scan_grippers()/find_left()/find_right(). "
            "Check USB connection, dialout permissions, and ModemManager not grabbing "
            "the CH343 serial ports."
        )

    # --- 2. Enumerate GSPS tactile sensors, map to side by USB hub ---
    try:
        gsps = Sensor.scanSerialNumber()  # {serial: cam_id}
    except Exception as e:
        raise RuntimeError(f"xensesdk Sensor.scanSerialNumber() failed: {e}") from e

    # collect (usb_port, serial) per hub so we can order sensors deterministically
    by_hub: dict[str, list[tuple[str, str]]] = {}
    for sn, cam in gsps.items():
        node = _video_node(cam)
        hub, port = _usb_hub_and_port(_sysfs_device_dir("video4linux", node))
        if hub is None:
            logger.warn(f"Could not resolve USB hub for tactile sensor {sn} ({node}); skipping.")
            continue
        by_hub.setdefault(hub, []).append((port or node, sn))

    for side, dev in sides.items():
        entries = sorted(by_hub.get(dev.usb_hub, []))  # by USB port ascending
        dev.tactile_sns = [sn for _, sn in entries]
        logger.info(
            f"[{side}] gripper={dev.firmware_sn} hub={dev.usb_hub} "
            f"wrist={dev.wrist_camera_name} tactile={dev.tactile_sns}"
        )

    return sides
