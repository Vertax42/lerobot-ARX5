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
"""Pre-warming of the xensesdk per-serial config cache for tactile sensors.

A Sunplus (0x1300) flash read resets/re-enumerates the sensor; doing that
concurrently during a parallel camera connect races the SDK's non-thread-safe
flash lib and moves camera nodes mid-open. ``prewarm_tactile_config_cache`` reads
each *uncached* Sunplus sensor's flash sequentially and up-front, encrypting the
result into the SDK's per-serial cache dir (``CONFIG_CACHE_DIR`` == the camera's
``_XENSE_CONFIG_CACHE_DIR`` == ``~/.xensesdk/config``). The actual camera open then
loads config from that cache (``camera_xense.py`` passes ``config_path``) with no
flash read, so the connect — parallel or not — never triggers a device reset.

Kept in the shared Xense camera package so any robot with Xense tactile cameras
can reuse it.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Any

from .configuration_xense import XenseTactileCameraConfig

# Config-cache key xensesdk uses for its per-serial config cache. Mirrors the
# constant baked into the SDK (xensesdk.core.ctx_builders); if it ever drifts the
# pre-warm just fails to decrypt and the SDK falls back to its own flash read.
_XENSE_CONFIG_CACHE_PSWD = "Wz8mmWz2ALJ6X5Ic"


def _wait_nodes_settle(serials, logger, timeout_s: float = 15.0) -> None:
    """Wait until each serial's ``/dev/v4l/by-id`` capture node is back + openable
    after a flash-read reset re-enumerated it."""
    deadline = time.perf_counter() + timeout_s
    for sn in serials:
        settled = False
        while time.perf_counter() < deadline:
            matches = glob.glob(f"/dev/v4l/by-id/*{sn}*-video-index0")
            if matches:
                try:
                    fd = os.open(os.path.realpath(matches[0]), os.O_RDWR)
                    os.close(fd)
                    settled = True
                    break
                except OSError:
                    pass
            time.sleep(0.2)
        if not settled:
            logger.warning(
                f"  Sensor {sn} V4L2 node did not settle within {timeout_s:.0f}s after pre-warm"
            )


def prewarm_tactile_config_cache(camera_configs: dict[str, Any], logger) -> None:
    """Warm the xensesdk per-serial config cache for tactile sensors **before**
    opening any camera.

    A Sunplus (0x1300) flash read resets/re-enumerates the sensor. Doing that
    concurrently (the parallel camera connect) on a cold cache races the SDK's
    non-thread-safe flash lib and moves camera nodes mid-open. Reading here,
    sequentially and only for **uncached Sunplus** sensors, makes cold start
    safe; then we wait for the nodes to settle. A warm cache is just a cheap
    ``exists()`` stat — no flash read, no reset, no extra cache decrypt (the SDK
    still reads the cache once at connect)."""
    serials = [
        cfg.serial_number
        for cfg in camera_configs.values()
        if isinstance(cfg, XenseTactileCameraConfig) and getattr(cfg, "serial_number", None)
    ]
    if not serials:
        return
    try:
        from xensesdk.core.ctx_builders import CONFIG_CACHE_DIR
        from xensesdk.flash import FlashClient
        from xensesdk.flash.sunplus_backend import is_sunplus
        from xensesdk.utils.encrypt import encrypt_config_file
    except Exception as e:  # SDK without the Sunplus/xbin path — nothing to do.
        logger.debug(f"Config pre-warm unavailable ({e}); skipping")
        return

    uncached = [sn for sn in serials if is_sunplus(sn) and not (CONFIG_CACHE_DIR / sn).exists()]
    if not uncached:
        return  # warm cache: cheap stat only, no flash read / reset

    logger.info(
        f"  Pre-warming config cache (cold start) for {len(uncached)} Sunplus sensor(s): {uncached}"
    )
    client = FlashClient()
    try:
        for sn in uncached:
            try:
                patch = client.read_patch(serial_number=sn)  # reads flash -> resets device
                CONFIG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                encrypt_config_file(
                    patch, CONFIG_CACHE_DIR / sn, password=_XENSE_CONFIG_CACHE_PSWD, format="xbin"
                )
            except Exception as e:
                logger.warning(f"  Config pre-warm failed for {sn}: {e}")
    finally:
        client.cleanup()

    _wait_nodes_settle(uncached, logger)
