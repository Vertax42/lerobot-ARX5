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
"""Provide the still images the OpenCV camera tests use as a stand-in camera.

``tests/artifacts/cameras/`` has never existed in this fork — ``.gitattributes``
declares LFS rules for it, but no binary was ever committed, so every test that
opened one of these files failed with a ConnectionError that looked like a broken
camera stack rather than a missing fixture.

These particular files are safe to generate. The OpenCV tests only care about the
image *dimensions* (the config's width/height are checked against what comes back,
and reads go through MockLoopingVideoCapture); no test asserts anything about the
pixels. They are fixtures, not recorded baselines.

The RealSense ``.bag`` is a different matter — it is a captured device stream that
cannot be synthesised — and the ``.safetensors`` under
``tests/artifacts/{datasets,image_transforms}`` are regression baselines whose
whole purpose is to predate the current code. Regenerating those locally would
make their tests compare the current behaviour against itself. Those skip instead.

Generated rather than committed so no binaries (and no LFS objects) enter the repo
for something reproducible in a millisecond.
"""

from pathlib import Path

import numpy as np
import pytest

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "cameras"

# Must stay in sync with TEST_IMAGE_SIZES in test_opencv.py, plus the default.
IMAGE_SIZES = ("128x128", "160x120", "320x180", "480x270")


def _write_png(path: Path, width: int, height: int) -> None:
    import cv2

    # A deterministic gradient rather than noise: if one of these ever ends up
    # compared against something, an obviously synthetic image is easier to
    # recognise than plausible-looking static.
    xs = np.linspace(0, 255, width, dtype=np.uint8)
    ys = np.linspace(0, 255, height, dtype=np.uint8)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = xs[None, :]
    img[:, :, 1] = ys[:, None]
    img[:, :, 2] = 128
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"failed to write test image {path}")


@pytest.fixture(scope="session", autouse=True)
def camera_test_images() -> None:
    """Create the PNGs the OpenCV camera tests open, if they are not present.

    Session-scoped and autouse: the paths are module-level constants baked into
    parametrize lists at import time, so they have to exist before collection
    finishes rather than being injected per test.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for size in IMAGE_SIZES:
        width, height = (int(v) for v in size.split("x"))
        path = ARTIFACTS_DIR / f"image_{size}.png"
        if not path.exists():
            _write_png(path, width, height)
