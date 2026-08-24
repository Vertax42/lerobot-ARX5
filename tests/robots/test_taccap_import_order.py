#!/usr/bin/env python
"""The TacCap SDK must load before torchvision, and the entry scripts enforce it.

torchvision ships a vendored libjpeg that claims the `LIBJPEG_8.0` symbol
version but carries none of the `jpeg12_*` symbols conda's libtiff needs.
Whichever of the two loads first wins the version slot, so once torchvision is
in, every later `import xense.taccap` — which reaches libtiff through
libopencv_videoio -> libopencv_imgcodecs — dies with:

    ImportError: .../libtiff.so.6: undefined symbol: jpeg12_write_raw_data,
                 version LIBJPEG_8.0

The failure is nasty because it is invisible until something actually asks for
the SDK: `XenseWristCamera.connect()` imports FisheyeUndistorter lazily, so a
recording session dies at camera-connect time, not at startup.

The fix is an import of `xense.taccap` above every lerobot import in the entry
scripts that pull torchvision. These tests fail if someone moves or deletes it.
"""

import subprocess
import sys

import pytest

ENTRY_SCRIPTS = [
    "lerobot_record",
    "lerobot_replay",
    "lerobot_teleoperate",
]


def _sdk_importable_after(module: str) -> tuple[bool, str]:
    """Import `module` in a fresh interpreter, then try to import the SDK.

    A subprocess is the point: the conflict is decided by what a *process* has
    already loaded, so it cannot be observed once this one has the SDK in.
    """
    code = f"import {module}\nimport xense.taccap\nprint('ok')\n"
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return p.returncode == 0, (p.stderr or "")[-400:]


@pytest.mark.parametrize("module", [f"lerobot.scripts.{s}" for s in ENTRY_SCRIPTS])
def test_entry_script_leaves_the_taccap_sdk_importable(module):
    pytest.importorskip("xense.taccap", reason="TacCap SDK not installed")
    ok, err = _sdk_importable_after(module)
    assert ok, (
        f"After importing {module}, `import xense.taccap` failed. The pre-import "
        f"above the lerobot imports in that script is what prevents this — check "
        f"it was not moved or removed.\n{err}"
    )


def test_the_conflict_is_real_without_the_ordering():
    """Guard the guard: if this ever passes, the underlying conflict is gone and
    the pre-imports (plus this file) can go with it."""
    pytest.importorskip("xense.taccap", reason="TacCap SDK not installed")
    pytest.importorskip("torchvision")
    ok, _ = _sdk_importable_after("torchvision")
    if ok:
        pytest.skip(
            "torchvision no longer shadows conda's libjpeg — the import-order "
            "workaround in the entry scripts and tests/conftest.py is obsolete "
            "and should be removed."
        )
