# lerobot-xense — Claude working notes

XenseRobotics Physical-AI platform: a lerobot fork carrying the arm robots
(ARX5, Flexiv Rizon4, Elite CS66) plus the Xense tactile gripper and its wrist
camera. Sister repo to `xense-taccap-lerobot`, which carries the TacCap-Gripper
handheld rig; the two share the `taccap-gripper` SDK submodule and several of
these traps.

## `import xense.taccap` must precede torchvision

torchvision ships a vendored `libjpeg` that claims the `LIBJPEG_8.0` symbol
version but carries **none** of the `jpeg12_*` symbols conda's `libtiff` needs.
Whichever loads first wins the version slot, so once torchvision is in, every
later `import xense.taccap` — which reaches libtiff via
`libopencv_videoio` → `libopencv_imgcodecs` → `libtiff` — dies with:

```text
ImportError: .../libtiff.so.6: undefined symbol: jpeg12_write_raw_data, version LIBJPEG_8.0
```

`lerobot_record.py`, `lerobot_replay.py`, `lerobot_teleoperate.py` and
`tests/conftest.py` each carry a `contextlib.suppress(ImportError)` import of
`xense.taccap` **above every lerobot import** for exactly this. Moving one below
them puts the bug back; `tests/robots/test_taccap_import_order.py` fails if that
happens.

**Why it hid for so long:** nothing fails at startup. `XenseWristCamera` imports
`FisheyeUndistorter` lazily inside `connect()` (`cameras/xense/camera_wrist.py`),
so a recording session came up fine and then died at camera-connect time. In the
test suite it looked like 11 broken fisheye tests rather than one load-order
problem.

Only the entry points that **both** pull torchvision (through `lerobot.datasets`)
and touch the SDK need the block. `lerobot-calibrate`, `-find-cameras`,
`-find-port`, `-setup-motors` and `-info` never load torchvision;
`-dataset-viz`, `-edit-dataset` and `-annotate-reward` do load it but never
import the SDK, so adding the block there would only bolt an SDK dependency onto
pure dataset tooling. Verify with a fresh interpreter before adding it anywhere
new rather than sprinkling it.

Things that do **not** work, so nobody re-tries them:

- **`LD_PRELOAD` of conda's libjpeg.** torchvision's copy is auditwheel-renamed
  (`libjpeg.4af9affd.so.8`), so it is not competing on soname but on the symbol
  version — preloading cannot outrank it.
- **Moving the lazy import in `camera_wrist.py` to module top level.** That
  module is itself imported _after_ `lerobot.datasets`, so the order is unchanged
  — it only moves the same ImportError from `connect()` to package import, which
  is strictly worse.
- **Dropping libtiff from the SDK.** It arrives through `libopencv_videoio`'s
  own `DT_NEEDED` on `libopencv_imgcodecs`; the SDK never calls an imgcodecs API
  and does not link it explicitly. Removing it means replacing `cv::VideoCapture`
  with a hand-written V4L2 capture path — and MJPEG decoding then needs a JPEG
  library anyway, so the class of conflict returns.

## Wrist camera: this repo owns the UVC device, not the SDK

`open_cameras` appears nowhere here. `XenseWristCamera` opens the device through
the LeRobot camera framework and calls `FisheyeUndistorter.apply()` itself, using
intrinsics the arm reads off the gripper's MCU and hands over with
`set_fisheye_calibration()`.

This is why the SDK's `wrist_color_mode` default (RGB since SDK `6b33678`) does
not reach us: it only applies when the SDK owns the camera. `camera_wrist.py`'s
`_postprocess_image` still receives BGR straight from OpenCV, which is exactly
what `FisheyeUndistorter.apply()` expects, and the base class converts to RGB
afterwards.
