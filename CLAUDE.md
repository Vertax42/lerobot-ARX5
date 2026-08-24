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

## Repo weight: attribute it to a ref, not to a path

The repo was 324 MiB against an 8.8 MiB working tree until 2026-08-25. All of it
was history, and 278.6 MiB of that hung off a **single stale branch**,
`dev/taccap-gripper` — 7 vendored `dist/xensesdk-2.0.0-*.whl` builds (196 MiB;
93 + 68 + 11 + 11 + 8.5 + 2.9 + 2.5 compressed) plus upstream lerobot's
`tests/data/**/*.arrow` fixtures (82 MiB). `main` and every tag referenced none of
it. Deleting the branch took the repo to 44.65 MiB with no rewrite of `main`, no
force-push, and nothing for collaborators to re-clone.

The branch was the pre-slimming snapshot of the line of work that now lives in the
sister repo `xense-taccap-lerobot` (36 MiB, zero wheels and zero `tests/data` in
history — it was rewritten there). Both share root commit `007ffa89`. The sister is
ahead at `db838fb6`, which never existed here, so this branch was dead, not
diverged. Its only surviving copy is
`/home/vertax/dev-taccap-gripper-b2154ab2-20260825.bundle` (308 MB, verified,
tip `b2154ab2`, 1641 commits) — git cannot record that, because the branch is gone
from both sides.

**Diagnose per-ref before touching anything.** A largest-blob listing over `--all`
tells you _what_ is big but not _which ref keeps it alive_, and it will walk you
straight into rewriting `main` for blobs `main` never had. Ask instead what each ref
costs over `main`:

```bash
git rev-list main --objects | awk '{print $1}' | sort -u > /tmp/main_objs
for r in $(git for-each-ref --format='%(refname)' refs/remotes refs/tags); do
  git rev-list "$r" --objects 2>/dev/null | awk '{print $1}' | sort -u > /tmp/r_objs
  sz=$(comm -13 /tmp/main_objs /tmp/r_objs | git cat-file --batch-check='%(objectsize:disk)' \
       | awk '/^[0-9]+$/{s+=$1} END{printf "%.1f", s/1048576}')
  echo "$r +${sz} MiB unique-vs-main"
done
```

Then bundle the branch, delete it on both sides, `git remote prune origin`,
`git reflog expire --expire-unreachable=now --all`, and only then `git gc
--prune=now`. Skipping the reflog expiry leaves every object still reachable and
`gc` reclaims nothing.

Things that do **not** work, so nobody re-tries them:

- **Adding `dist/` to `.gitignore`.** It is already there (`.gitignore:33`) and was
  there while all 7 wheels went in — they were forced past it. Ignoring a path has
  no effect on blobs already committed, and deleting the file does not shrink the
  pack.
- **Deleting the branch on GitHub and expecting the server to shrink.** The ref goes
  away and fresh clones get the small pack, but GitHub's own pack only shrinks when
  they gc; a support ticket is the only way to force it. Local size drops
  immediately, server size does not.
- **`filter-repo` on `main` for the residual 13 MiB.** What is left is
  `libPXREARobotSDK.so` (8 MiB) and `arx5_interface.cpython-311-*.so` (5 MiB),
  historical residue from before those SDKs became the `third_party/XenseVR-PC-Service`
  and `third_party/ARX5_SDK` submodules — neither is in `HEAD` any more. Rewriting
  `main` breaks every outstanding clone and PR to land 13 MiB on a 45 MiB repo,
  which is already the same order as the sister repo's 36 MiB.
