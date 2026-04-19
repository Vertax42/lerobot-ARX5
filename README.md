# LeRobot — Minimal

This is the `minimal` branch of [lerobot-xense](https://github.com/Vertax42/lerobot-xense), a
lightweight kernel stripped down from the full fork. It keeps the core dataset / record /
replay / teleoperate pipeline, the Xense tactile camera integration, and the `spacemouse` /
`pico4` teleops; it drops the rest of the vendor hardware integrations (Flexiv, ARX5,
Franka, XGripper, TRLC, Vive, phone), all VLA policies (Gr00t / Pi0 / Pi05 / SmolVLA), the
RL / async-inference / gRPC transport stacks, and the sim envs (LIBERO / MetaWorld / Aloha /
PushT).

## What you get

- `datasets/` — LeRobot dataset format, record & replay.
- `cameras/` — OpenCV, RealSense, and **Xense tactile**.
- `robots/` — base `Robot` class + `MockRobot`.
- `teleoperators/` — base `Teleoperator` + `keyboard` + `gamepad` + `spacemouse` + `pico4`
  + `MockTeleop`. **Pico4** additionally requires `xensevr_pc_service_sdk` — see
  [Manual install: `xensevr_pc_service_sdk`](#manual-install-xensevr_pc_service_sdk) below.
- CLI entry points: `lerobot-record`, `lerobot-replay`, `lerobot-teleoperate`, `lerobot-test`,
  `lerobot-calibrate`, `lerobot-find-cameras`, `lerobot-find-port`, `lerobot-info`,
  `lerobot-dataset-viz`, `lerobot-edit-dataset`, `lerobot-annotate-reward`,
  `lerobot-imgtransform-viz`.

## What was removed

- `src/lerobot/policies/`, `src/lerobot/processor/`, `src/lerobot/rl/`,
  `src/lerobot/async_inference/`, `src/lerobot/transport/`.
- All vendor `robots/` (arx5_*, bi_arx5, flexiv_rizon4*, bi_flexiv_rizon4_rt,
  pylibfranka_research3, xense_flare, xense_multisensor, bi_xense_flare_grippers).
- Vendor `teleoperators/` (bi_pico4, bi_trlc, btgamepad, phone, trlc_leader,
  vive_tracker, xense_flare). `pico4` and `spacemouse` are kept.
- `envs/` (sim env factory + aloha / pusht / libero / metaworld configs).
- `model/` (placo kinematics).
- `scripts/lerobot_train.py`, `scripts/lerobot_eval.py`, `scripts/lerobot_find_joint_limits.py`.
- `third_party/` — except `xensesdk` (required by `cameras/xense`).
- `setup_env.sh`, `setup_can.sh`, `gripper_fix.md`, `docker/`, `docs/`, `docs-requirements.txt`.

## Install

We use **mamba** (via [Miniforge](https://github.com/conda-forge/miniforge)) to manage the
Python environment. Plain `conda` also works — swap `mamba` for `conda` in the commands below.

### 1. Clone the repo

```bash
git clone --recursive -b minimal https://github.com/Vertax42/lerobot-xense.git
cd lerobot-xense
```

The `--recursive` flag pulls `third_party/xensesdk` (the only submodule left on this branch).

### 2. Create the environment

```bash
mamba env create -f conda_environment.yaml
mamba activate lerobot-minimal
```

This installs Python 3.10 plus the system-level toolchain (`cmake`, `ninja`, `cxx-compiler`,
`pybind11`, `libstdcxx-ng`, `libhidapi`, `ffmpeg`) needed to build the `pico4` pybind
extension and run `spacemouse` / Xense / media pipelines.

### 3. Install lerobot in editable mode

```bash
pip install -e .
```

### 4. Optional extras

```bash
pip install -e '.[xense]'          # xensesdk (Xense tactile cameras — see below)
pip install -e '.[intelrealsense]' # RealSense cameras
pip install -e '.[gamepad]'        # gamepad teleop (pygame + hidapi)
pip install -e '.[dev,test]'       # dev & test tooling
```

> `pyspacemouse` is already in the core dependencies, so `spacemouse` teleop works out of
> the box (requires `libhidapi`, which is in `conda_environment.yaml`).

---

## Manual install: `xensevr_pc_service_sdk`

The `pico4` teleop uses `xensevr_pc_service_sdk`, a Python binding around the C++ SDK
distributed with `XenseVR-PC-Service`. The `minimal` branch does **not** ship the
`third_party/XenseVR-PC-Service` submodule. If you need `pico4`, install the SDK manually
on your machine.

### Prerequisites

Ensure the mamba env from step 2 is active — it already provides `cmake`, `cxx-compiler`,
`pybind11`, and `libstdcxx-ng`:

```bash
mamba activate lerobot-minimal
```

### Build steps

```bash
# 1. Clone the XenseVR-PC-Service sources somewhere on your machine
#    (outside this repo is fine — pick any directory).
git clone git@github.com:Vertax42/XenseVR-PC-Service.git ~/XenseVR-PC-Service

# 2. Build the C SDK
bash ~/XenseVR-PC-Service/RoboticsService/PXREARobotSDK/build.sh

# 3. Copy the built SDK into the pybind project that ships with this repo
CSDK=~/XenseVR-PC-Service/RoboticsService/PXREARobotSDK
PYBIND=src/lerobot/teleoperators/pico4/xensevr-pc-service-pybind
mkdir -p "$PYBIND/include" "$PYBIND/lib"
cp "$CSDK/PXREARobotSDK.h"        "$PYBIND/include/"
cp -r "$CSDK/nlohmann"            "$PYBIND/include/"
cp "$CSDK/build/libPXREARobotSDK.so" "$PYBIND/lib/"

# 4. Build & install the Python bindings into the active env
pip install "$PYBIND" --no-build-isolation

# 5. Verify
python -c "import xensevr_pc_service_sdk; print(xensevr_pc_service_sdk.__file__)"
```

After this, `from lerobot.teleoperators.pico4 import Pico4` imports cleanly and
`lerobot-teleoperate --teleop.type=pico4 ...` works.

---

## Manual install: `xensesdk` (Xense tactile cameras)

The `third_party/xensesdk` submodule ships the native SDK. Build + install it into the
active mamba env before `pip install -e '.[xense]'`:

```bash
git submodule update --init third_party/xensesdk
# follow third_party/xensesdk/README — typically cmake build + pip install
pip install -e '.[xense]'
```

---

## Quick sanity check

```bash
python -c "import lerobot; print('lerobot OK')"
python -c "from lerobot.robots import Robot, RobotConfig; \
           from lerobot.teleoperators import Teleoperator; \
           from lerobot.cameras import Camera; print('base classes OK')"

lerobot-find-cameras --help
lerobot-find-port --help
lerobot-record --help
lerobot-teleoperate --help
```

A mock teleoperation loop for smoke-testing:

```bash
lerobot-teleoperate --robot.type=mock --teleop.type=keyboard --fps=30
```

## Where to go for the full fork

- Upstream: https://github.com/huggingface/lerobot
- Full vendor fork (xense / flexiv / arx5 / VLA policies): `main` branch of this repo.

## License

Apache-2.0 — see `LICENSE`.
