# LeRobot — Minimal

This is the `minimal` branch of [lerobot-xense](https://github.com/Vertax42/lerobot-xense), a
lightweight kernel stripped down from the full fork. It keeps the core dataset / record /
replay / teleoperate pipeline and the Xense tactile camera integration, and drops every other
vendor hardware integration (Flexiv, ARX5, Franka, XGripper, Pico4, TRLC, Vive, SpaceMouse,
phone), all VLA policies (Gr00t / Pi0 / Pi05 / SmolVLA), the RL / async-inference / gRPC
transport stacks, and the sim envs (LIBERO / MetaWorld).

## What you get

- `datasets/` — LeRobot dataset format, record & replay.
- `cameras/` — OpenCV, RealSense, and **Xense tactile**.
- `robots/` — base `Robot` class + `MockRobot`.
- `teleoperators/` — base `Teleoperator` + `keyboard` + `gamepad` + `MockTeleop`.
- CLI entry points: `lerobot-record`, `lerobot-replay`, `lerobot-teleoperate`, `lerobot-test`,
  `lerobot-calibrate`, `lerobot-find-cameras`, `lerobot-find-port`, `lerobot-info`,
  `lerobot-dataset-viz`, `lerobot-edit-dataset`, `lerobot-annotate-reward`,
  `lerobot-imgtransform-viz`.

## What was removed

- `src/lerobot/policies/`, `src/lerobot/processor/`, `src/lerobot/rl/`,
  `src/lerobot/async_inference/`, `src/lerobot/transport/`.
- All vendor `robots/` (arx5_*, bi_arx5, flexiv_rizon4*, bi_flexiv_rizon4_rt,
  pylibfranka_research3, xense_flare, xense_multisensor, bi_xense_flare_grippers).
- All vendor `teleoperators/` (bi_pico4, bi_trlc, btgamepad, phone, pico4, spacemouse,
  trlc_leader, vive_tracker, xense_flare).
- `envs/` (sim env factory + aloha / pusht / libero / metaworld configs).
- `model/` (placo kinematics).
- `scripts/lerobot_train.py`, `scripts/lerobot_eval.py`, `scripts/lerobot_find_joint_limits.py`.
- `third_party/` — except `xensesdk` (required by `cameras/xense`).
- `setup_env.sh`, `setup_can.sh`, `gripper_fix.md`, `conda_environment.yaml`, `docker/`,
  `docs/`, `docs-requirements.txt`.

## Install

```bash
# Clone + pick up the xensesdk submodule (only submodule left)
git clone --recursive -b minimal https://github.com/Vertax42/lerobot-xense.git
cd lerobot-xense

# Create a venv (Python 3.10+) and install
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Optional extras (examples)
pip install -e '.[xense]'          # xensesdk for tactile cameras (build xensesdk first)
pip install -e '.[intelrealsense]' # RealSense
pip install -e '.[gamepad]'        # gamepad teleop (pygame + hidapi)
pip install -e '.[dev,test]'       # development + tests
```

Building `xensesdk` from the submodule follows its own README in `third_party/xensesdk`.

## Quick sanity check

```bash
python -c "import lerobot; print('lerobot OK')"
python -c "from lerobot.robots import Robot, RobotConfig; from lerobot.teleoperators import Teleoperator; from lerobot.cameras import Camera; print('base classes OK')"

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
