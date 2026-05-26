# `taccap_gripper` — Handheld data-collection device

Single-arm handheld data-collection pipeline. The device is **self-driven**:
there is no separate teleoperator — the robot itself produces both the
observation (state being recorded) and the demonstration action.
`lerobot-record` dispatches this case through the same path as
`xense_flare`, so no `--teleop.*` flags are needed on the CLI.

Components:

- **Gripper:** TacCap-Gripper handheld unit (motor jaw, two embedded
  visuotactile sensors, wrist UVC camera, encoder, IMU). Driven by the
  `xense.taccap` SDK (`taccap-gripper` PyPI package).
- **Pose:** Pico4 Ultra **independent motion tracker** mounted on top
  of the gripper. Reached via `xensevr_pc_service_sdk` and read by
  `lerobot.teleoperators.pico4.tracker.Pico4TrackerReader`.
- **Cameras:** plain LeRobot `cameras/` framework. The wrist UVC camera
  is **auto-wired** via `GripperEndpoints.wrist_video` reported by the
  SDK at `connect()` time — no need to hard-code `/dev/videoN`. Tactile
  sensors come through `cameras/xense/` keyed by OG serial.

The device is passive: `send_action()` is a no-op; the motor is never
enabled. The operator drives the jaw mechanically and walks the device
through demonstrations.

## Coordinate frame

Recorded pose is in **raw Pico4 native** frame (X right, Y up, Z toward
the headset operator at Unity-launch time). This is **not** any
robot's base frame. Downstream policies must reframe explicitly.

The Pico4 origin is fixed at Unity-app launch and stays put as long as
the app keeps running. **Do not restart the Unity client between
episodes** or all subsequent recordings will be in a different origin.

## Hardware bring-up sequence

1. Plug the TacCap-Gripper into the host (USB).
2. Power on the Pico4 Ultra headset; pair the motion tracker.
3. Launch the Unity VR Client app on the headset (this freezes the
   coordinate origin).
4. Start the XenseVR PC Service on the host.
5. Run any of the scripts below.

## Calibration workflow (do once per device)

### 1. Find the jaw encoder endpoints

```bash
python -m lerobot.robots.taccap_gripper.calibrate_gripper_range
```

Squeeze the jaw fully closed, then fully open. When `min`/`max`
stabilise, press Ctrl+C. Copy the two printed values into your
`TaccapGripperConfig`:

```python
gripper_closed_rad=<min>,
gripper_open_rad=<max>,
```

If the encoder convention is reversed for your unit, swap the two
values manually — the normalisation in the robot is sign-aware.

### 2. Sanity-check the Pico4 tracker

```bash
python -m lerobot.robots.taccap_gripper.calibrate_tracker
# or, pin to a specific tracker SN:
python -m lerobot.robots.taccap_gripper.calibrate_tracker LHR-XXXXXXXX
```

Watch the `raw xyz` move smoothly when you wave the gripper. The
`ee xyz` is `raw` after the rigid `tracker_to_ee_*` mount transform —
identity by default. Measure your physical mount offset and put it in
the config (`tracker_to_ee_pos`, `tracker_to_ee_quat`).

## Standalone smoke test

Verifies the robot stack independently of `lerobot-record`:

```bash
# Default: gripper + auto-wired wrist camera (V4L2 path from SDK).
python -m lerobot.robots.taccap_gripper.taccap_gripper_example \
    --closed-rad <closed> --open-rad <open>

# Gripper only, skip the wrist camera:
python -m lerobot.robots.taccap_gripper.taccap_gripper_example \
    --no-wrist-cam --closed-rad <closed> --open-rad <open>

# + Pico4 tracker:
python -m lerobot.robots.taccap_gripper.taccap_gripper_example \
    --tracker --closed-rad <closed> --open-rad <open>

# + tactile sensors (left + right OG):
python -m lerobot.robots.taccap_gripper.taccap_gripper_example \
    --tracker --tactile --closed-rad <closed> --open-rad <open>
```

The script prints 10 observation frames (scalar fields + image
shapes), then disconnects. Use it as the first port of call when
something looks wrong end-to-end.

## End-to-end recording

`taccap_gripper` is in the self-driven-robot dispatch list inside
`lerobot/scripts/lerobot_record.py`, so the record loop reads the demo
action straight off `robot.get_action()` (same path as `xense_flare`).
**No `--teleop.*` flags.**

```bash
lerobot-record \
    --robot.type=taccap_gripper \
    --robot.id=right \
    --robot.gripper_closed_rad=<closed> \
    --robot.gripper_open_rad=<open> \
    --robot.cameras='{tactile_left: {type: xense, serial_number: OG000XXX, fps: 30, width: 400, height: 700}, tactile_right: {type: xense, serial_number: OG000YYY, fps: 30, width: 400, height: 700}}' \
    --dataset.repo_id=<your_org>/<your_dataset> \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=10 \
    --dataset.single_task='Pick up the object'
```

The wrist camera is **not** listed in `--robot.cameras` — it is
auto-wired by `enable_wrist_camera=True` (default). Override the
defaults with `--robot.wrist_camera_width=…` / `_height` / `_fps` if
needed, or set `--robot.enable_wrist_camera=false` to skip.

The `xrt.init()` call is process-singleton-guarded inside
`Pico4TrackerReader`, so any future re-use within the same process is
safe (e.g. for visualisation alongside recording).

## What gets recorded per frame

| Key | Source | Shape / type |
|---|---|---|
| `tcp.x`, `tcp.y`, `tcp.z` | Pico4 tracker → EE | float (m) |
| `tcp.r1`..`tcp.r6` | 6-D rotation of EE | float |
| `gripper.pos` | TacCap encoder, normalised | float ∈ [0, 1] |
| `imu.accel.{x,y,z}` (opt) | TacCap IMU | float (m/s²) |
| `imu.gyro.{x,y,z}` (opt) | TacCap IMU | float (rad/s) |
| `imu.mag.{x,y,z}` (opt) | TacCap IMU | float (µT) |
| `<camera_name>` per camera | `cameras/` framework | uint8 (H, W, 3) |

The 6-D rotation convention matches `xense_flare`/`vive_tracker`:
`r1..r3` is the first column of the rotation matrix, `r4..r6` is the
second column.

## Files in this package

- `taccap_gripper.py` — the `Robot` subclass. `get_observation()` and
  `get_action()` both surface pose + gripper + optional IMU + cameras.
- `config_taccap_gripper.py` — `RobotConfig` dataclass.
- `taccap_gripper_example.py` — standalone smoke test (above).
- `calibrate_gripper_range.py` — find jaw encoder endpoints.
- `calibrate_tracker.py` — sanity-check the Pico4 tracker.

The Pico4 tracker reader is shared with future devices and lives at
`src/lerobot/teleoperators/pico4/tracker.py`.

The integration point in the record script is
`src/lerobot/scripts/lerobot_record.py`:

- `RecordConfig.__post_init__` allows `teleop=None` when
  `robot.type == "taccap_gripper"`.
- The dispatch in `record()` routes `taccap_gripper` to
  `xense_flare_record_loop()` (the function is generic — it only calls
  `robot.get_observation()` and `robot.get_action()`).
