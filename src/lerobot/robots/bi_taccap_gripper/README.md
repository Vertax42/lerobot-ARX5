# bi_taccap_gripper

Bimanual TacCap-Gripper handheld data-collection rig — two `taccap_gripper` units
(left + right) driven as one robot. Passive/self-driven: `send_action()` is a no-op
(jaw motors stay disabled, encoders read-only); pose comes from a per-side Pico4
Ultra tracker, tactile + wrist cameras go through the standard `cameras` framework.

Implemented with the **reimplement-with-prefixes** pattern (cf. `bi_elite_cs66_rt`):
one `Robot` class, per-side handles in dicts keyed `"left"`/`"right"`, and every
observation/action key is `left_`/`right_` prefixed. Per-side reading logic matches
the single [`taccap_gripper`](../taccap_gripper/README.md).

## Observation / action schema

Per side `{s}` ∈ {left, right}:

| Key | When | Meaning |
|---|---|---|
| `{s}_tcp.x/y/z`, `{s}_tcp.r1..r6` | `{s}_enable_tracker` | Pico4 → EE 6D pose |
| `{s}_gripper.pos` | `{s}_enable_gripper` | normalised jaw, 0=closed / 1=open |
| `{s}_imu.{accel,gyro,mag}.{x,y,z}` | `{s}_enable_imu` | IMU |
| `{s}_wrist` | `{s}_enable_wrist_camera` | wrist UVC frame |
| `{s}_tactile_0` / `{s}_tactile_1` | from `cameras` | tactile frames |

`action_features` = the pose + `{s}_gripper.pos` subset (no cameras).

## Config (generic fields, serials via CLI)

Flat `left_*` / `right_*` fields mirroring `TaccapGripperConfig` — no presets, no
serials committed. Pass identities on the CLI:
`--robot.{left,right}_firmware_sn`, `--robot.{left,right}_tracker_sn`,
`--robot.{left,right}_wrist_camera_index_or_path`, and tactile sensors via
`--robot.cameras='{left_tactile_0: {type: xense, serial_number: GSPS…}, …}'`
(pre-prefixed keys). Trackers default **on** — pin both Pico4 SNs, or disable with
`--robot.{left,right}_enable_tracker=false`.

## Usage

- **Live Rerun visualization** (`lerobot-teleoperate` + `mock_teleop` placeholder)
  and **recording** (`lerobot-record`, no teleop → `self_driven_record_loop`,
  shifted-frame) — see ready-to-run commands with this machine's serials in the
  repo-root [`cli_commands.md`](../../../../cli_commands.md).

Prerequisite: `xense.taccap` must import in the `lerobot-xense` env
(`bash ./setup_env.sh --install`).
