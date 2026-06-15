# Lerobot-integration with BiARX5

## Prerequisites

### Hugging Face CLI login

Required before running any command with `--dataset.push_to_hub=true`:

```bash
huggingface-cli login
```

Paste your HuggingFace access token (with write permission) when prompted.
The token is stored at `~/.cache/huggingface/token` and persists across sessions.

## Config-driven teleop & record (recommended)

As the number of robots and datasets grows, prefer storing parameters in a YAML
under [`recipes/`](../../../recipes/) and passing its path, instead of
copy-pasting a long command from this file. Both `lerobot-teleoperate` and
`lerobot-record` accept `--config_path` (they share the same draccus parser):

```bash
# Teleoperate from a saved recipe
lerobot-teleoperate --config_path=recipes/teleop/bi_elite_cs66_rt/diagonal.yaml

# Record from a saved recipe
lerobot-record --config_path=recipes/record/bi_elite_cs66_rt/test.yaml

# Override anything ad-hoc — CLI flags win over the YAML
lerobot-record --config_path=recipes/record/bi_elite_cs66_rt/test.yaml \
    --dataset.num_episodes=1 --resume=true
```

Recipes are split by CLI then by robot type (`recipes/{teleop,record}/<robot_type>/<name>.yaml`)
and spell out every honored knob with its value; station hardware (cameras, SNs,
mount geometry, poses) stays in the `bi_mount_type` preset. One file per station
(teleop) / dataset (record) means a single source of truth plus provenance —
recipes are committed, so `git log recipes/` shows exactly what produced each
dataset. See [`recipes/README.md`](../../../recipes/README.md). The per-robot
flag lists below remain valid as a reference and for one-off runs.


## TacCap-Gripper lerobot-teleoperate command

`taccap_gripper` / `bi_taccap_gripper` are **self-driven** (sensors only) — there is
no taccap teleoperator. Running them through `lerobot-teleoperate` just streams
`get_observation()` to Rerun (data-stream + viz); `--teleop.type=mock_teleop` is an
**ignored placeholder**. Serials are not baked into the config (generic
`left_*`/`right_*` fields) — pass them here. Prerequisite: `xense.taccap` must import
in the env (`bash ./setup_env.sh --install`).

### Bimanual (`bi_taccap_gripper`) — cameras + gripper only (no Pico4 trackers)

```bash
lerobot-teleoperate \
    --robot.type=bi_taccap_gripper \
    --robot.left_firmware_sn=TCGU01A24Z0003m \
    --robot.right_firmware_sn=TCGU01A24Z0004m \
    --robot.left_enable_tracker=false \
    --robot.right_enable_tracker=false \
    --robot.left_wrist_camera_index_or_path=/dev/v4l/by-id/usb-Xense_Robotics_Co._Ltd._XCA24Z0003m_01.00.00-video-index0 \
    --robot.right_wrist_camera_index_or_path=/dev/v4l/by-id/usb-Xense_Robotics_Co._Ltd._XCA24Z0004m_01.00.00-video-index0 \
    --robot.cameras='{left_tactile_0: {type: xense, serial_number: GSPS01A24Z0003, fps: 30, output_types: [rectify], width: 400, height: 700}, left_tactile_1: {type: xense, serial_number: GSPS01A24Z0004, fps: 30, output_types: [rectify], width: 400, height: 700}, right_tactile_0: {type: xense, serial_number: GSPS01A24Z0005, fps: 30, output_types: [rectify], width: 400, height: 700}, right_tactile_1: {type: xense, serial_number: GSPS01A24Z0006, fps: 30, output_types: [rectify], width: 400, height: 700}}' \
    --teleop.type=mock_teleop \
    --fps=30 \
    --display_data=true
```

To add 6D pose, drop the two `*_enable_tracker=false` lines and pin both Pico4 SNs
(trackers default ON; `tracker_sn=None` is ambiguous with two units — needs the
XenseVR PC service):

```bash
    --robot.left_tracker_sn=<L-pico4-sn> \
    --robot.right_tracker_sn=<R-pico4-sn> \
```

### Single (`taccap_gripper`)

```bash
lerobot-teleoperate \
    --robot.type=taccap_gripper \
    --robot.firmware_sn=TCGU01A24Z0003m \
    --robot.enable_tracker=false \
    --robot.wrist_camera_index_or_path=/dev/v4l/by-id/usb-Xense_Robotics_Co._Ltd._XCA24Z0003m_01.00.00-video-index0 \
    --robot.cameras='{tactile_left: {type: xense, serial_number: GSPS01A24Z0003, fps: 30, output_types: [rectify], width: 400, height: 700}, tactile_right: {type: xense, serial_number: GSPS01A24Z0004, fps: 30, output_types: [rectify], width: 400, height: 700}}' \
    --teleop.type=mock_teleop \
    --fps=30 \
    --display_data=true
```

> Notes: wrist cams use the `by-id` path (USB serial `01.00.00` is non-unique).
> Tactile uses a **single** `output_types: [rectify]` so each sensor returns one
> `(700,400,3)` image (`width:400 height:700`); for force/depth change `output_types`
> and dims. Tactile L/R-finger mapping and `rectify` are unverified on hardware.
> Recording (no teleop) uses `lerobot-record --robot.type=bi_taccap_gripper …`
> (same robot flags, swap `--teleop.*` for `--dataset.*`).

## BiARX5 Robot lerobot-teleoperate command

```bash
lerobot-teleoperate \
    --robot.type=bi_arx5 \
    --robot.enable_tactile_sensors=true \
    --teleop.type=mock_teleop \
    --fps=30 \
    --debug_timing=false \
    --display_data=true
```

## Bimanual Flexiv Rizon4 RT + Bi-Pico4 teleoperate command

```bash
lerobot-teleoperate \
    --robot.type=bi_flexiv_rizon4_rt \
    --robot.bi_mount_type=forward \
    --teleop.type=bi_pico4 \
    --fps=30 \
    --display_data=true
```

```bash
lerobot-teleoperate \
    --robot.type=bi_flexiv_rizon4_rt \
    --robot.bi_mount_type=forward_dewu \
    --teleop.type=bi_pico4 \
    --fps=30 \
    --display_data=true
```


## Bimanual Elite CS66 RT + Bi-Pico4 teleoperate command

```bash
lerobot-teleoperate \
    --robot.type=bi_elite_cs66_rt \
    --robot.bi_mount_type=diagonal \
    --robot.left_robot_ip=192.168.8.53 \
    --robot.right_robot_ip=192.168.8.223 \
    --teleop.type=bi_pico4 \
    --fps=30 \
    --display_data=true


lerobot-teleoperate \
    --robot.type=bi_elite_cs66_rt \
    --robot.left_robot_ip=192.168.8.53 \
    --robot.right_robot_ip=192.168.8.223 \
    --teleop.type=bi_pico4 \
    --fps=30 \
    --dryrun=true
```

## BiARX5 Robot lerobot-record command

```bash
lerobot-record \
    --robot.type=bi_arx5 \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Xense/xense_bi_arx5_test \
    --dataset.num_episodes=100 \
    --dataset.single_task="tie shoelaces" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=true \
    --resume=false \
    --dataset.push_to_hub=false
```

## BiARX5 Robot lerobot-annotate-reward command

```bash
lerobot-annotate-reward \
    --repo-id Xense/xense_bi_arx5_tie_shoelaces \
    --new-repo-id Vertax/test_annotated \
    --push-to-hub
```

## Franka robot lerobot-record command

```bash
lerobot-record \
  --robot.type=pylibfranka_research3 \
  --robot.control_mode=cartesian_impedance \
  --teleop.type=btgamepad \
  --dataset.repo_id=franka_btgamepad/ceshi20260209 \
  --dataset.num_episodes=2 \
  --dataset.single_task="pick" \
  --dataset.fps=30 \
  --resume=false \
  --dataset.push_to_hub=false \
  --display_data=true
```

## Bimanual Flexiv Rizon4 RT + Bi-Pico4 lerobot-record command

### forward mount (side-by-side)

```bash
lerobot-record \
    --robot.type=bi_flexiv_rizon4_rt \
    --robot.bi_mount_type=forward \
    --robot.left_robot_sn=Rizon4s-063458 \
    --robot.right_robot_sn=Rizon4s-063670 \
    --teleop.type=bi_pico4 \
    --dataset.repo_id=Xense/assemble_box_with_phone_stand0430_merged \
    --dataset.num_episodes=5 \
    --dataset.single_task="Assemble the packaging by folding the flat box into shape, placing the metal phone stand inside, and closing the box properly" \
    --dataset.fps=30 \
    --dataset.episode_time_s=600 \
    --dataset.reset_time_s=120 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --resume=true \
    --dataset.push_to_hub=true \
    --display_data=false
```

```bash
lerobot-record \
    --robot.type=bi_flexiv_rizon4_rt \
    --robot.bi_mount_type=forward_dewu \
    --robot.left_robot_sn=Rizon4s-063458 \
    --robot.right_robot_sn=Rizon4s-063670 \
    --teleop.type=bi_pico4 \
    --dataset.repo_id=Xense/shoe_insole_retrieval_and_packing0515 \
    --dataset.num_episodes=10 \
    --dataset.single_task="Open the shoe tongue, take the insole out of the shoe, put the insole back into the shoe, and pack the shoe into the shoebox" \
    --dataset.fps=30 \
    --dataset.episode_time_s=600 \
    --dataset.reset_time_s=120 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --resume=true \
    --dataset.push_to_hub=true \
    --display_data=false
```

```bash
lerobot-record \
    --robot.type=bi_flexiv_rizon4_rt \
    --robot.bi_mount_type=forward_dewu \
    --robot.left_robot_sn=Rizon4s-063458 \
    --robot.right_robot_sn=Rizon4s-063670 \
    --teleop.type=bi_pico4 \
    --dataset.repo_id=Xense/newbalance_shoe_insole_retrieval_and_packing0515 \
    --dataset.num_episodes=10 \
    --dataset.single_task="Open the shoe tongue, take the insole out of the shoe, put the insole back into the shoe, and pack the shoe into the shoebox" \
    --dataset.fps=30 \
    --dataset.episode_time_s=600 \
    --dataset.reset_time_s=120 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --resume=true \
    --dataset.push_to_hub=true \
    --display_data=false
```

### side mount (facing each other)

```bash
lerobot-record \
    --robot.type=bi_flexiv_rizon4_rt \
    --robot.bi_mount_type=side \
    --robot.left_robot_sn=Rizon4-063423 \
    --robot.right_robot_sn=Rizon4-062855 \
    --teleop.type=bi_pico4 \
    --dataset.repo_id=Vertax/bi_flexiv_rt_pick_and_place \
    --dataset.num_episodes=50 \
    --dataset.single_task="pick up the cube and place it in the box" \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=20 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --resume=false \
    --dataset.push_to_hub=false \
    --display_data=false
```

**Current controller mapping during recording:**

| Button    | Action                                              |
| --------- | --------------------------------------------------- |
| Right `A` | Reset both arms to start pose (recording continues) |

Other record-control shortcuts are currently keyboard-driven:

| Input         | Action                                              |
| ------------- | --------------------------------------------------- |
| `Left Arrow`  | Discard current episode and re-record               |
| `Right Arrow` | Finish current episode early                        |
| `Esc`         | Stop the recording session                          |
| `Space`       | Reset both arms to start pose (recording continues) |

## ARX5 Robot lerobot-record command (use trlc_leader teleop)

```bash
lerobot-record \
    --robot.type=arx5_follower \
    --robot.control_mode=joint_control \
    --robot.arm_port=can3 \
    --teleop.type=trlc_leader \
    --teleop.port="/dev/ttyTRLC0" \
    --teleop.joint_signs "[1,1,1,1,1,1]" \
    --teleop.start_joints "[0.0,0.0,0.0,0.0,0.0,0.0]" \
    --dataset.repo_id=Vertax/arx5_trlc_pick_and_place \
    --dataset.num_episodes=50 \
    --dataset.single_task="pick up the cube and place it in the box" \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=15 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --resume=false \
    --dataset.push_to_hub=false \
    --display_data=true
```

## Bimanual Elite CS66 RT + Bi-Pico4 lerobot-record command

### Test record (smoke test: 2 short episodes, local only)

Recommended form — run the saved recipe (see
[`recipes/record/bi_elite_cs66_rt/test.yaml`](../../../recipes/record/bi_elite_cs66_rt/test.yaml)):

```bash
lerobot-record --config_path=recipes/record/bi_elite_cs66_rt/test.yaml
```

Equivalent explicit command (for one-off runs / reference):

```bash
lerobot-record \
    --robot.type=bi_elite_cs66_rt \
    --robot.bi_mount_type=diagonal \
    --robot.left_robot_ip=192.168.8.53 \
    --robot.right_robot_ip=192.168.8.223 \
    --teleop.type=bi_pico4 \
    --dataset.repo_id=Xense/bi_elite_cs66_rt_test \
    --dataset.num_episodes=2 \
    --dataset.single_task="pick up the cube and place it in the box" \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=15 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --resume=false \
    --dataset.push_to_hub=false \
    --display_data=true
```
