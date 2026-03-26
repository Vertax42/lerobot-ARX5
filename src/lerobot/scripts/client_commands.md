# Lerobot-integration with BiARX5

## Prerequisites

### Hugging Face CLI login

Required before running any command with `--dataset.push_to_hub=true`:

```bash
huggingface-cli login
```

Paste your HuggingFace access token (with write permission) when prompted.
The token is stored at `~/.cache/huggingface/token` and persists across sessions.

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

```bash
lerobot-teleoperate \
    --robot.type=arx5_follower \
    --robot.control_mode=cartesian_control \
    --robot.enable_tactile_sensors=false \
    --teleop.type=mock_teleop \
    --fps=30 \
    --debug_timing=false \
    --display_data=true
```

## ARX5 Robot lerobot-teleoperate command(use trlc_leader teleop)

```bash
lerobot-teleoperate \
    --robot.type=arx5_follower \
    --robot.control_mode=joint_control \
    --robot.enable_tactile_sensors=false \
    --teleop.type=trlc_leader \
    --teleop.port="/dev/ttyTRLC0" \
    --teleop.joint_signs "[1,1,1,1,1,1]" \
    --teleop.start_joints "[0.0,0.0,0.0,0.0,0.0,0.0]" \
    --fps=30 \
    --debug_timing=false \
    --display_data=true
```

```bash
lerobot-teleoperate \
    --robot.type=mock_robot \
    --robot.control_mode=joint_control \
    --robot.n_motors=6 \
    --robot.use_gripper=true \
    --teleop.type=trlc_leader \
    --teleop.port="/dev/ttyTRLC0" \
    --teleop.joint_signs "[1,1,1,1,1,1]" \
    --teleop.start_joints "[0.0,0.0,0.0,0.0,0.0,0.0]" \
    --fps=30 \
    --debug_timing=false \
    --display_data=true
``````

## Flexiv Rizon4 Robot with Flare Gripper teleoperate by Pico4 command

```bash
lerobot-teleoperate \
    --robot.type=flexiv_rizon4 \
    --robot.gripper_mac_addr="e2b26adbb104" \
    --robot.gripper_type="flare_gripper" \
    --robot.control_mode=cartesian_motion_force_control \
    --teleop.type=pico4 \
    --fps=30 \
    --display_data=true \
    --dryrun=true
```

## Bimanual Flexiv Rizon4 RT + Bi-Pico4 teleoperate command

```bash
lerobot-teleoperate \
    --robot.type=bi_flexiv_rizon4_rt \
    --robot.bi_mount_type=forward \
    --robot.left_robot_sn=Rizon4s-063458 \
    --robot.right_robot_sn=Rizon4s-063670 \
    --teleop.type=bi_pico4 \
    --fps=60 \
    --display_data=true
```

## Flexiv Rizon4 Robot with Xense Flare teleoperate by Xense Flare command

```bash
lerobot-teleoperate \
    --robot.type=flexiv_rizon4 \
    --robot.gripper_mac_addr="e2b26adbb104" \
    --robot.gripper_type="flare_gripper" \
    --robot.control_mode=cartesian_motion_force_control \
    --teleop.type=xense_flare \
    --teleop.mac_addr="6ebbc5f53240" \
    --fps=10 \
    --display_data=true \
    --dryrun=true
```

## Flexiv Rizon4 Robot with Flare Gripper lerobot-record by Pico4 command

```python
lerobot-record \
    --robot.type=flexiv_rizon4 \
    --robot.gripper_mac_addr="e2b26adbb104" \
    --robot.gripper_type="flare_gripper" \
    --robot.control_mode=cartesian_motion_force_control \
    --teleop.type=pico4 \
    --dataset.repo_id=flexiv_pico4/ceshi20260202 \
    --dataset.num_episodes=2 \
    --dataset.single_task="pick up cubes in rgb order from the table and place them in the blue box" \
    --dataset.fps=10 \
    --resume=false \
    --dataset.push_to_hub=false \
    --display_data=true
```

## Flexiv Rizon4 Robot with Flare Gripper lerobot-record by Beitong Gamepad command

```python
lerobot-record \
    --robot.type=flexiv_rizon4 \
    --robot.gripper_mac_addr="e2b26adbb104" \
    --robot.gripper_type="flare_gripper" \
    --robot.control_mode=cartesian_motion_force_control \
    --teleop.type=btgamepad \
    --dataset.repo_id=flexiv_pico4/ceshi20260204 \
    --dataset.num_episodes=2 \
    --dataset.single_task="pick up cubes in rgb order from the table and place them in the blue box" \
    --dataset.fps=10 \
    --resume=false \
    --dataset.push_to_hub=false \
    --display_data=true
```

## xense_flare Robot teleoperate by Mock Teleop command

### 1e892b82baa5 -another mac addr

```python
lerobot-teleoperate \
    --robot.type=xense_flare \
    --robot.mac_addr="6ebbc5f53240" \
    --teleop.type=mock_teleop \
    --fps=20 \
    --display_data=true \
    --debug_timing=true \
    --dryrun=false
```

## Xense-Flare Robot lerobot-record command

```python
lerobot-record \
    --robot.type=xense_flare \
    --robot.mac_addr=6ebbc5f53240 \
    --dataset.repo_id=Vertax/xense_flare_pick_and_place_cube_20260113 \
    --dataset.num_episodes=50 \
    --dataset.single_task="pick up cubes in rgb order from the table and place them in the blue box" \
    --dataset.fps=20 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=xense_flare \
    --robot.mac_addr=6ebbc5f53240 \
    --dataset.repo_id=Vertax/xense_flare_open_lock_20260108 \
    --dataset.num_episodes=20 \
    --dataset.single_task="open the lock with the key" \
    --dataset.fps=20 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=xense_flare \
    --robot.mac_addr=6ebbc5f53240 \
    --dataset.repo_id=Vertax/xense_flare_wipe_vase_20260113 \
    --dataset.num_episodes=20 \
    --dataset.single_task="wipe the vase" \
    --dataset.fps=20 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=xense_flare \
    --robot.mac_addr=6ebbc5f53240 \
    --dataset.repo_id=Vertax/xense_flare_pick_and_place_cubes_20260104 \
    --dataset.num_episodes=50 \
    --dataset.single_task="pick up cubes in rgb order from the table and place them in the blue box" \
    --dataset.fps=20 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=xense_flare \
    --robot.mac_addr=6ebbc5f53240 \
    --dataset.repo_id=Vertax/xense_flare_cucumber_peeling \
    --dataset.num_episodes=3 \
    --dataset.single_task="peel a cucumber with a vegetable peeler" \
    --dataset.fps=20 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=xense_flare \
    --robot.mac_addr=6ebbc5f53240 \
    --dataset.repo_id=Vertax/xense_flare_replay_test \
    --dataset.num_episodes=2 \
    --dataset.single_task="xense flare traj replay test" \
    --dataset.fps=20 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

## BiARX5 Robot lerobot-record command

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/xense_bi_arx5_tie_shoelaces \
    --dataset.num_episodes=100 \
    --dataset.single_task="tie shoelaces" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=false \
    --resume=true \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --robot.enable_tactile_sensors=true \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Xense/xense_bi_arx5_tie_shoelaces \
    --dataset.num_episodes=5 \
    --dataset.single_task="tie shoelaces" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --robot.enable_tactile_sensors=true \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/xense_bi_arx5_tie_shoelaces_tactile \
    --dataset.num_episodes=100 \
    --dataset.single_task="tie shoelaces" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/xense_bi_arx5_tie_shoelaces_high_quality \
    --dataset.num_episodes=100 \
    --dataset.single_task="tie shoelaces" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=false \
    --resume=true \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/xense_bi_arx5_tie_shoelaces_1027 \
    --dataset.num_episodes=100 \
    --dataset.single_task="tie shoelaces" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=false \
    --resume=true \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/xense_bi_arx5_tie_white_shoelaces_1028 \
    --dataset.num_episodes=100 \
    --dataset.single_task="tie shoelaces" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=false \
    --resume=true \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/xense_bi_arx5_tie_white_shoelaces_1030_no_adjust \
    --dataset.num_episodes=100 \
    --dataset.single_task="tie shoelaces" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=false \
    --resume=true \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --robot.enable_tactile_sensors=true \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/lerobot040_test_bi_arx5 \
    --dataset.num_episodes=5 \
    --dataset.single_task="test" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

---

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --robot.enable_tactile_sensors=true \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/lerobot040_pick_and_place_chip_bi_arx5_1204 \
    --dataset.num_episodes=10 \
    --dataset.single_task="pick up a potato chip and place it into the chips container" \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --robot.enable_tactile_sensors=true \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/bi_arx5_video_encode_test \
    --dataset.single_task="test video encoding" \
    --dataset.fps=30 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=10 \
    --dataset.num_episodes=5 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto
```

## BiARX5 Robot lerobot-replay command

```python
lerobot-replay \
    --robot.type=bi_arx5 \
    --dataset.repo_id=Vertax/lerobot040_test_bi_arx5 \
    --dataset.episode=0
```

## BiARX5 Robot lerobot-annotate-reward command

```python
lerobot-annotate-reward \
    --repo-id Xense/xense_bi_arx5_tie_shoelaces \
    --new-repo-id Vertax/test_annotated \
    --push-to-hub
```

**Note on preview_time:**

Adjust `--robot.preview_time` to reduce jittering:

- 0.03-0.05s: Smoother motion, more delay (recommended for stable movements)
- 0.01-0.02s: More responsive, but may cause jittering
- 0.0: No preview (only for teleoperation/recording)

## Franka robot lerobot-record command

```python
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

```python
lerobot-record \
    --robot.type=pylibfranka_research3 \
    --robot.control_mode=cartesian_impedance \
    --teleop.type=pico4 \
    --dataset.repo_id=flexiv_pico4/ceshi20260225 \
    --dataset.num_episodes=2 \
    --dataset.single_task="pick" \
    --dataset.fps=10 \
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
    --dataset.repo_id=Vertax/bi_flexiv_rt_test_demo_20260326 \
    --dataset.num_episodes=50 \
    --dataset.single_task="test demo" \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=20 \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=auto \
    --resume=false \
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

**Controller button mapping during recording:**

| Button | Action |
|---|---|
| Right `A` | Reset both arms to start pose (recording continues) |
| Left `X` | Discard current episode and re-record |
| Left `Y` | Finish current episode early |
| Right `B` | Stop the recording session |

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
