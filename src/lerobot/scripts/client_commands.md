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
