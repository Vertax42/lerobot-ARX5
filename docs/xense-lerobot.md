# Xense SO101

```bash
# Calibrate

lerobot-calibrate \
 --robot.type=so101_follower \
 --robot.port=/dev/ttyACM1 \
 --robot.id=vertax_follower_arm

lerobot-calibrate \
 --teleop.type=so101_leader \
 --teleop.port=/dev/ttyACM0 \
 --teleop.id=vertax_leader_arm

# Teleoperate only with the arm
lerobot-teleoperate \
 --robot.type=so101_follower \
 --robot.port=/dev/ttyACM1 \
 --robot.id=vertax_follower_arm \
 --teleop.type=so101_leader \
 --teleop.port=/dev/ttyACM0 \
 --teleop.id=vertax_leader_arm \
 --display_data=true

# Teleoperate with cameras
lerobot-teleoperate \
 --robot.type=so101_follower \
 --robot.port=/dev/ttyACM1 \
 --robot.id=vertax_follower_arm \
 --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}" \
 --teleop.type=so101_leader \
 --teleop.port=/dev/ttyACM0 \
 --teleop.id=vertax_leader_arm \
 --display_data=true
 
# # realsense depth cameras
# lerobot-teleoperate \
#  --robot.type=so101_follower \
#  --robot.port=/dev/ttyACM1 \
#  --robot.id=vertax_follower_arm \
#  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, side: {type: intelrealsense, serial_number_or_name: 834412071827, width: 640, height: 480, fps: 30}}" \
#  --teleop.type=so101_leader \
#  --teleop.port=/dev/ttyACM0 \
#  --teleop.id=vertax_leader_arm \
#  --display_data=true

# record datasets

lerobot-record \
 --robot.type=so101_follower \
 --robot.port=/dev/ttyACM1 \
 --robot.id=vertax_follower_arm \
 --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}" \
 --teleop.type=so101_leader \
 --teleop.port=/dev/ttyACM0 \
 --teleop.id=vertax_leader_arm \
 --display_data=true \
 --dataset.repo_id=Vertax/xense-so101-place-by-colors \
 --dataset.num_episodes=50 \
 --dataset.episode_time_s=120 \
 --dataset.single_task="Grab multipule cubes and plcae by colors"


# Train
lerobot-train \
 --dataset.repo_id=Vertax/xense-so101-place-by-colors  \
 --policy.type=act \
 --output_dir=outputs/train/act_xense-so101-place-by-colors  \
 --job_name=act_xense-so101-place-by-colors \
 --policy.device=cuda \
 --wandb.enable=true \
 --policy.repo_id=Vertax/act_xense-so101-place-by-colors_policy

# finetune smolvla
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=Vertax/xense-so101-place-by-colors \
  --batch_size=256 \
  --steps=30000 \
  --output_dir=outputs/train/smolvla_xense-so101-place-by-colors \
  --job_name=smolvla_xense-so101-place-by-colors \
  --policy.device=cuda \
  --policy.repo_id=Vertax/smolvla_xense-so101-place-by-colors_policy \
  --wandb.enable=true

# Eval with act model
lerobot-record \
 --robot.type=so101_follower \
 --robot.port=/dev/ttyACM1 \
 --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}" \
 --robot.id=vertax_follower_arm \
 --display_data=false \
 --dataset.repo_id=Vertax/eval_xense-so101-place-by-colors \
 --dataset.single_task="Grab multipule cubes and plcae by colors" \
 --teleop.type=so101_leader \
 --teleop.port=/dev/ttyACM0 \
 --teleop.id=vertax_leader_arm \
 --policy.path=outputs/train/act_xense-so101-place-by-colors/checkpoints/last/pretrained_model \
 --dataset.episode_time_s=180 \
 --dataset.push_to_hub=false

# Eval with fine-tuned smolvla model
lerobot-record \
 --robot.type=so101_follower \
 --robot.port=/dev/ttyACM1 \
 --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}" \
 --robot.id=vertax_follower_arm \
 --display_data=false \
 --dataset.repo_id=Vertax/eval_xense-so101-place-by-colors \
 --dataset.single_task="Grab multipule cubes and plcae by colors" \
 --dataset.episode_time_s=180 \
 --dataset.num_episodes=10 \
 --policy.path=outputs/train/smolvla_xense-so101-place-by-colors/checkpoints/last/pretrained_model \
 --dataset.push_to_hub=false
 
 
# Eval with fine-tuned model with async inference
# policy server
python src/lerobot/scripts/server/policy_server.py \
    --host=127.0.0.1 \
    --port=8080

# policy client
python src/lerobot/scripts/server/robot_client.py \
    --server_address=127.0.0.1:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=vertax_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}" \
    --task="Grab multipule cubes and plcae by colors" \
    --policy_type=smolvla \
    --pretrained_name_or_path=Vertax/smolvla_xense-so101-place-by-colors_policy \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
 ```
