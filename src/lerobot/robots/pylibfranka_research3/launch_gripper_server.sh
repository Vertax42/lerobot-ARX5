#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

cd ~/ros2_ws/src/lerobot_test/src

python -m lerobot.robots.xensegripper.gripper_server_xense \
  --id 7e0b26fa1cbe \
  --port 7001


  # --id 7ec0c7f50ea6 \
  # --id 7e0b26fa1cbe \
