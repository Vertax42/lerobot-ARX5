# BiARX5 Robot lerobot-teleoperate command

lerobot-teleoperate \
    --robot.type=bi_arx5 \
    --teleop.type=mock_teleop \
    --robot.cameras='{"head": {"type": "opencv", "index_or_path": "/dev/video16", "fps": 30, "width": 640, "height": 480}, "left_wrist": {"type": "opencv", "index_or_path": "/dev/video10", "fps": 30, "width": 640, "height": 480}, "right_wrist": {"type": "opencv", "index_or_path": "/dev/video4", "fps": 30, "width": 640, "height": 480}}' \
    --fps=30 \
    --display_data=true