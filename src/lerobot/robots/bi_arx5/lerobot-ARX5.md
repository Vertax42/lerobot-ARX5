# ARX_X5 CAN Bus Diagram

```
┌─────────────────┐    CAN Bus    ┌─────────────────┐
│   Controller    │◄─────────────►│  Motor ID: 1    │ (Joint 1)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 2    │ (Joint 2)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 4    │ (Joint 3)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 5    │ (Joint 4)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 6    │ (Joint 5)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 7    │ (Joint 6)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 8    │ (Gripper)
└─────────────────┘               └─────────────────┘
```

# default configurations X5 in ` include/app/config.h `
```cpp
// joint_names: [0: joint1, 1: joint2, 2: joint3, 3: joint4, 4: joint5, 5: joint6]
// motors: [0: EC_A4310, 1: EC_A4310, 2: EC_A4310, 3: DM_J4310, 4: DM_J4310, 5: DM_J4310]
RobotConfigFactory()
    {
        configurations["X5"] = std::make_shared<RobotConfig>(
            "X5",                                                          // robot_model
            (VecDoF(6) << -3.14, -0.05, -0.1, -1.6, -1.57, -2).finished(), // joint_pos_min
            (VecDoF(6) << 2.618, 3.50, 3.20, 1.55, 1.57, 2).finished(),    // joint_pos_max
            (VecDoF(6) << 5.0, 5.0, 5.5, 5.5, 5.0, 5.0).finished(),        // joint_vel_max
            (VecDoF(6) << 30.0, 40.0, 30.0, 15.0, 10.0, 10.0).finished(),  // joint_torque_max
            (Pose6d() << 0.6, 0.6, 0.6, 1.8, 1.8, 1.8).finished(),         // ee_vel_max
            0.3,                                                           // gripper_vel_max
            1.5,                                                           // gripper_torque_max
            0.088,                                                         // gripper_width
            5.03,                                                          // gripper_open_readout
            6,                                                             // joint_dof
            std::vector<int>{1, 2, 4, 5, 6, 7},                            // motor_id
            std::vector<MotorType>{MotorType::EC_A4310, MotorType::EC_A4310, MotorType::EC_A4310, MotorType::DM_J4310,
                                   MotorType::DM_J4310, MotorType::DM_J4310}, // motor_type
            8,                                                                // gripper_motor_id
            MotorType::DM_J4310,                                              // gripper_motor_type
            (Eigen::Vector3d() << 0, 0, -9.807).finished(),                   // gravity_vector
            "base_link",                                                      // base_link_name
            "eef_link",                                                       // eef_link_name
            std::string(SDK_ROOT) + "/models/X5.urdf"                         // urdf_path
        );
    }
ControllerConfigFactory()
    {
        configurations["joint_controller_6"] = std::make_shared<ControllerConfig>(
            "joint_controller",                                           // controller_type
            (VecDoF(6) << 80.0, 70.0, 70.0, 30.0, 30.0, 20.0).finished(), // default_kp
            (VecDoF(6) << 2.0, 2.0, 2.0, 1.0, 1.0, 0.7).finished(),       // default_kd
            2.0,                                                          // default_gripper_kp
            0.1,                                                          // default_gripper_kd
            20,                                                           // over_current_cnt_max
            0.002,                                                        // controller_dt
            true,                                                         // gravity_compensation
            true,                                                         // background_send_recv
            true,                                                         // shutdown_to_passive
            "linear",                                                     // interpolation_method
            0.0                                                           // default_preview_time
        );
    }
```

## for openpi-client compatibility
<!-- pip install opencv-python==4.9.0.80
pip install opencv-python-headless==4.9.0.80 -->

# Lerobot-integration with ARX_X5
## BiARX5 Robot lerobot-teleoperate command
lerobot-teleoperate \
    --robot.type=bi_arx5 \
    --teleop.type=mock_teleop \
    --fps=30 \
    --debug_timing=false \
    --display_data=true


## BiARX5 Robot lerobot-record command
lerobot-record \
    --robot.type=bi_arx5 \
    --teleop.type=mock_teleop \
    --dataset.repo_id=Vertax/xense_bi_arx5_pick_and_place_cube \
    --dataset.num_episodes=100 \
    --dataset.single_task="pick rgb cubes and place them in the blue box" \
    --dataset.fps=60 \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true

## BiARX5 Robot lerobot-replay command
lerobot-replay \
    --robot.type=bi_arx5 \
    --dataset.repo_id=Vertax/xense_bi_arx5_pick_and_place_cube \
    --dataset.episode=0

## BiARX5 Robot lerobot-train command act
lerobot-train \
  --dataset.repo_id=Vertax/bi_arx5_pick_and_place_cube \
  --policy.type=act \
  --output_dir=outputs/train/act_bi_arx5_pick_and_place_cube \
  --job_name=act_bi_arx5_pick_and_place_cube \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=Vertax/act_bi_arx5_pick_and_place_cube \
  --batch_size=32 \
  --steps=200000 \
  --policy.push_to_hub=true \
  --wandb.disable_artifact=true 

## BiARX5 Robot lerobot-train command diffusion
lerobot-train \
  --dataset.repo_id=Vertax/bi_arx5_pick_and_place_cube \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_bi_arx5_pick_and_place_cube \
  --job_name=diffusion_bi_arx5_pick_and_place_cube \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=Vertax/diffusion_bi_arx5_pick_and_place_cube \
  --batch_size=16 \
  --steps=100000 \
  --policy.push_to_hub=true \
  --wandb.disable_artifact=true

## resume
lerobot-train \
  --policy.path=outputs/train/act_bi_arx5_pick_and_place_cube/checkpoints/last/pretrained_model \
  --resume=true

## BiARX5 act policy lerobot-eval command
lerobot-record  \
  --robot.type=bi_arx5 \
  --robot.inference_mode=true \
  --robot.preview_time=0.01 \
  --robot.id=bi_arx5 \
  --dataset.episode_time_s=600 \
  --display_data=false \
  --dataset.repo_id=Vertax/eval_act_bi_arx5_pick_and_place_cube \
  --dataset.single_task="pick and place cube" \
  --policy.path=outputs/train/act_bi_arx5_pick_and_place_cube/checkpoints/last/pretrained_model
  <!-- Local model path alternative: -->
  <!-- --policy.path=outputs/train/act_bi_arx5_pick_and_place_cube/checkpoints/last/pretrained_model -->

**Note on preview_time:** Adjust `--robot.preview_time` to reduce jittering:
- 0.03-0.05s: Smoother motion, more delay (recommended for stable movements)
- 0.01-0.02s: More responsive, but may cause jittering
- 0.0: No preview (only for teleoperation/recording)

## BiARX5 diffusion policy lerobot-eval command
lerobot-record  \
  --robot.type=bi_arx5 \
  --robot.inference_mode=true \
  --robot.preview_time=0.0 \
  --robot.id=bi_arx5 \
  --dataset.fps=30 \
  --dataset.episode_time_s=600 \
  --display_data=false \
  --dataset.repo_id=Vertax/eval_diffusion_bi_arx5_pick_and_place_cube \
  --dataset.single_task="pick and place cube" \
  --policy.path=outputs/train/diffusion_bi_arx5_pick_and_place_cube/checkpoints/last/pretrained_model