
import argparse
import asyncio 
import numpy as np 
from xense_franka import RobotInterface, FrankaController
import time 
import json 

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# FR3 constants
FR3_JOINT_POS_MIN_SAFE = np.array([-2.3476, -1.5454, -2.4937, -2.7714, -2.51, 0.7773, -2.7045], dtype=np.float32)
FR3_JOINT_POS_MAX_SAFE = np.array([2.3476, 1.5454, 2.4937, -0.4225, 2.51, 4.2841, 2.7045], dtype=np.float32)
FR3_JOINT_VEL_SAFE = np.array([2, 1, 1.5, 1.25, 3, 1.5, 3])
FR3_TORQUE_LIMIT = np.array([87, 87, 87, 87, 12, 12, 12])

def decode_json(path): 

    data = json.load(path)
    print(data.keys())

def plot_episode_trajectory(
        commanded_poses,
        achieved_poses,
        achieved_vels,
        achieved_torques,
        achieved_real,
        achieved_vels_real,
        achieved_torques_real,
        goal_pose,
        path
):
    """Synchronous plotting function - runs in background thread."""
    timesteps = np.arange(len(commanded_poses))
    num_joints = 7

    # Create 7x3 grid
    fig, axes = plt.subplots(7, 3, figsize=(18,14))

    # Title with success/failure
    # For the joint-reach task, success defined as L2 distance btwn current and target joint positions < 0.1 rad
    final_error = np.linalg.norm(achieved_real[-1, :] - goal_pose)
    success = final_error < 0.1
    print('Final error: ', final_error, 'Success: ', success)
    status_str = "SUCCESS" if success else "FAILURE"
    color = "green" if success else "red"
    fig.suptitle(f"{status_str}", fontsize=16, fontweight='bold', color=color)

    for joint_idx in range(num_joints):
        # Column 1: position
        ax_pos = axes[joint_idx, 0]
        ax_pos.plot(timesteps, commanded_poses[:, joint_idx], 'k-', linewidth=1.5, label="Commanded", alpha=0.8)
        ax_pos.plot(timesteps, achieved_poses[:, joint_idx], 'r-', linewidth=1.5, label="Sim", alpha=0.8)
        ax_pos.plot(timesteps, achieved_real[:, joint_idx], 'b-', linewidth=1.5, label="Real", alpha=0.8)
        ax_pos.axhline(y=FR3_JOINT_POS_MIN_SAFE[joint_idx], color='black', linestyle='--', linewidth=1, label="Safe min", alpha=0.6)
        ax_pos.axhline(y=FR3_JOINT_POS_MAX_SAFE[joint_idx], color='black', linestyle='--', linewidth=1, label="Safe max", alpha=0.6)
        # goal pose if provided
        if goal_pose is not None:
            ax_pos.axhline(y=goal_pose[joint_idx], color='green', linestyle='--', linewidth=2, label="Goal", alpha=0.9)
        # labels
        ax_pos.set_ylabel(f'Joint {joint_idx+1}\nPosition (rad)', fontsize=9)
        ax_pos.grid(True, alpha=0.3)
        if joint_idx == 0:
            ax_pos.set_title('Position', fontsize=12, fontweight="bold")
            ax_pos.legend(loc='upper right', fontsize=7)
        if joint_idx == 6:
            ax_pos.set_xlabel('Timestep', fontsize=10)

        # Column 2: Velocity
        if achieved_vels is not None:
            ax_vel = axes[joint_idx, 1]
            ax_vel.plot(timesteps, achieved_vels[:, joint_idx], 'r-', linewidth=1.5, label='Sim', alpha=0.8)
            ax_vel.plot(timesteps, achieved_vels_real[:, joint_idx], 'b-', linewidth=1.5, label='Real', alpha=0.8)
            ax_vel.axhline(y=-FR3_JOINT_VEL_SAFE[joint_idx], color='black', linestyle='--', linewidth=1, label="Safe min", alpha=0.6)
            ax_vel.axhline(y=FR3_JOINT_VEL_SAFE[joint_idx], color='black', linestyle='--', linewidth=1, label="Safe max", alpha=0.6)
            ax_vel.set_ylabel(f"Joint {joint_idx+1}\nVelocity (rad/s)", fontsize=9)
            if joint_idx == 0:
                ax_vel.set_title('Velocity', fontsize=12, fontweight='bold')
                ax_vel.legend(loc='upper right', fontsize=7)
            if joint_idx == 6:
                ax_vel.set_xlabel('Timestep', fontsize=10)

        # Column 3: Torque
        if achieved_torques is not None:
            ax_torque = axes[joint_idx, 2]
            ax_torque.plot(timesteps, achieved_torques[:, joint_idx], 'r-', linewidth=1.5, label='Sim', alpha=0.8)
            ax_torque.plot(timesteps, achieved_torques_real[:, joint_idx], 'b-', linewidth=1.5, label='Real', alpha=0.8)
            ax_torque.axhline(y=-FR3_TORQUE_LIMIT[joint_idx], color='black', linestyle='--', linewidth=1, label="Min", alpha=0.6)
            ax_torque.axhline(y=FR3_TORQUE_LIMIT[joint_idx], color='black', linestyle='--', linewidth=1, label="Max", alpha=0.6)
            ax_torque.set_ylabel(f"Joint {joint_idx+1}\nTorque (Nm)", fontsize=9)
            if joint_idx == 0:
                ax_torque.set_title('Torque', fontsize=12, fontweight='bold')
                ax_torque.legend(loc='upper right', fontsize=7)
            if joint_idx == 6:
                ax_torque.set_xlabel('Timestep', fontsize=10) 
        
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    orig_path = Path(path)
    save_path = orig_path.with_name(f"replay_{orig_path.stem}.png")
    print(f'Saving to: {save_path}')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


async def main(commanded_poses, achieved_poses, Kp, Kd):
    robot = RobotInterface("172.16.0.2") 
    controller = FrankaController(robot)
    
    await controller.start()

    # tests the 1kHz connection with the robot 
    # prints average frequency / jitter 
    await controller.test_connection()

    init_pos = [0, 0, 0.3, -1.57079, 0, 1.57079, -0.7853]

    # this moves to certain position by offline trajectory generated by Ruckig
    await controller.move(init_pos)

    init_pos = achieved_poses[0]
    await controller.move(init_pos)

    # you can switch controllers / set gains at run-time 
    controller.switch("impedance")
    base =  np.array([1, 1, 1, 1, 0.6, 0.6, 0.6])

    controller.kp = base * Kp
    controller.kd = base * Kd
    # this enforces 50Hz update rate for impedance controller target
    controller.set_freq(50) 

    achieved_real = []
    achieved_vels_real = []
    achieved_torques_real = []
    try:
        for cnt in range(len(commanded_poses)):
            # delta = np.sin(cnt / 50.0 * np.pi) * 0.1
            # init = controller.initial_qpos
            target = commanded_poses[cnt]
            print(target)

            await controller.set("q_desired", target[:7])

            # save robot pose
            state = robot.state
            achieved_real.append(state['qpos'][:7])
            achieved_vels_real.append(state['qvel'][:7])
            achieved_torques_real.append(state['last_torque'][:7])


    except (Exception, SystemExit) as e:
        print(f"Error during execution at step {len(achieved_real)}: {e}")
        # Return partial trajectory on failure
        return np.array(achieved_real), np.array(achieved_vels_real), np.array(achieved_torques_real) if achieved_real else None, None

    print('Finished trajectory!')
    return np.array(achieved_real), np.array(achieved_vels_real), np.array(achieved_torques_real)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal relative action robot control")
    parser.add_argument("--path", type=str, default="./examples/successful_rollouts_K128_D32.json", help="Path to trajectory")
    parser.add_argument("--Kp", type=float, default=128.0, help="Controller KP")
    parser.add_argument("--Kd", type=float, default=32.0, help="Controller KD")
    args = parser.parse_args()

    # fp = "./examples/successful_rollouts_K128_D32.json"
    
    # load json 
    if '.json' in args.path:
        with open(args.path, "r") as r: 
            data = json.load(r)
        commanded_poses = data['episodes'][0]['commanded_joint_pos']
        achieved_poses = data['episodes'][0]['achieved_joint_pos']
        goal_pose = None 
        achieved_vels = None
        achieved_torques = None
    elif '.npz' in args.path:
        data = np.load(args.path)
        commanded_poses = data['joint_pos_commanded']
        achieved_poses = data['joint_pos_achieved']
        goal_pose = data['joint_pos_goal'][0]
        achieved_vels = data['joint_vel_achieved']
        achieved_torques = data['joint_torque_applied']

    # Run robot control and get achieved trajectory
    achieved_real, achieved_vels_real, achieved_torques_real = asyncio.run(main(commanded_poses, achieved_poses, args.Kp, args.Kd))

    # Plot outside of async context
    if achieved_real is not None and len(achieved_real) > 0:
        print('Plotting...')
        plot_episode_trajectory(
            commanded_poses, achieved_poses, achieved_vels, achieved_torques, achieved_real, achieved_vels_real, achieved_torques_real, goal_pose, args.path
        )
    else:
        print('No trajectory data to plot') 
