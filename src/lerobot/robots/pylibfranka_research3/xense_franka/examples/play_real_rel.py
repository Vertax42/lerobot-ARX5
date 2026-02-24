#!/usr/bin/env python3
"""
Minimal Real Robot Script for Relative Joint Position Actions

Observation space (21 dims):
  - joint_position_error (7): target_joints - current_joints
  - joint_vel (7): current velocity
  - last_action (7): previous action

Action space:
  - Relative: target = current_pos + clamp(action, -1, 1) * scale
  - Per-joint scaling: joints 1-4 use base scale, joints 5-7 use scale * ratio

Usage:
    # With sampled target (from training distribution)
    python play_real_rel.py \
        --checkpoint path/to/best_agent.pt \
        --action_scale 0.5

    # With uniform scaling (same scale for all joints)
    python play_real_rel.py \
        --checkpoint path/to/best_agent.pt \
        --action_scale 0.5 \
        --uniform

    # With custom ratio for wrist joints (default: 0.5)
    python play_real_rel.py \
        --checkpoint path/to/best_agent.pt \
        --action_scale 0.5 \
        --ratio 0.3

    # With explicit target
    python play_real_rel.py \
        --checkpoint path/to/best_agent.pt \
        --action_scale 0.5 \
        --target_joints 0.0 -0.569 0.0 -2.810 0.0 3.037 0.741

    # Sample start position (like simulation reset)
    python play_real_rel.py \
        --checkpoint path/to/best_agent.pt \
        --action_scale 0.5 \
        --sample_start

    # With explicit start and target positions
    python play_real_rel.py \
        --checkpoint path/to/best_agent.pt \
        --action_scale 0.5 \
        --start_joints 0.0 0.0 0.0 -1.6 0.0 2.5 0.0 \
        --target_joints 0.1 0.1 0.1 -1.5 0.1 2.6 0.1

    # Save trajectory plot and data (single rollout)
    python play_real_rel.py \
        --checkpoint path/to/best_agent.pt \
        --action_scale 0.5 \
        --save_path trajectory.png

    # Run multiple rollouts with automatic saving
    python play_real_rel.py \
        --checkpoint path/to/best_agent.pt \
        --action_scale 0.5 \
        --num_rollouts 10 \
        --sample_start
    # Saves rollout_001.npz, rollout_001.png, etc. to {checkpoint_dir}/rollouts/
"""

import argparse
import asyncio
import numpy as np
import torch
from pathlib import Path

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# FR3 Constants
# ============================================================================

FR3_DEFAULT_JOINT_POS = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741], dtype=np.float32)
FR3_JOINT_POS_MIN = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159], dtype=np.float32)
FR3_JOINT_POS_MAX = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159], dtype=np.float32)

# Safe limits (used for training distribution)
FR3_JOINT_POS_MIN_SAFE = np.array([-2.3476, -1.5454, -2.4937, -2.7714, -2.51, 0.7773, -2.7045], dtype=np.float32)
FR3_JOINT_POS_MAX_SAFE = np.array([2.3476, 1.5454, 2.4937, -0.4225, 2.51, 4.2841, 2.7045], dtype=np.float32)
FR3_DEFAULT_JOINT_POS_SAFE = (FR3_JOINT_POS_MIN_SAFE + FR3_JOINT_POS_MAX_SAFE) / 2.0

# Velocity and torque limits (for plotting)
FR3_JOINT_VEL_SAFE = np.array([2, 1, 1.5, 1.25, 3, 1.5, 3], dtype=np.float32)
FR3_TORQUE_LIMIT = np.array([87, 87, 87, 87, 12, 12, 12], dtype=np.float32)

# Training samples ±0.5 rad around safe default
TRAINING_SAMPLING_RANGE = 0.5


def sample_target_from_training_distribution() -> np.ndarray:
    """Sample target joints from the same distribution used in training."""
    r = np.random.uniform(-1.0, 1.0, size=7)
    target = FR3_DEFAULT_JOINT_POS_SAFE + r * TRAINING_SAMPLING_RANGE
    target = np.clip(target, FR3_JOINT_POS_MIN_SAFE, FR3_JOINT_POS_MAX_SAFE)
    return target.astype(np.float32)


def sample_start_from_training_distribution() -> np.ndarray:
    """Sample start position using reset_joints_by_scale logic from simulation.

    In simulation, reset_joints_by_scale:
    1. Takes default_joint_pos (FR3_DEFAULT_JOINT_POS_SAFE)
    2. Scales each joint by a random value in position_range (0.5, 1.5)
    3. Clamps to soft_joint_pos_limits

    Note: Joints with default=0 will always start at 0 with this approach.
    """
    position_range = (0.5, 1.5)
    scale = np.random.uniform(position_range[0], position_range[1], size=7)
    start = FR3_DEFAULT_JOINT_POS_SAFE * scale
    start = np.clip(start, FR3_JOINT_POS_MIN_SAFE, FR3_JOINT_POS_MAX_SAFE)
    return start.astype(np.float32)


def save_trajectory_data(
        save_path: str,
        start_pose: np.ndarray,
        goal_pose: np.ndarray,
        commanded_poses: np.ndarray,
        achieved_poses: np.ndarray,
        achieved_vels: np.ndarray,
        achieved_torques: np.ndarray,
        action_scales: np.ndarray,
        Kp: float,
        Kd: float,
        duration: float
):
    """Save trajectory data to npz for later comparison with simulation.

    Args:
        save_path: Path to save the npz file
        start_pose: Sampled/provided start joint positions (7,)
        goal_pose: Goal joint positions (7,)
        commanded_poses: Commanded joint positions (N, 7)
        achieved_poses: Achieved joint positions (N, 7)
        achieved_vels: Achieved joint velocities (N, 7)
        achieved_torques: Applied torques (N, 7)
        action_scales: Per-joint action scales (7,)
        Kp: Stiffness gain
        Kd: Damping gain
        duration: Episode duration in seconds
    """
    final_error = np.linalg.norm(achieved_poses[-1, :] - goal_pose)
    is_success = final_error < 0.1

    np.savez(
        save_path,
        start_pose=start_pose,
        goal_pose=goal_pose,
        commanded_poses=commanded_poses,
        achieved_poses=achieved_poses,
        achieved_vels=achieved_vels,
        achieved_torques=achieved_torques,
        action_scales=action_scales,
        Kp=Kp,
        Kd=Kd,
        duration=duration,
        final_error=final_error,
        is_success=is_success
    )
    print(f"Saved trajectory data to: {save_path}")


def plot_episode_trajectory(
        commanded_poses: np.ndarray,
        achieved_poses: np.ndarray,
        achieved_vels: np.ndarray,
        achieved_torques: np.ndarray,
        goal_pose: np.ndarray,
        save_path: str
):
    """Plot episode trajectory with position, velocity, and torque for each joint.

    Args:
        commanded_poses: Commanded joint positions (N, 7)
        achieved_poses: Achieved joint positions (N, 7)
        achieved_vels: Achieved joint velocities (N, 7)
        achieved_torques: Applied torques (N, 7)
        goal_pose: Goal joint positions (7,)
        save_path: Path to save the plot
    """
    timesteps = np.arange(len(commanded_poses))
    num_joints = 7

    # Create 7x3 grid
    fig, axes = plt.subplots(7, 3, figsize=(18, 14))

    # Title with success/failure
    final_error = np.linalg.norm(achieved_poses[-1, :] - goal_pose)
    success = final_error < 0.1
    status_str = "SUCCESS" if success else "FAILURE"
    color = "green" if success else "red"
    fig.suptitle(f"{status_str} (Final error: {final_error:.4f} rad)", fontsize=16, fontweight='bold', color=color)

    for joint_idx in range(num_joints):
        # Column 1: Position
        ax_pos = axes[joint_idx, 0]
        ax_pos.plot(timesteps, commanded_poses[:, joint_idx], 'k-', linewidth=1.5, label="Commanded", alpha=0.8)
        ax_pos.plot(timesteps, achieved_poses[:, joint_idx], 'b-', linewidth=1.5, label="Achieved", alpha=0.8)
        ax_pos.axhline(y=FR3_JOINT_POS_MIN_SAFE[joint_idx], color='gray', linestyle='--', linewidth=1, label="Safe min", alpha=0.6)
        ax_pos.axhline(y=FR3_JOINT_POS_MAX_SAFE[joint_idx], color='gray', linestyle='--', linewidth=1, label="Safe max", alpha=0.6)
        ax_pos.axhline(y=goal_pose[joint_idx], color='green', linestyle='--', linewidth=2, label="Goal", alpha=0.9)
        ax_pos.set_ylabel(f'Joint {joint_idx+1}\nPosition (rad)', fontsize=9)
        ax_pos.grid(True, alpha=0.3)
        if joint_idx == 0:
            ax_pos.set_title('Position', fontsize=12, fontweight="bold")
            ax_pos.legend(loc='upper right', fontsize=7)
        if joint_idx == 6:
            ax_pos.set_xlabel('Timestep', fontsize=10)

        # Column 2: Velocity
        ax_vel = axes[joint_idx, 1]
        ax_vel.plot(timesteps, achieved_vels[:, joint_idx], 'b-', linewidth=1.5, label='Achieved', alpha=0.8)
        ax_vel.axhline(y=-FR3_JOINT_VEL_SAFE[joint_idx], color='gray', linestyle='--', linewidth=1, label="Safe min", alpha=0.6)
        ax_vel.axhline(y=FR3_JOINT_VEL_SAFE[joint_idx], color='gray', linestyle='--', linewidth=1, label="Safe max", alpha=0.6)
        ax_vel.set_ylabel(f"Joint {joint_idx+1}\nVelocity (rad/s)", fontsize=9)
        ax_vel.grid(True, alpha=0.3)
        if joint_idx == 0:
            ax_vel.set_title('Velocity', fontsize=12, fontweight='bold')
            ax_vel.legend(loc='upper right', fontsize=7)
        if joint_idx == 6:
            ax_vel.set_xlabel('Timestep', fontsize=10)

        # Column 3: Torque
        ax_torque = axes[joint_idx, 2]
        ax_torque.plot(timesteps, achieved_torques[:, joint_idx], 'b-', linewidth=1.5, label='Applied', alpha=0.8)
        ax_torque.axhline(y=-FR3_TORQUE_LIMIT[joint_idx], color='gray', linestyle='--', linewidth=1, label="Limit", alpha=0.6)
        ax_torque.axhline(y=FR3_TORQUE_LIMIT[joint_idx], color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax_torque.set_ylabel(f"Joint {joint_idx+1}\nTorque (Nm)", fontsize=9)
        ax_torque.grid(True, alpha=0.3)
        if joint_idx == 0:
            ax_torque.set_title('Torque', fontsize=12, fontweight='bold')
            ax_torque.legend(loc='upper right', fontsize=7)
        if joint_idx == 6:
            ax_torque.set_xlabel('Timestep', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    print(f'Saving plot to: {save_path}')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Core Functions
# ============================================================================

def load_policy(checkpoint_path: str, device: str = "cpu"):
    """
    Load policy from skrl checkpoint.

    Returns policy state dict and state preprocessor (running mean/var).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    policy_state = checkpoint["policy"]
    state_preprocessor = checkpoint.get("state_preprocessor", None)

    return policy_state, state_preprocessor, device


def build_observation(joint_pos: np.ndarray, joint_vel: np.ndarray,
                      target_joints: np.ndarray, last_action: np.ndarray) -> np.ndarray:
    """
    Build observation vector (21 dims).

    Order: [joint_vel(7), last_action(7), joint_position_error(7)]

    This matches the observation space in rel_cat_joint_reach_env_cfg.py
    """
    # joint_position_error = goal - current (direction to move)
    joint_position_error = target_joints - joint_pos

    obs = np.concatenate([
        joint_vel,             # 7
        last_action,           # 7
        joint_position_error,  # 7
    ])
    return obs.astype(np.float32)


def normalize_observation(obs: np.ndarray, running_mean: np.ndarray,
                          running_var: np.ndarray, epsilon: float = 1e-8,
                          clip_range: float = 5.0) -> np.ndarray:
    """Apply running standard scaler normalization (same as training).

    Includes clipping to [-clip_range, clip_range] to match simulation.
    """
    normalized = (obs - running_mean) / np.sqrt(running_var + epsilon)
    normalized = np.clip(normalized, -clip_range, clip_range)
    return normalized.astype(np.float32)


def process_action(raw_action: np.ndarray, current_joint_pos: np.ndarray,
                   action_scales: np.ndarray) -> np.ndarray:
    """
    Process raw policy output to joint position command.

    Relative action: target = current_pos + clamp(raw, -1, 1) * scale
    Uses per-joint action scales.
    """
    # Clamp raw action to [-1, 1] (same as training)
    clamped = np.clip(raw_action, -1.0, 1.0)

    # Compute target position (per-joint scaling)
    target = current_joint_pos + clamped * action_scales

    # Safety: clip to joint limits
    target = np.clip(target, FR3_JOINT_POS_MIN, FR3_JOINT_POS_MAX)

    return target


def compute_action_scales(base_scale: float, ratio: float) -> np.ndarray:
    """
    Compute per-joint action scales.

    Joints 1-4: base_scale
    Joints 5-7: base_scale * ratio
    """
    scales = np.array([
        base_scale,           # joint 1
        base_scale,           # joint 2
        base_scale,           # joint 3
        base_scale,           # joint 4
        base_scale * ratio,   # joint 5
        base_scale * ratio,   # joint 6
        base_scale * ratio,   # joint 7
    ], dtype=np.float32)
    return scales


def build_policy_network(policy_state, device: str):
    """Build and load policy network from checkpoint state dict."""
    obs_dim = 21
    action_dim = 7

    policy = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 64),
        torch.nn.ELU(),
        torch.nn.Linear(64, 64),
        torch.nn.ELU(),
        torch.nn.Linear(64, action_dim),
    ).to(device)

    policy[0].weight.data = policy_state["net_container.0.weight"]
    policy[0].bias.data = policy_state["net_container.0.bias"]
    policy[2].weight.data = policy_state["net_container.2.weight"]
    policy[2].bias.data = policy_state["net_container.2.bias"]
    policy[4].weight.data = policy_state["policy_layer.weight"]
    policy[4].bias.data = policy_state["policy_layer.bias"]
    policy.eval()

    return policy


def dry_run_policy(policy_state, state_preprocessor, device: str,
                   trajectory_path: str, action_scales: np.ndarray):
    """
    Dry run: verify policy outputs match saved simulation trajectory.

    Uses obs_raw from npz, normalizes (with clipping), runs policy,
    clips & scales action, and compares with saved action_delta.

    Args:
        action_scales: Per-joint action scales (7,). Use compute_action_scales() to create.
    """
    print(f"\n{'='*60}")
    print("DRY RUN MODE - Verifying policy against saved trajectory")
    print(f"{'='*60}\n")

    # Load trajectory
    print(f"Loading trajectory: {trajectory_path}")
    data = np.load(trajectory_path)

    # Check required keys
    required_keys = ['obs_raw', 'action_delta']
    missing = [k for k in required_keys if k not in data.keys()]
    if missing:
        print(f"ERROR: Missing required keys: {missing}")
        print(f"Available keys: {list(data.keys())}")
        return False

    obs_raw = data['obs_raw']
    action_delta = data['action_delta']

    num_steps = len(obs_raw)
    print(f"Trajectory length: {num_steps} steps")
    if 'is_success' in data:
        print(f"Success: {data['is_success']}")

    # Get normalization parameters
    if state_preprocessor is not None:
        running_mean = state_preprocessor["running_mean"].cpu().numpy()
        running_var = state_preprocessor["running_variance"].cpu().numpy()
    else:
        print("ERROR: No state_preprocessor in checkpoint")
        return False

    # Build policy
    policy = build_policy_network(policy_state, device)

    # Full pipeline: obs_raw -> normalize (with clip) -> policy -> clip -> scale
    obs_normalized = normalize_observation(obs_raw, running_mean, running_var)
    obs_tensor = torch.from_numpy(obs_normalized).to(device)

    with torch.no_grad():
        raw_actions = policy(obs_tensor).cpu().numpy()

    clamped = np.clip(raw_actions, -1, 1)
    computed_delta = clamped * action_scales  # Per-joint scaling

    # Compare
    errors = np.abs(computed_delta - action_delta)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Action scales: {action_scales}")
    print(f"  Joints 1-4: {action_scales[0]}")
    print(f"  Joints 5-7: {action_scales[4]}")
    print(f"\nAction delta (clipped & scaled):")
    print(f"  Max error:  {errors.max():.2e}")
    print(f"  Mean error: {errors.mean():.2e}")

    print(f"\nPer-joint max errors:")
    for j in range(7):
        print(f"  Joint {j+1}: {errors[:, j].max():.2e}")

    all_pass = errors.max() < 1e-5
    print(f"\n{'='*60}")
    if all_pass:
        print("✓ PASS - Policy inference matches simulation")
    else:
        print("✗ FAIL - Discrepancies detected")
    print(f"{'='*60}\n")

    return all_pass


async def run_policy(policy_state, state_preprocessor, device: str,
                     target_joints: np.ndarray, action_scales: np.ndarray,
                     robot_ip: str, Kp: float, Kd: float,
                     start_joints: np.ndarray = None, duration: float = 8.0):
    """Main async control loop at exactly 50Hz.

    Args:
        duration: Episode duration in seconds (default: 8.0s = 400 steps at 50Hz)

    Returns:
        Tuple of (commanded_poses, achieved_poses, achieved_vels, achieved_torques, goal_pose)
        as numpy arrays for plotting.
    """
    from xense_franka import RobotInterface, FrankaController

    # Initialize robot and controller
    robot = RobotInterface(robot_ip)
    controller = FrankaController(robot)

    await controller.start()

    # Move to start position if specified
    if start_joints is not None:
        print(f"Moving to start position: {start_joints}")
        await controller.move(start_joints)

    # Switch to impedance control with gains
    controller.switch("impedance")
    base = np.array([1, 1, 1, 1, 0.6, 0.6, 0.6])
    controller.kp = base * Kp
    controller.kd = base * Kd
    controller.set_freq(50)  # Enforce exact 50Hz

    # Build policy network
    policy = build_policy_network(policy_state, device)

    # Get normalization parameters
    obs_dim = 21
    if state_preprocessor is not None:
        running_mean = state_preprocessor["running_mean"].cpu().numpy()
        running_var = state_preprocessor["running_variance"].cpu().numpy()
        print(f"Using observation normalization (mean shape: {running_mean.shape})")
    else:
        running_mean = np.zeros(obs_dim)
        running_var = np.ones(obs_dim)
        print("WARNING: No state preprocessor found, using identity normalization")

    # State
    last_action = np.zeros(7, dtype=np.float32)
    step = 0

    # Pre-allocate trajectory arrays (avoid dynamic list resizing)
    max_steps = int(duration * 50)  # 50Hz control rate
    commanded_poses = np.zeros((max_steps, 7), dtype=np.float32)
    achieved_poses = np.zeros((max_steps, 7), dtype=np.float32)
    achieved_vels = np.zeros((max_steps, 7), dtype=np.float32)
    achieved_torques = np.zeros((max_steps, 7), dtype=np.float32)

    print(f"\n{'='*60}")
    print(f"Starting control loop at 50 Hz (enforced by controller)")
    print(f"Action scales: {action_scales}")
    print(f"  Joints 1-4: {action_scales[0]}")
    print(f"  Joints 5-7: {action_scales[4]}")
    print(f"Target joints: {target_joints}")
    print(f"Kp: {Kp}, Kd: {Kd}")
    print(f"Duration: {duration}s ({max_steps} steps at 50Hz)")
    print(f"Press Ctrl+C to stop early")
    print(f"{'='*60}\n")

    try:
        for step in range(max_steps):
            # 1. Read robot state (use controller's cached state, not robot.state directly)
            state = controller.state
            joint_pos = np.array(state['qpos'][:7], dtype=np.float32)
            joint_vel = np.array(state['qvel'][:7], dtype=np.float32)
            joint_torque = np.array(state['last_torque'][:7], dtype=np.float32)

            # 2. Build observation
            obs = build_observation(joint_pos, joint_vel, target_joints, last_action)

            # 3. Normalize observation (same as training)
            obs_normalized = normalize_observation(obs, running_mean, running_var)
            obs_tensor = torch.from_numpy(obs_normalized).unsqueeze(0).to(device)

            # 4. Get action from policy
            with torch.no_grad():
                raw_action = policy(obs_tensor)[0].cpu().numpy()

            # 5. Process action (relative, per-joint scaling)
            target_action = process_action(raw_action, joint_pos, action_scales)

            # 6. Apply to robot - automatically waits to maintain 50Hz
            await controller.set("q_desired", target_action)

            # 7. Collect trajectory data (direct array assignment, no copy)
            commanded_poses[step] = target_action
            achieved_poses[step] = joint_pos
            achieved_vels[step] = joint_vel
            achieved_torques[step] = joint_torque

            # 8. Update state
            last_action = raw_action

            # 9. Print status
            if (step + 1) % 50 == 0:
                error = np.linalg.norm(joint_pos - target_joints)
                print(f"Step {step+1:4d}/{max_steps} | Error: {error:.4f} rad | "
                      f"Raw: [{raw_action[0]:+.3f}, {raw_action[1]:+.3f}, ...] | "
                      f"Pos: [{joint_pos[0]:+.3f}, {joint_pos[1]:+.3f}, ...]")

        # Episode complete
        final_error = np.linalg.norm(joint_pos - target_joints)
        print(f"\n{'='*60}")
        print(f"Episode complete after {max_steps} steps ({duration}s)")
        print(f"Final error: {final_error:.4f} rad")
        print(f"Success: {'Yes' if final_error < 0.1 else 'No'} (threshold: 0.1 rad)")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print(f"\n\nStopped early at step {step}")

    await controller.stop()

    # Return trajectory data (slice to actual steps completed)
    # step+1 because step is 0-indexed, so after completing step N we have N+1 data points
    return (
        commanded_poses[:step + 1],
        achieved_poses[:step + 1],
        achieved_vels[:step + 1],
        achieved_torques[:step + 1],
        target_joints
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Minimal relative action robot control")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--action_scale", type=float, required=True, help="Action scale for joints 1-4")
    parser.add_argument("--uniform", action="store_true", help="Use same scale for all joints")
    parser.add_argument("--ratio", type=float, default=0.5,
                        help="Ratio for joints 5-7 (default: 0.5, ignored if --uniform)")
    parser.add_argument("--target_joints", type=float, nargs=7, default=None,
                        help="Target joint positions (7 values)")
    parser.add_argument("--start_joints", type=float, nargs=7, default=None,
                        help="Start joint positions (7 values). Robot will move here before running.")
    parser.add_argument("--sample_start", action="store_true",
                        help="Sample start position from training distribution (same as simulation reset)")
    parser.add_argument("--robot_ip", type=str, default="172.16.0.2", help="Robot IP address")
    parser.add_argument("--Kp", type=float, default=128.0, help="Impedance controller stiffness gain")
    parser.add_argument("--Kd", type=float, default=32.0, help="Impedance controller damping gain")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="Episode duration in seconds (default: 8.0s = 400 steps at 50Hz)")
    parser.add_argument("--dry_run", type=str, default=None,
                        help="Path to trajectory .npz file for dry run verification (no robot)")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save trajectory plot (e.g., trajectory.png). If not specified, no plot is saved.")
    parser.add_argument("--num_rollouts", type=int, default=None,
                        help="Number of rollouts to run. Data and plots saved to {checkpoint_dir}/rollouts/")
    args = parser.parse_args()

    # Compute per-joint action scales
    ratio = 1.0 if args.uniform else args.ratio
    action_scales = compute_action_scales(args.action_scale, ratio)

    print(f"Loading checkpoint: {args.checkpoint}")
    policy_state, state_preprocessor, device = load_policy(args.checkpoint, args.device)

    # Dry run mode: verify policy against saved trajectory
    if args.dry_run is not None:
        dry_run_policy(policy_state, state_preprocessor, device,
                       args.dry_run, action_scales)
        return

    # Multi-rollout mode
    if args.num_rollouts is not None:
        # Create rollouts directory next to checkpoint
        checkpoint_path = Path(args.checkpoint)
        rollouts_dir = checkpoint_path.parent / "rollouts"
        rollouts_dir.mkdir(exist_ok=True)
        print(f"\n{'='*60}")
        print(f"MULTI-ROLLOUT MODE: {args.num_rollouts} rollouts")
        print(f"Saving to: {rollouts_dir}")
        print(f"{'='*60}\n")

        successes = 0
        for i in range(args.num_rollouts):
            print(f"\n{'='*60}")
            print(f"ROLLOUT {i+1}/{args.num_rollouts}")
            print(f"{'='*60}")

            # Sample start and target for each rollout (unless explicitly provided)
            if args.start_joints is not None:
                start_joints = np.array(args.start_joints, dtype=np.float32)
            elif args.sample_start:
                start_joints = sample_start_from_training_distribution()
                print(f"Sampled start: {start_joints}")
            else:
                start_joints = None

            if args.target_joints is not None:
                target_joints = np.array(args.target_joints, dtype=np.float32)
            else:
                target_joints = sample_target_from_training_distribution()
                print(f"Sampled target: {target_joints}")

            # Run rollout
            commanded_poses, achieved_poses, achieved_vels, achieved_torques, goal_pose = asyncio.run(
                run_policy(
                    policy_state, state_preprocessor, device, target_joints,
                    action_scales, args.robot_ip, args.Kp, args.Kd, start_joints,
                    args.duration
                )
            )

            if len(achieved_poses) > 0:
                # Save data
                data_path = rollouts_dir / f"rollout_{i+1:03d}.npz"
                save_trajectory_data(
                    str(data_path),
                    start_joints if start_joints is not None else achieved_poses[0],
                    goal_pose,
                    commanded_poses,
                    achieved_poses,
                    achieved_vels,
                    achieved_torques,
                    action_scales,
                    args.Kp,
                    args.Kd,
                    args.duration
                )

                # Save plot
                plot_path = rollouts_dir / f"rollout_{i+1:03d}.png"
                plot_episode_trajectory(
                    commanded_poses,
                    achieved_poses,
                    achieved_vels,
                    achieved_torques,
                    goal_pose,
                    str(plot_path)
                )

                # Track success
                final_error = np.linalg.norm(achieved_poses[-1, :] - goal_pose)
                if final_error < 0.1:
                    successes += 1

        # Summary
        print(f"\n{'='*60}")
        print(f"ROLLOUT SUMMARY")
        print(f"{'='*60}")
        print(f"Total rollouts: {args.num_rollouts}")
        print(f"Successes: {successes}/{args.num_rollouts} ({100*successes/args.num_rollouts:.1f}%)")
        print(f"Data saved to: {rollouts_dir}")
        print(f"{'='*60}\n")
        return

    # Single rollout mode
    # Sample or use provided start position
    if args.start_joints is not None:
        start_joints = np.array(args.start_joints, dtype=np.float32)
        print(f"Using provided start position: {start_joints}")
    elif args.sample_start:
        start_joints = sample_start_from_training_distribution()
        print(f"Sampled start from training distribution: {start_joints}")
    else:
        start_joints = None

    # Sample target from training distribution if not provided
    if args.target_joints is None:
        target_joints = sample_target_from_training_distribution()
        print(f"Sampled target from training distribution: {target_joints}")
    else:
        target_joints = np.array(args.target_joints, dtype=np.float32)

    print(f"Running with:")
    print(f"  robot_ip: {args.robot_ip}")
    print(f"  action_scale: {args.action_scale}")
    if start_joints is not None:
        print(f"  start_joints: {start_joints}")
    print(f"  target_joints: {target_joints}")
    print(f"  Kp: {args.Kp}, Kd: {args.Kd}")
    print(f"  duration: {args.duration}s")
    if args.save_path:
        print(f"  save_path: {args.save_path}")

    # Run the control loop and get trajectory data
    commanded_poses, achieved_poses, achieved_vels, achieved_torques, goal_pose = asyncio.run(
        run_policy(
            policy_state, state_preprocessor, device, target_joints,
            action_scales, args.robot_ip, args.Kp, args.Kd, start_joints,
            args.duration
        )
    )

    # Save data and/or plot if save_path is specified
    if args.save_path and len(achieved_poses) > 0:
        # Save data (change extension to .npz)
        save_path = Path(args.save_path)
        data_path = save_path.with_suffix('.npz')
        save_trajectory_data(
            str(data_path),
            start_joints if start_joints is not None else achieved_poses[0],
            goal_pose,
            commanded_poses,
            achieved_poses,
            achieved_vels,
            achieved_torques,
            action_scales,
            args.Kp,
            args.Kd,
            args.duration
        )

        # Save plot
        print("Plotting trajectory...")
        plot_episode_trajectory(
            commanded_poses,
            achieved_poses,
            achieved_vels,
            achieved_torques,
            goal_pose,
            str(save_path)
        )
    elif args.save_path:
        print("No trajectory data to plot/save")


if __name__ == "__main__":
    main()
