# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os

from isaaclab.app import AppLauncher
from copy import deepcopy
import yaml
from cmaes import CMA
import json
# make the plots into mp4
import imageio
import time

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--gripper", action="store_true", help="Use gripper")
parser.add_argument("--name", type=str, default="test", help="Name of the experiment")
parser.add_argument("--dir", type=str, default=".", help="Directory for results")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import matplotlib.pyplot as plt
from tqdm import trange



def _symmetrize(A):
    # Works with any batch shape ending in (3,3)
    return 0.5 * (A + A.transpose(-1, -2))

def _pack_symmetric_3x3(theta):
    """
    theta: (*, 6) -> (*, 3, 3)
      maps (t0..t5) to [[t0, t3, t4],
                        [t3, t1, t5],
                        [t4, t5, t2]]
    """
    t0, t1, t2, t3, t4, t5 = torch.unbind(theta, dim=-1)
    row0 = torch.stack([t0, t3, t4], dim=-1)
    row1 = torch.stack([t3, t1, t5], dim=-1)
    row2 = torch.stack([t4, t5, t2], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def logm_spd(I, eps=1e-9):
    """
    Matrix log for SPD 3x3 via eigen-decomposition.
    I: (*, 3, 3) symmetric positive definite
    returns: (*, 3, 3) symmetric
    """
    I = _symmetrize(I)
    # eigh: symmetric/hermitian eigen-decomposition with guaranteed real eigenvalues
    w, V = torch.linalg.eigh(I)                           # (*,3), (*,3,3)
    w = torch.clamp(w, min=eps)
    logw = torch.log(w)
    return (V * logw[..., None, :]) @ V.transpose(-1, -2)


def expm_sym(S):
    """
    Matrix exp for real symmetric 3x3 via eigen-decomposition.
    S: (*, 3, 3) symmetric
    returns: (*, 3, 3) SPD
    """
    S = _symmetrize(S)
    w, V = torch.linalg.eigh(S)
    expw = torch.exp(w)
    return (V * expw[..., None, :]) @ V.transpose(-1, -2)


# ---- main API --------------------------------------------------------------

def inertia_from_log_delta(I0, theta: np.ndarray, eps=1e-9):
    """
    Log-Euclidean inertia update.

    Args:
      I0:    (*, 3, 3) baseline inertia (SPD)
      theta: (*, 6)    unconstrained parameters for a symmetric delta in log-space

    Returns:
      I_new: (*, 3, 3) updated inertia, symmetric positive-definite
    """
    # ensure double precision for stability (you can remove if you prefer float32)
    dtype = torch.float32 #if I0.dtype == torch.float64 or theta.dtype == torch.float64 else torch.float32
    I0 = I0.to(dtype)
    theta = torch.from_numpy(theta).to(dtype)

    print(I0.shape, theta.shape)

    S0 = logm_spd(I0, eps=eps)                 # log(I0)
    dS = _pack_symmetric_3x3(theta)            # symmetric delta
    print(S0.shape, dS.shape)
    S_new = S0 + dS
    I_new = expm_sym(S_new)                    # exp(S0 + dS) -> SPD
    # tiny symmetrize to kill numeric skew
    return _symmetrize(I_new)

# np.random.seed(0)


print(f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/FR3/fr3.usd".replace("5.1", "4.5"))

FRANKA_FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/FR3/fr3.usd".replace("5.1", "4.5"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fr3_joint1": 0.0,
            "fr3_joint2": 0.0,
            "fr3_joint3": 0.0,
            "fr3_joint4": -1.5707899999999999,
            "fr3_joint5": 0.0,
            "fr3_joint6": 1.5707899999999999,
            "fr3_joint7": -0.7853,
            "fr3_finger_joint.*": 0.04,
        },
        # pos=(0.0, -1.0, 0.0),

    ),

    actuators={
        "fr3_joint1": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint1"],
            effort_limit_sim=87.0,
            velocity_limit_sim=1000.0,
            stiffness=80, #joints[0][1],
            damping=4, #joints[0][2],
            # armature=0, #0.195,
            # friction=0, #1.137,
        ),
        "fr3_joint2": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint2"],
            effort_limit_sim=87.0,
            velocity_limit_sim=1000.0,
            stiffness=80, #joints[1][1],
            damping=4, #joints[1][2],
            # armature=0, #0.195,
            # friction=0, #1.137,
        ),
        "fr3_joint3": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint3"],
            effort_limit_sim=87.0,
            velocity_limit_sim=1000.0,
            stiffness=80, #joints[2][1],
            damping=4, #joints[2][2],
            # armature=0, #0.195,
            # friction=0, #1.137,
        ),
        "fr3_joint4": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint4"],
            effort_limit_sim=87.0,
            velocity_limit_sim=1000.0,
            stiffness=80, #joints[3][1],
            damping=4, #joints[3][2],
            # armature=0, #0.195,
            # friction=0, #1.137,
        ),
        "fr3_joint5": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint5"],
            effort_limit_sim=12.0,
            velocity_limit_sim=1000.0,
            stiffness=80, #joints[4][1],
            damping=4, #joints[4][2],
            # armature=0, #0.074,
            # friction=0, #0.763,
        ),
        "fr3_joint6": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint6"],
            effort_limit_sim=12.0,
            velocity_limit_sim=1000.0,
            stiffness=80, #joints[5][1],
            damping=4, #joints[5][2],
            # armature=0, #0.074,
            # friction=0, #0.44,
        ),
        "fr3_joint7": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint7"],
            effort_limit_sim=12.0,
            velocity_limit_sim=1000.0,
            stiffness=80, #joints[6][1],
            damping=4, #joints[6][2],
            # armature=0, #0.074,
            # friction=0, #0.248,
        ),
        "fr3_hand": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"],
            effort_limit_sim=200.0,
            velocity_limit_sim=1000.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },

soft_joint_pos_limit_factor=1.0,
)

"""Configuration of Franka Emika Panda robot."""



class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    # ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(size = (0.1, 0.1)))

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=100.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    Franka = FRANKA_FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Franka")




def run_simulator(name, sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    print(sim_dt)
    sim_time = 0.0
    count = 0

    init_pos = [0.0, 0.0, 0.0, -1.5707899999999999, 0.0, 1.5707899999999999, -0.7853]

    lower = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0519])
    upper = np.array([+2.7437, +1.7837, +2.9007, +0.1518, +2.8065, +4.5169, +3.0519])

    # name = args_cli.name

    os.makedirs(f"final_grid/{name}", exist_ok=True)

    real_data = np.load(f"examples/sysid_left_more/sysid_{name}.npz")
    # print(real_+d)
    real_data = {k: v for k, v in real_data.items()}
    print(real_data.keys())


    joint_pos = np.zeros(9)
    joint_pos[:7] = real_data["qpos"][0, :7] #+ np.random.uniform(-1, 1, 7)
    joint_pos[:7] = np.clip(joint_pos[:7], lower, upper)
    joint_pos[7] = 0.04
    joint_pos[8] = 0.04
    joint_vel = np.zeros(9)

    torch_joint_pos = torch.from_numpy(joint_pos).to(torch.float32).to(args_cli.device).unsqueeze(0)
    torch_joint_vel = torch.from_numpy(joint_vel).to(torch.float32).to(args_cli.device).unsqueeze(0)

    scene["Franka"].write_joint_state_to_sim(torch_joint_pos, torch_joint_vel)
    scene.reset()

    if args_cli.gripper:
        joint_num = 9
    else:
        joint_num = 7

    body_num = 11

    bounds = np.zeros((6 * joint_num, 2))
    bounds[:, 0] = 0
    bounds[0*joint_num:1*joint_num, 1] = 10. # stiffness
    bounds[1*joint_num:2*joint_num, 1] = 10. # damping
    bounds[2*joint_num:3*joint_num, 1] = 0.5 # armature
    bounds[3*joint_num:4*joint_num, 0] = 0.01 # friction coefficient
    bounds[3*joint_num:4*joint_num, 1] = 1.0 # friction coefficient
    bounds[4*joint_num:5*joint_num, 1] = 1.0 # dynamic friction coefficient
    bounds[5*joint_num:6*joint_num, 1] = 1.0 # viscous friction coefficient

    # randomly sample from the bounds for initial guess
    init_guess = np.random.uniform(bounds[:, 0], bounds[:, 1])

    # print(bounds)

    optimizer = CMA(mean = init_guess, sigma = 3.0, \
        bounds = bounds,
            population_size = args_cli.num_envs)


    plots = []

    real_data["qpos"][:, 7:9] *= 0.5

    lowest_loss = 1e6

    # specify the seed used for random initialization
    rand_run_int = 42

    from tqdm import trange as original_trange

    pbar = original_trange(200, desc="Optimization", unit=" iter")

    for iii in pbar:

        iter_start = time.time()

        # try randomized pd gains
        t_ask = time.time()
        pd_gains = []
        for j in range(args_cli.num_envs):
            pd_gain  = optimizer.ask()
            pd_gains.append(pd_gain)
        pd_gains = np.stack(pd_gains, axis = 0)
        t_ask = time.time() - t_ask

        t_write = time.time()
        # write the joint armature, friction coefficient, dynamic friction coefficient, and viscous friction coefficient to the simulator
        scene["Franka"].write_joint_stiffness_to_sim(torch.from_numpy(2 ** pd_gains[:, 0:joint_num  ]).to(args_cli.device).type(torch.float32), \
            joint_ids = torch.tensor([i for i in range(joint_num)]).type(torch.int).to(args_cli.device))
        scene["Franka"].write_joint_damping_to_sim(torch.from_numpy(2 **pd_gains[:, joint_num:2*joint_num]).to(args_cli.device).type(torch.float32), \
            joint_ids = torch.tensor([i for i in range(joint_num)]).type(torch.int).to(args_cli.device))
        scene["Franka"].write_joint_armature_to_sim(torch.from_numpy(pd_gains[:,2*joint_num:3*joint_num] ).to(args_cli.device).type(torch.float32), \
            joint_ids = torch.tensor([i for i in range(joint_num)]).type(torch.int).to(args_cli.device))
        
        static_friction = torch.from_numpy(pd_gains[:, 3*joint_num:4*joint_num]).to(args_cli.device).type(torch.float32)
        dynamic_ratio = torch.from_numpy(pd_gains[:, 4*joint_num:5*joint_num]).to(args_cli.device).type(torch.float32)
        dynamic_friction = static_friction * dynamic_ratio
        viscous_friction = torch.from_numpy(pd_gains[:, 5*joint_num:6*joint_num]).to(args_cli.device).type(torch.float32)

        scene["Franka"].write_joint_friction_coefficient_to_sim(static_friction, dynamic_friction, viscous_friction, \
                                                                joint_ids = torch.tensor([i for i in range(joint_num)]).type(torch.int).to(args_cli.device))
        t_write = time.time() - t_write

        costs = np.zeros(args_cli.num_envs)


        joint_pos = np.zeros(9)
        joint_pos[:7] = real_data["qpos"][0, :7] #+ np.random.uniform(-1, 1, 7)
        joint_pos[7:9] = 0.04
        joint_vel = np.zeros(9)

        torch_joint_pos = torch.from_numpy(joint_pos).to(torch.float32).to(args_cli.device).unsqueeze(0)
        torch_joint_vel = torch.from_numpy(joint_vel).to(torch.float32).to(args_cli.device).unsqueeze(0)

        scene["Franka"].write_joint_state_to_sim(torch_joint_pos, torch_joint_vel)
        scene.reset()

        joint_positions = []
        joint_velocities = [] 
        joint_targets = []

        # get the joint position from the real data
        # reset.
        t_sim = time.time()
        for count in range(real_data["qpos"].shape[0]):

            # wave
            wave_action = torch.zeros(args_cli.num_envs, 9)
            wave_action[:, :7] = torch.from_numpy(real_data["qdes"][count][:7]).to(args_cli.device).unsqueeze(0)
            wave_action[:, 7:9] = 0.04
            # wave_action[:, 7] = 0.04 #.04
            # wave_action[:, 8] = 0.04 #.04

            # wave_action[:, 0:7] += magnitude * np.sin(2 * np.pi * frequency * (sim_time - reset_sim_time))
            # wave_action[:, 0:7] += noise_magnitude * torch.randn_like(wave_action[:, 0:7])

            for _ in range(2):
                scene["Franka"].set_joint_position_target(wave_action)

                scene.write_data_to_sim()
                sim.step()

                sim_time += sim_dt
                count += 1
                scene.update(sim_dt)

            joint_positions.append(scene["Franka"].data.joint_pos.clone().cpu().squeeze().numpy())
            joint_velocities.append(scene["Franka"].data.joint_vel.clone().cpu().squeeze().numpy())

            joint_targets.append(wave_action.clone().cpu().squeeze().numpy())
        t_sim = time.time() - t_sim

        joint_positions = np.stack(joint_positions, axis = 1)[..., :joint_num]
        joint_velocities = np.stack(joint_velocities, axis = 1)[..., :joint_num]
        joint_targets = np.stack(joint_targets, axis = 1)[..., :joint_num]

        real_joint_positions = real_data["qpos"][:, :joint_num]
        real_joint_velocities = real_data["qvel"][:, :joint_num]

        # Normalize by min/max of actual response curves
        pos_min = real_joint_positions.min(axis=0, keepdims=True)
        pos_max = real_joint_positions.max(axis=0, keepdims=True)
        pos_range = pos_max - pos_min
        pos_range = np.where(pos_range < 1e-6, 1.0, pos_range)  # avoid division by zero
        
        vel_min = real_joint_velocities.min(axis=0, keepdims=True)
        vel_max = real_joint_velocities.max(axis=0, keepdims=True)
        vel_range = vel_max - vel_min
        vel_range = np.where(vel_range < 1e-6, 1.0, vel_range)  # avoid division by zero
        
        normalized_real_pos = (real_joint_positions - pos_min) / pos_range
        normalized_real_vel = (real_joint_velocities - vel_min) / vel_range

        # Compute all 4 metrics for each environment
        t_loss = time.time()
        mse_pos_losses = []
        mse_vel_losses = []
        spectral_pos_losses = []
        spectral_vel_losses = []
        
        for i in range(args_cli.num_envs):
            # Normalize simulated data using same normalization as real data
            normalized_sim_pos = (joint_positions[i] - pos_min) / pos_range
            normalized_sim_vel = (joint_velocities[i] - vel_min) / vel_range
            
            # MSE losses
            mse_pos = np.mean((normalized_sim_pos - normalized_real_pos)**2)
            mse_vel = np.mean((normalized_sim_vel - normalized_real_vel)**2)
            
            # Spectral MSE losses
            spectral_pos = spectral_mse(normalized_sim_pos, normalized_real_pos)
            spectral_vel = spectral_mse(normalized_sim_vel, normalized_real_vel)
            
            mse_pos_losses.append(mse_pos)
            mse_vel_losses.append(mse_vel)
            spectral_pos_losses.append(spectral_pos)
            spectral_vel_losses.append(spectral_vel)

        # compute the cost (combined)
        cost = [spectral_pos_losses[i] + spectral_vel_losses[i] for i in range(args_cli.num_envs)]
        costs += cost
        t_loss = time.time() - t_loss


        argidx = np.argmin(costs)

        # make a list of (pd_gain, cost)
        pd_gains_cost = []
        for i in range(args_cli.num_envs):
            pd_gains_cost.append((pd_gains[i], costs[i]))

        t_opt = time.time()
        optimizer.tell(pd_gains_cost)
        t_opt = time.time() - t_opt




        rjps = []
        sjps = []
        jts = []


        rjps.append(real_joint_positions)
        sjps.append(joint_positions)
        jts.append(joint_targets)

        rjps = np.concatenate(rjps, axis = 0)
        sjps = np.concatenate(sjps, axis = 1)
        jts = np.concatenate(jts, axis = 1)

        # print(rjps.shape, sjps.shape, jts.shape)

        valid_costs = costs

        # calculate the min cost
        min_cost = np.min(valid_costs)
        argidx = np.argmin(valid_costs)
        min_pd_gain = pd_gains_cost[argidx][0]
        p_gain = 2**pd_gains[argidx, :joint_num]
        d_gain = 2**pd_gains[argidx, joint_num:2*joint_num]
        armature = pd_gains[argidx, 2*joint_num:3*joint_num]
        friction = pd_gains[argidx, 3*joint_num:4*joint_num]
        dynamic_friction = pd_gains[argidx, 3*joint_num:4*joint_num] * pd_gains[argidx, 4*joint_num:5*joint_num]
        viscous_friction = pd_gains[argidx, 5*joint_num:6*joint_num]

        t_plot = time.time()

        info = {
            "p_gain": p_gain.tolist(),
            "d_gain": d_gain.tolist(),
            "armature": armature.tolist(),
            "friction": friction.tolist(),
            "dynamic_friction": dynamic_friction.tolist(),
            "viscous_friction": viscous_friction.tolist(),
            "body_mass": pd_gains[argidx, 6*joint_num:6*joint_num + 1*body_num].tolist(),
            "body_inertia": pd_gains[argidx, 6*joint_num + 1*body_num:6*joint_num + 1*body_num + 6 * body_num].tolist(),
            "cost": min_cost,
            "mse_pos": mse_pos_losses[argidx],
            "mse_vel": mse_vel_losses[argidx],
            "spectral_mse_pos": spectral_pos_losses[argidx],
            "spectral_mse_vel": spectral_vel_losses[argidx]
        }
        # save as json (pretty print)

        if min_cost < lowest_loss:
            lowest_loss = min_cost
            with open(f"final_grid/{name}/best_seed{rand_run_int}.json", "w") as f:
                json.dump(info, f, indent=4)

        min_pd_gain = pd_gains_cost[argidx][0]
        actual_p_gains = []
        for i in range(joint_num):
            actual_p_gains.append(2**(min_pd_gain[i]))
        actual_d_gains = []
        for i in range(joint_num):
            actual_d_gains.append(2**(min_pd_gain[i+joint_num]))

        # Only plot at the last iteration
        if iii == 199:
            plot = draw_real_plot(name, min_cost, iii, sjps[argidx], jts[argidx], rjps, \
                    actual_p_gains, actual_d_gains, \
                    pd_gains[argidx, 2*joint_num:3*joint_num], \
                    pd_gains[argidx, 3*joint_num:4*joint_num], \
                    pd_gains[argidx, 4*joint_num:5*joint_num], \
                    pd_gains[argidx, 5*joint_num:6*joint_num], joint_num)

            # update the mean and sigma
            plots.append(plot)

            imageio.mimsave(f'final_grid/{name}/plots.mp4', plots, fps=10)
        
        t_plot = time.time() - t_plot
        
        iter_time = time.time() - iter_start
        
        # Update tqdm description with timing and loss info
        from tqdm import tqdm as tqdm_main
        desc = f"{name} | MSE Pos: {mse_pos_losses[argidx]:.5f} | MSE Vel: {mse_vel_losses[argidx]:.5f} | Spectral Pos: {spectral_pos_losses[argidx]:.5f} | Spectral Vel: {spectral_vel_losses[argidx]:.5f} | Loss: {min_cost:.5f}"
        pbar.set_description(desc)



def spectral_mse(x, y):
    """
    Compute the mean squared error between the Fourier spectra
    of two multi-dimensional trajectories.

    Args:
        x: np.ndarray, shape (t, D)
        y: np.ndarray, shape (T, D)

    Returns:
        float: average spectral MSE across all dimensions.
    """
    # FFT along the time axis (axis 0)
    Xf = np.fft.fft(x, axis=0)
    Yf = np.fft.fft(y, axis=0)

    # Compute squared magnitude difference
    mse_per_dim = np.mean(np.abs(Xf - Yf)**2, axis=0)

    # wegithed sum
    weights = np.array([1] * x.shape[1])

    # Average over all dimensions
    return np.sum(mse_per_dim * weights) / np.sum(weights)


def draw_real_plot(name, cost, generation, joint_positions, joint_targets, real_joint_positions, \
    actual_p_gains, actual_d_gains, actual_armature, actual_friction, actual_dynamic_friction, actual_viscous_friction, num_joints):
    print(joint_positions.shape, joint_targets.shape, real_joint_positions.shape)
    fig, axs = plt.subplots(num_joints, 1, figsize=(20, 20))
    for i in range(num_joints):
        axs[i].plot(joint_positions[:, i], label = "joint_positions", linewidth=2)
        axs[i].plot(real_joint_positions[:, i], label = "real_joint_positions", linewidth=2)
        axs[i].plot(joint_targets[:, i], label = "qdes", linestyle = "--", linewidth=2)
        axs[i].set_ylabel(f"Joint {i+1}")
    # add the actual p and d gains below the plot # :.2f for every element in the list
    p_gains_txt = f"Actual P gains: {', '.join([f'{x:.2f}' for x in actual_p_gains])}"
    d_gains_txt = f"Actual D gains: {', '.join([f'{x:.2f}' for x in actual_d_gains])}"
    armature_txt = f"Actual armature: {', '.join([f'{x:.2f}' for x in actual_armature])}"
    friction_txt = f"Actual friction coefficient: {', '.join([f'{x:.2f}' for x in actual_friction])}"
    dynamic_friction_txt = f"Actual dynamic friction coefficient: {', '.join([f'{x:.2f}' for x in actual_dynamic_friction])}"
    viscous_friction_txt = f"Actual viscous friction coefficient: {', '.join([f'{x:.2f}' for x in actual_viscous_friction])}"
    plt.figtext(0.5, 0.01, p_gains_txt, ha='center', fontsize=12)
    plt.figtext(0.5, 0.02, d_gains_txt, ha='center', fontsize=12)
    plt.figtext(0.5, 0.03, armature_txt, ha='center', fontsize=12)
    plt.figtext(0.5, 0.04, friction_txt, ha='center', fontsize=12)
    plt.figtext(0.5, 0.05, dynamic_friction_txt, ha='center', fontsize=12)
    plt.figtext(0.5, 0.06, viscous_friction_txt, ha='center', fontsize=12)
    plt.legend()
    axs[0].set_title(f"Generation {generation} | Cost: {cost:.5f}")
    plt.savefig(f"final_grid/{name}/plot.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

    # tight layout
    plt.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    data = data[..., 1:4]
    plt.close()
    return data


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt = 0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator

    # kps = [ 16, 32, 64, 128, 256, 512 ]
    kps = [ 16, 32, 64, 128, 160, 256, 512 ]
    kds = [ 1, 2, 4, 8, 12, 16, 24 ]

    for kp in kps:
        for kd in kds:
            name = f"K{kp}_D{kd}"

            run_simulator(name, sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()