"""
This script collects reference trajectory data under different impedance control gains.
It moves the Franka robot arm in a sinusoidal pattern while logging joint positions, velocities,
desired positions, and control torques. The collected data is saved to a .npz file for further analysis.
"""


import asyncio 
import numpy as np 
from xense_franka.robot import RobotInterface
from xense_franka import FrankaController


async def main():
    robot = RobotInterface(None) 
    # robot = RobotInterface()
    controller = FrankaController(robot)

    # save robot pose
    state = robot.state
    achieved_real= state['qpos']
    print(achieved_real)
    achieved_vels_real= state['qvel']
    print(achieved_vels_real)
    achieved_torques_real= state['last_torque']
    print(achieved_torques_real)
    ee = state['ee']
    print(ee)
        
if __name__ == "__main__":
    asyncio.run(main())