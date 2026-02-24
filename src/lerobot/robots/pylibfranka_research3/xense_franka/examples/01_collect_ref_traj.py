"""
This script collects reference trajectory data under different impedance control gains.
It moves the Franka robot arm in a sinusoidal pattern while logging joint positions, velocities,
desired positions, and control torques. The collected data is saved to a .npz file for further analysis.
"""


import asyncio 
import numpy as np 
from xense_franka.robot import RobotInterface
from xense_franka import FrankaController
import time 
import os 
import matplotlib.pyplot as plt


async def main():
    robot = RobotInterface("172.16.0.2") 
    # robot = RobotInterface()
    controller = FrankaController(robot)

    await controller.start()


    base = np.array([1, 1, 1, 1, 0.6, 0.6, 0.6])

    # kps = [ 16, 32, 64, 128, 256, 512 ]
    kps = [ 160 ]
    kds = [ 1, 2, 4, 8, 12, 16, 24 ]

    kp_kd_pairs = [ (kp, kd) for kp in kps for kd in kds ]

    
    for kp, kd in kp_kd_pairs:
        
        with controller.state_lock:
            controller.kp = base * 80
            controller.kd = base * 4
            print("Moving to initial position...")
        await controller.move([0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853])


        print(f"Testing with kp={kp}, kd={kd}")

        await asyncio.sleep(2.0)

        # run the controller test 
        controller.switch("impedance")
        controller.set_freq(50)

        with controller.state_lock:
            controller.kp = base * kp
            controller.kd = base * kd

        logs = { 
            'qpos': [], 
            'qvel': [],
            'qdes': [], 
            'ctrl': [], 
        }


        for cnt in range(200): 

            logs['qpos'].append(controller.robot.data.qpos.copy())
            logs['qvel'].append(controller.robot.data.qvel.copy())
            logs['ctrl'].append(controller.robot.data.ctrl.copy())
            logs['qdes'].append(controller.q_desired.copy())

            delta = np.sin(cnt / 50.0 * np.pi) * 0.15
            init = controller.initial_qpos
            await controller.set("q_desired", delta + init)

        await asyncio.sleep(1.0)

        for key in logs.keys():
            logs[key] = np.stack(logs[key])
        
        os.makedirs("./examples/sysid_left_more/", exist_ok=True)
        np.savez(f"./examples/sysid_left_more/sysid_K{int(kp)}_D{int(kd)}.npz", **logs)
        
if __name__ == "__main__":
    asyncio.run(main())