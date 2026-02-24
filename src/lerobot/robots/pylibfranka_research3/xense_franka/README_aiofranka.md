# aiofranka

<div align="center">
  <img width="340" src="assets/image.png">
</div>
<p align="center">
  <a href="https://pypi.org/project/aiofranka/">
    <img src="https://img.shields.io/pypi/v/aiofranka" alt="CI">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="CI">
  </a>
</p>

**aiofranka** is an asyncio-based Python library for controlling Franka Emika robots. It provides a high-level, asynchronous interface that combines **`pylibfranka`** for official low-level control interface (1kHz torque control), **`MuJoCo`** for kinematics/dynamics computation, **`Ruckig`** for  smooth trajectory generation.

The library is designed for research applications requiring precise, real-time control with minimal latency and maximum flexibility.

📚 **[Documentation](https://improbableai.com/aiofranka)**





## Installation

Make sure you can access Franka Desk GUI from your machine's browser by typing in the robot's IP (e.g. 172.16.0.2). Then, install: 


```bash
pip install aiofranka
```

Or for development:
```bash
git clone https://github.com/Improbable-AI/aiofranka.git
cd aiofranka
pip install -e .
```

## Quick Start

```bash 
python test.py 
```

Basic usage pattern:

```python
import asyncio 
import numpy as np 
from aiofranka import RobotInterface, FrankaController

async def main():
    # Connect to robot (use IP address for real robot, None for simulation)
    robot = RobotInterface("172.16.0.2") 
    controller = FrankaController(robot)
    
    # Start the 1kHz control loop
    await controller.start()

    # Test connection quality
    await controller.test_connection()

    # Move to home position using smooth trajectory
    await controller.move([0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853])

    # Switch to impedance control
    controller.switch("impedance")
    controller.kp = np.ones(7) * 80.0
    controller.kd = np.ones(7) * 4.0
    controller.set_freq(50)  # 50Hz update rate for set() commands
    
    for cnt in range(100): 
        delta = np.sin(cnt / 50.0 * np.pi) * 0.1
        init = controller.initial_qpos
        await controller.set("q_desired", delta + init)

    # Switch to operational space control (OSC)
    controller.switch("osc")
    controller.set_freq(50)  

    for cnt in range(100): 
        delta = np.sin(cnt / 50.0 * np.pi) * 0.1
        init = controller.initial_ee 

        desired_ee = np.eye(4) 
        desired_ee[:3, :3] = init[:3, :3]
        desired_ee[:3, 3] = init[:3, 3] + np.array([0, delta, 0])

        await controller.set("ee_desired", desired_ee)

    # Stop control loop
    await controller.stop()

if __name__ == "__main__":
    asyncio.run(main()) 
```

## Core Concepts

### Asyncio-based Design

The library uses Python's `asyncio` for non-blocking control. The control loop runs at 1kHz in the background while your code sends commands asynchronously:

```python
# Control loop runs in background at 1kHz
await controller.start()

# Your code can await other operations without blocking the control loop
await asyncio.sleep(1.0)
await controller.set("q_desired", target)
```

### Rate Limiting

Use `set_freq()` to enforce strict timing for command updates:

```python
controller.set_freq(50)  # Set 50Hz update rate

# This will automatically sleep to maintain 50Hz timing
for i in range(100):
    await controller.set("q_desired", compute_target())
```


### State Access

Robot state is continuously updated at 1kHz and accessible via `controller.state`:

```python
state = controller.state  # Thread-safe access
# Contains: qpos, qvel, ee, jac, mm, last_torque
print(f"Joint positions: {state['qpos']}")
print(f"End-effector pose: {state['ee']}")  # 4x4 homogeneous transform
```

## Controllers

### 1. Impedance Control (Joint Space)

Controls joint positions with spring-damper behavior:

```python
controller.switch("impedance")
controller.kp = np.ones(7) * 80.0   # Position gains
controller.kd = np.ones(7) * 4.0    # Damping gains

await controller.set("q_desired", target_joint_angles)
```

**Use case**: Precise joint-space motions, compliant behavior


### 2. Operational Space Control (Task Space)

Controls end-effector pose in Cartesian space:

```python
controller.switch("osc")
controller.ee_kp = np.array([300, 300, 300, 1000, 1000, 1000])  # [xyz, rpy]
controller.ee_kd = np.ones(6) * 10.0

desired_ee = np.eye(4)  # 4x4 homogeneous transform
desired_ee[:3, 3] = [0.4, 0.0, 0.5]  # Position
await controller.set("ee_desired", desired_ee)
```

**Use case**: Cartesian trajectories, end-effector tracking



## License

MIT License - see LICENSE file

## Citation

If you use this library in your research, please cite:

```bibtex
@software{aiofranka,
  author = {Improbable AI Lab},
  title = {aiofranka: Asyncio-based Franka Robot Control},
  year = {2025},
  url = {https://github.com/Improbable-AI/aiofranka}
}
```

## Acknowledgments

- Built on [libfranka](https://frankarobotics.github.io/docs/) by Franka Emika
- Uses [MuJoCo](https://mujoco.org/) physics engine
- Trajectory generation with [Ruckig](https://github.com/pantor/ruckig)