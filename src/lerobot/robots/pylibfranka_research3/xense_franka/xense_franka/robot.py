import os
from pathlib import Path
import numpy as np 
import time
import requests 
from xense_franka.client import FrankaLockUnlock
from scipy.spatial.transform import Rotation

CUR_DIR = Path(__file__).parent.resolve()


class RobotInterface: 
    """
    High-level interface for Franka FR3 robot control.
    
    This class provides a unified interface for real robot control (via pylibfranka).
    It handles low-level communication, state synchronization, and kinematics/dynamics
    computation using pylibfranka's model.
    
    Attributes:
        real (bool): True if connected to real robot
        robot (pylibfranka.Robot): Real robot interface
        model: Robot model for kinematics/dynamics (from pylibfranka)
        torque_controller: Active torque control interface
        
    Examples:
        Real robot:
            >>> robot = RobotInterface("172.16.0.2")
            >>> robot.start()
            >>> state = robot.state
            >>> robot.step(np.zeros(7))  # Send zero torques
            >>> robot.stop()
        
    Caveats:
        - Must call start() before step() on real robot
        - Collision behavior is set to high thresholds by default
        - State is synced from robot on every access (thread-safe)
    """

    def __init__(self, ip: str): 
        """
        Initialize robot interface.
        
        Args:
            ip (str): Robot IP address (e.g., "172.16.0.2") for real robot.
                           
        Raises:
            RuntimeError: If pylibfranka is not installed
            ConnectionError: If cannot connect to robot at given IP
            
        Note:
            Collision thresholds are set to [100.0] * 7 for joints
            and [100.0] * 6 for Cartesian space. Adjust via robot.robot.set_collision_behavior()
            for more conservative behavior.
        """

        self.real = True
        self.torque_controller = None
        self._robot_state = None

        import pylibfranka
        self.robot = pylibfranka.Robot(ip, pylibfranka.RealtimeConfig.kIgnore)
        self.model = self.robot.load_model()

        self.robot.set_collision_behavior(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        )
        
        self._sync_state() 
        
    def start(self): 
        """
        Start torque control mode on the real robot.
        
        This must be called before sending torque commands.
        
        Raises:
            RuntimeError: If robot is not ready or already in control mode
            
        Caveat:
            After calling start(), you must send torque commands at ~1kHz to maintain
            control. Use FrankaController for automatic control loop management.
        """
        self.torque_controller = self.robot.start_torque_control()

    def stop(self): 
        """
        Stop torque control mode on the real robot.
        
        This gracefully terminates the control session.
        
        Caveat:
            Robot will hold position briefly then release brakes. Ensure robot
            is in a safe configuration before stopping.
        """
        self.robot.stop()

    def _sync_state(self): 
        """Sync state from real robot"""
        if self.torque_controller is None:
            self._robot_state = self.robot.read_once()
        else:
            self._robot_state, _ = self.torque_controller.readOnce()

    @property 
    def state(self): 
        """
        Get current robot state with kinematics and dynamics.
        
        Returns:
            dict: Dictionary containing:
                - qpos (np.ndarray): Joint positions [rad] (7,)
                - qvel (np.ndarray): Joint velocities [rad/s] (7,)
                - ee (np.ndarray): End-effector pose as 4x4 homogeneous transform
                                  [[R, p], [0, 1]] where R is rotation, p is position
                - jac (np.ndarray): End-effector Jacobian (6, 7) - [linear; angular]
                - mm (np.ndarray): Joint-space mass matrix (7, 7)
                - coriolis (np.ndarray): Coriolis forces [Nm] (7,)
                - last_torque (np.ndarray): Last commanded torques [Nm] (7,)
                - ext_wrench (np.ndarray): External wrench [N, N, N, Nm, Nm, Nm] (6,)
                - robot_state: Raw robot state from pylibfranka
                
        Note:
            State is synchronized from real robot on every access.
            Uses pylibfranka model for kinematics/dynamics computation.
            
        Example:
            >>> state = robot.state
            >>> print(f"Joint 1 position: {state['qpos'][0]:.3f} rad")
            >>> print(f"EE position: {state['ee'][:3, 3]}")
            >>> print(f"EE orientation: {state['ee'][:3, :3]}")
        """
        self._sync_state()
        
        robot_state = self._robot_state
        
        # Get end-effector pose from robot state
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4).T
        
        # Get Jacobian from model (6x7, column-major -> need to reshape)
        jac = np.array(self.model.zero_jacobian(robot_state)).reshape(6, 7, order='F')
        
        # Get mass matrix from model
        mm = np.array(self.model.mass(robot_state)).reshape(7, 7, order='F')
        
        # Get coriolis forces from model
        coriolis = np.array(self.model.coriolis(robot_state))

        state = { 
            "qpos": np.array(robot_state.q),
            "qvel": np.array(robot_state.dq), 
            "O_T_EE": np.array(robot_state.O_T_EE),
            "ee": O_T_EE,
            "jac": jac,
            "mm": mm,
            "coriolis": coriolis,
            "last_torque": np.array(robot_state.tau_J_d),
            "ext_wrench": np.array(robot_state.O_F_ext_hat_K),
            "robot_state": robot_state,
        }

        return state

    def step(self, torque: np.ndarray): 
        """
        Send torque command to robot.
        
        Args:
            torque (np.ndarray): Joint torques [Nm] (7,)
            
        Raises:
            RuntimeError: If robot not started or communication error
            
        Note:
            Sends torque command via pylibfranka at current timestep.
            
        Caveats:
            - Must be called at ~1kHz to maintain control
            - Large torque changes may trigger safety limits
            - Torques should respect robot limits: |tau_i| < 87 Nm for joints 1-4,
              |tau_i| < 12 Nm for joints 5-7
              
        Example:
            >>> robot.step(np.zeros(7))  # Send zero torques
        """
        import pylibfranka
        torque_command = pylibfranka.Torques(torque.tolist())
        torque_command.motion_finished = False
        self.torque_controller.writeOnce(torque_command)


if __name__ == "__main__": 

    robot = RobotInterface("192.168.99.111")
    while True: 

        # zero_torque = np.zeros(7)
        # robot.step(zero_torque)
        # time.sleep(0.1)
        # print(zero_torque)
        state = robot.state
        print("qpos:", state["qpos"])
        print("qvel:", state["qvel"])
        print("O_T_EE:", state["O_T_EE"])
        print("ext_wrench:", state["ext_wrench"])
        print("last_torque:", state["last_torque"])
        time.sleep(0.5)