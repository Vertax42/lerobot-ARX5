import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import sys 
sys.path.append(".")
import threading
import asyncio
import time
from tqdm import trange
from pathlib import Path
import numpy as np 
import time 
from xense_franka.robot import RobotInterface
from ruckig import InputParameter, Ruckig, Trajectory, Result

CUR_DIR = Path(__file__).parent.resolve()



class FrankaController: 
    """
    High-level asyncio controller for Franka robots with multiple control modes.
    
    This controller runs a 1kHz torque control loop in the background using asyncio,
    while allowing you to send high-level commands asynchronously. Supports three
    control modes:
    
    1. Impedance Control: Joint-space spring-damper control
    2. Operational Space Control (OSC): Task-space control with null-space
    3. Direct Torque Control: Raw torque commands
    
    The controller automatically handles:
    - Background control loop at 1kHz
    - Torque rate limiting for safety
    - Thread-safe state updates
    - Rate-limited command updates
    - Smooth trajectory generation
    
    Attributes:
        robot (RobotInterface): Robot interface instance
        type (str): Current controller type ("impedance", "osc", "torque")
        running (bool): Whether control loop is active
        state (dict): Current robot state (updated at 1kHz)
        
        # Impedance control gains
        kp (np.ndarray): Joint position stiffness [Nm/rad] (7,)
        kd (np.ndarray): Joint damping [Nm⋅s/rad] (7,)
        
        # OSC gains
        ee_kp (np.ndarray): EE stiffness [N/m for xyz, Nm/rad for rpy] (6,)
        ee_kd (np.ndarray): EE damping [N⋅s/m for xyz, Nm⋅s/rad for rpy] (6,)
        null_kp (np.ndarray): Null-space stiffness [Nm/rad] (7,)
        null_kd (np.ndarray): Null-space damping [Nm⋅s/rad] (7,)
        
        # Target states
        q_desired (np.ndarray): Desired joint positions [rad] (7,)
        ee_desired (np.ndarray): Desired EE pose as 4x4 transform
        torque (np.ndarray): Direct torque command [Nm] (7,)
        
        # Safety
        clip (bool): Enable torque rate limiting (default: True)
        torque_diff_limit (float): Max torque rate [Nm/s] (default: 990)
        
    Examples:
        Basic usage:
            >>> robot = RobotInterface("172.16.0.2")
            >>> controller = FrankaController(robot)
            >>> await controller.start()
            >>> controller.switch("impedance")
            >>> controller.set_freq(50)
            >>> await controller.set("q_desired", target_joints)
            >>> await controller.stop()
        
        OSC control:
            >>> controller.switch("osc")
            >>> controller.ee_kp = np.array([300, 300, 300, 1000, 1000, 1000])
            >>> controller.set_freq(50)
            >>> desired_ee = np.eye(4)
            >>> desired_ee[:3, 3] = [0.4, 0.0, 0.5]
            >>> await controller.set("ee_desired", desired_ee)
        
        Direct torque:
            >>> controller.switch("torque")
            >>> controller.torque = np.zeros(7)  # Zero torques
        
    Caveats:
        - Must await controller.start() before sending commands
        - Use set_freq() before set() to enforce timing
        - High gains can cause instability or safety triggers
        - Switching controllers resets initial state
        - State access is thread-safe but copy if modifying
        - Control loop must run continuously at ~1kHz
    """

    def __init__(self, robot: RobotInterface):
        """
        Initialize controller with robot interface.
        
        Args:
            robot (RobotInterface): Initialized robot interface
            
        Note:
            Controller is initialized in "impedance" mode with conservative gains.
            Call start() to begin the control loop, then switch() to change modes.
            
        Default Gains:
            - Impedance: kp=80, kd=4 (per joint)
            - OSC: ee_kp=[100]*6, ee_kd=[4]*6
            - Null-space: null_kp=1, null_kd=1
        """
        self.robot = robot

        self.initialize() 
        self.state_lock = threading.Lock()

        self.type = "impedance"
        self.running = False
        self.task = None
        self.clip = True 


        self.kp, self.kd = np.ones(7) * 80, np.ones(7) * 4
        self.ki = np.ones(7) * 0.1  # Integral gains
        self.error_integral = np.zeros(7)  # Accumulated error
        self.ee_kp, self.ee_kd = np.ones(6) * 100, np.ones(6) * 4
        self.null_kp, self.null_kd = np.ones(7) * 1, np.ones(7) * 1


        self.track = False 
        self.torque_diff_limit = 990.
        self.torque_limit = np.array([87, 87, 87, 87, 12, 12, 12])  # Nm
        
        # Rate limiting for .set() method
        self._update_freq = 50.0  # Default 50Hz
        self._last_update_time = {}
        self._pending_updates = {}

        self.state = None 

        self.verbose = False


    async def test_connection(self): 
        """
        Test control loop timing and diagnose connection quality.
        
        Runs for 5 seconds and prints statistics about control loop performance:
        - Actual frequency (should be ~1000 Hz)
        - Mean/std/min/max loop time
        - Jitter (max - min)
        
        Use this to verify your setup is working correctly before running experiments.
        
        Example Output:
            Control loop stats (last 1000 iterations):
              Frequency: 1000.2 Hz (target: 1000 Hz)
              Mean dt: 1.000 ms, Std: 0.015 ms
              Min dt: 0.985 ms, Max dt: 1.025 ms
              Jitter (max-min): 0.040 ms
              
        Caveats:
            - High jitter (>0.5ms) indicates system load or network issues
            - Frequency <990 Hz suggests performance problems
            - Run on a dedicated realtime system for best results
        """

        self.track = True 
        await asyncio.sleep(5)
        self.track = False

    def initialize(self): 

        # Get initial state
        initial_state = self.robot.state
        self.initial_ee = initial_state['ee']
        self.initial_qpos = deepcopy(initial_state['qpos'])
        self.initial_qvel = deepcopy(initial_state['qvel'])
        self.last_torque = initial_state['last_torque']

        self.q_desired= self.initial_qpos
        self.ee_desired = self.initial_ee

    def _update_desired(self, desired):
        """
        Update the desired joint positions, kp and kd
        This function is called by the server when a client sends new values
        Thread-safe update of shared state.
        
        Args:
            desired: Desired joint positions (7-element array)
            kp: Joint position stiffness gains (7-element array)
            kd: Joint velocity damping gains (7-element array)
        """
        with self.state_lock:
            self.q_desired = np.array(desired) if type(desired) == list else desired
    
    def set_freq(self, freq: float):
        """
        Set the update frequency for rate-limited set() calls.
        
        This enforces strict timing for subsequent set() calls, automatically
        sleeping to maintain the specified frequency. Prevents sending commands
        too fast and ensures smooth, consistent control.
        
        Args:
            freq (float): Desired update frequency in Hz (typically 10-100 Hz)
            
        Note:
            Must be called BEFORE the first set() call to take effect.
            Each attribute tracked by set() has independent timing.
            
        Examples:
            >>> controller.set_freq(50)  # 50 Hz updates
            >>> for i in range(100):
            ...     await controller.set("q_desired", compute_target())
            ...     # Automatically sleeps to maintain 50 Hz
            
        Caveats:
            - Don't set freq > 200 Hz (unnecessary and may cause timing issues)
            - Lower freq = smoother motion but slower response
            - Higher freq = faster response but requires more computation
            - Timing is per-attribute (q_desired and ee_desired tracked separately)
        """
        self._update_freq = freq
    
    async def set(self, attr: str, value):
        """
        Rate-limited setter that enforces strict timing for control updates.
        
        This method ensures updates to controller attributes happen at the
        frequency specified by set_freq(). It compensates for drift by tracking
        the target time for each update, guaranteeing consistent timing even
        if your computation time varies.
        
        Args:
            attr (str): Attribute name to set. Common values:
                       - "q_desired": Joint position target (impedance mode)
                       - "ee_desired": End-effector pose target (OSC mode)
                       - "torque": Direct torque command (torque mode)
            value: Value to set. Type depends on attr:
                  - q_desired: np.ndarray (7,) [rad]
                  - ee_desired: np.ndarray (4, 4) homogeneous transform
                  - torque: np.ndarray (7,) [Nm]
                  
        Note:
            - Automatically sleeps to maintain frequency set by set_freq()
            - First call for an attribute initializes timing
            - Each attribute has independent timing tracking
            - Thread-safe update of shared state
            
        Examples:
            Impedance control:
                >>> controller.set_freq(50)
                >>> for i in range(100):
                ...     target = initial_q + np.sin(i / 50.0 * np.pi) * 0.1
                ...     await controller.set("q_desired", target)
            
            OSC control:
                >>> controller.set_freq(100)
                >>> desired_ee = np.eye(4)
                >>> desired_ee[:3, 3] = [0.5, 0.0, 0.3]
                >>> await controller.set("ee_desired", desired_ee)
            
        Caveats:
            - Must call set_freq() before first set() call
            - If computation takes longer than 1/freq, timing will slip
            - Don't mix set() and direct attribute assignment
            - Don't call set() faster than the specified frequency
        """
        current_time = time.perf_counter()
        dt = 1.0 / self._update_freq
        
        # Initialize tracking for this attribute if first time
        if attr not in self._last_update_time:
            self._last_update_time[attr] = current_time
            # Sleep for the first update too to maintain consistent timing
            await asyncio.sleep(dt)
            with self.state_lock:
                setattr(self, attr, value)
            self._last_update_time[attr] = current_time + dt
            return
        
        # Calculate target time for this update
        target_time = self._last_update_time[attr] + dt
        
        # Sleep until target time
        sleep_time = target_time - current_time
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        
        with self.state_lock:
            setattr(self, attr, value)
        
        # Update last update time to target (not actual) to avoid drift
        self._last_update_time[attr] = target_time


    async def _run(self): 
        """Run the control loop continuously in the background"""
        self.running = True
        
        loop_times = []
        last_time = time.perf_counter()
        log_interval = 1000  # Log every 1000 iterations (1 second at 1000Hz)
        iteration = 0
        
        try:
            while self.running: 

                
                t0 = time.time() 
                self.step()
                
                # Track timing
                if self.track:
                    current_time = time.perf_counter()
                    dt = current_time - last_time
                    loop_times.append(dt)
                    last_time = current_time
                    iteration += 1
                    
                    # Log statistics every log_interval iterations
                    if iteration % log_interval == 0:
                        loop_times_array = np.array(loop_times)
                        mean_dt = np.mean(loop_times_array) * 1000  # Convert to ms
                        std_dt = np.std(loop_times_array) * 1000
                        min_dt = np.min(loop_times_array) * 1000
                        max_dt = np.max(loop_times_array) * 1000
                        actual_freq = 1.0 / np.mean(loop_times_array)
                        
                        print(f"Control loop stats (last {log_interval} iterations):")
                        print(f"  Frequency: {actual_freq:.1f} Hz (target: 1000 Hz)")
                        print(f"  Mean dt: {mean_dt:.3f} ms, Std: {std_dt:.3f} ms")
                        print(f"  Min dt: {min_dt:.3f} ms, Max dt: {max_dt:.3f} ms")
                        print(f"  Jitter (max-min): {max_dt - min_dt:.3f} ms")
                        
                        loop_times.clear()

                dt = time.time() - t0
                await asyncio.sleep(0)  # Yield control to event loop
        except Exception as e:
            self.running = False
            print(f"Error in control loop: {e}")
            # diff = (self.torque - self.last_torque)/1e-3
            # diff = np.abs(diff)

            # if np.any(diff > 1000.):
            #     # print what axis is causing the issue
            #     arg_idxs = np.where(diff > 1000.)[0]
            #     print(f"High torque rate of change detected on axes: {arg_idxs}")
            #     print((self.torque - self.last_torque)/1e-3) 
            sys.exit(1)  # Kill the entire script
    
    async def start(self):
        """
        Start the 1kHz background control loop.
        
        This creates an asyncio task that runs the control loop continuously
        at ~1000 Hz. The loop reads robot state, computes control torques based
        on the current controller type, and sends torque commands.
        
        Returns:
            asyncio.Task: The background control loop task
            
        Note:
            - Blocks for 1 second to ensure loop starts successfully
            - Starts robot torque control mode automatically
            - Control loop runs until stop() is called
            - You can send commands via set() while loop is running
            
        Example:
            >>> await controller.start()
            >>> # Control loop now running in background
            >>> controller.set_freq(50)
            >>> await controller.set("q_desired", target)
            >>> await controller.stop()
            
        Caveats:
            - Must be awaited (async function)
            - Don't call start() multiple times without stop()
            - If loop crashes, robot will trigger safety stop
            - Check terminal for error messages if robot stops unexpectedly
        """

        print("starting robot!")
        self.robot.start()

        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())
        await asyncio.sleep(1)  # Yield to ensure the task starts
        return self.task
    
    async def stop(self):
        """
        Stop the background control loop and robot.
        
        Gracefully terminates the control loop task and stops robot torque control.
        Robot will hold position briefly then release brakes.
        
        Note:
            - Blocks for 1 second to ensure clean shutdown
            - Cancels asyncio control loop task
            - Stops robot torque control mode
            - Safe to call multiple times
            
        Example:
            >>> await controller.start()
            >>> # ... do control ...
            >>> await controller.stop()
            
        Caveat:
            Ensure robot is in a safe configuration before stopping. Robot
            will briefly hold position then release brakes.
        """
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                print("Control loop task cancelled.") 

        self.robot.stop()
        print("robot stopped")
        await asyncio.sleep(1)  # Yield to ensure the task starts

    def switch(self, controller_type: str):
        """
        Switch between control modes at runtime.
        
        Changes the active controller without stopping the control loop. Resets
        the initial state (initial_qpos, initial_ee) to current robot state and
        clears rate-limiting timing.
        
        Args:
            controller_type (str): Controller type to switch to:
                - "impedance": Joint-space impedance control
                - "pid": Joint-space PID control with integral term
                - "osc": Operational space control (task space)
                - "torque": Direct torque control
                
        Example:
            >>> controller.switch("impedance")
            >>> controller.kp = np.ones(7) * 80
            >>> # ... run impedance control ...
            >>> 
            >>> controller.switch("osc")
            >>> controller.ee_kp = np.array([300, 300, 300, 1000, 1000, 1000])
            >>> # ... run OSC control ...
            
        Note:
            - Can be called while control loop is running
            - Resets q_desired to current position
            - Resets ee_desired to current end-effector pose
            - Clears timing state from previous set() calls
            - Resets integral term when switching to/from PID
            
        Caveats:
            - Switching causes brief discontinuity in control
            - Adjust gains after switching for smooth transition
            - Don't switch rapidly (< 1 Hz) as it resets state
        """
        self.type = controller_type
        self.initialize()
        # Reset integral term when switching controllers
        self.error_integral = np.zeros(7)
        # Reset timing state when switching controllers
        self._last_update_time.clear()

        if self.verbose:
            print("==================================")
            print(f"Switched to {controller_type} controller.")
            print("==================================")

    def step(self): 
        self.state = self.robot.state
        if self.type == "impedance":
            self._impedance_step(self.state)
        elif self.type == "pid":
            self._pid_step(self.state)
        elif self.type == "osc":
            self._osc_step(self.state)
        elif self.type == "torque":
            self._torque_step(self.state)
        else:
            raise ValueError(f"Unknown controller type: {self.type}")

    def _torque_step(self, state):

        self.robot.step(self.torque)

    def _osc_step(self, state): 
        """
        Operational Space Control (Cartesian impedance control).
        
        Uses the same implementation as pylibfranka_controllers for computing
        Cartesian impedance torques with proper orientation error handling.
        """
        jac = state['jac']
        ee = state['ee']
        q = state['qpos']
        dq = state['qvel']
        mm = state['mm']
        last_torque = state['last_torque']
        coriolis = state['coriolis']

        with self.state_lock:
            ee_goal = self.ee_desired

        # Current position and orientation
        position = ee[:3, 3]
        orientation = R.from_matrix(ee[:3, :3])
        
        # Desired position and orientation
        position_d = ee_goal[:3, 3]
        orientation_d = R.from_matrix(ee_goal[:3, :3])

        # Compute 6D error
        error = np.zeros(6)
        error[:3] = position - position_d  # Position error
        
        # Orientation error using quaternion (same as pylibfranka_controllers)
        orientation_quat = orientation.as_quat()  # [qx, qy, qz, qw]
        orientation_d_quat = orientation_d.as_quat()
        
        # Handle quaternion sign ambiguity
        if np.dot(orientation_d_quat, orientation_quat) < 0.0:
            orientation_quat = -orientation_quat
        
        orientation_corrected = R.from_quat(orientation_quat)
        error_quaternion = orientation_corrected.inv() * orientation_d
        error_quat = error_quaternion.as_quat()
        error[3:] = error_quat[:3]  # Vector part of quaternion
        error[3:] = -ee[:3, :3] @ error[3:]  # Transform to base frame

        # Build stiffness and damping matrices
        cartesian_stiffness = np.diag(self.ee_kp)
        cartesian_damping = np.diag(self.ee_kd)
        
        # Compute task-space control torque (same as official pylibfranka_controllers)
        tau_task = jac.T @ (-cartesian_stiffness @ error - 
                           cartesian_damping @ (jac @ dq))

        # Total torque with coriolis compensation
        tau_d = tau_task + coriolis

        # Torque rate limiting
        if self.clip:
            diff = (tau_d - last_torque) / 1e-3
            diff = np.clip(diff, -self.torque_diff_limit, self.torque_diff_limit)
            tau_d = last_torque + diff * 1e-3

        self.robot.step(tau_d)

    


    def _pid_step(self, robot_state):
        """
        PID control in joint space with integral term for steady-state error.
        
        Computes: τ = -Kp*(q - q_d) + Ki*∫e*dt - Kd*dq + coriolis
        (Based on official impedance example with integral term)
        """
        # Get state variables
        q = np.array(robot_state['qpos'])
        dq = np.array(robot_state['qvel'])
        last_torque = robot_state['last_torque']
        coriolis = robot_state['coriolis']

        # Get current target (thread-safe)
        with self.state_lock:
            kp = self.kp
            ki = self.ki
            kd = self.kd
            q_goal = self.q_desired
    
        # Compute error (same convention as official example)
        position_error = q - q_goal
        
        # Update integral term (dt = 1ms = 0.001s)
        # Note: integral uses negative error for consistency
        self.error_integral += (-position_error) * 1e-3
        
        # Anti-windup: clamp integral term
        integral_limit = 10.0  # Nm*s (adjust as needed)
        self.error_integral = np.clip(self.error_integral, -integral_limit, integral_limit)

        # Compute PID control with coriolis compensation
        tau_task = -kp * position_error + ki * self.error_integral - kd * dq
        tau_d = tau_task + coriolis
        
        # Torque rate limiting
        if self.clip:
            diff = (tau_d - last_torque) / 1e-3
            diff = np.clip(diff, -self.torque_diff_limit, self.torque_diff_limit)
            tau_d = last_torque + diff * 1e-3

        self.torque = tau_d
        self.robot.step(tau_d)

    def _impedance_step(self, robot_state): 
        """
        Joint-space impedance control with coriolis compensation.
        
        Computes: τ = -Kp*(q - q_d) - Kd*dq + coriolis
        (Same as official joint_impedance_example.py)
        """
        # Get state variables
        q = np.array(robot_state['qpos'])
        dq = np.array(robot_state['qvel'])
        last_torque = robot_state['last_torque']
        coriolis = robot_state['coriolis']

        # Get current target from trajectory (thread-safe)
        with self.state_lock:
            kp = self.kp
            kd = self.kd
            q_goal = self.q_desired
    
        # Compute error to desired equilibrium joint configuration
        position_error = q - q_goal

        # Compute joint-space impedance control (same as official example)
        tau_task = -kp * position_error - kd * dq

        # Add coriolis compensation
        tau_d = tau_task + coriolis
        
        # Torque rate limiting
        if self.clip:
            diff = (tau_d - last_torque) / 1e-3
            diff = np.clip(diff, -self.torque_diff_limit, self.torque_diff_limit)
            tau_d = last_torque + diff * 1e-3
            tau_d = np.clip(tau_d, -self.torque_limit, self.torque_limit)

        self.torque = tau_d

        self.robot.step(tau_d)


    async def move(self, qpos = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.7853],
                   vel = np.ones(7) * 0.1,
                   acc = np.ones(7) * 0.5):
        """
        Move robot to target joint position using smooth trajectory.
        
        Generates a time-optimal, jerk-limited trajectory using Ruckig online
        trajectory generation, then executes it at 50 Hz. Automatically switches
        to impedance mode if not already active.
        
        Args:
            qpos (list | np.ndarray): Target joint positions [rad] (7,)
                                     Default: Home position
                                     
        Note:
            - Uses Ruckig for smooth, time-optimal trajectories
            - Respects velocity, acceleration, and jerk limits
            - Switches to impedance control automatically
            - Executes trajectory at 50 Hz (20ms updates)
            - Duration depends on distance and limits
            
        Trajectory Limits:
            - Max velocity: 10 rad/s per joint
            - Max acceleration: 5 rad/s² per joint
            - Max jerk: 1 rad/s³ per joint
            
        Examples:
            Move to home position:
                >>> await controller.move()
            
            Move to custom position:
                >>> target = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
                >>> await controller.move(target)
            
            Move to current position + offset:
                >>> current = controller.state['qpos']
                >>> await controller.move(current + np.array([0.1, 0, 0, 0, 0, 0, 0]))
                
        Caveats:
            - Large motions take longer (trajectory is time-optimal)
            - Don't call while other control is active
            - Blocks until motion completes
            - Switches to impedance mode (resets controller state)
            - May fail if target is at joint limits or in collision
        """
        self.type = "impedance"
        print("setting impedance controller for move...")

        inp = InputParameter(7)

        print('getting current state for ruckig...')
        inp.current_position = self.robot.state['qpos']
        inp.current_velocity = self.robot.state['qvel']
        print("got current state")
        inp.current_acceleration = np.zeros(7)

        inp.target_position = np.array(qpos)
        inp.target_velocity = np.zeros(7)
        inp.target_acceleration = np.zeros(7)

        inp.max_velocity = vel
        inp.max_acceleration = acc
        inp.max_jerk = np.ones(7)

        print("set input parameters for ruckig")
        
        otg = Ruckig(7)
        trajectory = Trajectory(7)

        print("calculating trajectory...")
        result = otg.calculate(inp, trajectory)

        print(f"Generated trajectory with result: {result}")
        print(trajectory.duration * 50)

        # create a trajectory to the desired qpos  (linear interpolation)
        for i in range(int(trajectory.duration * 50)):
            print(i, trajectory.duration * 50)
            q_desired, _, _ = trajectory.at_time(i / 50.0)
            await self.set("q_desired", q_desired)

        # await self.stabilize()

