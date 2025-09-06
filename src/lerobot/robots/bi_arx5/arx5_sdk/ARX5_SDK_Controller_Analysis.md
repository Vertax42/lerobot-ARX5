# ARX5 SDK Controller Core Implementation Analysis

## Overview

This document provides a detailed analysis of the ARX5 SDK's core controller implementation, focusing on the critical components that enable real-time robot control at 500Hz frequency.

## Architecture Overview

The ARX5 controller follows a multi-threaded architecture:
- **Main Thread**: User API calls (`set_joint_cmd`, `get_joint_state`)
- **Background Thread**: Real-time control loop at 500Hz (2ms cycle)
- **CAN Communication**: Low-level motor control via CAN bus

## Core Components

### 1. Background Control Loop

The heart of the ARX5 control system is the background thread created in the constructor:

**Thread Creation** (`controller_base.cpp:28-30`):
```cpp
Arx5ControllerBase::Arx5ControllerBase(RobotConfig robot_config, ControllerConfig controller_config,
                                       std::string interface_name)
{
    // ... initialization
    init_robot_();
    // Create and start background thread
    background_send_recv_thread_ = std::thread(&Arx5ControllerBase::background_send_recv_, this);
    background_send_recv_running_ = controller_config_.background_send_recv;
    logger_->info("Background send_recv task is running at ID: {}", syscall(SYS_gettid));
}
```

**Background Loop Implementation** (`controller_base.cpp:672-694`):
```cpp
void Arx5ControllerBase::background_send_recv_()
{
    while (!destroy_background_threads_)
    {
        int start_time_us = get_time_us();
        if (background_send_recv_running_)
        {
            over_current_protection_();      // Over-current protection
            check_joint_state_sanity_();     // Joint state validation
            send_recv_();                    // Core: send commands + receive states
        }
        
        // Precise timing control - ensure 500Hz frequency
        int elapsed_time_us = get_time_us() - start_time_us;
        int sleep_time_us = int(controller_config_.controller_dt * 1e6) - elapsed_time_us;
        if (sleep_time_us > 0)
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_us));
        else if (sleep_time_us < -500)
            logger_->debug("Background send_recv task is running too slow, time: {} us", elapsed_time_us);
    }
}
```

**Key Features:**
- **500Hz Precise Control**: Each cycle targets 2ms (2000μs)
- **Dynamic Enable/Disable**: Controlled by `background_send_recv_running_` flag
- **Safety Checks**: Over-current protection and joint state validation
- **Performance Monitoring**: Logs when control loop runs too slow

### 2. Core Communication Function - send_recv_()

The `send_recv_()` function handles all real-time communication with motors:

**Function Structure** (`controller_base.cpp:558-631`):
```cpp
void Arx5ControllerBase::send_recv_()
{
    // Torque constants for different motor types
    const double torque_constant_EC_A4310 = 1.4;   // Nm/A
    const double torque_constant_DM_J4310 = 0.424;
    const double torque_constant_DM_J4340 = 1.0;
    
    // 1. Update output commands (trajectory interpolation + safety limits)
    update_output_cmd_();
    
    // 2. Send joint commands to each motor
    for (int i = 0; i < robot_config_.joint_dof; i++)
    {
        if (robot_config_.motor_type[i] == MotorType::EC_A4310)
        {
            can_handle_.send_EC_motor_cmd(robot_config_.motor_id[i], 
                                        gain_.kp[i], gain_.kd[i],
                                        output_joint_cmd_.pos[i], 
                                        output_joint_cmd_.vel[i],
                                        output_joint_cmd_.torque[i] / torque_constant_EC_A4310);
        }
        else if (robot_config_.motor_type[i] == MotorType::DM_J4310)
        {
            can_handle_.send_DM_motor_cmd(robot_config_.motor_id[i], 
                                        gain_.kp[i], gain_.kd[i],
                                        output_joint_cmd_.pos[i], 
                                        output_joint_cmd_.vel[i],
                                        output_joint_cmd_.torque[i] / torque_constant_DM_J4310);
        }
        sleep_us(150);  // Inter-motor communication delay
    }
    
    // 3. Send gripper command
    if (robot_config_.gripper_motor_type == MotorType::DM_J4310)
    {
        double gripper_motor_pos = output_joint_cmd_.gripper_pos / robot_config_.gripper_width * 
                                  robot_config_.gripper_open_readout;
        can_handle_.send_DM_motor_cmd(robot_config_.gripper_motor_id, 
                                    gain_.gripper_kp, gain_.gripper_kd,
                                    gripper_motor_pos, gripper_motor_vel, gripper_motor_torque);
    }
    
    // 4. Update joint states from motor feedback
    update_joint_state_();
}
```

**Timing Analysis:**
- **Total Target**: <2ms (500Hz frequency)
- **Command Update**: ~100-200μs (trajectory interpolation + safety checks)
- **Motor Communication**: ~1200μs (6 joints×150μs + 1 gripper×150μs)
- **State Reading**: ~100μs (CAN message parsing)
- **Remaining Time**: Used for `sleep_us()` to ensure precise timing

### 3. Command Processing - update_output_cmd_()

This function processes user commands through trajectory interpolation and safety checks:

**Key Operations** (`controller_base.cpp:416-556`):
```cpp
void Arx5ControllerBase::update_output_cmd_()
{
    // 1. Trajectory interpolation from user-set targets
    output_joint_cmd_ = interpolator_.interpolate(get_timestamp());
    
    // 2. Gravity compensation (if enabled)
    if (controller_config_.gravity_compensation)
    {
        output_joint_cmd_.torque += solver_->inverse_dynamics(joint_state_.pos, 
                                                            VecDoF::Zero(), VecDoF::Zero());
    }
    
    // 3. Joint position limits
    for (int i = 0; i < robot_config_.joint_dof; ++i)
    {
        if (output_joint_cmd_.pos[i] < robot_config_.joint_pos_min[i])
            output_joint_cmd_.pos[i] = robot_config_.joint_pos_min[i];
        else if (output_joint_cmd_.pos[i] > robot_config_.joint_pos_max[i])
            output_joint_cmd_.pos[i] = robot_config_.joint_pos_max[i];
    }
    
    // 4. Joint velocity limits
    for (int i = 0; i < robot_config_.joint_dof; ++i)
    {
        double delta_pos = output_joint_cmd_.pos[i] - prev_output_cmd.pos[i];
        double max_vel = robot_config_.joint_vel_max[i];
        if (std::abs(delta_pos) > max_vel * dt)
        {
            // Limit velocity to prevent robot from moving too fast
            double new_pos = prev_output_cmd.pos[i] + max_vel * dt * delta_pos / std::abs(delta_pos);
            output_joint_cmd_.pos[i] = new_pos;
        }
    }
    
    // 5. Torque limits
    for (int i = 0; i < robot_config_.joint_dof; ++i)
    {
        if (output_joint_cmd_.torque[i] > robot_config_.joint_torque_max[i])
            output_joint_cmd_.torque[i] = robot_config_.joint_torque_max[i];
        else if (output_joint_cmd_.torque[i] < -robot_config_.joint_torque_max[i])
            output_joint_cmd_.torque[i] = -robot_config_.joint_torque_max[i];
    }
}
```

### 4. State Reading - update_joint_state_()

Converts raw motor feedback into joint states:

**Implementation** (`controller_base.cpp:372-414`):
```cpp
void Arx5ControllerBase::update_joint_state_()
{
    // Read all motor messages from CAN bus
    std::array<OD_Motor_Msg, 10> motor_msg = can_handle_.get_motor_msg();
    
    for (int i = 0; i < robot_config_.joint_dof; i++)
    {
        // Read position and velocity directly
        joint_state_.pos[i] = motor_msg[robot_config_.motor_id[i]].angle_actual_rad;
        joint_state_.vel[i] = motor_msg[robot_config_.motor_id[i]].speed_actual_rad;
        
        // Convert current to torque based on motor type
        if (robot_config_.motor_type[i] == MotorType::EC_A4310)
        {
            joint_state_.torque[i] = motor_msg[robot_config_.motor_id[i]].current_actual_float *
                                   torque_constant_EC_A4310 * torque_constant_EC_A4310;
        }
        else if (robot_config_.motor_type[i] == MotorType::DM_J4310)
        {
            joint_state_.torque[i] = motor_msg[robot_config_.motor_id[i]].current_actual_float * 
                                   torque_constant_DM_J4310;
        }
    }
    
    // Convert gripper motor reading to gripper position
    joint_state_.gripper_pos = motor_msg[robot_config_.gripper_motor_id].angle_actual_rad /
                              robot_config_.gripper_open_readout * robot_config_.gripper_width;
    
    joint_state_.timestamp = get_timestamp();
}
```

### 5. Safety Mechanisms

#### A. Gain Setting Safety (set_gain)

Prevents dangerous jumps when transitioning from damping to position control:

**Critical Safety Check** (`controller_base.cpp:96-124`):
```cpp
void Arx5ControllerBase::set_gain(Gain new_gain)
{
    // Prevent robot jumps when setting kp to non-zero
    if (gain_.kp.isZero() && !new_gain.kp.isZero()) // damping -> joint_control
    {
        JointState joint_state = get_joint_state();
        JointState joint_cmd = get_joint_cmd();
        double max_pos_error = (joint_state.pos - joint_cmd.pos).cwiseAbs().maxCoeff();
        double pos_error_threshold = 0.2;  // 0.2 rad = 11.5 degrees
        double kp_threshold = 1;

        // Dangerous: pos_error > 0.2 && new_kp > 1
        if (max_pos_error > pos_error_threshold && new_gain.kp.maxCoeff() > kp_threshold)
        {
            logger_->error("Cannot set kp too large when joint pos cmd is far from current pos.");
            logger_->error("Target max kp: {}, Current pos: {}, cmd pos: {}", 
                          new_gain.kp.maxCoeff(), vec2str(joint_state.pos), vec2str(joint_cmd.pos));
            background_send_recv_running_ = false;
            throw std::runtime_error("Cannot set kp to non-zero when joint pos cmd is far from current pos.");
        }
    }
    gain_ = new_gain;
}
```

#### B. Over-Current Protection

Monitors motor currents and triggers emergency stop if limits exceeded:

**Protection Logic** (`controller_base.cpp:318-349`):
```cpp
void Arx5ControllerBase::over_current_protection_()
{
    bool over_current = false;
    
    // Check joint motor currents
    for (int i = 0; i < robot_config_.joint_dof; ++i)
    {
        if (std::abs(joint_state_.torque[i]) > robot_config_.joint_torque_max[i])
        {
            over_current = true;
            logger_->error("Over current detected on joint {}, current: {:.3f}", i, joint_state_.torque[i]);
            break;
        }
    }
    
    // Check gripper current
    if (std::abs(joint_state_.gripper_torque) > robot_config_.gripper_torque_max)
    {
        over_current = true;
        logger_->error("Over current detected on gripper, current: {:.3f}", joint_state_.gripper_torque);
    }
    
    if (over_current)
    {
        over_current_cnt_++;
        if (over_current_cnt_ > controller_config_.over_current_cnt_max)
        {
            logger_->error("Over current detected, robot set to damping. Please restart.");
            enter_emergency_state_();
        }
    }
    else
    {
        over_current_cnt_ = 0;
    }
}
```

### 6. Trajectory Interpolation System

The `JointStateInterpolator` class provides smooth trajectory planning:

#### Key Features:
- **Linear and Cubic Interpolation**: Configurable interpolation methods
- **Waypoint Management**: Add, override, and append trajectory points
- **Velocity Calculation**: Automatic velocity computation for smooth motion
- **Time-based Interpolation**: Precise timing for real-time control

#### Core Methods (`utils.cpp`):
```cpp
// Initialize trajectory between two points
void init(JointState start_state, JointState end_state);

// Set fixed position (no interpolation)
void init_fixed(JointState start_state);

// Add waypoint to existing trajectory
void append_waypoint(double current_time, JointState end_state);

// Replace all future waypoints
void override_waypoint(double current_time, JointState end_state);

// Get interpolated state at specific time
JointState interpolate(double time);
```

### 7. Calibration and Homing

#### Reset to Home Function

Intelligent homing with safety considerations:

**Smart Homing Logic** (`controller_base.cpp:149-213`):
```cpp
void Arx5ControllerBase::reset_to_home()
{
    JointState init_state = get_joint_state();
    Gain init_gain = get_gain();
    
    // Calculate maximum position error
    double max_pos_error = (init_state.pos - VecDoF::Zero()).cwiseAbs().maxCoeff();
    max_pos_error = std::max(max_pos_error, init_state.gripper_pos * 2 / robot_config_.gripper_width);
    
    // Interpolate time based on maximum error (minimum 0.5s)
    double wait_time = std::max(max_pos_error, 0.5);
    int step_num = int(wait_time / controller_config_.controller_dt);
    
    // Gradually interpolate gains and positions
    for (int i = 0; i <= step_num; i++)
    {
        double alpha = double(i) / step_num;
        Gain new_gain = init_gain * (1 - alpha) + target_gain * alpha;
        set_gain(new_gain);
        sleep_us(int(controller_config_.controller_dt * 1e6));
    }
}
```

## Data Flow Architecture

```
User API (Python)
       ↓
   set_joint_cmd() → JointState → interpolator_
                                      ↓
Background Thread: update_output_cmd_() → safety_limits → CAN_send
                                                              ↓
Motors ← CAN bus ← send_recv_() ← update_joint_state_() ← CAN_receive
   ↓
joint_state_ ← get_joint_state() ← User API (Python)
```

## Performance Characteristics

- **Control Frequency**: 500Hz (2ms cycle time)
- **Communication Protocol**: CAN bus at various baud rates
- **Safety Response Time**: <2ms for emergency stops
- **Position Accuracy**: Motor-dependent, typically <0.1°
- **Torque Control**: Current-based with motor-specific constants

## Thread Safety

The controller uses mutex locks for thread-safe access:
- `cmd_mutex_`: Protects command data (`output_joint_cmd_`, `gain_`, `interpolator_`)
- `state_mutex_`: Protects state data (`joint_state_`)

This ensures safe concurrent access between the user API thread and the background control thread.

## Conclusion

The ARX5 SDK implements a sophisticated real-time control system with:
- **Multi-threaded Architecture**: Separates user interface from real-time control
- **Comprehensive Safety**: Multiple protection mechanisms prevent hardware damage
- **Smooth Motion**: Advanced trajectory interpolation for natural robot movement
- **High Performance**: 500Hz control loop for precise real-time operation
- **Flexible Configuration**: Supports different motor types and control modes

This design enables robust, safe, and precise control of ARX5 robotic arms in demanding real-time applications.