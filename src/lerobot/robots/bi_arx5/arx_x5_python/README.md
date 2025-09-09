# arx_x5_python

## 目录
- [介绍](#介绍)
- [安装](#安装)
- [使用方法](#使用方法)
- [接口指南](#接口指南)

### example

## 📖介绍
ARX L5Pro SDK for Python

## 🔧安装
### 检查依赖
#### 安装can

```
sudo apt update && sudo apt install can-utils
```
#### 安装keyboard库
```
sudo pip3 install keyboard
```
#### 安装pybind11
```
git clone https://github.com/pybind/pybind11.git && cd pybind11 && mkdir build && cd build && cmake .. && make && sudo make install
```

#### 编译python接口
* cd到仓库目录下，执行:
    ```
    build.sh
    ```

## 🚀使用方法
### arx_can配置
```
sudo -S slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up
```

### example使用
#### 环境变量
```
source ./setup.sh
```
### 运行
```
python3 test_single_arm.py
```
### keyboard节点的运行
```
sudo su
source ./setup.sh
python3 test_keyboard.py
```

### 二次开发
* 把bimanual setup.sh移到自己的工程下即可

## 📚接口指南

### statePositionControl
```cpp
#include <vector>
#include <memory>
#include <unistd.h> // for usleep
#include "ControllerBase.h" // Assumed header for class definition
#include "CanBase.h"       // Assumed header for CAN-related types
#include "solve.h"         // Assumed header for Interpolation function

void arx::x5::ControllerBase::statePositionControl() {
    CanFrame frame;
    double current_pos[6];
    double solve_position[6];
    double output_vel[6];

    // Get current joint positions
    for (int i = 0; i < 6; ++i) {
        current_pos[i] = joint_status_[i].position;
    }

    // Get target positions
    for (int i = 0; i < 6; ++i) {
        solve_position[i] = joint_target_[i].position;
    }

    // Apply joint limits
    for (int i = 0; i < 6; ++i) {
        solve_position[i] = limit(solve_position[i], lower_Joint[i], upper_Joint[i]);
    }

    // Compute velocity using interpolation
    for (int i = 0; i < 6; ++i) {
        solve::Interpolation(&solve_position[i], &temp_position[i], &output_vel[i], max_acc_, max_vel_, 200.0);
    }

    // Apply control for each joint
    for (int i = 0; i < 6; ++i) {
        auto motor = joint_ptr_[i];
        double grav_torque = gravity_compensation_torque_[i];
        double pos_error = (i < 3) ? (solve_position[i] - current_pos[i]) : 0.0;
        double vel = (i < 3) ? joint_status_[i].velocity : 0.0; // Though for i=3, it fetched but not used
        double kd_gain = (i < 3) ? 0.6 : 0.0;
        double kp_gain = (i < 3) ? 1.5 : 0.0;
        double control_torque = grav_torque + pos_error * kp_gain - vel * kd_gain;

        double motor_kp, motor_kd;
        switch (i) {
            case 0:
            case 1:
            case 2:
                motor_kp = 150.0;
                motor_kd = 1.0;
                break;
            case 3:
                motor_kp = 50.0;
                motor_kd = 0.8;
                break;
            case 4:
                motor_kp = 25.0;
                motor_kd = 0.8;
                break;
            case 5:
                motor_kp = 10.0;
                motor_kd = 1.0;
                break;
        }

        frame.can_id = motor->applyControl(motor_kp, motor_kd, temp_position[i], output_vel[i], control_torque);
        arx_can_->sendFrame(frame);
        usleep(200);
    }

    CatchPositionCtrl(this);
}
```

### stateGravityCompensation
```cpp
#include <vector>
#include <memory>
#include <unistd.h> // for usleep
#include "ControllerBase.h" // Assumed header for class definition
#include "CanBase.h"       // Assumed header for CAN-related types

void arx::x5::ControllerBase::stateGravityCompensation() {
    CanFrame frame;
    
    // Apply gravity compensation torque for each joint
    for (int i = 0; i < 6; ++i) {
        auto motor = joint_ptr_[i]; // Access shared_ptr to motor
        auto torque = gravity_compensation_torque_[i]; // Access torque value
        frame.can_id = motor->applyTorque(0, 0, 0, 0, torque); // Call motor-specific function with zeroed parameters
        arx_can_->sendFrame(frame); // Send CAN frame
        usleep(200); // Delay for 200 microseconds
    }
    
    // Update joint positions
    for (int i = 0; i < 6; ++i) {
        temp_position[i] = joint_status_[i].position; // Store current joint position
    }
    
    CatchSoft(this); // Call to handle soft limits or safety checks
}
```

### MotorType2 applyTorque MotorType::DM_J4310 joint[3, 4, 5]
```cpp
#include "MotorType2.h" // Assumed header for class definition
#include "CanFrame.h"   // Assumed header for CanFrame type

CanFrame arx::hw_interface::MotorType2::packMotorMsg(double k_p, double k_d, double position, double velocity, double torque) {
    // Restrict parameters to bounds
    double bounded_kp = restrictBound(k_p, 500.0, 0.0);
    double bounded_kd = restrictBound(k_d, 5.0, 0.0);
    double bounded_pos = restrictBound(position, 12.5, -12.5);
    double bounded_vel = restrictBound(velocity, 45.0, -45.0);
    double bounded_tor = restrictBound(torque, 10.0, -10.0);

    // Convert floats to unsigned integers with scaling
    uint16_t pos_tmp = float_to_uint(static_cast<float>(bounded_pos), -12.5f, 12.5f, 16); // 0x10 = 16 bits
    uint16_t vel_tmp = float_to_uint(static_cast<float>(bounded_vel), -45.0f, 45.0f, 12); // 0xc = 12 bits
    uint16_t kp_tmp = float_to_uint(static_cast<float>(bounded_kp), 0.0f, 500.0f, 12);
    uint16_t kd_tmp = float_to_uint(static_cast<float>(bounded_kd), 0.0f, 5.0f, 12);
    uint16_t tor_tmp = float_to_uint(static_cast<float>(bounded_tor), -10.0f, 10.0f, 12);

    // Pack into CanFrame
    CanFrame frame;
    frame.can_dlc = 8; // Data length code = 8 bytes
    frame.can_id = motor_id_;
    frame.data[0] = (pos_tmp >> 8) & 0xFF;
    frame.data[1] = pos_tmp & 0xFF;
    frame.data[2] = (vel_tmp >> 4) & 0xFF;
    frame.data[3] = ((kp_tmp >> 8) & 0x0F) | ((vel_tmp & 0x0F) << 4);
    frame.data[4] = kp_tmp & 0xFF;
    frame.data[5] = (kd_tmp >> 4) & 0xFF;
    frame.data[6] = ((tor_tmp >> 8) & 0x0F) | ((kd_tmp & 0x0F) << 4);
    frame.data[7] = tor_tmp & 0xFF;

    return frame;
}

```
### MotorType4 applyControl MotorType::EC_A4310 joint[0, 1, 2]
```cpp
#include <cstdint>
#include "MotorType4.h" // Assumed header for MotorType4 and MotorDlcBase
#include "CanBase.h"   // Assumed header for CanFrame

namespace arx::hw_interface {

CanFrame MotorType4::packMotorMsg(double k_p, double k_d, double position, double velocity, double torque) {
    CanFrame frame;

    // Restrict input parameters to safe bounds
    double restricted_kp = MotorDlcBase::restrictBound(k_p, 500.0, 0.0);
    double restricted_kd = MotorDlcBase::restrictBound(k_d, 50.0, 0.0);
    double restricted_pos = MotorDlcBase::restrictBound(position, 12.5, -12.5);
    double restricted_vel = MotorDlcBase::restrictBound(velocity, 18.0, -18.0);
    double restricted_torque = MotorDlcBase::restrictBound(torque, 30.0, -30.0);

    // Convert to unsigned integers with specified ranges and bit widths
    uint16_t pos_tmp = float_to_uint(static_cast<float>(restricted_pos), -12.5, 12.5, 16);
    uint16_t vel_tmp = float_to_uint(static_cast<float>(restricted_vel), -18.0, 18.0, 12);
    uint16_t kp_tmp = float_to_uint(static_cast<float>(restricted_kp), 0.0, 500.0, 12);
    uint16_t kd_tmp = float_to_uint(static_cast<float>(restricted_kd), 0.0, 50.0, 12);
    uint16_t tor_tmp = float_to_uint(static_cast<float>(restricted_torque), -30.0, 30.0, 12);

    // Pack data into CAN frame (8 bytes)
    frame.can_dlc = 8; // Data length code: 8 bytes
    frame.can_id = motor_id; // Use motor-specific ID
    frame.data[0] = static_cast<uint8_t>((kp_tmp & 0xFFFF) >> 7); // High 5 bits of kp
    frame.data[1] = static_cast<uint8_t>(((kd_tmp >> 8) & 0x1) | ((kp_tmp & 0x7F) << 1)); // Combine kd high bit and kp low 7 bits
    frame.data[2] = static_cast<uint8_t>(kd_tmp & 0xFF); // Low byte of kd
    frame.data[3] = static_cast<uint8_t>(pos_tmp >> 8); // High byte of position
    frame.data[4] = static_cast<uint8_t>(pos_tmp & 0xFF); // Low byte of position
    frame.data[5] = static_cast<uint8_t>(vel_tmp >> 4); // High 8 bits of velocity
    frame.data[6] = static_cast<uint8_t>((tor_tmp >> 8) | (vel_tmp << 4)); // Combine torque high and velocity low
    frame.data[7] = static_cast<uint8_t>(tor_tmp & 0xFF); // Low byte of torque

    return frame;
}
} // namespace arx::hw_interface
```
