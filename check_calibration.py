#!/usr/bin/env python3
"""
Script to check calibration mismatch between motors and calibration file.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.teleoperators.arx5_leader import ARX5LeaderConfig, ARX5Leader

def main():
    # Create ARX5Leader instance
    config = ARX5LeaderConfig(
        port="/dev/ttyACM0",
        id="arx5_leader"
    )
    
    leader = ARX5Leader(config)
    
    print("=" * 80)
    print("Checking ARX5 Leader Calibration")
    print("=" * 80)
    
    # Connect to motors (without calibration check)
    leader.bus.connect()
    
    # Read calibration from file
    print("\n1. Calibration from file:")
    if leader.calibration:
        for motor, calib in leader.calibration.items():                                             
            print(f"  {motor}:")
            print(f"    ID: {calib.id}")
            print(f"    Drive_Mode: {calib.drive_mode}")
            print(f"    Homing_Offset: {calib.homing_offset}")
            print(f"    Min_Position_Limit: {calib.range_min}")
            print(f"    Max_Position_Limit: {calib.range_max}")
    else:
        print("  No calibration file found!")
    
    # Read calibration from motors
    print("\n2. Calibration from motors:")
    motor_calib = leader.bus.read_calibration()
    for motor, calib in motor_calib.items():
        print(f"  {motor}:")
        print(f"    ID: {calib.id}")
        print(f"    Drive_Mode: {calib.drive_mode}")
        print(f"    Homing_Offset: {calib.homing_offset}")
        print(f"    Min_Position_Limit: {calib.range_min}")
        print(f"    Max_Position_Limit: {calib.range_max}")
    
    # Compare
    print("\n3. Differences:")
    if leader.calibration:
        found_diff = False
        for motor in leader.calibration.keys():
            file_calib = leader.calibration[motor]
            motor_calib_val = motor_calib[motor]
            
            diffs = []
            if file_calib.drive_mode != motor_calib_val.drive_mode:
                diffs.append(f"Drive_Mode: file={file_calib.drive_mode}, motor={motor_calib_val.drive_mode}")
            if file_calib.homing_offset != motor_calib_val.homing_offset:
                diffs.append(f"Homing_Offset: file={file_calib.homing_offset}, motor={motor_calib_val.homing_offset}")
            if file_calib.range_min != motor_calib_val.range_min:
                diffs.append(f"Min_Position_Limit: file={file_calib.range_min}, motor={motor_calib_val.range_min}")
            if file_calib.range_max != motor_calib_val.range_max:
                diffs.append(f"Max_Position_Limit: file={file_calib.range_max}, motor={motor_calib_val.range_max}")
            
            if diffs:
                found_diff = True
                print(f"  {motor}:")
                for diff in diffs:
                    print(f"    {diff}")
        
        if not found_diff:
            print("  No differences found! Calibration matches.")
    else:
        print("  Cannot compare - no calibration file found!")
    
    print("\n4. is_calibrated:", leader.bus.is_calibrated)
    
    # Disconnect
    leader.bus.disconnect()
    print("\n" + "=" * 80)
    print("Check complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

