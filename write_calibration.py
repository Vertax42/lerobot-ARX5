#!/usr/bin/env python3
"""
Script to write calibration from file to motors.
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
    print("Writing Calibration to ARX5 Leader Motors")
    print("=" * 80)
    
    if not leader.calibration:
        print("\nERROR: No calibration file found!")
        print("Please run calibration first: lerobot-calibrate --teleop.type=arx5_leader")
        return
    
    # Connect to motors
    print("\nConnecting to motors...")
    leader.bus.connect()
    print("✓ Connected")
    
    # Write calibration
    print("\nWriting calibration to motors...")
    leader.bus.write_calibration(leader.calibration)
    print("✓ Calibration written")
    
    # Verify
    print("\nVerifying...")
    if leader.bus.is_calibrated:
        print("✓ Calibration verified! Motors match calibration file.")
    else:
        print("✗ Verification failed! Motors still don't match calibration file.")
    
    # Disconnect
    leader.bus.disconnect()
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

if __name__ == "__main__":
    main()

