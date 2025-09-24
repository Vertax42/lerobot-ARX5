#!/usr/bin/env python3
"""
Test script for move_joint_trajectory function in BiARX5 robot.

This script tests the complete workflow:
1. Connect to robot
2. Switch to normal position control mode
3. Move both arms to target position in 2 seconds
4. Switch back to gravity compensation mode
5. Disconnect safely
"""

import sys
import time
import logging
import select
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from lerobot.robots.bi_arx5.bi_arx5 import BiARX5
from lerobot.robots.bi_arx5.config_bi_arx5 import BiARX5Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_move_joint_trajectory():
    """Test the move_joint_trajectory function with the specified workflow."""

    # Target joint positions for both arms
    # Format: [joint1, joint2, joint3, joint4, joint5, joint6, gripper]
    # Note: gripper value 0.0 means fully closed, 1.0 means fully open
    target_pose = [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.08]

    # Duration for the movement (2 seconds)
    duration = 2.0

    # Create robot configuration
    config = BiARX5Config()

    # Initialize robot
    robot = BiARX5(config)

    try:
        logger.info("=" * 60)
        logger.info("Starting BiARX5 move_joint_trajectory test")
        logger.info("=" * 60)

        # Step 1: Connect to robot
        logger.info("Step 1: Connecting to robot...")
        robot.connect()
        logger.info("✓ Robot connected successfully")

        # Step 2: Switch to normal position control mode
        logger.info("Step 2: Switching to normal position control mode...")
        robot.set_to_normal_position_control()
        logger.info("✓ Switched to normal position control mode")

        # Step 3: Move both arms to target position
        logger.info("Step 3: Moving both arms to target position...")
        logger.info(f"Target pose: {target_pose}")
        logger.info(f"Duration: {duration} seconds")
        logger.info("Note: gripper value 0.0 = fully closed, 1.0 = fully open")

        # Prepare target poses for both arms
        # You can set different gripper values for left and right arms if needed
        target_poses = {"left": target_pose, "right": target_pose}

        # Execute the trajectory
        start_time = time.time()
        robot.move_joint_trajectory(
            target_joint_poses=target_poses,
            durations=duration,
            easing="ease_in_out_quad",
        )
        end_time = time.time()

        actual_duration = end_time - start_time
        logger.info(f"✓ Movement completed in {actual_duration:.2f} seconds")

        # Step 4: Switch back to gravity compensation mode
        logger.info("Step 4: Switching back to gravity compensation mode...")
        robot.set_to_gravity_compensation_mode()
        logger.info("✓ Switched back to gravity compensation mode")

        # Step 5: Interactive mode - wait for user commands
        logger.info("Step 5: Robot is now in gravity compensation mode")
        logger.info("You can manually move the arms to observe the result")
        logger.info("Commands:")
        logger.info("  - Type 'r' + Enter: Return to home position (all zeros)")
        logger.info("  - Press Ctrl+C: Exit the test")

        try:
            while True:
                try:
                    # Use select to check if input is available (non-blocking)

                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        user_input = input().strip().lower()
                        if user_input == "r":
                            logger.info("Using smooth_go_home() method...")
                            robot.smooth_go_home(duration=2.0)
                        else:
                            logger.info(
                                "Unknown command. Type 'r' to return home or Ctrl+C to exit."
                            )
                    else:
                        time.sleep(0.1)  # Small sleep to prevent high CPU usage
                except EOFError:
                    # Handle Ctrl+D or input stream closing
                    break
        except KeyboardInterrupt:
            logger.info("Ctrl+C detected, continuing to exit...")

        logger.info("=" * 60)
        logger.info("Test completed successfully!")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt received. Stopping test...")
    except Exception as e:
        logger.error(f"Error during test: {e}")
        raise
    finally:
        # Always disconnect safely
        try:
            if robot.is_connected:
                logger.info("Disconnecting robot...")
                robot.disconnect()
                logger.info("✓ Robot disconnected safely")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")


if __name__ == "__main__":
    test_move_joint_trajectory()
