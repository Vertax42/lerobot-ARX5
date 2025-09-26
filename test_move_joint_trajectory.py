#!/usr/bin/env python3
"""
Test script for smooth_go_start function in BiARX5 robot.

This script tests the complete workflow:
1. Connect to robot
2. Use smooth_go_start() to move both arms to start position [0,0,0,0,0,0,0]
   (This automatically handles mode switching and trajectory execution)
3. Interactive mode for manual testing
4. Disconnect safely
"""

import sys
import time
import logging
import select
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from lerobot.robots.bi_arx5.bi_arx5 import BiARX5  # noqa: E402
from lerobot.robots.bi_arx5.config_bi_arx5 import BiARX5Config  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_move_joint_trajectory():
    """Test the smooth_go_start function with the specified workflow."""

    # Duration for the movement (2 seconds)
    duration = 2.0

    # Create robot configuration
    config = BiARX5Config()

    # Initialize robot
    robot = BiARX5(config)

    try:
        logger.info("=" * 60)
        logger.info("Starting BiARX5 smooth_go_start test")
        logger.info("=" * 60)

        # Step 1: Connect to robot
        logger.info("Step 1: Connecting to robot...")
        robot.connect()
        logger.info("✓ Robot connected successfully")

        # Step 2: Wait for user command to start movement
        logger.info("Step 2: Ready to move arms to start position")
        logger.info(f"Duration: {duration} seconds")
        logger.info("This will automatically:")
        logger.info("  - Switch to normal position control mode")
        logger.info("  - Move both arms to start position [0,0,0,0,0,0,0]")
        logger.info("  - Switch back to gravity compensation mode")
        logger.info("")
        logger.info("Press 's' + Enter to start the movement, or Ctrl+C to exit:")

        # Wait for user input to start movement
        while True:
            try:
                user_input = input().strip().lower()
                if user_input == "s":
                    logger.info("Starting smooth_go_start() movement...")
                    break
                else:
                    logger.info("Press 's' + Enter to start, or Ctrl+C to exit:")
            except KeyboardInterrupt:
                logger.info("Movement cancelled by user")
                return
            except EOFError:
                logger.info("Input stream closed, exiting...")
                return

        # Step 3-4: Execute the smooth start movement
        start_time = time.time()
        robot.smooth_go_start(duration=duration, easing="ease_in_out_quad")
        end_time = time.time()

        actual_duration = end_time - start_time
        logger.info(f"✓ smooth_go_start() completed in {actual_duration:.2f} seconds")

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
