#!/usr/bin/env python3

"""
Test script for BiARX5 integration with lerobot-record system
"""

import argparse
import sys
import logging
from lerobot.robots import make_robot_from_config
from lerobot.robots.bi_arx5.config_bi_arx5 import BiARX5Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_robot_creation():
    """Test creating BiARX5 robot from config"""
    try:
        config = BiARX5Config(
            left_arm_model="X5",
            left_arm_port="can1",
            right_arm_model="X5",
            right_arm_port="can3",
            log_level="INFO",
        )

        robot = make_robot_from_config(config)
        logger.info(f"✅ Successfully created robot: {robot.name}")
        logger.info(f"✅ Robot type: {type(robot).__name__}")
        logger.info(f"✅ Action features: {list(robot.action_features.keys())}")
        logger.info(
            f"✅ Observation features: {list(robot.observation_features.keys())}"
        )

        return True
    except Exception as e:
        logger.error(f"❌ Failed to create robot: {e}")
        return False


def test_config_registration():
    """Test if BiARX5Config is properly registered"""
    try:
        from lerobot.robots.config import RobotConfig

        config = RobotConfig.from_robot_type("bi_arx5")
        logger.info(
            f"✅ Successfully loaded config from robot type: {type(config).__name__}"
        )
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load config from robot type: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test BiARX5 integration")
    parser.add_argument(
        "--connect",
        action="store_true",
        help="Test actual robot connection (requires hardware)",
    )
    args = parser.parse_args()

    logger.info("🚀 Testing BiARX5 integration with lerobot-record...")

    # Test 1: Config registration
    logger.info("\n1. Testing config registration...")
    config_test = test_config_registration()

    # Test 2: Robot creation
    logger.info("\n2. Testing robot creation...")
    robot_test = test_robot_creation()

    # Test 3: Connection (optional)
    if args.connect:
        logger.info("\n3. Testing robot connection...")
        try:
            config = BiARX5Config()
            robot = make_robot_from_config(config)
            robot.connect()
            logger.info("✅ Robot connected successfully")
            logger.info(f"✅ Is connected: {robot.is_connected}")
            robot.disconnect()
            logger.info("✅ Robot disconnected successfully")
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            logger.info("This is expected if ARX5 hardware is not connected")

    # Summary
    logger.info(f"\n📊 Test Results:")
    logger.info(f"Config registration: {'✅' if config_test else '❌'}")
    logger.info(f"Robot creation: {'✅' if robot_test else '❌'}")

    if config_test and robot_test:
        logger.info("\n🎉 BiARX5 is successfully integrated with lerobot-record!")
        logger.info("\nYou can now use it with:")
        logger.info("lerobot-record --robot.type=bi_arx5 \\")
        logger.info("  --robot.left_arm_port=can1 \\")
        logger.info("  --robot.right_arm_port=can3 \\")
        logger.info("  --dataset.repo_id=your_username/your_dataset \\")
        logger.info("  --dataset.single_task='Your task description' \\")
        logger.info("  --dataset.num_episodes=10")
        return 0
    else:
        logger.error("❌ Integration test failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
