#!/usr/bin/env python

"""
Test script to validate BiARX5 robot integration with LeRobot record system
"""

import sys
import logging
from pathlib import Path

# Add the lerobot source to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from lerobot.robots.bi_arx5 import BiARX5, BiARX5Config
from lerobot.robots.utils import make_robot_from_config
from lerobot.datasets.utils import hw_to_dataset_features

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_robot_factory():
    """Test robot factory registration"""
    logger.info("Testing robot factory registration...")

    # Create config - type is automatically determined from class
    config = BiARX5Config()
    logger.info(f"Config type: {config.type}")

    try:
        # Test factory function
        robot = make_robot_from_config(config)
        logger.info(f"✓ Robot factory works: {type(robot)}")
        return robot
    except Exception as e:
        logger.error(f"✗ Robot factory failed: {e}")
        raise


def test_robot_features(robot):
    """Test robot observation and action features"""
    logger.info("Testing robot features...")

    # Test observation features
    obs_features = robot.observation_features
    logger.info(f"Observation features: {list(obs_features.keys())}")

    # Test action features
    action_features = robot.action_features
    logger.info(f"Action features: {list(action_features.keys())}")

    # Verify expected features exist
    expected_joints = [
        f"{arm}_joint_{i}.pos" for arm in ["left", "right"] for i in range(1, 7)
    ]
    expected_grippers = ["left_gripper.pos", "right_gripper.pos"]
    expected_features = expected_joints + expected_grippers

    for feature in expected_features:
        if feature not in action_features:
            logger.error(f"✗ Missing action feature: {feature}")
            return False
        if feature not in obs_features:
            logger.error(f"✗ Missing observation feature: {feature}")
            return False

    logger.info("✓ All expected features present")
    return True


def test_dataset_features(robot):
    """Test dataset feature conversion"""
    logger.info("Testing dataset feature conversion...")

    try:
        # Test conversion to dataset features
        action_features = hw_to_dataset_features(
            robot.action_features, "action", use_video=True
        )
        obs_features = hw_to_dataset_features(
            robot.observation_features, "observation", use_video=True
        )

        logger.info(f"Dataset action features: {len(action_features)}")
        logger.info(f"Dataset observation features: {len(obs_features)}")

        # Check camera features are properly handled
        camera_features = [
            k
            for k in obs_features.keys()
            if "camera" in k.lower() or any(cam in k for cam in ["head", "wrist"])
        ]
        logger.info(f"Camera features: {camera_features}")

        logger.info("✓ Dataset feature conversion works")
        return True

    except Exception as e:
        logger.error(f"✗ Dataset feature conversion failed: {e}")
        raise


def main():
    """Main test function"""
    logger.info("=== BiARX5 LeRobot Integration Test ===")

    try:
        # Test 1: Robot factory
        robot = test_robot_factory()

        # Test 2: Robot features
        if not test_robot_features(robot):
            sys.exit(1)

        # Test 3: Dataset features
        if not test_dataset_features(robot):
            sys.exit(1)

        logger.info("=== All tests passed! ===")
        logger.info("BiARX5 is ready for LeRobot record functionality")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
