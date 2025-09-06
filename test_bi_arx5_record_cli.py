#!/usr/bin/env python

"""
Test script for BiARX5 record functionality
This script tests the record command line interface without actually recording.
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add the lerobot source to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Mock ARX5 SDK before importing
from unittest.mock import Mock, MagicMock
import numpy as np

# Mock the necessary classes and functions
mock_arx5 = Mock()
mock_arx5.Arx5JointController = MagicMock
mock_arx5.RobotConfigFactory = Mock()
mock_arx5.ControllerConfigFactory = Mock() 
mock_arx5.JointState = Mock()
mock_arx5.LogLevel = Mock()

# Mock factory methods
mock_config_factory = Mock()
mock_robot_config = Mock()
mock_robot_config.joint_dof = 6
mock_config_factory.get_config.return_value = mock_robot_config
mock_arx5.RobotConfigFactory.get_instance.return_value = mock_config_factory

mock_controller_config_factory = Mock()
mock_controller_config = Mock()
mock_controller_config_factory.get_config.return_value = mock_controller_config
mock_arx5.ControllerConfigFactory.get_instance.return_value = mock_controller_config_factory

# Mock JointState
def mock_joint_state_init(dof):
    instance = Mock()
    instance.pos = Mock(return_value=np.zeros(dof))
    instance.gripper_pos = 0.0
    return instance

mock_arx5.JointState.side_effect = mock_joint_state_init

# Set up the mock in sys.modules
sys.modules['lerobot.robots.bi_arx5.arx5_sdk.python.arx5_interface'] = mock_arx5

# Now import record functionality
from lerobot.configs import parser
from lerobot.record import RecordConfig
from lerobot.robots.bi_arx5 import BiARX5Config

def test_record_config_parsing():
    """Test record configuration parsing with BiARX5"""
    print("Testing record configuration parsing...")
    
    # Create test arguments similar to the record command
    test_args = [
        "test_script.py",
        "--robot.type=bi_arx5",
        "--robot.left_arm_model=X5",
        "--robot.left_arm_port=can0",
        "--robot.right_arm_model=X5", 
        "--robot.right_arm_port=can1",
        '--robot.cameras={"head": {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30}}',
        "--dataset.repo_id=test/bi_arx5_dataset",
        "--dataset.single_task=Test task",
        "--dataset.num_episodes=1",
        "--dataset.episode_time_s=10",
        "--dataset.reset_time_s=5",
        "--dataset.push_to_hub=false"
    ]
    
    # Parse arguments
    try:
        config = parser.parse(RecordConfig, cli_args=test_args[1:])
        print(f"✓ Config parsing successful")
        print(f"  Robot type: {config.robot.type}")
        print(f"  Left arm model: {config.robot.left_arm_model}")
        print(f"  Right arm model: {config.robot.right_arm_model}")
        print(f"  Dataset repo: {config.dataset.repo_id}")
        print(f"  Task: {config.dataset.single_task}")
        print(f"  Episodes: {config.dataset.num_episodes}")
        print(f"  Cameras: {list(config.robot.cameras.keys())}")
        return config
    except Exception as e:
        print(f"✗ Config parsing failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_record_dry_run():
    """Test record functionality without actually recording"""
    print("\nTesting record dry run...")
    
    # Use temporary directory for dataset
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Create test config programmatically
        config = RecordConfig(
            robot=BiARX5Config(
                left_arm_model="X5",
                left_arm_port="can0",
                right_arm_model="X5",
                right_arm_port="can1",
                cameras={
                    "head": {
                        "type": "opencv",
                        "index_or_path": "/dev/video0",
                        "width": 640,
                        "height": 480,
                        "fps": 30
                    }
                }
            ),
            dataset=RecordConfig.__dataclass_fields__['dataset'].type(
                repo_id="test/bi_arx5_dataset",
                single_task="Test BiARX5 recording",
                root=temp_dir,
                num_episodes=1,
                episode_time_s=1,  # Very short for testing
                reset_time_s=1,
                push_to_hub=False
            )
        )
        
        # Test robot creation
        from lerobot.robots.utils import make_robot_from_config
        robot = make_robot_from_config(config.robot)
        print(f"✓ Robot creation successful: {type(robot)}")
        
        # Test feature compatibility
        from lerobot.datasets.utils import hw_to_dataset_features
        action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
        
        print(f"✓ Feature conversion successful")
        print(f"  Action features: {len(action_features)}")
        print(f"  Observation features: {len(obs_features)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dry run failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)

def main():
    """Main test function"""
    print("=== BiARX5 Record Functionality Test ===")
    
    try:
        # Test 1: Configuration parsing
        config = test_record_config_parsing()
        
        # Test 2: Dry run
        success = test_record_dry_run()
        
        if success:
            print("\n=== All record tests passed! ===")
            print("BiARX5 is ready for LeRobot record functionality")
            print("\nTo run actual recording with hardware:")
            print("lerobot-record \\")
            print("  --robot.type=bi_arx5 \\")
            print("  --robot.left_arm_model=X5 \\")
            print("  --robot.left_arm_port=can0 \\")
            print("  --robot.right_arm_model=X5 \\")
            print("  --robot.right_arm_port=can1 \\")
            print('  --robot.cameras=\'{"head": {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30}}\' \\')
            print("  --dataset.repo_id=your_username/bi_arx5_dataset \\")
            print("  --dataset.single_task=\"Your task description\" \\")
            print("  --dataset.num_episodes=10")
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()