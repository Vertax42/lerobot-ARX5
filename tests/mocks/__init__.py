# Re-exported so the device-class resolver can find the test robot from the
# package. The resolver derives candidate module names from the class's `name`
# with underscores stripped ("mock_robot_test" -> "mockrobottest"), which never
# matches the real module `mock_robot`; it checks the package itself first, so
# exporting here is what makes make_robot_from_config work for the test mock.
from tests.mocks.mock_robot import MockRobot, MockRobotConfig

__all__ = ["MockRobot", "MockRobotConfig"]
