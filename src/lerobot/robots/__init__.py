from .config import RobotConfig
from .robot import Robot
from .utils import make_robot_from_config

# Import robot modules to register them
from . import bi_arx5
from . import bi_so100_follower
from . import hope_jr
from . import koch_follower
from . import so100_follower
from . import so101_follower
