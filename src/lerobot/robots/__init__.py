from .config import RobotConfig
from .robot import Robot
from .utils import make_robot_from_config

# Import robot modules to register them
try:
    from . import bi_arx5
except ImportError:
    pass  # bi_arx5 requires arx5_interface which may not be available

try:
    from . import arx5_follower
except ImportError:
    pass  # arx5_follower requires arx5_interface which may not be available

from . import bi_so100_follower
from . import hope_jr
from . import koch_follower
from . import so100_follower
from . import so101_follower
