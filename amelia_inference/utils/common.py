from enum import Enum
from easydict import EasyDict

# Base paths

# Global variables
class AgentType(Enum):
    AIRCRAFT = 0
    VEHICLE  = 1
    UNKNOWN  = 2

DATA_DIR = "./datasets/amelia"
VERSION = "a10v08"
TRAJ_DATA_DIR = f"{DATA_DIR}/traj_data_{VERSION}/proc_trajectories" # Trajectory from which to load scenes
