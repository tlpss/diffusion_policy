import os 
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController

if __name__ == "__main__":
    with SharedMemoryManager() as shm_manager:
        with RTDEInterpolationController(shm_manager=shm_manager, robot_ip='10.42.0.162') as robot:
            time.sleep(2)
            print(robot.get_state())

