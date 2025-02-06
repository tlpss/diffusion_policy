import os 
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.robotiq_gripper import RobotiqController

if __name__ == "__main__":
    with SharedMemoryManager() as shm_manager:

        gripper= RobotiqController(shm_manager=shm_manager, robot_ip='10.42.0.162')
        gripper.start()
        time.sleep(2)
        gripper.schedule_waypoint(0.02,time.time()+2)
        time.sleep(2)        
        gripper.schedule_waypoint(0.08,time.time()+2)
        time.sleep(2)
        gripper.stop()
   


