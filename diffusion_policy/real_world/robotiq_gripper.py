import socket
import time
from typing import Optional, Any, Callable, List, Dict, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from concurrent.futures import Future, ThreadPoolExecutor


### gripper drivers copied from airo mono ###
def rescale_range(x: float, from_min: float, from_max: float, to_min: float, to_max: float) -> float:
    return to_min + (x - from_min) / (from_max - from_min) * (to_max - to_min)

import enum
import time
import warnings
from typing import Callable, Optional


class ACTION_STATUS_ENUM(enum.Enum):
    EXECUTING = 1
    SUCCEEDED = 2
    TIMEOUT = 3



def wait_for_condition_with_timeout(
    check_condition: Callable[..., bool], timeout: float = 10, sleep_resolution: float = 0.1
) -> None:
    """helper function to wait on completion of hardware interaction with a timeout to avoid blocking forever."""

    while not check_condition():
        time.sleep(sleep_resolution)
        timeout -= sleep_resolution
        if timeout < 0:
            raise TimeoutError()
        

class AwaitableAction:
    def __init__(
        self,
        termination_condition: Callable[..., bool],
        default_timeout: float = 30.0,
        default_sleep_resolution: float = 0.1,
    ):
        """

        Args:
            termination_condition (Callable[..., bool]): Any  callable that returns True when the action is completed.
            In the simplest case, it can be a lambda that returns True.
            It can also be true when the gripper is in the desired position.
            Or it could be true after a certain amount of time has passed since the action was started.

            default_timeout (float, optional): The max waiting time before the wait returns and raises a warning.
            Select an appropriate default value for the command you are creating this AwaitableAction for.

            default_sleep_resolution (float, optional): The length of the time.sleep() in each iteration.
            Select an appropriate default value for the command you are creating this AwaitableAction for.

            Note that the scope of this action is to send 1 command, do some other things, then wait for the command to finish.
            If you send multiple commands to the gripper, there is no guarantee that the gripper will execute them in the order you send them,
             as the gripper might preempt intermediate commands after it has finished the current command.
            Such preemption will also not be detected by this action, and hence the wait of a preempted action will either timeout or succeed by accident.

        """
        self.status = ACTION_STATUS_ENUM.EXECUTING
        self.is_action_done = termination_condition
        self._default_timeout = default_timeout
        self._default_sleep_resolution = default_sleep_resolution

    def wait(self, timeout: Optional[float] = None, sleep_resolution: Optional[float] = None) -> ACTION_STATUS_ENUM:
        """Busy waiting until the termination condition returns true, or until timeout.

        Args:
            timeout (float, optional): The max waiting time before the wait returns and raises a warning.
            This prevents infinite loops. Defaults to the value set during creation of the awaitable, which is usually an appropriate value for the command.

            sleep_resolution (float, optional): The length of the time.sleep() in each iteration.
            higher values will take up less CPU resources but will also cause more latency between the action finishing and
            this method to return. Defaults to the value set during creation of the awaitable, which is usually an appropriate value for the command.

            Keep in mind that the time.sleep() function has limited accuracy, so the actual sleeping time will be usually higher
            due to scheduling activities of the OS. Take a look [here](https://github.com/airo-ugent/airo-mono/pull/21#discussion_r1132520057)
            for some realistic numbers on the error. The error is independent of the sleep time and is in the order of 0.2ms.
            So the lower the sleep time, the higher the relative error becomes.


        Returns:
            ACTION_STATUS_ENUM: _description_
        """
        # see #airo-robots/scripts/measure_sleep_accuracy.py for a script that measures the sleep accuracy and
        # the result of some measurements.
        sleep_resolution = sleep_resolution or self._default_sleep_resolution
        timeout = timeout or self._default_timeout
        assert (
            sleep_resolution > 0.001
        ), "sleep resolution must be at least 1 ms, otherwise the relative error of a sleep becomes too large to be meaningful"
        if not self.status == ACTION_STATUS_ENUM.EXECUTING:
            return self.status
        while True:
            time.sleep(sleep_resolution)
            timeout -= sleep_resolution
            if self.is_action_done():
                self.status = ACTION_STATUS_ENUM.SUCCEEDED
                return self.status
            if timeout < 0:
                warnings.warn("Action timed out. Make sure this was expected.")
                return ACTION_STATUS_ENUM.TIMEOUT

    def is_done(self) -> bool:
        return self.status == ACTION_STATUS_ENUM.SUCCEEDED
    
@dataclass
class ParallelPositionGripperSpecs:
    """
    all values are in metric units:
    - the position of the gripper is expressed as the width between the fingers in meters
    - the speed in meters/second
    - the force in Newton
    """

    max_width: float
    min_width: float
    max_force: float
    min_force: float
    max_speed: float
    min_speed: float


class ParallelPositionGripper(ABC):
    """
    Base class for a position-controlled, 2 finger parallel gripper.

    These grippers typically allow to set a speed and maximum applied force before moving,
    and attempt to move to specified positions under these constraints.

    all values are in metric units:
    - the position of the gripper is expressed as the width between the fingers in meters
    - the speed in meters/second
    - the force in Newton
    """

    def __init__(self, gripper_specs: ParallelPositionGripperSpecs) -> None:
        self._gripper_specs = gripper_specs

    @property
    def gripper_specs(self) -> ParallelPositionGripperSpecs:
        return self._gripper_specs

    @property
    @abstractmethod
    def speed(self) -> float:
        """speed with which the fingers will move in m/s"""
        # no need to raise NotImplementedError thanks to ABC

    @speed.setter
    @abstractmethod
    def speed(self, new_speed: float) -> None:
        """sets the moving speed [m/s]."""
        # this function is delibarately not templated
        # as one always requires this to happen synchronously.

    @property
    @abstractmethod
    def max_grasp_force(self) -> float:
        """max force the fingers will apply in Newton"""

    @max_grasp_force.setter
    @abstractmethod
    def max_grasp_force(self, new_force: float) -> None:
        """sets the max grasping force [N]."""
        # this function is delibarately not templated
        # as one always requires this to happen synchronously.

    @abstractmethod
    def get_current_width(self) -> float:
        """the current opening of the fingers in meters"""

    @abstractmethod
    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None) -> AwaitableAction:
        """
        move the fingers to the desired width between the fingers[m].
        Optionally provide a speed and/or force, that will be used from then on for all move commands.
        """

    def open(self) -> AwaitableAction:
        return self.move(self.gripper_specs.max_width)

    def close(self) -> AwaitableAction:
        return self.move(0.0)

    def is_an_object_grasped(self) -> bool:
        """
        Heuristics to check if an object is grasped, usually by looking at motor currents.
        """
        raise NotImplementedError

class Robotiq2F85(ParallelPositionGripper):
    """
    Implementation of the gripper interface for a Robotiq 2F-85 gripper that is connected to a UR robot and is controlled with the Robotiq URCap.

    The API is available at TCP port 63352 of the UR controller and wraps the Modbus registers of the gripper, as described in
    https://dof.robotiq.com/discussion/2420/control-robotiq-gripper-mounted-on-ur-robot-via-socket-communication-python.
    The control sequence is gripper motor <---- gripper registers<--ModbusSerial(rs485)-- UR controller <--TCP-- remote control

    This class does 2 things:
    - it communicates over TCP using the above mentioned API to read/write values from/to the gripper's registers.
    - it rescales all those register values into metric units, as required by the gripper interface

    For more info on how to install the URCap and connecting the gripper's RS-485 connection to a UR robot, see the manual, section 4.8
    https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_e-Series_PDF_20190206.pdf

    Make sure you can control the gripper using the robot controller (polyscope) before using this script.
    """

    # values obtained from the manual
    # see https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_e-Series_PDF_20190206.pdf
    specs = ParallelPositionGripperSpecs(0.085, 0.0, 220, 25, 0.15, 0.02)

    def __init__(self, host_ip: str, port: int = 63352, fingers_max_stroke: Optional[float] = None) -> None:
        """
        host_ip: the IP adress of the robot to which the gripper is connected.

        fingers_max_stroke:
        allow for custom max stroke width (if you have fingertips that are closer together).
        the robotiq will always calibrate such that max opening = 0 and min opening = 255 for its register.
        see manual p42
        """
        if fingers_max_stroke:
            self._gripper_specs.max_width = fingers_max_stroke
        self.host_ip = host_ip
        self.port = port

        self._check_connection()

        if not self.gripper_is_active():
            self._activate_gripper()

        super().__init__(self.specs)

    def get_current_width(self) -> float:
        register_value = int(self._communicate("GET POS").split(" ")[1])
        width = rescale_range(register_value, 0, 230, self._gripper_specs.max_width, self._gripper_specs.min_width)
        return width

    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None) -> AwaitableAction:
        if speed:
            self.speed = speed
        if force:
            self.max_grasp_force = force
        self._set_target_width(width)
        # this sleep is required to make sure that the OBJ STATUS
        # of the gripper is already in 'moving' before entering the wait loop.
        time.sleep(0.01)

        def move_done_condition() -> bool:
            done = abs(self.get_current_width() - width) < 0.002
            done = done or self.is_an_object_grasped()
            return done

        return AwaitableAction(move_done_condition)

    def is_an_object_grasped(self) -> bool:
        return int(self._communicate("GET OBJ").split(" ")[1]) == 2

    @property
    def speed(self) -> float:
        speed_register_value = self._read_speed_register()
        return rescale_range(
            speed_register_value, 0, 255, self._gripper_specs.min_speed, self._gripper_specs.max_speed
        )

    @speed.setter
    def speed(self, value: float) -> None:
        speed = np.clip(value, self.gripper_specs.min_speed, self.gripper_specs.max_speed)
        speed_register_value = int(
            rescale_range(speed, self._gripper_specs.min_speed, self._gripper_specs.max_speed, 0, 255)
        )
        self._communicate(f"SET SPE {speed_register_value}")

        def is_value_set() -> bool:
            return self._is_target_value_set(self._read_speed_register(), speed_register_value)

        wait_for_condition_with_timeout(is_value_set)

    @property
    def max_grasp_force(self) -> float:
        force_register_value = self._read_force_register()
        # 0 force has a special meaning, cf manual.
        return rescale_range(
            force_register_value, 1, 255, self._gripper_specs.min_force, self._gripper_specs.max_force
        )

    @max_grasp_force.setter
    def max_grasp_force(self, value: float) -> None:
        force = np.clip(value, self.gripper_specs.min_force, self.gripper_specs.max_force)
        force_register_value = int(
            rescale_range(force, self._gripper_specs.min_force, self._gripper_specs.max_force, 1, 255)
        )
        self._communicate(f"SET FOR {force_register_value}")

        def is_value_set() -> bool:
            return self._is_target_value_set(force_register_value, self._read_force_register())

        wait_for_condition_with_timeout(is_value_set)



    ####################
    # Private methods #
    ####################

    def _set_target_width(self, target_width_in_meters: float) -> None:
        """Sends target width to gripper"""
        target_width_in_meters = np.clip(
            target_width_in_meters, self._gripper_specs.min_width, self._gripper_specs.max_width
        )
        # 230 is 'force closed', cf _write_target_width_to_register.
        target_width_register_value = round(
            rescale_range(target_width_in_meters, self._gripper_specs.min_width, self._gripper_specs.max_width, 230, 0)
        )
        self._write_target_width_to_register(target_width_register_value)

    def _write_target_width_to_register(self, target_width_register_value: int) -> None:
        """
        Takes values in range 0 -255
        3 is actually fully open, 230 is fully closed in operating mode (straight fingers) and 255 is fully closed in encompassed mode.
        For the default fingertips, the range 3 - 230 maps approximately to 0mm - 85mm with a quasi linear relation of 0.4mm / unit
        (experimental findings, but manual also mentions this 0.4mm/unit relation)

        For custom finger tips with a different stroke, it won't be 0.4mm/unit anymore,
        but 230 will still be fully closed due to the self-calibration of the gripper.

        """
        self._communicate(f"SET  POS {target_width_register_value}")

        def is_value_set() -> bool:
            return self._is_target_value_set(target_width_register_value, self._read_target_width_register())

        wait_for_condition_with_timeout(is_value_set)

    def _read_target_width_register(self) -> int:
        return int(self._communicate("GET PRE").split(" ")[1])

    def _read_speed_register(self) -> int:
        return int(self._communicate("GET SPE").split(" ")[1])

    def _read_force_register(self) -> int:
        return int(self._communicate("GET FOR").split(" ")[1])

    def is_gripper_moving(self) -> bool:
        # Moving == 0 => detected OR position reached
        return int(self._communicate("GET OBJ").split(" ")[1]) == 0

    def _check_connection(self) -> None:
        """validate communication with gripper is possible.
        Raises:
            ConnectionError
        """
        if not self._communicate("GET STA").startswith("STA"):
            raise ConnectionError("Could not connect to gripper")

    def _communicate(self, command: str) -> str:
        """Helper function to communicate with gripper over a tcp socket.
        Args:
            command (str): The GET/SET command string.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.host_ip, self.port))
                s.sendall(("" + str.strip(command) + "\n").encode())

                data = s.recv(2**10)
                return data.decode()[:-1]
            except Exception as e:
                raise (e)

    def _activate_gripper(self) -> None:
        """Activates the gripper, sets target position to "Open" and sets GTO flag."""
        self._communicate("SET ACT 1")
        wait_for_condition_with_timeout(self.gripper_is_active)
        # initialize gripper
        self._communicate("SET GTO 1")  # enable Gripper
        self.speed = self._gripper_specs.max_speed
        self.force = self._gripper_specs.min_force

    def _deactivate_gripper(self) -> None:
        self._communicate("SET ACT 0")
        wait_for_condition_with_timeout(lambda: self._communicate("GET STA") == "STA 0")

    def gripper_is_active(self) -> bool:
        return self._communicate("GET STA") == "STA 3"

    @staticmethod
    def _is_target_value_set(target: int, value: int) -> bool:
        """helper to compare target value to current value and make the force / speed request synchronous"""
        return abs(target - value) < 5


def get_empirical_data_on_opening_angles(robot_ip: str) -> None:
    """used to verify the relation between finger width and register values."""
    gripper = Robotiq2F85(robot_ip)
    gripper._write_target_width_to_register(230)
    input("press")
    for position in [0, 50, 100, 150, 200, 250]:
        gripper._write_target_width_to_register(position)
        input(f"press key for moving to next position {position}")

### airo mono copy  until here ###
import multiprocessing as mp    
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
import time 

class Command(enum.Enum):
    STOP = 0
    MOVE = 1
    
class RobotiqController(mp.Process):
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 robot_ip,
                 frequency=200,
                 ):
        super().__init__()
        self.shm_manager = shm_manager
        self.robot_ip = robot_ip

        # build action queue:
        example = {
            "cmd" : Command.MOVE.value,
            "target_width" : 0.05,
            "target_time": 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256 
        )

        # build observation ring buffer:
        example = {
            "current_width" : np.zeros((1,), dtype=np.float32),
            "is_object_grasped": np.zeros((1,), dtype=np.float32),
            "gripper_receive_timestamp": time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=200,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        
        self.frequency = frequency

    def run(self):
        # breakpoint()
        # print("running")
        # self.ready_event.set()
        # while 1:
        #     time.sleep(1)

        print("Connecting to gripper")
        self.gripper = Robotiq2F85(self.robot_ip)
        print("Gripper connected")

        try:
            target_width = self.gripper.get_current_width()
            self.ready_event.set()
            iter_idx = 0
            stopped = False
            while not stopped:
                self.gripper.move(target_width)

                while not self.input_queue.empty():
                    action = self.input_queue.get()
                    cmd = action["cmd"]
                    if cmd == Command.MOVE.value:
                        target_width = action["target_width"]
                        target_width = float(target_width)
                        assert isinstance(target_width, float)
                    elif cmd == Command.STOP.value:
                        stopped = True
                    else:
                        raise ValueError(f"Unknown command: {cmd}")

                self.collect_observation()
                time.sleep(1 / self.frequency)
            
            if iter_idx == 0:
                self.ready_event.set()
            iter_idx += 1
        finally:
            # make gripper stop moving
            self.gripper.move(self.gripper.get_current_width()).wait()
            self.ready_event.set()
            # no need to deactivate controller?
    
    def collect_observation(self):
        current_width = np.array([self.gripper.get_current_width()])
        current_width = np.expand_dims(current_width, axis=0)

        is_object_grasped = np.array([self.gripper.is_an_object_grasped()*1.0])
        is_object_grasped = np.expand_dims(is_object_grasped, axis=0)
        observation = {
        "current_width": current_width,
        "is_object_grasped": is_object_grasped,
        "gripper_receive_timestamp": time.time()
        }
        self.ring_buffer.put(observation)
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(10)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        


    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
# ========= command methods ============
    def schedule_waypoint(self, width, target_time):
        assert target_time > time.time()
        width = float(width)

        message = {
            'cmd': Command.MOVE.value,
            'target_width': width,
            'target_time': target_time
        }
        self.input_queue.put(message)

