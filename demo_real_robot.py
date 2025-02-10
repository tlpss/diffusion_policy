"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Move your SpaceMouse to move the robot EEF in 3D
Press SpaceMouse right button to close gripper.
Press SpaceMouse left button to open gripper.

If you add --teleop-2d flag, you can move the robot in 2D plane only, useful for debugging tasks such as 'pushT'. 


This script assumes a setup with a UR5e, a Robotiq Gripper and >1 RealSense cameras.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from diffusion_policy.real_world.utils import convert_rotvec_to_6D_representation, convert_6D_rotation_to_rotation_matrix

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option("--teleop-2d", "-t2d", default=False, is_flag=True, help="Enable 2D translation mode.")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency, teleop_2d):

    # hw reset for all realsense cameras
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()
    
    dt = 1/frequency
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            Spacemouse(shm_manager=shm_manager,max_value=500,deadzone=0.1) as sm, \
            RealEnv(
                output_dir=output, 
                robot_ip=robot_ip, 
                # recording resolution
                obs_image_resolution=(1280,720),
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager
            ) as env:
            cv2.setNumThreads(1)

            # realsense exposure
            # env.realsense.set_exposure(exposure=120, gain=0)
            # # realsense white balance
            # env.realsense.set_white_balance(white_balance=5900)

            time.sleep(1.0)
            print('Ready!')
            robot_state = env.get_robot_state()
            gripper_state = env.get_gripper_state()
            state = np.concatenate([robot_state["robot_eef_pose_6d_rot"], gripper_state["current_width"]])
            target_pose = state
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c') and not is_recording:
                        # Start recording
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s') and is_recording:
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                stage = key_counter[Key.space]

                # visualize
                vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )

                cv2.imshow('default', vis_img)
                cv2.pollKey()

                precise_wait(t_sample)
                # get teleop command
                sm_state = sm.get_motion_state_transformed()
                # print(sm_state)
                dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)
                
                if teleop_2d:
                    if not sm.is_button_pressed(0):
                        # translation mode
                        drot_xyz[:] = 0
                    else:
                        dpos[:] = 0
                    if not sm.is_button_pressed(1):
                        # 2D translation mode
                        dpos[2] = 0    

                else: 
                    gripper_idx = 9
                    if sm.is_button_pressed(0):
                        target_pose[gripper_idx] += 0.01
                        target_pose[gripper_idx] = min(0.084, target_pose[gripper_idx])
                    if sm.is_button_pressed(1):
                        target_pose[gripper_idx] -= 0.01
                        target_pose[gripper_idx] = max(0.0, target_pose[gripper_idx])


                # update target pose
                target_pose[:3] += dpos
            
                drot = st.Rotation.from_euler('xyz', drot_xyz)
                current_rotmat = convert_6D_rotation_to_rotation_matrix(target_pose[3:9])
                # apply delta rotation
                rotvec = (drot * st.Rotation.from_matrix(current_rotmat)).as_rotvec()
                # convert back to 6D representation
                rot_6d = convert_rotvec_to_6D_representation(rotvec)
                target_pose[3:9] = rot_6d
                
              
     
                # execute teleop command
                env.exec_actions(
                    actions=[target_pose], 
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage])
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
