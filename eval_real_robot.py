"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.utils import convert_6D_rotation_to_rotation_matrix, convert_rotvec_to_6D_representation
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.dataset.real_image_dataset import position_and_rot6d_to_se3,position_and_axis_angle_to_se3, se3_to_position_and_rot6d

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option("--teleop-2d", "-t2d", default=False, is_flag=True, help="Enable 2D translation mode.")
@click.option("--delta-action", "-da", default=False, is_flag=True, help="Use delta action instead of absolute action for policy inference.")
def main(input, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency, teleop_2d, delta_action):

    # reset cameras
    # hw reset for all realsense cameras
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()
    

    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # set the policy vision encoder transforms to match the dataset
    # can try to extract from image transforms but for now this is manually.
    cfg.policy.obs_encoder.resize_shape=[240,240]
    cfg.policy.obs_encoder.crop_shape=[224,224]
    cfg.policy.obs_encoder.random_crop=False
    print(f"updating the obs encoder to 1) resize to {cfg.policy.obs_encoder.resize_shape} 2) crop to {cfg.policy.obs_encoder.crop_shape}")
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    action_offset = 0
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager,deadzone=0.1) as sm, RealEnv(
            output_dir=output, 
            robot_ip=robot_ip, 
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            init_joints=init_joints,
            enable_multi_cam_vis=True,
            record_raw_video=True,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager) as env:
            cv2.setNumThreads(1)

            # Should be the same as demo
            # realsense exposure
            # env.realsense.set_exposure(exposure=120, gain=0)
            # # realsense white balance
            # env.realsense.set_white_balance(white_balance=5900)

            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            obs = env.get_obs()
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                del result

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                robot_state = env.get_robot_state()
                gripper_state = env.get_gripper_state()
                state = np.concatenate([robot_state["robot_eef_pose_6d_rot"], gripper_state["current_width"]])
                target_pose = state
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = np.minimum(vis_img, match_img)

                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    key_stroke = cv2.pollKey()
                    if key_stroke == ord('q'):
                        # Exit program
                        env.end_episode()
                        exit(0)
                    elif key_stroke == ord('c'):
                        # Exit human control loop
                        # hand control over to the policy
                        break

                    precise_wait(t_sample)
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
                    # clip target pose
                    # target_pose[:2] = np.clip(target_pose[:2], [-0.05, -0.5], [0.3, -0.17])
                    # if not in range of the robot, do not execute
                    # this is for UR3e
                    if np.linalg.norm(target_pose[:2]) > 0.49:
                        target_pose[:2] = target_pose[:2]/ np.linalg.norm(target_pose[:2]) * 0.49
                    # execute teleop command
                    env.exec_actions(
                        actions=[target_pose], 
                        timestamps=[t_command_target-time.monotonic()+time.time()])
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    term_area_start_timestamp = float('inf')
                    perv_target_pose = None
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        print('get_obs')
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()
                            print('Inference latency:', time.time() - s)
                        
                        # convert policy action to env actions
                        if delta_action:
                            # get the curent obs pose (actions are relative to this pose)
                            current_pose = obs['robot_eef_pose_6d_rot'][-1]
                            current_gripper = obs['gripper_width'][-1]
                            # to torch
                            current_pose = torch.from_numpy(current_pose).float()
                            current_gripper = torch.from_numpy(current_gripper).float()
                            action = torch.from_numpy(action).float()

                            current_pose = current_pose.unsqueeze(0)
                            current_pose = position_and_rot6d_to_se3(current_pose[:,:3], current_pose[:,3:]).squeeze(0)
                            action_dim = action.shape[-1]
                            if action_dim == 10:
                                # x,y,z rot6d, gripper
                                robot_actions = position_and_rot6d_to_se3(action[:,:3], action[:,3:9])
                                gripper_actions = action[:,9]
                                pass
                            elif action_dim == 7:
                                # x,y,z, axis angle, gripper
                                robot_actions = position_and_axis_angle_to_se3(action[:,:3], action[:,3:6])
                                gripper_actions = action[:,6]
                            else:
                                raise ValueError(f'Unsupported action dim {action_dim}')

                            abs_robot_actions = current_pose @ robot_actions
                            abs_robot_positions, abs_robot_rot6ds = se3_to_position_and_rot6d(abs_robot_actions)
                            abs_robot_actions = torch.cat([abs_robot_positions, abs_robot_rot6ds], dim=-1)
                        
                            gripper_actions = gripper_actions.unsqueeze(1)
                            abs_gripper_actions = current_gripper + gripper_actions
                            abs_actions = torch.cat([abs_robot_actions, abs_gripper_actions],dim=-1)
                            # to numpy
                            this_target_poses = abs_actions.numpy()
                            

                            

                        else:
                            # actions are [x,y,z,rxx,rxy,rxz,ryx,ryy,ryz,width]
                            # convert 6D rotation to rotation matrix
                            this_target_poses = action
                           

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # clip actions
                        # this_target_poses[:,:2] = np.clip(
                        #     this_target_poses[:,:2], [-0.05, -0.5], [0.3, -0.17])

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])


                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            env.end_episode()
                            print('Stopped.')
                            break

                        # auto termination
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        #TODO: override with my terminal state?
                        term_pose = np.array([ 3.40948500e-01,  2.17721816e-01,  4.59076878e-02,  2.22014183e+00, -2.22184883e+00, -4.07186655e-04])
                        curr_pose = obs['robot_eef_pose'][-1]
                        dist = np.linalg.norm((curr_pose - term_pose)[:2], axis=-1)
                        if dist < 0.03:
                            # in termination area
                            curr_timestamp = obs['timestamp'][-1]
                            if term_area_start_timestamp > curr_timestamp:
                                term_area_start_timestamp = curr_timestamp
                            else:
                                term_area_time = curr_timestamp - term_area_start_timestamp
                                if term_area_time > 0.5:
                                    terminate = True
                                    print('Terminated by the policy!')
                        else:
                            # out of the area
                            term_area_start_timestamp = float('inf')

                        if terminate:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
