from typing import Dict, List, Optional
import torch
import numpy as np
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import cv2
import json
import hashlib
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.common.normalize_util import (
    concatenate_normalizer,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    array_to_stats
)
from tqdm import tqdm
import torchvision.transforms as transforms

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix, matrix_to_axis_angle


def position_and_rot6d_to_se3(positions,rotations_6d):
        assert len(positions.shape) == 2
        assert len(rotations_6d.shape) == 2
        poses = torch.eye(4).unsqueeze(0).repeat(positions.shape[0],1,1)
        poses[:,:3,:3] = rotation_6d_to_matrix(rotations_6d)
        poses[:,:3,3] = positions
        return poses

def position_and_axis_angle_to_se3(positions,rotations):
        assert len(positions.shape) == 2
        assert len(rotations.shape) == 2
        poses = torch.eye(4).unsqueeze(0).repeat(positions.shape[0],1,1)
        poses[:,:3,:3] = axis_angle_to_matrix(rotations)
        poses[:,:3,3] = positions
        return poses

def se3_to_position_and_rot6d(poses):
        assert len(poses.shape) == 3
        positions = poses[:,:3,3]
        rotations = matrix_to_rotation_6d(poses[:,:3,:3])
        return positions, rotations

def se3_to_position_and_axis_angle(poses):
        assert len(poses.shape) == 3
        positions = poses[:,:3,3]
        rotations = matrix_to_axis_angle(poses[:,:3,:3])
        return positions, rotations

class RealImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            image_buffer_resolution: List[int], # (h,w) in which the images need to be stored in replay buffer.
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            n_latency_steps=0,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            delta_action=False,
            no_proprioception=False,
            image_transforms: Optional[List[torch.nn.Module]] = None,
        ):
        assert os.path.isdir(dataset_path)
        
        replay_buffer = None

        replay_buffer_shape_meta = copy.deepcopy(shape_meta)

        # ensure that these keys are always in the replay buffer, even if it is not in the shape meta
        if not "robot_eef_pose_6d_rot" in replay_buffer_shape_meta['obs']:
            replay_buffer_shape_meta['obs']['robot_eef_pose_6d_rot'] = {
                'shape': [9],
                'type': 'low_dim'
            }
        if not "gripper_width" in replay_buffer_shape_meta['obs']:
            replay_buffer_shape_meta['obs']['gripper_width'] = {
                'shape': [1],
                'type': 'low_dim'
            }

        print("Replay buffer shape meta:")
        print(replay_buffer_shape_meta)
        if use_cache:
            # fingerprint shape_meta
            shape_meta_json = json.dumps(OmegaConf.to_container(replay_buffer_shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            image_buffer_resolution=image_buffer_resolution,
                            shape_meta=replay_buffer_shape_meta,
                            store=zarr.MemoryStore()
                        )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                image_buffer_resolution=image_buffer_resolution,
                shape_meta=replay_buffer_shape_meta,
                store=zarr.MemoryStore()
            )
        
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
    
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_delta_action = delta_action
        self.image_transforms = image_transforms
        self.no_proprioception = no_proprioception # cannot just drop the keys, since the keys are used to build the action
        self.image_buffer_resolution = image_buffer_resolution
        if self.image_transforms is not None:
            assert all(isinstance(x, torch.nn.Module) for x in self.image_transforms), "image_transforms must be a list of torch.nn.Module"
            self._image_transform = transforms.Compose([x for x in self.image_transforms])

        self.use_sampler_cache = False # somehow takes up more memory than expected..
        if self.use_sampler_cache:
            self.sampler_cached = list()
            for i in tqdm(range(len(sampler)), desc='caching sampler'):
                self.sampler_cached.append(self._get_from_sampler(i))
            
        
        # get dummy batch to get the shape of the observations and check them against the shape meta
        dummy_batch = self[0]
        for key in dummy_batch['obs'].keys():
            assert dummy_batch['obs'][key].shape[1:] == self.shape_meta['obs'][key]['shape'], f"shape mismatch for key {key}: {dummy_batch['obs'][key].shape[1:]} vs {self.shape_meta['obs'][key]['shape']}"
        assert dummy_batch['action'].shape[1:] == self.shape_meta['action']['shape'], f"shape mismatch for action: {dummy_batch['action'].shape[1:]} vs {self.shape_meta['action']['shape']}"
        print("meta shape matches dataset shape")
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

      # enumerate the dataset and save low_dim data
      # needed bc some of the data elements are only constructed at runtime.
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch['obs'][key]))
            data_cache['action'].append(copy.deepcopy(batch['action']))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            data_cache[key] = data_cache[key].reshape(B*T, D)

        
        # action
        # do not normalize the orientation representation (3:-1)
        action_normalizers = []
        action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][:,:3])))
        action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache['action'][:,3:-1])))
        action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][:,-1:])))

        action_normalizer = concatenate_normalizer(action_normalizers)
        normalizer['action'] = action_normalizer
        
        # obs
        for key in self.lowdim_keys:
            data = data_cache[key]

            if key == "robot_eef_pose_6d_rot" or key == "robot_eef_pose":
                # do not normalize the orientation of the robot pose
                key_normalizers = []
                key_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data[:,:3])))
                key_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data[:,3:])))
                key_normalizer = concatenate_normalizer(key_normalizers)

            else:
                stat = array_to_stats(data)
                key_normalizer = get_range_normalizer_from_stat(stat)
            normalizer[key] = key_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def _get_from_sampler(self, idx: int) -> Dict[str, torch.Tensor]:
        T_slice = slice(self.n_obs_steps)
        data = self.sampler.sample_sequence(idx)
        for key in data.keys():
            if not 'action' in key:
                data[key] = data[key][T_slice]                
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # this is a very expensive call! 0.005 seconds..
        #threadpool_limits(1)


        if not self.use_sampler_cache:
            data = self._get_from_sampler(idx)
        else:
            data = self.sampler_cached[idx]
        
        obs_dict = dict()
        for key in self.rgb_keys:
            if key in data:
                # move channel last to channel first
                # T,H,W,C
                # convert uint8 image to float32
                obs_dict[key] = np.moveaxis(data[key],-1,1
                    ).astype(np.float32) / 255.
                # T,C,H,W
                # save ram
                del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            # save ram
            del data[key]

      
        # temporarily add the proprio keys if they are not in the shape meta
        if not "robot_eef_pose_6d_rot" in obs_dict:
            obs_dict["robot_eef_pose_6d_rot"] = data["robot_eef_pose_6d_rot"].astype(np.float32)
        if not "gripper_width" in obs_dict:
            obs_dict["gripper_width"] = data["gripper_width"].astype(np.float32)
        
        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        # convert observations and actions to torch

        action = torch.from_numpy(action)
        obs_dict = dict_apply(obs_dict, torch.from_numpy)


        # convert actions to SE3
        gripper_actions = action[:,-1:]
        robot_actions = action[:,:-1]

        if robot_actions.shape[1] == 6:
            # position, axis-angle
            robot_actions = position_and_axis_angle_to_se3(robot_actions[:,:3],robot_actions[:,3:])
        elif robot_actions.shape[1] == 9:
            # position, rot6d
            robot_actions = position_and_rot6d_to_se3(robot_actions[:,:3],robot_actions[:,3:])
        
        else: 
            raise ValueError("source action shape not supported")
        
        if self.use_delta_action:
            # make all actions relative to the last observation.
            # and represent them as twist (x,y,z,roll,pitch,yaw)

            # determine robot obs key as the single key that has 'robot' in the name
            robot_reference_pose = None
            if "robot_eef_pose" in obs_dict:
                robot_reference_pose = obs_dict["robot_eef_pose"][-1]
                reference_position,reference_axis_angle = robot_reference_pose[:3].unsqueeze(0),robot_reference_pose[3:].unsqueeze(0)
                robot_reference_pose = position_and_axis_angle_to_se3(reference_position,reference_axis_angle)
                robot_reference_pose = robot_reference_pose.squeeze(0)
                
            elif "robot_eef_pose_6d_rot" in obs_dict:
                robot_reference_pose = obs_dict["robot_eef_pose_6d_rot"][-1]
                reference_position,reference_rot6d = robot_reference_pose[:3].unsqueeze(0),robot_reference_pose[3:].unsqueeze(0)
                robot_reference_pose = position_and_rot6d_to_se3(reference_position,reference_rot6d)
                robot_reference_pose = robot_reference_pose.squeeze(0)
            else: 
                raise ValueError("No robot pose key found in obs_dict")

            gripper_reference_position = obs_dict["gripper_width"][-1]

            # for each chunk in the action:
            # build SE3 matrix
            # multiply by the inverse of the last observation to get the relative pose wrt the last observation.
            robot_actions = torch.matmul(torch.linalg.inv(robot_reference_pose),robot_actions)
            gripper_actions = gripper_actions - gripper_reference_position
            
        if self.shape_meta['action']['shape'] == (7,):
            # position, axis angle, gripper
            robot_actions = se3_to_position_and_axis_angle(robot_actions)
            robot_actions = torch.cat(robot_actions,dim=1)

        elif self.shape_meta['action']['shape'] == (10,):
            # position, rot6d, gripper
            robot_actions = se3_to_position_and_rot6d(robot_actions)
            robot_actions = torch.cat(robot_actions,dim=1)
        else:
            raise ValueError("Action shape meta size not supported")
            
           
        action = torch.cat([robot_actions, gripper_actions],dim=1)

        torch_data = {
            'obs': obs_dict,
            'action': action
        }

        for key in self.rgb_keys:
            if key in torch_data['obs'] and self.image_transforms is not None:
            # apply the image tranfsorm
                torch_data['obs'][key] = self._image_transform(torch_data['obs'][key])


        # drop the propriokey if configured, remove robot and gripper keys
        if self.no_proprioception:
            if "robot_eef_pose" in torch_data['obs']:
                torch_data['obs'].pop("robot_eef_pose")
            if "robot_eef_pose_6d_rot" in torch_data['obs']:
                torch_data['obs'].pop("robot_eef_pose_6d_rot")
            if "gripper_width" in torch_data['obs']:
                torch_data['obs'].pop("gripper_width")
        return torch_data

def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[...,idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = actions
    return zarr_arr

def _get_replay_buffer(dataset_path, image_buffer_resolution, shape_meta, store):
    # parse shape meta
    rgb_keys = list()
    lowdim_keys = list()
    out_resolutions = dict()
    lowdim_shapes = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = tuple(attr.get('shape'))
        if type == 'rgb':
            rgb_keys.append(key)
            c,h,w = shape
            out_resolutions[key] = (w,h)
        elif type == 'low_dim':
            lowdim_keys.append(key)
            lowdim_shapes[key] = tuple(shape)
            

    # load data
    cv2.setNumThreads(4)
    with threadpool_limits(4):
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions=image_buffer_resolution, # do not resize here. want to resize in the dataset.
            lowdim_keys=lowdim_keys + ['action'],
            image_keys=rgb_keys
        )


    return replay_buffer
