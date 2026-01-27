import os
import random
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils
import numpy as np
from robomimic.utils.dataset import SequenceDataset
from PIL import Image
import copy 
import cv2
import h5py
import json

class PlaydataSequenceDataset(SequenceDataset):
    def __init__(
            self,
            hdf5_path,
            obs_keys,
            dataset_keys,
            goal_obs_gap,
            frame_stack=1,
            seq_length=1,
            pad_frame_stack=True,
            pad_seq_length=True,
            get_pad_mask=False,
            goal_mode=None,
            hdf5_cache_mode=None,
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=None,
            load_next_obs=True,
            perform_aug=True,
            aug_p = 0.5,
            mix_agent_demo = False,
            demo_path=None,
            same_configuration=False,
            json_path=None,
            train=True
    ):
        """
        Dataset class for fetching sequences of experience.
        Length of the fetched sequence is equal to (@frame_stack - 1 + @seq_length)

        Args:
            hdf5_path (str): path to hdf5

            obs_keys (tuple, list): keys to observation items (image, object, etc) to be fetched from the dataset

            dataset_keys (tuple, list): keys to dataset items (actions, rewards, etc) to be fetched from the dataset

            frame_stack (int): numbers of stacked frames to fetch. Defaults to 1 (single frame).

            seq_length (int): length of sequences to sample. Defaults to 1 (single frame).

            pad_frame_stack (int): whether to pad sequence for frame stacking at the beginning of a demo. This
                ensures that partial frame stacks are observed, such as (s_0, s_0, s_0, s_1). Otherwise, the
                first frame stacked observation would be (s_0, s_1, s_2, s_3).

            pad_seq_length (int): whether to pad sequence for sequence fetching at the end of a demo. This
                ensures that partial sequences at the end of a demonstration are observed, such as
                (s_{T-1}, s_{T}, s_{T}, s_{T}). Otherwise, the last sequence provided would be
                (s_{T-3}, s_{T-2}, s_{T-1}, s_{T}).

            get_pad_mask (bool): if True, also provide padding masks as part of the batch. This can be
                useful for masking loss functions on padded parts of the data.

            goal_mode (str): either "last" or None. Defaults to None, which is to not fetch goals

            hdf5_cache_mode (str): one of ["all", "low_dim", or None]. Set to "all" to cache entire hdf5
                in memory - this is by far the fastest for data loading. Set to "low_dim" to cache all
                non-image data. Set to None to use no caching - in this case, every batch sample is
                retrieved via file i/o. You should almost never set this to None, even for large
                image datasets.

            hdf5_use_swmr (bool): whether to use swmr feature when opening the hdf5 file. This ensures
                that multiple Dataset instances can all access the same hdf5 file without problems.

            hdf5_normalize_obs (bool): if True, normalize observations by computing the mean observation
                and std of each observation (in each dimension and modality), and normalizing to unit
                mean and variance in each dimension.

            filter_by_attribute (str): if provided, use the provided filter key to look up a subset of
                demonstrations to load

            load_next_obs (bool): whether to load next_obs from the dataset
        """

        self.hdf5_path = os.path.expanduser(hdf5_path)
        self.hdf5_use_swmr = hdf5_use_swmr
        self.hdf5_normalize_obs = hdf5_normalize_obs
        self._hdf5_file = None

        self.goal_obs_gap = goal_obs_gap

        assert hdf5_cache_mode in ["all", "low_dim", None]
        self.hdf5_cache_mode = hdf5_cache_mode

        self.load_next_obs = load_next_obs
        self.filter_by_attribute = filter_by_attribute

        # get all keys that needs to be fetched
        self.obs_keys = tuple(obs_keys)
        self.dataset_keys = tuple(dataset_keys)

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.goal_mode = goal_mode
        if self.goal_mode is not None:
            assert self.goal_mode in ["nstep", "last"]

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.load_demo_info(filter_by_attribute=self.filter_by_attribute)

        # maybe prepare for observation normalization
        self.obs_normalization_stats = None
        if self.hdf5_normalize_obs:
            self.obs_normalization_stats = self.normalize_obs()

        # maybe store dataset in memory for fast access
        if self.hdf5_cache_mode in ["all", "low_dim"]:
            obs_keys_in_memory = self.obs_keys
            if self.hdf5_cache_mode == "low_dim":
                # only store low-dim observations
                obs_keys_in_memory = []
                for k in self.obs_keys:
                    if ObsUtils.key_is_obs_modality(k, "low_dim"):
                        obs_keys_in_memory.append(k)
            self.obs_keys_in_memory = obs_keys_in_memory

            self.hdf5_cache = self.load_dataset_in_memory(
                demo_list=self.demos,
                hdf5_file=self.hdf5_file,
                obs_keys=self.obs_keys_in_memory,
                dataset_keys=self.dataset_keys,
                load_next_obs=self.load_next_obs
            )

            if self.hdf5_cache_mode == "all":
                # cache getitem calls for even more speedup. We don't do this for
                # "low-dim" since image observations require calls to getitem anyways.
                print("SequenceDataset: caching get_item calls...")
                self.getitem_cache = [self.get_item(i) for i in LogUtils.custom_tqdm(range(len(self)))]

                # don't need the previous cache anymore
                del self.hdf5_cache
                self.hdf5_cache = None
        else:
            self.hdf5_cache = None

        self.close_and_delete_hdf5_handle()
        self.perform_aug = perform_aug
        self.aug_p = aug_p
        self.mix_agent_demo = mix_agent_demo
        
        self.task_demo_id_mapping = None
        if self.mix_agent_demo:
            # open json file to get task demo_id mapping
            json_path = hdf5_path.replace('.hdf5', '_task_demo_id_mapping.json')
            with open(json_path, 'r') as f:
                self.task_demo_id_mapping = json.load(f)
        else:
            self.task_demo_id_mapping = None
        
        # open hdf5 file
        robot_dataset_file = h5py.File(self.hdf5_path, "r")
        # get human indeces
        try:
            self.start_robot_demo_idx = robot_dataset_file['data'].attrs['start_robot_demo_idx']
        except:
            self.start_robot_demo_idx = -1
        robot_dataset_file.close()
        
        # path to demonstration dataset
        self.demo_path = demo_path
        self.same_configuration = same_configuration
        if self.demo_path is not None:
            
            if not same_configuration:
                self.demo_dataset = h5py.File(self.demo_path, "r")
                self.json_path = json_path
                with open(self.json_path, 'r') as f:
                    self.demo_task_demo_id_mapping = json.load(f)
                # open json file to get task demo_id mapping for demonstration dataset
                # json_path = self.demo_path.replace('.hdf5', '_low_level_human_demo_task_demo_id_mapping.json')
                # with open(json_path, 'r') as f:
                #     self.demo_task_demo_id_mapping = json.load(f)
            else:
                # load only demos with same initial target location
                self.demo_dataset = h5py.File(self.demo_path, "r")
                self.json_path = json_path
                with open(self.json_path, 'r') as f:
                    self.demo_task_demo_id_mapping = json.load(f)
        self.train = train
            

    def recolor_arm(self, img_seq, color, new_color):
        # pil_img = Image.fromarray(img)
        # pil_img.save("original_img.png")
        
        new_img_seq = copy.deepcopy(img_seq)
        
        mask_img = new_img_seq<=color
        mask_img = np.sum(mask_img, axis=3)==3
        mask_img[:, :, :30] = False        
        mask_img[:, :, 100:] = False
        # erosion to remove isolated points
        mask_img = np.array(cv2.erode(np.array(mask_img, np.uint8), np.ones((3,3), np.uint8)), np.bool)
        
                            
        new_img_seq[mask_img, :]=new_color
        # pil_img = Image.fromarray(new_img)
        # pil_img.save("new_img.png")
        
        return new_img_seq
        
    def get_sequence_from_human_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1, prefix="obs"):
        
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        demo_length = self.demo_dataset['data'][demo_id].attrs['num_samples']
        assert index_in_demo < demo_length, "index_in_demo {} out of range for demo_length {}".format(index_in_demo, demo_length)

        # determine begin and end of sequence
        seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
        seq_end_index = min(demo_length, index_in_demo + seq_length)

        # determine sequence padding
        seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)  # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + seq_length - demo_length)  # pad for sequence length

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        # fetch observation from the dataset file
        seq = dict()
        for k in keys:
            k_with_prefix = f"{prefix}/{k}"
            data = self.demo_dataset['data'][demo_id][k_with_prefix]
            seq[k_with_prefix] = data[seq_begin_index: seq_end_index]

        seq = TensorUtils.pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(bool)

        obs = {k.split('/')[1]: seq[k] for k in seq}  # strip the prefix
        if self.get_pad_mask:
            obs["pad_mask"] = pad_mask

        return obs
        

    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """

        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]
            
            
        demo_indx = int(demo_id.split("_")[-1])
        # human_demo = True if demo_indx < self.start_robot_demo_idx else False
        # if human_demo:
        #     pass
        #     #print(f"Fetching from human demo: {demo_id}")
        # else:
        #     pass
        #     #print(f"Fetching from robot demo: {demo_id}")

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        # end at offset index if not padding for seq length
        demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            num_frames_to_stack=self.n_frame_stack - 1, # note: need to decrement self.n_frame_stack by one
            seq_length=self.seq_length
        )
        
        if 'actions_robot' in meta.keys() and meta['actions_robot'].shape[0] <= self.seq_length:
            # pad actions_robot to have at least seq_length
            pad_len = self.seq_length - meta['actions_robot'].shape[0] 
            # add zero actions_robot at the end
            actions_robot_pad = np.zeros((pad_len, meta['actions_robot'].shape[1]))
            meta['actions_robot'] = np.concatenate([meta['actions_robot'], actions_robot_pad], axis=0)
            
        
        human_demo = False
        if self.mix_agent_demo:
            # get agent task_name
            task_name = self.hdf5_file['data'][demo_id].attrs['task']
            # pick random demo_id from robot demos for the same task
            demonstrator_id = np.random.choice(list(set(self.task_demo_id_mapping[task_name]).intersection(set(self._demo_id_to_demo_length.keys()))))
            demo_len = self.hdf5_file['data'][demonstrator_id].attrs['num_samples']
            goal_index = None
            if self.goal_mode == "nstep":
                goal_index = min(index_in_demo + random.randint(self.goal_obs_gap[0], self.goal_obs_gap[1]) , demo_len) - 1
            if self.goal_mode == "last":
                goal_index = demo_len - 1
            
            human_demo = True if int(demonstrator_id.split("_")[-1]) < self.start_robot_demo_idx else False
                
        else:
            # used when training high-level policies or low-level policies without mixing demos
            if self.demo_path is None:
                # determine goal index
                goal_index = None
                if self.goal_mode == "nstep":
                    goal_index = min(index_in_demo + random.randint(self.goal_obs_gap[0], self.goal_obs_gap[1]) , demo_length) - 1
                if self.goal_mode == "last":
                    goal_index = demo_length - 1
                
                human_demo = True if demo_indx < self.start_robot_demo_idx else False
            else:
                # get task_name from agent
                task_name = self.hdf5_file['data'][demo_id].attrs['task']
                # pick random demo_id from human demos for the same task
                if not self.same_configuration:
                    split_key = 'train' if self.train else 'val'
                    
                    demonstrator_id = self.demo_task_demo_id_mapping[demo_id][split_key].pop(0)
                    self.demo_task_demo_id_mapping[demo_id][split_key].append(demonstrator_id)
                    # demonstrator_id = np.random.choice(self.demo_task_demo_id_mapping[split_key][task_name])
                    demo_len = self.demo_dataset['data'][demonstrator_id].attrs['num_samples']
                    goal_index = None
                    if self.goal_mode == "nstep":
                        goal_index = min(index_in_demo + random.randint(self.goal_obs_gap[0], self.goal_obs_gap[1]) , demo_len) - 1
                    if self.goal_mode == "last":
                        goal_index = demo_len - 1
                    human_demo = True
                else:
                    split_key = 'train' if self.train else 'val'
                    # get list of human demos with same configuration
                    demonstrator_id = self.demo_task_demo_id_mapping[demo_id][split_key].pop(0)
                    self.demo_task_demo_id_mapping[demo_id][split_key].append(demonstrator_id)
                    demo_len = self.demo_dataset['data'][demonstrator_id].attrs['num_samples']
                    goal_index = None
                    if self.goal_mode == "nstep":
                        goal_index = min(index_in_demo + random.randint(self.goal_obs_gap[0], self.goal_obs_gap[1]) , demo_len) - 1
                    if self.goal_mode == "last":
                        goal_index = demo_len - 1
                    human_demo = True
                    
                

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs"
        )
        
        if 'robot0_eef_pos_3d_camera' in meta["obs"].keys():
            # new key name for 3D eef pos in camera frame
            new_key = 'robot0_eef_pos_3d_camera' #'robot0_eef_pos_3D_0'
            meta["obs"][new_key] = meta["obs"]['robot0_eef_pos_3d_camera']
            # del meta["obs"]['robot0_eef_pos_3d_camera']
          
        # reduce dimension 
        for key in meta["obs"].keys():
            if 'robot0_eef_pos' in key:
                state_key = key
                break  
          
        # check if state contains inf values
        if np.any(np.isinf(meta["obs"][state_key])):
            print(f"Found inf values in demo {demo_id} at index {index_in_demo}")
            print(meta["obs"][state_key])
            
            
        if len(meta["obs"][state_key].shape) == 3:
            for key in meta["obs"].keys():
                if 'agentview_image' not in key and 'robot0_eye_in_hand_image' not in key:
                    meta["obs"][key] = np.reshape(meta["obs"][key], (meta["obs"][key].shape[0], meta["obs"][key].shape[-1]))
        
        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="next_obs"
            )

            # if not human_demo:
            #     for obs_key in meta['next_obs'].keys():
            #         if not human_demo:
            #             if obs_key == 'robot0_eef_pos' or obs_key == 'robot0_eef_pos_future_traj':
            #                 meta['next_obs'][obs_key] = np.array([meta['next_obs'][obs_key]])


        if goal_index is not None:
            if not self.mix_agent_demo and self.demo_path is None:
                meta["goal_obs"] = self.get_obs_sequence_from_demo(
                    demo_id,
                    index_in_demo=goal_index,
                    keys=self.obs_keys,
                    num_frames_to_stack=self.n_frame_stack - 1,
                    seq_length=self.seq_length,
                    prefix="obs",
                )
            elif self.mix_agent_demo and self.demo_path is None:
                meta["goal_obs"] = self.get_obs_sequence_from_demo(
                    demonstrator_id,
                    index_in_demo=goal_index,
                    keys=self.obs_keys,
                    num_frames_to_stack=self.n_frame_stack - 1,
                    seq_length=self.seq_length,
                    prefix="obs",
                )
            elif not self.mix_agent_demo and self.demo_path is not None:
                meta["goal_obs"] = self.get_sequence_from_human_demo(
                    demonstrator_id,
                    index_in_demo=goal_index,
                    keys=['agentview_image'], #['agentview_image', 'robot0_eef_pos_3D_0'],
                    num_frames_to_stack=self.n_frame_stack - 1,
                    seq_length=self.seq_length,
                    prefix="obs"
                )
                
        
            # reduce dimension 
            if self.demo_path is None:
                # training high-level policies
                if len(meta["goal_obs"][state_key].shape) == 3:
                    for key in meta["goal_obs"].keys():
                        if 'agentview_image' not in key and 'robot0_eye_in_hand_image' not in key:
                            meta["goal_obs"][key] = np.reshape(meta["goal_obs"][key], (meta["goal_obs"][key].shape[0], meta["goal_obs"][key].shape[-1]))
            else:
                # if meta["goal_obs"].get('robot0_eef_pos_3D_0', None) is not None  and len(meta["goal_obs"]['robot0_eef_pos_3D_0'].shape) == 3:
                if meta["goal_obs"].get('robot0_eef_pos_3d_camera', None) is not None  and len(meta["goal_obs"]['robot0_eef_pos_3d_camera'].shape) == 3:
                    for key in meta["goal_obs"].keys():
                        if 'agentview_image' not in key and 'robot0_eye_in_hand_image' not in key:
                            meta["goal_obs"][key] = np.reshape(meta["goal_obs"][key], (meta["goal_obs"][key].shape[0], meta["goal_obs"][key].shape[-1]))
                
                
        
        aug = np.random.choice([1, 0], p=[self.aug_p, 1-self.aug_p])
        if human_demo and aug and self.perform_aug:
            # pick color for arm
            arm_color = [40,40,40]
            new_arm_color = np.random.randint(0, 255, size=3).tolist()
            if not self.mix_agent_demo and self.demo_path is None:
                # training high-level policies with human demos only
                meta["obs"]["agentview_image"] = self.recolor_arm(
                                                                meta["obs"]["agentview_image"][0], 
                                                                color=arm_color,
                                                                new_color=new_arm_color)[None]
            if "next_obs" in meta.keys():
                meta["next_obs"]["agentview_image"] = self.recolor_arm(
                                                                meta["next_obs"]["agentview_image"][0], 
                                                                color=arm_color,
                                                                new_color=new_arm_color)[None]
            if "goal_obs" in meta.keys():
                meta["goal_obs"]["agentview_image"] = self.recolor_arm(
                                                                meta["goal_obs"]["agentview_image"], 
                                                                color=arm_color,
                                                                new_color=new_arm_color)
                
        # replace actions_robot key with actions key
        # if len(meta["actions"][0]) == 40:
        #     meta["actions"] = meta["actions"][:, :20] 
            
        # if len(meta['obs']['robot0_eef_pos_future_traj'][0]) == 40:
        #     meta['obs']['robot0_eef_pos_future_traj'] = meta['obs']['robot0_eef_pos_future_traj'][:, :20]
        
        if "actions_robot" in meta.keys():
            meta["actions"] = meta["actions_robot"]
            del meta["actions_robot"]
        
        
        return meta