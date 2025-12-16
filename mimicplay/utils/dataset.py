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
            aug_p = 0.5
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
        
        # open hdf5 file
        robot_dataset_file = h5py.File(self.hdf5_path, "r")
        # get human indeces
        try:
            self.start_robot_demo_idx = robot_dataset_file['data'].attrs['start_robot_demo_idx']
        except:
            self.start_robot_demo_idx = -1
        robot_dataset_file.close()

    def recolor_arm(self, img, color, new_color):
        # pil_img = Image.fromarray(img)
        # pil_img.save("original_img.png")
        
        new_img = copy.deepcopy(img)
        mask_img = new_img<=color
        mask_img = np.sum(mask_img, axis=2)==3
        mask_img[:, :30] = False        
        mask_img[:, 100:] = False
        # erosion to remove isolated points
        mask_img = np.array(cv2.erode(np.array(mask_img, np.uint8), np.ones((3,3), np.uint8)), np.bool)
        
                               
        new_img[mask_img, :]=new_color
        # pil_img = Image.fromarray(new_img)
        # pil_img.save("new_img.png")
        
        return new_img
        
        

    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """

        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        demo_indx = int(demo_id.split("_")[-1])
        human_demo = True if demo_indx < self.start_robot_demo_idx else False

        if human_demo:
            pass
            #print(f"Fetching from human demo: {demo_id}")
        else:
            pass
            #print(f"Fetching from robot demo: {demo_id}")

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

        # determine goal index
        goal_index = None
        if self.goal_mode == "nstep":
            goal_index = min(index_in_demo + random.randint(self.goal_obs_gap[0], self.goal_obs_gap[1]) , demo_length) - 1
        if self.goal_mode == "last":
            goal_index = demo_length - 1

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs"
        )
        
        # reduce dimension 
        meta["obs"]["robot0_eef_pos"] = np.reshape(meta["obs"]["robot0_eef_pos"], (1,meta["obs"]["robot0_eef_pos"].shape[-1]))
        
        meta["obs"]["robot0_eef_pos_future_traj"] = np.reshape(meta["obs"]["robot0_eef_pos_future_traj"], (1, meta["obs"]["robot0_eef_pos_future_traj"].shape[-1]))
        
        meta["actions"] = np.reshape(meta["actions"], (1, meta["actions"].shape[-1]))

        # if not human_demo:
        #     for key in meta.keys():
        #         if key != 'obs':
        #             # set to float32
        #             meta[key] = np.array(meta[key], dtype=np.float32)
        #         elif key == 'obs':
        #             for obs_key in meta['obs'].keys():
        #                 meta['obs'][obs_key] = np.array(meta['obs'][obs_key], dtype=np.float32)
        #                 if obs_key == 'robot0_eef_pos' or obs_key == 'robot0_eef_quat':
        #                     meta['obs'][obs_key] = np.array(meta['obs'][obs_key], dtype=np.float32)
        
        # if not human_demo:
        #     for obs_key in meta['obs'].keys():
        #         if obs_key == 'robot0_eef_pos' or obs_key == 'robot0_eef_pos_future_traj':
        #             meta['obs'][obs_key] = np.array([meta['obs'][obs_key]])
        
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
            meta["goal_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=goal_index,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="obs",
            )
            # if not human_demo:
            #     for obs_key in meta['goal_obs'].keys():
            #         if obs_key == 'robot0_eef_pos' or obs_key == 'robot0_eef_pos_future_traj':
            #             meta['goal_obs'][obs_key] = np.array([meta['goal_obs'][obs_key]])
        
            # reduce dimension 
            meta["goal_obs"]["robot0_eef_pos"] = np.reshape(meta["goal_obs"]["robot0_eef_pos"], (1, meta["goal_obs"]["robot0_eef_pos"].shape[-1]))
            meta["goal_obs"]["robot0_eef_pos_future_traj"] = np.reshape(meta["goal_obs"]["robot0_eef_pos_future_traj"], (1, meta["goal_obs"]["robot0_eef_pos_future_traj"].shape[-1]))
        
        aug = np.random.choice([1, 0], p=[self.aug_p, 1-self.aug_p])
        if human_demo and aug and self.perform_aug:
            # pick color for arm
            arm_color = [40,40,40]
            new_arm_color = np.random.randint(0, 255, size=3).tolist()
            meta["obs"]["agentview_image"] = self.recolor_arm(meta["obs"]["agentview_image"][0], 
                                                              color=arm_color,
                                                              new_color=new_arm_color)[None]
            if "next_obs" in meta.keys():
                meta["next_obs"]["agentview_image"] = self.recolor_arm(meta["next_obs"]["agentview_image"][0], 
                                                              color=arm_color,
                                                              new_color=new_arm_color)[None]
            if "goal_obs" in meta.keys():
                meta["goal_obs"]["agentview_image"] = self.recolor_arm(meta["goal_obs"]["agentview_image"][0], 
                                                              color=arm_color,
                                                              new_color=new_arm_color)[None]
                
        return meta