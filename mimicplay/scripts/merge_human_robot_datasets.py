import argparse
import os
import h5py
import glob
import debugpy
import cv2
import json
from PIL import Image
import copy
import numpy as np
from tqdm import tqdm

DIM_ACTION_VECTOR = 60
TOT_NUM_VIEWS = 3
NUM_ACTIONS_PER_VIEW = DIM_ACTION_VECTOR // TOT_NUM_VIEWS
NUM_ACTION_PER_STEP = 2 * TOT_NUM_VIEWS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate human dataset')
    parser.add_argument('--dataset_paths', type=str, help='List of dataset paths to merge', nargs='+', default=["/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/human_dataset/hdf5/merged_dataset/human_all_demos.hdf5", "/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/panda_pick_place/hdf5_files/merged_dataset/panda_all_demos.hdf5"])
    parser.add_argument('--robots_name', default=["panda"], type=str, help='List of robot names corresponding to dataset paths', nargs='+')
    parser.add_argument('--config_path', default="scripts/config.json", type=str, help='Path to the config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--output_file_path', default=None, type=str, help='Path to the output file')
    args = parser.parse_args()

    if args.debug:
        debugpy.listen(("localhost", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")

    print(f"Dataset paths to merge: {args.dataset_paths}")
    if args.output_file_path is None:
        human_dataset_name = args.dataset_paths[0].split("/")[-1]
        robots_name_list = "_".join(args.robots_name)
        args.output_file_path = args.dataset_paths[0].replace(human_dataset_name, f"human_{robots_name_list}_robot_merged.hdf5")
        print(f"Output file path not provided. Using default: {args.output_file_path}")
        
    # open human dataset
    human_dataset_file = h5py.File(args.dataset_paths[0], "r")

    task_cnt = np.zeros(16)
    
    # for demo_key in human_dataset_file["data"].keys():
    #     task_id = human_dataset_file["data"][demo_key].attrs["task"]
    #     task_id_number = int(task_id.split("_")[-1])
    #     task_cnt[task_id_number] += 1
    # print(f"Human dataset task counts: {task_cnt}")
    
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # each timestep has 60 values, 20 actions for each view organized in the following way
    # [view1, view2, view3, view1, view2, view3, ...]
    start = config["human_keys"]["action_inds"][0]
    end = config["human_keys"]["action_inds"][1]
    action_indices = []
    for i in range(0, NUM_ACTIONS_PER_VIEW//2*config["num_view"]):
        if i == 0:
            action_indices.extend(list(range(start, end)))
        else:
            action_indices.extend(list(range(start + i*NUM_ACTION_PER_STEP, end + i*NUM_ACTION_PER_STEP)))
    
    # create new hdf5 file
    merged_dataset_file = h5py.File(args.output_file_path, "w")
    merged_dataset_file.create_group("data")
    
    # copy env_meta_data
    merged_dataset_file['data'].attrs['env_args'] = human_dataset_file['data'].attrs['env_args']
    
    # copy demo samples from human to merged
    offset = 0 
    num_samples = 0
    human_keys = config["human_keys"]
    for demo_key in tqdm(list(human_dataset_file["data"].keys())):
        # print(f"Copying human demo {demo_key} to merged dataset")
        
        if demo_key not in merged_dataset_file["data"]:
            merged_dataset_file["data"].create_group(demo_key)
            
            human_demo = human_dataset_file["data"][demo_key]
            
            # get only the specified human keys
            keys = human_keys["keys"]
            for key in keys:
                if 'actions' in key:
                    human_data = human_demo[key][:]
                    # select only the relevant action indices
                    human_data_selected = human_data[:, action_indices]
                    merged_dataset_file["data"][demo_key].create_dataset(key, data=human_data_selected)
                else:
                    human_data = human_demo[key][:]
                    merged_dataset_file["data"][demo_key].create_dataset(key, data=human_data)
                
            # obs_keys
            obs_keys = human_keys["obs_keys"]
            human_obs = human_demo['obs']
            merged_dataset_file["data"][demo_key].create_group('obs')
            for obs_key in obs_keys:
                
                if obs_key == 'robot0_eef_pos':
                    # take only the px values of interest
                    human_obs_data = human_obs[obs_key][:, :, start:end]
                elif obs_key == "robot0_eef_pos_future_traj":
                    human_obs_data = human_obs[obs_key][:, :, action_indices]
                else:
                    human_obs_data = human_obs[obs_key][:]
                merged_dataset_file["data"][demo_key]['obs'].create_dataset(obs_key, data=human_obs_data)
                
            # copy all attributes     
            for key in human_demo.attrs:
                # print(f"\tCopying attr {key}")
                merged_dataset_file["data"][demo_key].attrs[key] = human_demo.attrs[key]
                if 'num_samples' in human_demo.attrs:
                    num_samples += human_demo.attrs['num_samples']   
        
        offset += 1
    
    start_robot_demo_idx = copy.deepcopy(offset)
    
    # now copy robot demos from other datasets
    robot_keys = config["robot_keys"]
    for dataset_path in args.dataset_paths[1:]:
        # print(f"Processing robot dataset: {dataset_path}")
        robot_dataset_file = h5py.File(dataset_path, "r")
        
        for demo_key in tqdm(robot_dataset_file["data"].keys()):
            # change demo key to avoid overwriting human demos
            robot_demo_key = int(demo_key.split("_")[-1]) + offset
            new_demo_key = f"demo_{robot_demo_key}"
            # print(f"Copying robot demo {demo_key} to merged dataset as {new_demo_key}")
            
            robot_demo = robot_dataset_file["data"][demo_key]
            
            if new_demo_key not in merged_dataset_file["data"]:
                merged_dataset_file["data"].create_group(new_demo_key)
                
                keys = robot_keys["keys"]
                for key in keys:
                    robot_data = robot_demo[key][:]
                    merged_dataset_file["data"][new_demo_key].create_dataset(key, data=robot_data)
                    
                obs_keys = robot_keys["obs_keys"]
                robot_obs = robot_demo['obs']
                merged_dataset_file["data"][new_demo_key].create_group('obs')
                for obs_key in obs_keys:
                    robot_obs_data = robot_obs[obs_key][:]
                    
                    if obs_key in robot_keys['remap_keys'].keys():
                        new_obs_key = robot_keys['remap_keys'][obs_key]
                    else:
                        new_obs_key = obs_key
                    
                    merged_dataset_file["data"][new_demo_key]['obs'].create_dataset(new_obs_key, data=robot_obs_data)
                    
                # copy all attributes     
                for key in robot_demo.attrs:
                    # print(f"\tCopying attr {key}")
                    merged_dataset_file["data"][new_demo_key].attrs[key] = robot_demo.attrs[key]
                    if 'num_samples' in robot_demo.attrs:
                        num_samples += robot_demo.attrs['num_samples']
        
        offset += len(robot_dataset_file["data"].keys())
        robot_dataset_file.close()
    
    
    merged_dataset_file['data'].attrs['total'] = num_samples
       
    # remap mask/train and mask/valid indices
    mask_train = []
    mask_valid = []
    for mask_train_key in human_dataset_file['mask/train']:
        mask_train.append(mask_train_key.decode())
    for mask_valid_key in human_dataset_file['mask/valid']:
        mask_valid.append(mask_valid_key.decode())
    
    offset = len(human_dataset_file["data"].keys())
    
    for dataset_path in args.dataset_paths[1:]:
        # print(f"Processing robot dataset: {dataset_path}")
        robot_dataset_file = h5py.File(dataset_path, "r")
        
        robot_mask_train = list(robot_dataset_file['mask/train'])
        robot_mask_valid = list(robot_dataset_file['mask/valid'])
        
        # remap train indices
        for i in range(len(robot_mask_train)):
            original_indx = robot_mask_train[i].decode().split("_")[-1]
            new_indx = int(original_indx) + offset
            new_demo_key = f"demo_{new_indx}"
            mask_train.append(new_demo_key)
            
        # remap valid indices
        for i in range(len(robot_mask_valid)):
            original_indx = robot_mask_valid[i].decode().split("_")[-1]
            new_indx = int(original_indx) + offset
            new_demo_key = f"demo_{new_indx}"
            mask_valid.append(new_demo_key)
            
        offset += len(robot_dataset_file["data"].keys())
        robot_dataset_file.close()
        
    merged_dataset_file['data'].attrs['start_robot_demo_idx'] = start_robot_demo_idx
    merged_dataset_file.create_group('mask')
    merged_dataset_file['mask'].create_dataset('train', data=mask_train)
    merged_dataset_file['mask'].create_dataset('valid', data=mask_valid)
    print(f"Train demos: {len(mask_train)}, Valid demos: {len(mask_valid)}")
    
    merged_dataset_file.close()
    human_dataset_file.close()
    
    