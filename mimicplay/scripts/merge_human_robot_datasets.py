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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate human dataset')
    parser.add_argument('--dataset_paths', type=list, help='List of dataset paths to merge', nargs='+', default=["/user/frosa/multi_task_lfd/datasets/pick_place/human_rgb_pick_place/hdf5/merged_dataset/human_pick_place_all_demos.hdf5", "/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/panda_pick_place/hdf5_files/merged_dataset/panda_pick_place_all_demos.hdf5"])
    parser.add_argument('--output_file_path', default=None, type=str, help='Path to the output file')
    args = parser.parse_args()

    # debugpy.listen(("localhost", 5678))
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()
    # print("Debugger attached.")

    if args.output_file_path is None:
        human_dataset_name = args.dataset_paths[0].split("/")[-1]
        args.output_file_path = args.dataset_paths[0].replace(human_dataset_name, f"human_panda_robot_merged.hdf5")
        
    # open human dataset
    human_dataset_file = h5py.File(args.dataset_paths[0], "r")

    task_cnt = np.zeros(16)
    
    # for demo_key in human_dataset_file["data"].keys():
    #     task_id = human_dataset_file["data"][demo_key].attrs["task"]
    #     task_id_number = int(task_id.split("_")[-1])
    #     task_cnt[task_id_number] += 1
    # print(f"Human dataset task counts: {task_cnt}")
        

    # create new hdf5 file
    merged_dataset_file = h5py.File(args.output_file_path, "w")
    merged_dataset_file.create_group("data")
    
    # copy env_meta_data
    merged_dataset_file['data'].attrs['env_args'] = human_dataset_file['data'].attrs['env_args']
    
    # copy demo samples from human to merged
    offset = 0 
    num_samples = 0
    for demo_key in tqdm(human_dataset_file["data"].keys()):
        # print(f"Copying human demo {demo_key} to merged dataset")
        
        if demo_key not in merged_dataset_file["data"]:
            merged_dataset_file["data"].create_group(demo_key)
            
            human_demo = human_dataset_file["data"][demo_key]
            # copy all datasets
            for dset_key in human_demo.keys():
                # print(f"\tCopying dataset {dset_key}")
                
                if dset_key == 'obs':
                    human_obs = human_demo[dset_key]
                    merged_dataset_file["data"][demo_key].create_group(dset_key)
                    for obs_key in human_obs.keys():
                        # print(f"\t\tCopying obs key {obs_key}")
                        human_obs_data = human_obs[obs_key][:]
                        merged_dataset_file["data"][demo_key][dset_key].create_dataset(obs_key, data=human_obs_data)
                else:
                    human_data = human_demo[dset_key][:]
                    merged_dataset_file["data"][demo_key].create_dataset(dset_key, data=human_data)
                
            # copy all attributes     
            for key in human_demo.attrs:
                # print(f"\tCopying attr {key}")
                merged_dataset_file["data"][demo_key].attrs[key] = human_demo.attrs[key]
                if 'num_samples' in human_demo.attrs:
                    num_samples += human_demo.attrs['num_samples']   
        
        offset += 1
    
    start_robot_demo_idx = copy.deepcopy(offset)
    
    # now copy robot demos from other datasets
    for dataset_path in args.dataset_paths[1:]:
        # print(f"Processing robot dataset: {dataset_path}")
        robot_dataset_file = h5py.File(dataset_path, "r")
        
        for demo_key in tqdm(robot_dataset_file["data"].keys()):
            # change demo key to avoid overwriting human demos
            robot_demo_key = int(demo_key.split("_")[-1]) + offset
            new_demo_key = f"demo_{robot_demo_key}"
            # print(f"Copying robot demo {demo_key} to merged dataset as {new_demo_key}")
            
            
            if new_demo_key not in merged_dataset_file["data"]:
                merged_dataset_file["data"].create_group(new_demo_key)
                
                robot_demo = robot_dataset_file["data"][demo_key]
                # copy all datasets
                for dset_key in robot_demo.keys():
                    # print(f"\tCopying dataset {dset_key}")
                    
                    if dset_key == 'obs':
                        robot_obs = robot_demo[dset_key]
                        merged_dataset_file["data"][new_demo_key].create_group(dset_key)
                        for obs_key in robot_obs.keys():    
                            # print(f"\t\tCopying obs key {obs_key}")
                            robot_obs_data = robot_obs[obs_key][:]
                            merged_dataset_file["data"][new_demo_key][dset_key].create_dataset(obs_key, data=robot_obs_data)
                    else:
                        robot_data = robot_demo[dset_key][:]
                        if 'actions_robot' in dset_key:
                            new_dset_key = 'actions_robot'
                        elif 'actions' in dset_key:
                            new_dset_key = 'actions'
                        elif 'reward' in dset_key:
                            new_dset_key = 'rewards'
                        elif 'done' in dset_key:
                            new_dset_key = 'dones'
                        
                        merged_dataset_file["data"][new_demo_key].create_dataset(new_dset_key, data=robot_data)
                    
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
    
    