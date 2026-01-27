import pickle as pkl
import os
import h5py
import numpy as np
import glob
import argparse
import debugpy
import cv2
from utils import *
import copy
import numpy as np
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', 
                        type=str, 
                        default='/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/human_dataset/hdf5/merged_dataset/human_all_demos.hdf5',
                        help='Directory containing the raw data files.')    
    
    args = parser.parse_args()
    
    task_demo_id_mapping = {}
    task_demo_id_mapping['train'] = {}
    task_demo_id_mapping['val'] = {}
   
    # open hdf5 file
    with h5py.File(args.file_path, "r") as data_file:
        # print(data_file['data'].keys())
        print(f"data file keys: {data_file['mask'].keys()}")
        train_keys = data_file['mask']['train']
        print(f"train keys: {train_keys}")
        for train_key in train_keys:
            train_key = train_key.decode('utf-8')
            print(f"Processing train key: {train_key}")
            task_name = data_file['data'][train_key].attrs['task']
            if task_name not in task_demo_id_mapping['train']:
                task_demo_id_mapping['train'][task_name] = []
            task_demo_id_mapping['train'][task_name].append(train_key)
        
        val_keys = data_file['mask']['valid']
        print(f"val keys: {val_keys}")
        for val_key in val_keys:
            val_key = val_key.decode('utf-8')
            print(f"Processing val key: {val_key}")
            task_name = data_file['data'][val_key].attrs['task']
            if task_name not in task_demo_id_mapping['val']:
                task_demo_id_mapping['val'][task_name] = []
            task_demo_id_mapping['val'][task_name].append(val_key)
            
            
    # save mapping to a json file
    save_path = args.file_path.replace('.hdf5', '_low_level_human_demo_task_demo_id_mapping.json')
    print(f"Saving task demo id mapping to: {save_path}")
    with open(save_path, 'w') as f:
        json.dump(task_demo_id_mapping, f, indent=4)
    