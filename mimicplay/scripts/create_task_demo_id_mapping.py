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
                        default='/user/frosa/multi_task_lfd/datasets/pick_place/human_rgb_pick_place/hdf5/merged_dataset/human_panda_robot_merged.hdf5',
                        help='Directory containing the raw data files.')    
    
    args = parser.parse_args()
    
    task_demo_id_mapping = {}
   
    # open hdf5 file
    with h5py.File(args.file_path, "r") as data_file:
        # print(data_file['data'].keys())
        
        for demo_key in data_file['data'].keys():
            task_name = data_file['data'][demo_key].attrs['task']
            if task_name not in task_demo_id_mapping:
                task_demo_id_mapping[task_name] = []
            task_demo_id_mapping[task_name].append(demo_key)
            
    # save mapping to a json file
    save_path = args.file_path.replace('.hdf5', '_task_demo_id_mapping.json')
    with open(save_path, 'w') as f:
        json.dump(task_demo_id_mapping, f, indent=4)
    