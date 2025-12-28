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

def convert_from_3D_to_px_space(pos, camera_pos, camera_quat, fovy=60, img_width=320, img_height=200):
    model_matrix = np.zeros((3, 4))
    model_matrix[:3, :3] = quat2mat(camera_quat).T
    
    f = 0.5 * img_height / np.tan(fovy * np.pi / 360)
    camera_matrix = np.array(
        ((f, 0, img_width / 2), (0, f, img_height / 2), (0, 0, 1)))
    
    MVP_matrix = camera_matrix.dot(model_matrix)
    cam_coord = np.ones((4, 1))
    cam_coord[:3, 0] = pos - camera_pos

    clip = MVP_matrix.dot(cam_coord)
    row, col = clip[:2].reshape(-1) / clip[2]
    row, col = row, img_height - col

    point = (int(max(col, 0)), int(max(row, 0)))
    
    flip_points = np.zeros(2)
    flip_points[0] = int(img_height - point[0])
    flip_points[1] = int(img_width - point[1])
    
    return flip_points.astype(int)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', 
                        type=str, 
                        default='/user/frosa/multi_task_lfd/datasets/pick_place/human_rgb_pick_place/hdf5/merged_dataset/human_pick_place_all_demos.hdf5',
                        help='Directory containing the raw data files.')
    
    
    args = parser.parse_args()
    len_list = []
    # open hdf5 file
    with h5py.File(args.file_path, "r") as data_file:
        # print(data_file['data'].keys())
        
        for demo_key in data_file['data'].keys():
            #print(data_file['data'][demo_key]['actions'].shape[0])
            len_list.append(data_file['data'][demo_key]['actions'].shape[0])
            
    print(f"Average frame number {np.mean(len_list)}")
    print(f"Std frame {np.std(len_list)}")