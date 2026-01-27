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


Y_SPAWN_REGION = [[0.255, 0.195], [0.105, 0.045], [-0.045, -0.105], [-0.195, -0.255]]

# ID spawn region - Value list of variation
SPAWN_Task_MAP = {
    0: [0, 1, 2, 3],
    1: [4, 5, 6, 7],
    2: [8, 9, 10, 11],
    3: [12, 13, 14, 15]
}


def filter_hdf5_files(hdf5_files, task_folder, debug=False):
    task_id = int(task_folder.split('_')[-1])
    filtered_files = []
    
    
    for hdf5_file in hdf5_files:
        traj_num = int(hdf5_file.split("/")[-1].split(".")[0].split("traj")[-1])
        with h5py.File(hdf5_file, "r") as f:
            # check spawn position
            spawn_indx = -1
            target_obj_pos_y = f['data'].attrs['target_obj_pos'][1]
            for indx, y_region in enumerate(Y_SPAWN_REGION):
                if target_obj_pos_y <= y_region[0] and target_obj_pos_y >= y_region[1]:
                    spawn_indx = indx
                    break
            # for t, action in enumerate(f['data']['demo_0']['actions_robot']):
            #     if action[-1] == 1.0: # gripper closed
            #         # verify spawn index
            #         pos_y = f['data']['demo_0']['obs']['robot0_eef_pos_3d_world'][t-1][1]
            #         if traj_num == 10 and debug:
            #             print(f"Debug: Traj {traj_num}, pos_y: {pos_y}")
            #         for indx, y_region in enumerate(Y_SPAWN_REGION):
            #             if pos_y <= y_region[0] and pos_y >= y_region[1]:
            #                 spawn_indx = indx
            #                 break
            #         break
                    
            if spawn_indx == -1:
                raise ValueError(f"Could not determine spawn index for file {hdf5_file}")  
                
            # check if task_id and spawn_indx match
            include_file = False
            if task_id in SPAWN_Task_MAP[spawn_indx]:
                filtered_files.append(hdf5_file)
                if debug:
                    print(f"Including {hdf5_file} for task id {task_id} and spawn index {spawn_indx}.")
                include_file = True
            else:
                if debug:
                    print(f"Skipping {hdf5_file} as task id {task_id} not in SPAWN_Task_MAP for spawn index {spawn_indx}.")
                include_file = False
            if True: #and debug:
                # plot image
                pil_img = Image.fromarray(f['data']['demo_0']['obs']['agentview_image'][0])
                pil_img.save(f'debug_images/debug_spawn_task_{task_id}_spawn_{spawn_indx}_include_{include_file}_traj_number_{traj_num}.png')
                
                
    return filtered_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate human dataset')
    parser.add_argument('--task_folder', default="/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/ur5e_pick_place/hdf5_files/", type=str, help='Path to the task folder')
    parser.add_argument('--output_folder', default="/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/ur5e_pick_place/hdf5_files/merged_dataset", type=str, help='Path to the output folder')
    parser.add_argument('--robot_name', default="ur5e", type=str, help='Name of the robot')
    parser.add_argument('--dataset_type', default="all_demos", type=str, help='Type of dataset to generate [same_target_place / all_demos / one_spawn_per_task]')
    parser.add_argument('--config_path', default="config.json", type=str, help='Path to the config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()

    if args.debug:
        debugpy.listen(("localhost", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")

    real_robot = True if "real" in args.task_folder else False
    train_val_split = 0.8

    os.makedirs(args.output_folder, exist_ok=True)
    new_fout = h5py.File(os.path.join(args.output_folder, f"{args.robot_name}_{args.dataset_type}.hdf5"), "w")
    
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    task_folders = glob.glob(os.path.join(args.task_folder, "task_*"))
    task_folders = sorted(task_folders, key=lambda x: int(x.split("_")[-1]))
    
    demo_number = 0
    num_samples = 0
    train_demo_number = []
    val_demo_number = []
    file_cnt = 0
    for task_folder in task_folders:
        print(f"Processing task folder: {task_folder}")
        
        # get all the hdf5 files in the task folder
        hdf5_files = glob.glob(os.path.join(task_folder, "*.hdf5"))
        
        # order hdf5 files by numeric order based on filename
        if args.dataset_type == "same_target_place":
            hdf5_files = sorted(hdf5_files, key=lambda x: int(x.split("/")[-1].split(".")[0].split("traj")[-1]))[0:5:4] # 0 for train, 4 for test
        elif args.dataset_type == "all_demos":
            hdf5_files = sorted(hdf5_files, key=lambda x: int(x.split("/")[-1].split(".")[0].split("traj")[-1]))
        
        
        if args.dataset_type == "one_spawn_per_task":
            # filter hdf5 files based on spawn region
            hdf5_files = filter_hdf5_files(hdf5_files, task_folder, args.debug)
        
        
        
        train_num_files = int(len(hdf5_files) * train_val_split)
        
        for indx, hdf5_file in enumerate(hdf5_files):
            # print(f"\t{hdf5_file}")
            
            # open hdf5 file
            with h5py.File(hdf5_file, "r") as f:
                # read data from hdf5 file
                print(f"hdf5 file keys: {list(f['data'].keys())}")
                                            
                file_cnt += 1
                # get demo_0
                demo_0 = f["data"]["demo_0"]
                
                if 'num_samples' in demo_0.attrs:
                    num_samples += demo_0.attrs["num_samples"]
                else:
                    num_samples += demo_0['obs']['agentview_image'].shape[0]
                
                for key in demo_0.keys():
                    print(f"Key: {key}")
                    
                    if isinstance(demo_0[key], h5py.Dataset):
                        if 'dones' in key:
                            # convert to float64 to save space
                            new_fout.create_dataset(f"data/demo_{demo_number}/{key}", data=demo_0[key][...].astype(np.float64))
                        else:
                            new_fout.create_dataset(f"data/demo_{demo_number}/{key}", data=demo_0[key][...])
                    else:
                        for subkey in demo_0[key].keys():
                            # if '_0' in subkey:
                            #     new_subkey = subkey.replace('_0', f"_{demo_number}")
                            # else:
                            #     new_subkey = subkey
                            new_subkey = subkey
                            if "image" not in subkey:
                                print(f"\tSubkey: {subkey}")
                                
                                if len(demo_0[key][subkey].shape) == 2:
                                    new_fout.create_dataset(f"data/demo_{demo_number}/{key}/{new_subkey}", data=np.array(demo_0[key][subkey][...][:, None, :]))
                                else:
                                    new_fout.create_dataset(f"data/demo_{demo_number}/{key}/{new_subkey}", data=demo_0[key][subkey][...])
                                    
                            elif "image" in subkey:
                                print(f"\tImage Subkey: {subkey}")
                                images = []
                                for t, img in enumerate(demo_0[key][subkey]):
                                    if img.shape[:2] != (120, 120):
                                        crop_params = [0, 30, 120, 120]
                                        top, left = crop_params[0], crop_params[2]
                                        img_height, img_width = img.shape[0], img.shape[1]
                                        box_h, box_w = img_height - top - \
                                        crop_params[1], img_width - left - crop_params[3]
                                        img = img[top:top + box_h, left:left + box_w]
                                        final_img = cv2.resize(img, (120, 120))
                                        images.append(final_img) # save RGB image
                                        # if t == 0 or t == len(demo_0[key][subkey]) - 1:
                                        #     pil_img = Image.fromarray(final_img)
                                        #     pil_img.save(f"debug_demo_{demo_number}_{subkey}_{t}.png")
                                    else:
                                        final_img = copy.deepcopy(img)
                                        images.append(final_img)
                                          
                                new_fout.create_dataset(f"data/demo_{demo_number}/{key}/{new_subkey}", data=images)
                                
                
                if 'num_samples' in demo_0.attrs:
                    new_fout[f'data/demo_{demo_number}'].attrs["num_samples"] = demo_0.attrs["num_samples"]
                else:
                    new_fout[f'data/demo_{demo_number}'].attrs["num_samples"] = demo_0['obs']['agentview_image'].shape[0]
                new_fout[f'data/demo_{demo_number}'].attrs['task'] = task_folder.split('/')[-1]
                new_fout[f'data/demo_{demo_number}'].attrs['target_obj_pos'] = f['data'].attrs['target_obj_pos']    
                print(f"Added demo_{demo_number} from {hdf5_file} with length {new_fout[f'data/demo_{demo_number}'].attrs['num_samples']}")
                
                if args.dataset_type == "same_target_place": 
                    if demo_number % 2 == 0:
                        # 0, 2, 4, ... for training
                        train_demo_number.append(demo_number)
                    else:
                        # 1, 3, 5, ... for validation
                        val_demo_number.append(demo_number)
                elif args.dataset_type == "all_demos":
                    traj_num = int(hdf5_file.split("/")[-1].split(".")[0].split("traj")[-1])
                    if traj_num < train_num_files:
                        train_demo_number.append(demo_number)
                    else:
                        val_demo_number.append(demo_number)
                elif args.dataset_type == "one_spawn_per_task":
                    if indx < train_num_files:
                        train_demo_number.append(demo_number)
                    else:
                        val_demo_number.append(demo_number)

                demo_number += 1
                
    new_fout['data'].attrs['total'] = num_samples
    env_meta = {
        "env_name": "Libero_Kitchen_Tabletop_Manipulation",
        "env_version": "1.4.1",
        "type": 1,
        "env_kwargs": {
            "robots": [
                "Panda"
            ],
            "controller_configs": {
                "type": "OSC_POSE",
                "input_max": 1,
                "input_min": -1,
                "output_max": [
                    0.05,
                    0.05,
                    0.05,
                    0.5,
                    0.5,
                    0.5
                ],
                "output_min": [
                    -0.05,
                    -0.05,
                    -0.05,
                    -0.5,
                    -0.5,
                    -0.5
                ],
                "kp": 150,
                "damping_ratio": 1,
                "impedance_mode": "fixed",
                "kp_limits": [
                    0,
                    300
                ],
                "damping_ratio_limits": [
                    0,
                    10
                ],
                "position_limits": None,
                "orientation_limits": None,
                "uncouple_pos_ori": True,
                "control_delta": True,
                "interpolation": None,
                "ramp_ratio": 0.2
            },
            "bddl_file_name": None,
            "reward_shaping": False,
            "camera_names": [
                "agentview",
                "robot0_eye_in_hand"
            ],
            "camera_heights": 84,
            "camera_widths": 84,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_object_obs": True,
            "use_camera_obs": True,
            "camera_depths": False,
            "render_gpu_device_id": 0
        }
    }
    new_fout['data'].attrs['env_args'] = json.dumps(env_meta, indent=4)             
                
    new_fout.create_dataset('mask/train', data=[f"demo_{i}" for i in train_demo_number])
    new_fout.create_dataset('mask/valid', data=[f"demo_{i}" for i in val_demo_number])
    new_fout.close()
    
    print(f"Total number of demos: {demo_number} from {file_cnt} files with total length: {num_samples}")