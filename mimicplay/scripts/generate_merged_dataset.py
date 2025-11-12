import argparse
import os
import h5py
import glob
import debugpy
import cv2
import json
from PIL import Image
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate human dataset')
    parser.add_argument('--task_folder', default="/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/real_new_ur5e_pick_place/hdf5_files/", type=str, help='Path to the task folder')
    parser.add_argument('--output_folder', default="/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/real_new_ur5e_pick_place/hdf5_files/merged_dataset", type=str, help='Path to the output folder')
    parser.add_argument('--dataset_type', default="all_demos", type=str, help='Type of dataset to generate')
    args = parser.parse_args()

    # debugpy.listen(("localhost", 5678))
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()
    # print("Debugger attached.")

    os.makedirs(args.output_folder, exist_ok=True)
    new_fout = h5py.File(os.path.join(args.output_folder, f"ur5e_{args.dataset_type}.hdf5"), "w")
    
    
    task_folders = glob.glob(os.path.join(args.task_folder, "task_*"))
    task_folders = sorted(task_folders, key=lambda x: int(x.split("_")[-1]))
    
    demo_number = 0
    num_samples = 0
    train_demo_number = []
    val_demo_number = []
    file_cnt = 0
    for task_folder in task_folders:
        print(f"Processing task folder: {task_folder}")
        
        hdf5_files = glob.glob(os.path.join(task_folder, "*.hdf5"))
        
        # order hdf5 files by numeric order based on filename
        if args.dataset_type == "same_target_place":
            hdf5_files = sorted(hdf5_files, key=lambda x: int(x.split("/")[-1].split(".")[0].split("traj")[-1]))[0:5:4] # 0 for train, 4 for test
        elif args.dataset_type == "all_demos":
            hdf5_files = sorted(hdf5_files, key=lambda x: int(x.split("/")[-1].split(".")[0].split("traj")[-1]))
        
        
        for hdf5_file in hdf5_files:
            # print(f"\t{hdf5_file}")
            file_cnt += 1
            # open hdf5 file
            with h5py.File(hdf5_file, "r") as f:
                # read data from hdf5 file
                print(f"hdf5 file keys: {list(f['data'].keys())}")
                
                # get demo_0
                demo_0 = f["data"]["demo_0"]
                
                if 'num_samples' in demo_0.attrs:
                    num_samples += demo_0.attrs["num_samples"]
                else:
                    num_samples += demo_0['obs']['agentview_image'].shape[0]
                
                for key in demo_0.keys():
                    print(f"Key: {key}")
                    
                    if isinstance(demo_0[key], h5py.Dataset):
                        new_fout.create_dataset(f"data/demo_{demo_number}/{key}", data=demo_0[key][...])
                    else:
                        for subkey in demo_0[key].keys():
                            if '_0' in subkey:
                                new_subkey = subkey.replace('_0', f"_{demo_number}")
                            else:
                                new_subkey = subkey
                                
                            if subkey != "agentview_image" and subkey != 'front_image_0':
                                print(f"\tSubkey: {subkey}")
                                new_fout.create_dataset(f"data/demo_{demo_number}/{key}/{new_subkey}", data=demo_0[key][subkey][...])
                            elif subkey == "agentview_image" or subkey == 'front_image_0':
                                print(f"\tSubkey: {subkey}")
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
                    if traj_num < 20:
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