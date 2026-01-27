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
from PIL import Image

Y_SPAWN_REGION = [[0.255, 0.195], [0.105, 0.045], [-0.045, -0.105], [-0.195, -0.255]]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_path', 
                        type=str, 
                        default='/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/panda_pick_place/hdf5_files/merged_dataset/panda_all_demos.hdf5',
                        help='Directory containing the agent dataset.')
    parser.add_argument('--human_path', 
                        type=str, 
                        default='/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/human_dataset/hdf5/merged_dataset/human_all_demos.hdf5',
                        help='Directory containing the human dataset.')
    parser.add_argument('--file_name_note',
                        type=str,
                        default='')
    parser.add_argument('--diff_conf', 
                        action='store_true') 
    parser.add_argument('--debug', 
                        action='store_true',
                        help='If true, attach debugger.')
    args = parser.parse_args()
    
    map_id_for_same_conf = {}
    
    agent_name = args.agent_path.split('/')[-1].split('_')[0]
    
    if args.debug:
        debugpy.listen(5678)
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")
    
    os.makedirs('debug_images', exist_ok=True)
    
    # open hdf5 files
    with h5py.File(args.agent_path, "r") as agent_data_file, h5py.File(args.human_path, "r") as human_data_file:
        agent_keys = list(agent_data_file['data'].keys())
        agent_keys.sort(key = lambda x: int(x.split('_')[-1]))
        human_keys = list(human_data_file['data'].keys())
        human_keys.sort(key = lambda x: int(x.split('_')[-1]))
        
        train_keys = []
        val_keys = []
        for train_key in human_data_file['mask']['train']:
            train_keys.append(train_key.decode('utf-8'))
        for val_key in human_data_file['mask']['valid']:
            val_keys.append(val_key.decode('utf-8'))
        
        
        print(f"Number of agent demos: {len(agent_keys)}")
        print(f"Number of human demos: {len(human_keys)}")
        
        
        for agent_key in agent_keys:
            agent_task = agent_data_file['data'][agent_key].attrs['task']
            print(f"Processing agent demo: {agent_key} with task: {agent_task}")
            
            if agent_key not in map_id_for_same_conf:
                map_id_for_same_conf[agent_key] = {}
                map_id_for_same_conf[agent_key]['train'] = []
                map_id_for_same_conf[agent_key]['val'] = []
                
            # check where the target object in agent demo is placed
            # get the first time when the gripper is closed
            target_region_indx = -1

            # get target object position y from attributes
            target_obj_pos_y = agent_data_file['data'][agent_key].attrs['target_obj_pos'][1]
            for region_indx, region in enumerate(Y_SPAWN_REGION):
                if target_obj_pos_y <= region[0] and target_obj_pos_y >= region[1]:
                    print(f"Target obj pos y: {target_obj_pos_y}, Region indx: {region_indx}")
                    target_region_indx = region_indx
                    break
            
            assert target_region_indx != -1, "Could not find target region index"
                            
                
            for human_key in human_keys:
                human_task = human_data_file['data'][human_key].attrs['task']
                
                if human_task != agent_task:
                    continue
                
                
                if not args.diff_conf:
                    # check where the target object in human demo is placed
                    print(f"  Checking human demo: {human_key} with task: {human_task}")
                    human_demo_id = int(human_key.split('_')[-1])
                    
                    human_spawn_region = -(human_demo_id % 4) + 3 # 0 -> 3, 1 -> 2, 2 -> 1,
                    if human_spawn_region == target_region_indx:
                        print(f"    Found matching human demo: {human_key} with spawn region: {human_spawn_region}")
                        if human_key in train_keys:
                            map_id_for_same_conf[agent_key]['train'].append(human_key)
                        elif human_key in val_keys:
                            map_id_for_same_conf[agent_key]['val'].append(human_key)
                        
                        if args.debug:
                            demo_img = Image.fromarray(human_data_file['data'][human_key]['obs']['agentview_image'][0])
                            # demo_img.save(f'debug_images/human_demo_{human_key}_frame_0.png')
                            # concat both images
                            agent_img = Image.fromarray(agent_data_file['data'][agent_key]['obs']['agentview_image'][0])
                            
                            concat_img = Image.new('RGB', (agent_img.width + demo_img.width, agent_img.height))
                            concat_img.paste(agent_img, (0, 0))
                            concat_img.paste(demo_img, (agent_img.width, 0))
                            concat_img.save(f'debug_images/concat_demo_agent_{agent_key}_human_{human_key}.png')
                else:
                    # just map all human demos with same task
                    print(f"    Mapping human demo: {human_key} with task: {human_task}")
                    if human_key in train_keys:
                        map_id_for_same_conf[agent_key]['train'].append(human_key)
                    elif human_key in val_keys:
                        map_id_for_same_conf[agent_key]['val'].append(human_key)

    # save the mapping
    if not args.diff_conf:
        output_file = os.path.join(args.human_path.replace('.hdf5', f'_agent_{agent_name}_map_id_for_{args.file_name_note}.json'))
    else:
        output_file = os.path.join(args.human_path.replace('.hdf5', f'_agent_{agent_name}_map_id_for_same_task_{args.file_name_note}.json'))
    with open(output_file, 'w') as f:
        json.dump(map_id_for_same_conf, f)
                                 
            
            

    
    
    