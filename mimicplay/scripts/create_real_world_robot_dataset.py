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
from  PIL import Image

TARGET_IMG_SIZE = (120, 120)
CROP_PARAMS = [0, 30, 120, 120] # top, bottom, left, right
CAMERA_POS = np.array([[0.022843813138628592, -0.43800020977692405, 0.5643843146648674]])
CAMERA_QUAT = np.array([0.3603325062389276, 0.015749675284185274, -0.0008269422755895826, 0.9326905965230317])  # xyzw
T_table_bl = np.array([[-1, 0.0, 0, 0.01],
                        [0.0, -1.0, 0, 0.612],
                        [0, 0, 1, 0.120],
                        [0, 0, 0, 1]])

camera_intrinsic = np.array([[345.2712097167969, 0.0, 337.5007629394531],
                             [0.0, 345.2712097167969,
                              179.0137176513672],
                             [0, 0, 1]])

film_px_offset = np.array([[337.5007629394531],
                           [179.0137176513672]])

def normalize_angle(angle, tol=1e-1):
    """
    Normalize angle to (-π, π], where -π wraps to π
    """
    norm = (angle + np.pi) % (2 * np.pi) - np.pi
    if np.isclose(norm, -np.pi, atol=tol):
        norm = np.pi
    return norm

def crop_and_resize(image, target_size=(120, 120)):
    crop_params = CROP_PARAMS
    top, left = crop_params[0], crop_params[2]
    img_height, img_width = image.shape[0], image.shape[1]
    box_h, box_w = img_height - top - \
    crop_params[1], img_width - left - crop_params[3]

    image = image[top:top + box_h, left:left + box_w]
    image = cv2.resize(image, target_size)
    return image

def adjust_point_to_cropped_resized_image(point, original_img_shape, target_size=(120, 120), crop_params=CROP_PARAMS):
    # point: (row, col) format
    top, left = crop_params[0], crop_params[2]
    img_height, img_width = original_img_shape[0], original_img_shape[1]
    box_h, box_w = img_height - top - \
    crop_params[1], img_width - left - crop_params[3]
    scale_y = target_size[0] / box_h
    scale_x = target_size[1] / box_w
    adjusted_point = (int((point[0] - top) * scale_y), int((point[1] - left) * scale_x))
    return adjusted_point

def convert_action(next_pos, current_pos, next_quat, current_quat, gripper_state=None):
    """Convert action from world frame to robot frame using delta angles."""
    # delta_x, delta_y, delta_z end_effector w.r.t. the world frame
    # r, p, y end_effector orientation w.r.t. the world frame

    new_action = np.zeros(7, dtype=np.float64)

    # Position delta
    new_action[0:3] = next_pos[0:3] - current_pos[0:3]
    # print(f"Position delta: {new_action[0:3]} - Gripper state: {gripper_state}")
    # set mm changes to zeros
    # new_action[0:3] = np.where(np.abs(new_action[0:3]) < 0.01, 0, new_action[0:3])
    
    # Convert quaternions to rotation matrices
    R_ee_to_gripper = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ])

    R_current = R_ee_to_gripper @ quat2mat(current_quat)
    R_next    = R_ee_to_gripper @ quat2mat(next_quat)

    # Compute relative rotation matrix
    R_delta = R_next @ R_current.T

    # Convert relative rotation to Euler angles (delta angles)
    delta_euler = mat2euler(R_delta)

    # Normalize angles to [-π, π)
    delta_euler = [normalize_angle(a) for a in delta_euler]

    # print(f"delta_euler: {delta_euler}")

    new_action[3:6] = delta_euler
    new_action[6] = gripper_state

    return new_action.astype(np.float64)

def convert_from_3D_to_px_space(gripper_pose, camera_intrinsic, T_table_bl, T_camera_table,film_px_offset, img_width=320, img_height=200):
    # convert 3D position to pixel space using camera intrinsic and extrinsic

    gripper_pos_bl = np.array([gripper_pose[:3]]).T
    gripper_quat_bl = np.array(gripper_pose[3:-1])
    gripper_rot_bl = quat2mat(
        np.array(gripper_quat_bl))
    T_gripper_bl = np.concatenate(
        (gripper_rot_bl, gripper_pos_bl), axis=1)
    T_gripper_bl = np.concatenate(
        (T_gripper_bl, np.array([[0, 0, 0, 1]])), axis=0)

    TCP_table = T_table_bl @ T_gripper_bl

    tcp_camera = np.array([(T_camera_table @ TCP_table)[:3, -1]]).T
    tcp_camera_scaled = tcp_camera / tcp_camera[2][0]
    tcp_camera_scaled[0][0] = - tcp_camera_scaled[0][0]
    
    tcp_pixel_cord = np.array(camera_intrinsic @ tcp_camera_scaled, dtype=np.uint32)
    # x, y pixel coordinates (x: columns, y: rows)
    tcp_pixel_cord = tcp_pixel_cord.T[0][:2]
    # print(f"Projected pixel coordinates before flipping: {tcp_pixel_cord}")
    x_px = tcp_pixel_cord[0]
    y_px = tcp_pixel_cord[1]
    
    return np.array([y_px, x_px])  # return (row, col) format
    
    # return tcp_pixel_cord.T[0][:2]
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', 
                        type=str, 
                        default='/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/real_new_ur5e_pick_place',
                        help='Directory containing the raw data files.')
    parser.add_argument('--output_path', 
                        type=str,
                        default='/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/real_new_ur5e_pick_place/hdf5_files',
                        help='Path to save the output HDF5 file.')
    parser.add_argument('--convert_from_3D_to_px_space_flag', 
                        action='store_true', 
                        help='If set, include end-effector position in pixel space.')
    args = parser.parse_args()

    # debugpy.listen(5678)
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()

    task_list = glob.glob(os.path.join(args.data_dir, 'task_*'))
    
    task_list.sort(key=lambda x: int(x.split('_')[-1]))

    camera_quat = CAMERA_QUAT
    r_camera_table = quat2mat(
        np.array(camera_quat)).T
    p_camera_table = -np.matmul(r_camera_table, np.array(CAMERA_POS).T)
    T_camera_table = np.append(
        r_camera_table, p_camera_table, axis=1)

    for task in task_list:
        print(f'Processing task directory: {task}')
        data_files = glob.glob(os.path.join(task, '*.pkl'))
        data_files.sort(key=lambda x: int(x.split('_')[-1].split('traj')[-1].split('.')[0]))

        for data_file in data_files:
            demo_name = 'demo_0'
            
            print(f'Processing data file: {data_file}')
            with open(data_file, 'rb') as f:
                data = pkl.load(f)
                traj = data['traj']
            
            gripper_state = -1
            agent_view_list = []
            gripper_view_list = []
            eef_pos_list = []
            eef_quat_list = []
            gripper_state_list = []
            action_list = []
            reward_list = []
            done_list = []
            robot_future_eef_pos_list = []
            for t in range(len(traj)):
                if t == 0:
                    continue  # Skip the first timestep as it doesn't have a valid action
                
                # 1. Get the agent image
                agent_view = traj[t]['obs']['camera_front_image'][:,:,::-1]  # Convert BGR to RGB
                agent_view = crop_and_resize(agent_view, target_size=TARGET_IMG_SIZE)
                # 2. Get the gripper image
                try:
                    camera_gripper_image = cv2.flip(traj[t]['obs']['eye_in_hand_image'], 1)
                    camera_gripper_image = cv2.resize(camera_gripper_image, TARGET_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    # print(f"Error processing gripper image at timestep {t}: {e}")
                    pass

                # 3. Get the end-effector position and orientation
                if not args.convert_from_3D_to_px_space_flag:
                    eef_pos = traj[t]['obs']['eef_pos']
                    
                else:
                    print("Converting from 3D to pixel space...")
                    eef_pos = convert_from_3D_to_px_space(
                        gripper_pose = traj[t]['action'],
                        camera_intrinsic = camera_intrinsic,
                        T_table_bl = T_table_bl,
                        T_camera_table = T_camera_table,
                        film_px_offset = film_px_offset,
                        img_width = traj[t]['obs']['camera_front_image'].shape[1],
                        img_height = traj[t]['obs']['camera_front_image'].shape[0]
                    ) 
                    os.makedirs('debug_images', exist_ok=True)
                    cv2.circle( traj[t]['obs']['camera_front_image'], 
                               (eef_pos[1], eef_pos[0]), 
                               3, 
                               (0, 0, 255), 
                               3)
                    # pil_img = Image.fromarray(traj[t]['obs']['camera_front_image'][:,:,::-1])
                    # pil_img.save(f'debug_images/original_image_{t}.png')

                    eef_pos = adjust_point_to_cropped_resized_image(
                        point = eef_pos,
                        original_img_shape = traj[t]['obs']['camera_front_image'].shape,
                        target_size = TARGET_IMG_SIZE,
                        crop_params = CROP_PARAMS
                    )

                    # plot the adjusted point on the cropped and resized image
                    os.makedirs('debug_images', exist_ok=True)
                    to_plot = copy.deepcopy(agent_view)
                    cv2.circle(to_plot, 
                               (eef_pos[1], eef_pos[0]), 
                               3, 
                               (0, 0, 255), 
                               3)
                    # pil_img = Image.fromarray(to_plot)
                    # pil_img.save(f'debug_images/agent_view_cropped_resized_{t}.png')
                    eef_pos = np.array(eef_pos) / agent_view.shape[0]  # normalize to [0, 1]
                    
                    

                eef_quat = traj[t]['obs']['eef_quat']
                
                # 4. Get the gripper state
                gripper_q_pos = copy.deepcopy(gripper_state)

                num_future_frame = 10
                skip_len = 2
                robot_future_eef_pos = []
                to_plot = copy.deepcopy(agent_view)
                for i in range(num_future_frame):
                    # convert from 3D to 2D in image space
                    next_pos_px_space = convert_from_3D_to_px_space(
                        gripper_pose = traj[min(t + ((i+1)*skip_len), len(traj)-1)]['action'],
                        camera_intrinsic = camera_intrinsic,
                        T_table_bl = T_table_bl,
                        T_camera_table = T_camera_table,
                        film_px_offset = film_px_offset,
                        img_width = traj[t]['obs']['camera_front_image'].shape[1],
                        img_height = traj[t]['obs']['camera_front_image'].shape[0]
                    ) 
                    
                    # plot the projected point on the agent view image
                    # cv2.circle(np.asarray(traj[t]['obs']['camera_front_image']), (next_pos_px_space[1], next_pos_px_space[0]), 3, (0, 0, 255), 3)
                    # cv2.imwrite(f'agent_view_{t}_{i}.png', traj[t]['obs']['camera_front_image'][:,:,::-1])
                    
                    # adjust point to cropped and resized image
                    next_pos_space = adjust_point_to_cropped_resized_image(
                        point = next_pos_px_space,
                        original_img_shape = traj[t]['obs']['camera_front_image'].shape,
                        target_size = TARGET_IMG_SIZE,
                        crop_params = CROP_PARAMS
                     )
                   
                    # plot the adjusted point on the cropped and resized image
                    # cv2.circle(to_plot, (next_pos_space[1], next_pos_space[0]), 3, (0, 0, 255), 3)
                    # pil_img = Image.fromarray(to_plot)
                    # pil_img.save(f'debug_images/agent_view_cropped_resized_{t}_{i}.png')
                    
                    robot_future_eef_pos.extend(next_pos_px_space/agent_view.shape[0])  # normalize to [0, 1]
                
                robot_future_eef_pos_list.append(robot_future_eef_pos)

                # 5. Compute the action
                if t < len(traj) - 1:
                    next_eef_pos = traj[t + 1]['obs']['eef_pos']
                    next_eef_quat = traj[t + 1]['obs']['eef_quat']
                    action = convert_action(
                        next_pos = next_eef_pos,
                        current_pos = traj[t]['obs']['eef_pos'],
                        next_quat = next_eef_quat,
                        current_quat= traj[t]['obs']['eef_quat'],
                        gripper_state = traj[t]['action'][-1]
                    )
                    
                # update gripper state
                gripper_state = traj[t]['action'][-1]
                
                agent_view_list.append(agent_view)
                # gripper_view_list.append(camera_gripper_image)
                eef_pos_list.append(eef_pos)
                eef_quat_list.append(eef_quat)
                gripper_state_list.append(gripper_q_pos)
                if t < len(traj) - 1:
                    action_list.append(action)
                
                reward_list.append(traj[t]['reward'])
                done_list.append( 0 if t < len(traj) - 1 else 1 )
                
                
            # 6. Save to HDF5
            task_name = task.split('/')[-1]
            output_dir = os.path.join(args.output_path, task_name)
            os.makedirs(output_dir, exist_ok=True)
            trajectory_name = data_file.split('/')[-1].split('.')[0]
            output_file = os.path.join(output_dir, f'{trajectory_name}.hdf5')
            
            with h5py.File(output_file, 'w') as hf:
                obs_path = 'data/'+demo_name+'/obs'
                hf.create_dataset(obs_path+'/agentview_image', data=np.array(agent_view_list))
                # hf.create_dataset(obs_path+'/robot0_eye_in_hand_image', data=np.array(gripper_view_list))
                hf.create_dataset(obs_path+'/robot0_eef_pos', data=np.array(eef_pos_list))
                hf.create_dataset(obs_path+'/robot0_eef_quat', data=np.array(eef_quat_list))
                hf.create_dataset(obs_path+'/robot0_gripper_state', data=np.array(gripper_state_list))
                hf.create_dataset(obs_path+'/robot0_eef_pos_future_traj', data=np.array(robot_future_eef_pos_list))

                hf['data'].attrs['total'] = len(action_list)
                hf['data'].attrs['task'] = task_name

                hf.create_dataset('data/'+demo_name+'/action', data=np.array(action_list))
                hf.create_dataset('data/'+demo_name+'/reward', data=np.array(reward_list))
                hf.create_dataset('data/'+demo_name+'/done', data=np.array(done_list))
                
                
                
                
           
            
            
            
            
                
                