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
import multiprocessing as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

TARGET_IMG_SIZE = (120, 120)
CROP_PARAMS = [20, 25, 80, 75]  # top, bottom, left, right
CAMERA_FRONT_POS = np.array([0.45, -0.002826249197217832, 1.27])
CAMERA_FRONT_QUAT = np.array([0.26169506249574287, 0.25790267731943883, 0.6532651777140575, 0.6620018964346217])
fov_y = 60  # degrees
MAP_TARGET_OBJECT = {
    0: 'greenbox',
    1: 'yellowbox',
    2: 'bluebox',
    3: 'redbox'
}


def save_camera_projection(
    rgb_image,
    cam_pos_film,
    fov_y,
    img_width,
    img_height,
    output_path,
    title="Camera-space projection debug"
):
    """
    Save an RGB frame with an overlaid camera-space projection (OpenCV).

    Args:
        rgb_image (np.array): (H, W, 3) RGB image
        cam_pos_film (np.array): (3,) camera film coords (x right, y down, z forward)
        fov_y (float): vertical field of view in degrees
        img_width (int)
        img_height (int)
        output_path (str): path to save the image
    """

    # OpenCV uses BGR internally
    img = rgb_image.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    arrow_len = 20

    x, y, z = cam_pos_film

    if z <= 0:
        cv2.putText(
            img,
            "Point behind camera",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return

    # --- Intrinsics from FOV ---
    f = 0.5 * img_height / np.tan(fov_y * np.pi / 360)  # focal length in pixels

    cx = img_width / 2
    cy = img_height / 2

    # --- Projection ---
    u = int(round(f * (x / z) + cx))
    v = int(round(f * (y / z) + cy))

    # --- Draw ---
    if 0 <= u < img_width and 0 <= v < img_height:
        cv2.drawMarker(
            img,
            (u, v),
            color=(255, 0, 0),  # Red in RGB
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=20 if img_height > 120 else 10,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

        # Draw text
        cv2.putText(
            img,
            f"Projected point: ({u}, {v})",
            (0, 10), # upper-left corner of image
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4 if img_height > 120 else 0.2,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"3D coords (film frame): ({x:.2f}, {y:.2f}, {z:.2f})",
            (0, 30), # slightly below the previous text
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4 if img_height > 120 else 0.2,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        
        
        img = cv2.arrowedLine(img, (int(cx), int(cy)), (int(cx) + arrow_len, int(cy)), (255, 0, 0), 2, tipLength=0.3)
        img = cv2.arrowedLine(img, (int(cx), int(cy)), (int(cx), int(cy) + arrow_len), (0, 255, 0), 2, tipLength=0.3)

        
    else:
        cv2.putText(
            img,
            "Projection out of image bounds",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # Save (convert RGB → BGR for OpenCV)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


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
    # point: (row, col)
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
    

def convert_from_world_to_camera_space(pos, quat, camera_pos, camera_quat, fov_y, img_width, img_height, img, t, debug=False):
    """Convert the robot position 'pos' and orientation 'quat' from world space to camera space

    Args:
        pos (np.array): 3D position in world
        quat (np.array): Quaternion in world
        camera_pos (np.array): Camera position in world
        camera_quat (np.array): Camera orientation in world
        fov_y (float): Field of view in y direction (degrees)
        img_width (int): Image width in pixels
        img_height (int): Image height in pixels
    """
    # World → camera rotation
    R_wc = R.from_quat(camera_quat)
    R_cw = R_wc.inv()

    # Position: translate then rotate
    rel_pos = pos - camera_pos
    cam_pos = R_cw.apply(rel_pos)

    # Orientation: relative rotation
    obj_rot = R.from_quat(quat)
    cam_rot = R_cw * obj_rot

    # -------------------------------------------------
    # Convert to FILM FRAME (x right, y down, z forward)
    # -------------------------------------------------
    # Typical camera frame after rotation:
    #   x right, y up, z forward
    # We flip Y to make it point down
    film_flip = R.from_matrix(
        np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0,  1]
        ])
    )

    cam_pos_film = film_flip.apply(cam_pos)
    cam_rot_film = film_flip * cam_rot

    if debug:
        os.makedirs("tmp_img", exist_ok=True)
        save_camera_projection(
            rgb_image=img,
            cam_pos_film=cam_pos_film,
            fov_y=fov_y,
            img_width=img_width,
            img_height=img_height,
            output_path=f"tmp_img/debug_camera_projection_{t}.png",
        )
        
    # Bring cam_pos_film_px to film frame coordinates of cropped and resized image
    # --- Convert 3D to pixel coords in original image ---
    f = 0.5 * img_height / np.tan(fov_y * np.pi / 360)  # focal length in pixels
    fy = f
    fx = f
    cx = img_width / 2
    cy = img_height / 2

    X, Y, Z = cam_pos_film
    u_orig = fx * (X / Z) + cx
    v_orig = fy * (Y / Z) + cy

    # -------------------------------------------------
    # Adjust for crop + resize
    # -------------------------------------------------
    if TARGET_IMG_SIZE is not None:
        H_target, W_target = TARGET_IMG_SIZE
    else:
        H_target, W_target = img_height, img_width

    if CROP_PARAMS is not None:
        top, bottom, left, right = CROP_PARAMS
        crop_w = img_width - left - right
        crop_h = img_height - top - bottom
        u_crop = u_orig - left
        v_crop = v_orig - top
    else:
        crop_w = img_width
        crop_h = img_height
        u_crop = u_orig
        v_crop = v_orig

    # Resize scale
    sx = W_target / crop_w
    sy = H_target / crop_h
    u_new = u_crop * sx
    v_new = v_crop * sy

    # Back-project to 3D in cropped/resized film frame
    fx_new = fx * sx
    fy_new = fy * sy
    cx_new = W_target / 2
    cy_new = H_target / 2

    X_new = (u_new - cx_new) * Z / fx_new
    Y_new = (v_new - cy_new) * Z / fy_new
    cam_pos_film_resized = np.array([X_new, Y_new, Z])

    fy_new_degree = (np.arctan(0.5 * H_target / fy_new)) * (360 / np.pi)

    # -------------------------------------------------
    # Debug: plot on cropped/resized image
    # -------------------------------------------------
    if debug:
        cropped_img = img[top:top + crop_h, left:left + crop_w]
        resized_img = cv2.resize(cropped_img, (W_target, H_target))
        img = resized_img.copy() 
        
        # plot projection
        cv2.drawMarker(
            img,
            (int(u_new), int(v_new)),
            color=(255, 0, 0),  # Red in RGB
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=5,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

        # Draw text
        cv2.putText(
            img,
            f"({int(u_new)}, {int(v_new)})",
            (0, 10), # upper-left corner of image
            cv2.FONT_HERSHEY_SIMPLEX,
            0.2,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"({X_new:.2f}, {Y_new:.2f}, {Z:.2f})",
            (0, 30), # slightly below the previous text
            cv2.FONT_HERSHEY_SIMPLEX,
            0.2,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        
        
        img = cv2.arrowedLine(img, (int(cx_new), int(cy_new)), (int(cx_new) + 10, int(cy_new)), (255, 0, 0), 2, tipLength=0.3)
        img = cv2.arrowedLine(img, (int(cx_new), int(cy_new)), (int(cx_new), int(cy_new) + 10), (0, 255, 0), 2, tipLength=0.3)
        cv2.imwrite(f"tmp_img/debug_cropped_resized_camera_projection_{t}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        
    return cam_pos_film_resized, cam_rot_film.as_quat()





def process_single_data_file(task, data_file, args):
    demo_name = 'demo_0'

    with open(data_file, 'rb') as f:
        data = pkl.load(f)
        traj = data['traj']

    gripper_state = -1
    agent_view_list = []
    gripper_view_list = []
    eef_pos_3d_list_world = []
    eef_pos_3d_list_camera = []
    eef_pos_px_list = []
    eef_quat_list_world = []
    eef_quat_list_camera = []
    gripper_state_list = []
    action_list = []
    reward_list = []
    done_list = []
    robot_future_eef_pos_list = []

    for t in range(1, len(traj)):
        
        if t == 1:
            # get position of target object at the start of the demo
            target_object_indx = traj[t]['obs']['target-object']
            target_obj_name = MAP_TARGET_OBJECT[target_object_indx]
            target_obj_pos_world = traj[t]['obs'][f'{target_obj_name}_pos']
             
        # 1. Agent view
        agent_view = crop_and_resize(
            traj[t]['obs']['camera_front_image'],
            target_size=TARGET_IMG_SIZE
        )

        # 2. Gripper view
        camera_gripper_image = cv2.flip(
            traj[t]['obs']['eye_in_hand_image'], 1
        )
        camera_gripper_image = cv2.resize(
            camera_gripper_image,
            TARGET_IMG_SIZE,
            interpolation=cv2.INTER_LINEAR
        )

        # 3. EEF position
        eef_pos_3D_world = traj[t]['obs']['eef_pos']
        # 4. EEF orientation
        eef_quat_world = traj[t]['obs']['eef_quat']
        R_ee_to_gripper = np.array([[0, -1, 0],
                                    [1,  0, 0],
                                    [0,  0, 1]])
        eef_mat = R_ee_to_gripper @ quat2mat(eef_quat_world)
        eef_euler = [normalize_angle(a) for a in mat2euler(eef_mat)]
        eef_quat_world = mat2quat(euler2mat(eef_euler))

        eef_pos_3D_camera, eef_quat_camera = convert_from_world_to_camera_space(
            pos=eef_pos_3D_world.copy(),
            quat=eef_quat_world.copy(),
            camera_pos=CAMERA_FRONT_POS,
            camera_quat=CAMERA_FRONT_QUAT,
            fov_y=fov_y,
            img_width=traj[t]['obs']['camera_front_image'].shape[1],
            img_height=traj[t]['obs']['camera_front_image'].shape[0],
            img=traj[t]['obs']['camera_front_image'],
            t=t,
            debug=args.debug,
        )
        
        # Store EEF pos in pixel space
        eef_pos_px = convert_from_3D_to_px_space(
            pos=traj[t]['obs']['eef_pos'],
            camera_pos=CAMERA_FRONT_POS,
            camera_quat=CAMERA_FRONT_QUAT,
            fovy=fov_y,
            img_width=traj[t]['obs']['camera_front_image'].shape[1],
            img_height=traj[t]['obs']['camera_front_image'].shape[0],
        )
        eef_pos_px = adjust_point_to_cropped_resized_image(
            point=eef_pos_px,
            original_img_shape=traj[t]['obs']['camera_front_image'].shape,
            target_size=TARGET_IMG_SIZE,
            crop_params=CROP_PARAMS,
        )
        eef_pos_px = np.array(eef_pos_px) / agent_view.shape[0]

        # 5. Future EEF trajectory
        num_future_frame = 20 #10
        skip_len = 2
        robot_future_eef_pos = []

        for i in range(num_future_frame):
            idx = min(t + (i + 1) * skip_len, len(traj) - 1)
            next_px = convert_from_3D_to_px_space(
                pos=traj[idx]['obs']['eef_pos'],
                camera_pos=CAMERA_FRONT_POS,
                camera_quat=CAMERA_FRONT_QUAT,
                fovy=fov_y,
                img_width=traj[t]['obs']['camera_front_image'].shape[1],
                img_height=traj[t]['obs']['camera_front_image'].shape[0],
            )
            next_px = adjust_point_to_cropped_resized_image(
                point=next_px,
                original_img_shape=traj[t]['obs']['camera_front_image'].shape,
                target_size=TARGET_IMG_SIZE,
                crop_params=CROP_PARAMS,
            )
            robot_future_eef_pos.extend(
                np.array(next_px) / agent_view.shape[0]
            )

        robot_future_eef_pos_list.append(robot_future_eef_pos)

        # 6. Action
        if t < len(traj) - 1:
            action = convert_action(
                next_pos=traj[t + 1]['obs']['eef_pos'],
                current_pos=traj[t]['obs']['eef_pos'],
                next_quat=traj[t + 1]['obs']['eef_quat'],
                current_quat=traj[t]['obs']['eef_quat'],
                gripper_state=traj[t]['action'][-1],
            )
            action_list.append(action)
        elif t == len(traj) - 1:
            action = np.zeros(7, dtype=np.float64)
            action_list.append(action)

        gripper_state = traj[t]['action'][-1]

        agent_view_list.append(agent_view)
        gripper_view_list.append(camera_gripper_image)
        eef_pos_3d_list_camera.append(eef_pos_3D_camera)
        eef_quat_list_camera.append(eef_quat_camera)
        eef_pos_3d_list_world.append(eef_pos_3D_world)
        eef_quat_list_world.append(eef_quat_world)
        eef_pos_px_list.append(eef_pos_px)
        gripper_state_list.append([gripper_state])
        reward_list.append(traj[t]['reward'])
        done_list.append(0 if t < len(traj) - 1 else 1)

    # ===============================
    # SAVE HDF5
    # ===============================
    task_name = os.path.basename(task)
    output_dir = os.path.join(args.output_path, task_name)
    os.makedirs(output_dir, exist_ok=True)

    trajectory_name = os.path.splitext(os.path.basename(data_file))[0]
    output_file = os.path.join(output_dir, f'{trajectory_name}.hdf5')

    with h5py.File(output_file, 'w') as hf:
        obs_path = f'data/{demo_name}/obs'
        hf.create_dataset(obs_path + '/agentview_image', data=np.array(agent_view_list))
        hf.create_dataset(obs_path + '/robot0_eye_in_hand_image', data=np.array(gripper_view_list))
        
        hf.create_dataset(obs_path + '/robot0_eef_pos_3d_camera', data=np.array(eef_pos_3d_list_camera))
        hf.create_dataset(obs_path + '/robot0_eef_quat_camera', data=np.array(eef_quat_list_camera))
        hf.create_dataset(obs_path + '/robot0_eef_pos_3d_world', data=np.array(eef_pos_3d_list_world))
        hf.create_dataset(obs_path + '/robot0_eef_quat_world', data=np.array(eef_quat_list_world))
        hf.create_dataset(obs_path + '/robot0_eef_pos_px', data=np.array(eef_pos_px_list))
        
        hf.create_dataset(obs_path + '/robot0_gripper_qpos', data=np.array(gripper_state_list))
        hf.create_dataset(obs_path + '/robot0_eef_pos_future_traj', data=np.array(robot_future_eef_pos_list))

        hf['data'].attrs['total'] = len(action_list)
        hf['data'].attrs['task'] = task_name
        hf['data'].attrs['target_obj_pos'] = target_obj_pos_world
        
        hf.create_dataset(f'data/{demo_name}/actions_robot', data=np.array(action_list))
        hf.create_dataset(f'data/{demo_name}/actions', data=np.array(robot_future_eef_pos_list))
        hf.create_dataset(f'data/{demo_name}/rewards', data=np.array(reward_list))
        hf.create_dataset(f'data/{demo_name}/dones', data=np.array(done_list))
        
        print(f"Saved HDF5 file: {output_file}\n\tObs kyes under '{obs_path}': {list(hf[obs_path].keys())}\n\tData keys under 'data/{demo_name}': {list(hf[f'data/{demo_name}'].keys())}")


# ===============================
# MAIN
# ===============================
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', 
                        type=str, 
                        default='/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/ur5e_pick_place/',
                        help='Directory containing the raw data files.')
    parser.add_argument('--output_path', 
                        type=str,
                        default='/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/ur5e_pick_place/hdf5_files',
                        help='Path to save the output HDF5 file.')
    parser.add_argument('--plot', 
                        action='store_true')
    parser.add_argument('--debug', 
                        action='store_true')
    parser.add_argument('--num_workers', type=int, default=10)

    args = parser.parse_args()

    task_list = sorted(
        glob.glob(os.path.join(args.data_dir, 'task_*')),
        key=lambda x: int(x.split('_')[-1])
    )

    if args.debug:
        debugpy.listen(5678)
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")


    jobs = []
    for task in task_list:
        data_files = sorted(
            glob.glob(os.path.join(task, '*.pkl')),
            key=lambda x: int(x.split('_')[-1].split('traj')[-1].split('.')[0])
        )
        for data_file in data_files:
            jobs.append((task, data_file, args))

    print(f'Processing {len(jobs)} trajectories with {args.num_workers} workers')

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_single_data_file, task, data_file, args)
            for task, data_file, args in jobs
        ]

        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc='Processing trajectories',
            unit='traj'
        ):
            f.result()

    print('✅ All trajectories processed successfully.')  
                
                
                