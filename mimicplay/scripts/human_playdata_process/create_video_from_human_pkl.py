"""__summary__
Given a human playdata pkl file, create a video from the frames stored in the pkl.
"""
import argparse
import os
import pickle as pkl
import glob
from PIL import Image
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

MIN_DEPTH=0.10 
MAX_DEPTH=2.0

# Normalize and convert depth to 8-bit
def depth_to_bgr(depth):
    depth_norm = (depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH + 1e-8)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_3c = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
    # Optional colormap:
    # depth_3c = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    return depth_3c


# ---------------- Process single pkl file ----------------
def process_pkl_file(pkl_file, tmp_img, video_save_dir):
    print(f"Processing pkl file: {os.path.basename(pkl_file)}")
    
    with open(pkl_file, 'rb') as f:
        data = pkl.load(f)
        traj = data['traj']

    # create video writers
    video_writer_list = []
    camera_names = [name for name in traj[0]['obs'].keys() if 'camera' in name]
    height, width = traj[0]['obs']['camera_front_image'].shape[:2]

    for camera_name in camera_names:
        video_path = os.path.join(
            video_save_dir, f"{os.path.basename(pkl_file).split('.')[0]}_{camera_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        video_writer_list.append((camera_name, video_writer))

    # iterate over frames
    for t in range(len(traj)):
        # --- RGB images ---
        front_image = cv2.cvtColor(traj[t]['obs']['camera_front_image'], cv2.COLOR_BGR2RGB)
        lateral_right_image = cv2.cvtColor(
            cv2.imdecode(traj[t]['obs']['camera_lateral_right_image'], cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB
        )
        lateral_left_image = cv2.cvtColor(
            cv2.imdecode(traj[t]['obs']['camera_lateral_left_image'], cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB
        )

        # --- Depth images ---
        front_depth_bgr = depth_to_bgr(traj[t]['obs']['camera_front_depth'])
        lateral_right_depth_bgr = depth_to_bgr(traj[t]['obs']['camera_lateral_right_depth'])
        lateral_left_depth_bgr = depth_to_bgr(traj[t]['obs']['camera_lateral_left_depth'])

        # --- Save tmp images ---
        # Image.fromarray(front_image).save(os.path.join(tmp_img, f"{t:04d}_front.png"))
        # Image.fromarray(lateral_right_image).save(os.path.join(tmp_img, f"{t:04d}_lateral_right.png"))
        # Image.fromarray(lateral_left_image).save(os.path.join(tmp_img, f"{t:04d}_lateral_left.png"))

        # Image.fromarray(front_depth_bgr[:, :, 0]).save(os.path.join(tmp_img, f"{t:04d}_front_depth.png"))
        # Image.fromarray(lateral_right_depth_bgr[:, :, 0]).save(os.path.join(tmp_img, f"{t:04d}_lateral_right_depth.png"))
        # Image.fromarray(lateral_left_depth_bgr[:, :, 0]).save(os.path.join(tmp_img, f"{t:04d}_lateral_left_depth.png"))

        # --- Write frames to video ---
        for camera_name, video_writer in video_writer_list:
            if 'camera_front_image' in camera_name:
                frame = front_image[:, :, ::-1]
            elif 'camera_lateral_right_image' in camera_name:
                frame = lateral_right_image[:, :, ::-1]
            elif 'camera_lateral_left_image' in camera_name:
                frame = lateral_left_image[:, :, ::-1]
            elif 'camera_front_depth' in camera_name:
                frame = front_depth_bgr
            elif 'camera_lateral_right_depth' in camera_name:
                frame = lateral_right_depth_bgr
            elif 'camera_lateral_left_depth' in camera_name:
                frame = lateral_left_depth_bgr
            else:
                raise ValueError(f"Unknown camera name: {camera_name}")

            video_writer.write(frame)

    # release video writers
    for _, video_writer in video_writer_list:
        video_writer.release()

    # remove tmp images
    # tmp_images = glob.glob(os.path.join(tmp_img, '*.png'))
    # for img_file in tmp_images:
    #     os.remove(img_file)


# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_path', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    tasks_dir = glob.glob(os.path.join(args.task_path, 'task_**'))
    tasks_dir.sort(key=lambda x: int(x.split('_')[-1]))

    tmp_img = './tmp_img'
    os.makedirs(tmp_img, exist_ok=True)

    # ---------------- Parallel processing ----------------
    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = []
        for task_dir in tasks_dir[:1]:
            print(f"Processing task directory: {os.path.basename(task_dir)}")
            pkl_files = glob.glob(os.path.join(task_dir, 'traj*.pkl'))
            pkl_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('traj')[-1]))
            video_save_dir = os.path.join(task_dir, 'videos')
            os.makedirs(video_save_dir, exist_ok=True)

            for pkl_file in pkl_files[:4]:
                futures.append(executor.submit(process_pkl_file, pkl_file, tmp_img, video_save_dir))

        # wait for all to complete
        for future in as_completed(futures):
            future.result()  # raises exceptions if any