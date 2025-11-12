import pickle as pkl
import os
import glob
from PIL import Image
import cv2
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'X264')
DATA_FOLDER="/user/frosa/multi_task_lfd/datasets/pick_place/human_rgb_pick_place"
#DATA_FOLDER="/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/human_video/"


if __name__ == "__main__":
    # Codec that works with mp4 and VS Code preview
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    task_folders = glob.glob(os.path.join(DATA_FOLDER, "task_*"))
    for task_folder in task_folders:  # first task folder
        print(f"Processing {task_folder}")
        save_folder = os.path.join(task_folder, "videos")
        os.makedirs(save_folder, exist_ok=True)

        video_files = glob.glob(os.path.join(task_folder, "traj*.pkl"))
        for video_file in video_files:  # first 2 pkl files
            print(f"  Creating video for {video_file}")

            with open(video_file, "rb") as f:
                trj = pkl.load(f)["traj"]

            # use the shape of the first frame
            first_img = trj[0]["obs"]["camera_front_image"]
            height, width = first_img.shape[:2]

            # save as .mp4 (playable in VS Code and system player)
            out_path = os.path.join(save_folder, f"{os.path.basename(video_file).split('.pkl')[0]}.mp4")
            out = cv2.VideoWriter(out_path, fourcc, 20.0, (width, height))

            for t in range(len(trj)):
                # print( trj[t]['obs'].keys())
                if 'depth' in trj[t]['obs'].keys():
                    print(f"Depth image found in {task_folder}")
                img = trj[t]["obs"]["camera_front_image"][:,:,::-1]

                # ensure uint8 + 3 channels
                if img.dtype != "uint8":
                    img = img.astype("uint8")
                # if len(img.shape) == 2:
                #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                out.write(img)

            out.release()
            print(f"  Saved video: {out_path}")