import h5py
import numpy as np
import cv2
from PIL import Image
import argparse
import os
import debugpy

# view_id = 0 # change to 2 if drawing on second view
# FILE_NAME="/user/frosa/multi_task_lfd/datasets/pick_place/human_rgb_pick_place/hdf5/task_00/traj039.hdf5"

# with h5py.File(FILE_NAME, 'r') as f:
#     images = np.array(f['data/demo_0/obs/front_image_{}'.format(view_id)])
#     actions = np.array(f['data/demo_0/actions'])
#     # print(actions.shape)
#     num_timesteps = actions.shape[0]

# # Reshape the actions to [145, 10, 4]
# actions = actions.reshape((num_timesteps, 10, 2))

# if view_id == 0:
#     actions = actions[:, :, :2]
# elif view_id == 1:
#     actions = actions[:, :, 2:4]
# elif view_id == 2:
#     actions = actions[:, :, 4:]


# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# task_name=FILE_NAME.split('/')[-2]
# trajectory_name=FILE_NAME.split('/')[-1].split('.')[0]
# video = cv2.VideoWriter(f'output_{task_name}_{trajectory_name}.mp4', fourcc, 30.0, (120, 120))
# print(f"Video image {images.shape}.")

# for i in range(images.shape[0]):
#     img = images[i].copy()
    
#     crop_params = [0, 30, 120, 120]
#     top, left = crop_params[0], crop_params[2]
#     img_height, img_width = img.shape[0], img.shape[1]
#     box_h, box_w = img_height - top - \
#     crop_params[1], img_width - left - crop_params[3]
#     img = img[top:top + box_h, left:left + box_w]
#     img = cv2.resize(img, (120, 120))
    
#     action = actions[i]
#     print(action.shape)

#     action_unscaled = action * np.array([img.shape[1], img.shape[0]])
    
#     for pt in action_unscaled:
#         # save cropped image
#         # cv2.imwrite('cropped_img.png', im_in)
#         print(pt)
#         img = cv2.circle(img, (int(pt[1]), int(pt[0])), radius=5, color=(0, 255, 0), thickness=-1)

#     # pil_img = Image.fromarray(img)
#     # pil_img.save(f'temp_{i}.png')
    
#     video.write(img[:, :, ::-1])  # Convert RGB to BGR for OpenCV
    
# video.release()

# print("The video has been successfully saved as output.mp4")

def plot_actions_on_image(image, actions, t):
    img = image.copy()
    
    if image.shape != (120, 120, 3):
        raise ValueError("Image must be of shape (120, 120, 3)")
    
    
    action_unscaled = actions * np.array(img.shape[1])
    num_actions = int(action_unscaled.shape[0]/2)
    actions_reshaped = action_unscaled.reshape((num_actions, 2))
    print("Reshaped actions:", actions_reshaped.shape)
    
    for action in actions_reshaped:
        print("Action point:", action)
        img = cv2.circle(img, (int(action[1]), int(action[0])), radius=5, color=(0, 255, 0), thickness=-1)
    
    pil_img = Image.fromarray(img)
    pil_img.save(f'temp_{t}.png')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--file_name', type=str, default="/user/frosa/multi_task_lfd/datasets/pick_place/human_rgb_pick_place/hdf5/merged_dataset/human_panda_robot_merged.hdf5", help='Path to the hdf5 file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    if args.debug:
        debugpy.listen(5678)
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")
        
    # open the hdf5 file and process
    with h5py.File(args.file_name, 'r') as f:
        print("Keys in the hdf5 file:", list(f.keys()))
        
        # take human index 
        start_robot_demo_idx = f['data'].attrs['start_robot_demo_idx']
    
        # take human demo
        human_demo = f['data']['demo_0']
        print("Keys in human demo:", list(human_demo.keys()))
        
        for t, action in enumerate(human_demo['actions']):
            print("Action shape:", action.shape)
            img = np.array(human_demo['obs']['agentview_image'][t])
            plot_actions_on_image(img, action, t)
            
        # take robot demo
        robot_demo = f['data'][f'demo_{start_robot_demo_idx}']
        print("Keys in robot demo:", list(robot_demo.keys()))
        for t, action in enumerate(robot_demo['actions']):
            print("Action shape:", action.shape)
            img = np.array(robot_demo['obs']['agentview_image'][t])
            plot_actions_on_image(img, action, t + len(human_demo['actions']))
            
           