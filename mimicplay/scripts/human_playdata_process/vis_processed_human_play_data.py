import h5py
import numpy as np
import cv2
from PIL import Image

view_id = 0 # change to 2 if drawing on second view
FILE_NAME="/user/frosa/multi_task_lfd/datasets/pick_place/human_rgb_pick_place/hdf5/task_00/traj000.hdf5"

with h5py.File(FILE_NAME, 'r') as f:
    images = np.array(f['data/demo_0/obs/front_image_{}'.format(view_id)])
    actions = np.array(f['data/demo_0/actions'])
    hand_loc = np.array(f['data/demo_0/obs/hand_loc'])   
    print(hand_loc[:, :, 0]*120)
    print(hand_loc[:, :, 1]*120)
    # print(actions.shape)
    num_timesteps = actions.shape[0]

# Reshape the actions to [145, 10, 6]
actions = actions.reshape((num_timesteps, 10, 2))

if view_id == 0:
    actions = actions[:, :, :2]
    
elif view_id == 1:
    actions = actions[:, :, 2:4]
elif view_id == 2:
    actions = actions[:, :, 4:]


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
task_name=FILE_NAME.split('/')[-2]
trajectory_name=FILE_NAME.split('/')[-1].split('.')[0]
video = cv2.VideoWriter(f'output_{task_name}_{trajectory_name}_{view_id}.mp4', fourcc, 30.0, (120, 120))
print(f"Video image {images.shape}.")

for i in range(images.shape[0]):
    img = images[i].copy()
    
    if view_id == 0:
        crop_params = [0, 30, 120, 120]
        top, left = crop_params[0], crop_params[2]
        img_height, img_width = img.shape[0], img.shape[1]
        box_h, box_w = img_height - top - \
        crop_params[1], img_width - left - crop_params[3]
        img = img[top:top + box_h, left:left + box_w]
    img = cv2.resize(img, (120, 120))
    
    action = actions[i]
    # print(action.shape)
    action_unscaled = action * np.array([img.shape[1], img.shape[0]])
    
    for pt in action_unscaled:
        # save cropped image
        # cv2.imwrite('cropped_img.png', im_in)
        # print(pt)
        img = cv2.circle(img, (int(pt[1]), int(pt[0])), radius=5, color=(0, 255, 0), thickness=-1)

    # pil_img = Image.fromarray(img)
    # pil_img.save(f'temp_{i}.png')
    
    video.write(img[:, :, ::-1])  # Convert RGB to BGR for OpenCV
    
video.release()

print("The video has been successfully saved as output.mp4")

