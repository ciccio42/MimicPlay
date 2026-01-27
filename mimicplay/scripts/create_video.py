import h5py
import numpy as np
import cv2
from PIL import Image
import argparse
import os
import debugpy

DIM_ACTION_VECTOR = 60
TOT_NUM_VIEWS = 3
NUM_ACTIONS_PER_VIEW = DIM_ACTION_VECTOR // TOT_NUM_VIEWS
NUM_ACTION_PER_STEP = 2 * TOT_NUM_VIEWS

def plot_actions_on_image(image, actions, human):
    """Plot actions as green circles on the image and return the modified image."""
    img = image.copy()
    
    if image.shape != (120, 120, 3):
        raise ValueError("Image must be of shape (120, 120, 3)")
    
    # Scale actions to image coordinates
    action_unscaled = actions * img.shape[0]
    num_actions = int(action_unscaled.shape[0] / 2)
    actions_reshaped = action_unscaled.reshape((num_actions, 2))
    
    # Draw circles for each action point
    for action in actions_reshaped[:10]:  # Limit to first 10 actions for clarity
        img = cv2.circle(img, (int(action[1]), int(action[0])), 
                        radius=5, color=(0, 255, 0), thickness=-1)
    
    return img


def create_actions_video(hdf5_path, output_video='output_actions_video.mp4', fps=10):
    """Create a video from HDF5 dataset with actions plotted on images."""
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (120, 120))
    
    frames = []
    
    print(f"Processing dataset: {hdf5_path}")
    
    # Open HDF5 file
    human = 'human' in hdf5_path.lower()
    with h5py.File(hdf5_path, 'r') as f:
        data_group = f['data']
        
        # Process all demonstrations in order
        demo_names = sorted([key for key in data_group.keys() if key.startswith('demo_')])
        print(f"Found {len(demo_names)} demonstrations: {demo_names}")
        
        for demo_idx, demo_name in enumerate(demo_names[:10]):  # Limit to first 10 demos for brevity
            demo = data_group[demo_name]
            print(f"Processing {demo_name}...")
            
            if not human:
                actions = demo['obs']['robot0_eef_pos_future_traj'][:,0,:] #demo['actions'] if 'actions' in demo.keys() else demo['obs']['robot0_eef_pos_future_traj'] 
            else:
                start = 0
                end = 2
                action_indices = []
                for i in range(0, NUM_ACTIONS_PER_VIEW//2):
                    if i == 0:
                        action_indices.extend(list(range(start, end)))
                    else:
                        action_indices.extend(list(range(start + i*NUM_ACTION_PER_STEP, end + i*NUM_ACTION_PER_STEP)))
                actions = demo['actions'][:, action_indices]
            
            images = demo['obs']['agentview_image']
            
            print(f"  Actions shape: {actions.shape}, Images shape: {images.shape}")
            
            # Process each timestep in this demo
            for t in range(len(actions)):
                action = actions[t]
                img = np.array(images[t])
                
                # Plot actions and get modified image
                modified_img = plot_actions_on_image(img, action, human)
                frames.append(modified_img)
    
    # Write frames to video
    print(f"Writing {len(frames)} frames to video...")
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr_frame = frame[:, :, ::-1]
        video.write(bgr_frame)
    
    video.release()
    print(f"✅ Video saved as '{output_video}' ({len(frames)} frames, {fps} FPS)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create video with actions plotted on images from HDF5 dataset")
    
    parser.add_argument('--dataset_path', 
                        type=str, 
                        help='Path to the HDF5 dataset file')
    parser.add_argument('--output', '-o', 
                        type=str, 
                        default='output_actions_video.mp4', 
                        help='Output video filename (default: output_actions_video.mp4)')
    parser.add_argument('--fps', 
                        type=float, 
                        default=10.0, 
                        help='Frames per second (default: 10.0)')
    parser.add_argument('--debug', 
                        action='store_true', 
                        help='Enable debug mode with verbose output')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    
    if args.debug:
        print("Debug mode enabled.")
        debugpy.listen(("localhost", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
    
    create_actions_video(args.dataset_path, args.output, args.fps)
