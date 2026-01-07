from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./hand_object_detector')

import _init_paths
import os
import numpy as np
import argparse
import pprint
import pdb
import time
import json
import cv2
import torch
import shutil, copy
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, \
    vis_detections_filtered_objects_PIL, vis_detections_filtered_objects, calculate_center  # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
import h5py
import moviepy.editor as mpy
from moviepy.editor import *
import matplotlib.pyplot as plt
import debugpy
import glob

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

ZED_MINI_INTRINSIC_PARAMS = np.array(
    [[345.2712097167969, 0.0, 337.5007629394531],
     [0.0, 345.2712097167969, 179.0137176513672],
     [0, 0, 1]])
MIN_DEPTH=0.10 
MAX_DEPTH=2.0


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset_path',
                        help='dataset path',
                        default='/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/human_dataset', type=str)
    parser.add_argument('--output_path',
                        help='output folder',
                        default='/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/human_dataset/hdf5', type=str)
    parser.add_argument('--camera_names', dest='camera_names',
                        help='camera names',
                        default='camera_front,camera_lateral_right,camera_lateral_left', type=str)
    parser.add_argument('--use_depth', dest='use_depth',
                        help='whether to use depth information',
                        action='store_true')
    parser.add_argument('--debug', dest='debug',
                        help='whether to use debug mode',
                        action='store_true')
    
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--hdf5_read_mode',
                        help='online / offline',
                        default='online', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=8, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=132028, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        default=False)
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)
    parser.add_argument('--thresh_hand',
                        type=float, default=0.5,
                        required=False)
    parser.add_argument('--thresh_obj', default=0.5,
                        type=float,
                        required=False)
    parser.add_argument('--downsample_rate', default=1,
                        type=int, help='set to n: sample 1 frame every n frames')
    parser.add_argument('--crop', action='store_true',
                        help='whether crop input image')

    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def export_images_from_hdf5(hdf5_path='', target_path='images', demo_name='demo_1', front_image_index=None):
    shutil.rmtree(target_path)
    os.makedirs(target_path, exist_ok=True)
    h5py_file = h5py.File(hdf5_path, "r+")['data']
    for demo_index in h5py_file.keys():
        if demo_index != demo_name:
            continue
        os.makedirs(os.path.join(target_path), exist_ok=True)
        demo = h5py_file[demo_index]
        obs = demo['obs']
        front_image = obs['front_image_{}'.format(front_image_index)]
        print(front_image.shape)
        image = np.array(front_image).astype("uint8")
        print(image.max(), image.min())
        for img_idx in range(front_image.shape[0]):
            rgb_img = front_image[img_idx]
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(target_path, '{:06d}.png'.format(img_idx)), bgr_img)


def extractImages(pathIn, pathOut=None):
    count = 0
    frame_list = []
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    success = True
    while success:
        # vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line
        success, image = vidcap.read()
        # print ('Read a new frame: ', success)
        if success:
            frame_list.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.imwrite( pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
            count = count + 1
    print ('Read {} frames.'.format(len(frame_list)))
    return frame_list


def convert_from_px_to_3d_position(hand_locs_px, camera_name, depth_video_path, intrinsic_params=ZED_MINI_INTRINSIC_PARAMS, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, img_size=(120,120), crop_params=[0, 30, 120, 120], plot=False):

    # fix hand_locs_px with (0,0)
    for t in range(hand_locs_px.shape[0]):
        if hand_locs_px[t, 0, 0] == 0 and hand_locs_px[t, 0, 1] == 0:
            if t == 0:
                # set to top left corner
                hand_locs_px[t, 0, 0] = 0.0
                hand_locs_px[t, 0, 1] = 0.0
            else:
                # take the first in the past non-zero value
                for past_t in range(t-1, -1, -1):
                    if hand_locs_px[past_t, 0, 0] != 0 and hand_locs_px[past_t, 0, 1] != 0:
                        hand_locs_px[t, 0, 0] = hand_locs_px[past_t, 0, 0]
                        hand_locs_px[t, 0, 1] = hand_locs_px[past_t, 0, 1]
                        break
                hand_locs_px[t, 0, 0] = hand_locs_px[t-1, 0, 0]
                hand_locs_px[t, 0, 1] = hand_locs_px[t-1, 0, 1]
    

    ##### Bring px locations to original image size, then convert to 3D positions
    # Read depth video
    depth_frames = extractImages(depth_video_path)
    
    print(f"Converting {hand_locs_px.shape[0]} hand locations from px to 3D positions for camera {camera_name} ...")
    if camera_name == 'camera_front':
        # Bring px locations to original image size if cropped
        H_orig, W_orig = depth_frames[0].shape[:2]
        box_h = H_orig - crop_params[0] - crop_params[1]
        box_w = W_orig - crop_params[2] - crop_params[3]
        
        # row indx is hand_locs_px[:, 1], col indx is hand_locs_px[:, 0]
        unscaled_px_locations = hand_locs_px.copy()
        # width unscaling -> x direction -> columns index
        unscaled_px_locations[:, :, 1] = hand_locs_px[:, :, 1] * (box_w) + crop_params[2]
        # height unscaling -> y direction -> row index
        unscaled_px_locations[:, :, 0] = hand_locs_px[:, :, 0] * (box_h) + crop_params[0]
    else:
        # No cropping for lateral cameras
        H_orig, W_orig = depth_frames[0].shape[:2]
        unscaled_px_locations = hand_locs_px.copy()
        # width unscaling -> x direction -> columns index
        unscaled_px_locations[:, :, 1] = hand_locs_px[:, :, 1] * W_orig
        # height unscaling -> y direction -> row index
        unscaled_px_locations[:, :, 0] = hand_locs_px[:, :, 0] * H_orig    

    hand_loc_3d_positions = np.zeros((hand_locs_px.shape[0], hand_locs_px.shape[1], 3), dtype=np.float32)  # (T, num_hands, 3)
    

    for t in range(hand_locs_px.shape[0]):
        depth_frame = cv2.cvtColor(depth_frames[t], cv2.COLOR_RGB2GRAY)
        # Denormalize depth value
        depth_frame = depth_frame.astype(np.float32) / 255.0 * (max_depth - min_depth) + min_depth  
        
        # For each hand location, get depth value and convert to 3D position
        hand_px_loc = unscaled_px_locations[t][0]  # (y, x) (rows, columns)
        if True: #hand_px_loc[0] != 0 and hand_px_loc[1] != 0:
            u = hand_px_loc[1]
            v = hand_px_loc[0]
            hand_depth = depth_frame[int(v), int(u)]
            Z = hand_depth
            X = (u - intrinsic_params[0, 2]) * Z / intrinsic_params[0, 0]
            Y = (v - intrinsic_params[1, 2]) * Z / intrinsic_params[1, 1]
            
            hand_loc_3d_positions[t, 0, 0] = X
            hand_loc_3d_positions[t, 0, 1] = Y
            hand_loc_3d_positions[t, 0, 2] = Z
            # print(f"X,Y,Z: {hand_loc_3d_positions[t, 0, :]}")
            
            # plot circe on depth image
            if plot:
                depth_frame_bgr = cv2.cvtColor((depth_frame - min_depth) / (max_depth - min_depth + 1e-8) * 255.0, cv2.COLOR_GRAY2BGR)
                depth_frame_bgr = cv2.circle(depth_frame_bgr, (int(u), int(v)), radius=5, color=(0, 255, 0), thickness=-1)
                # label X,Y,Z values
                depth_frame_bgr = cv2.putText(depth_frame_bgr, f"X:{hand_loc_3d_positions[t, 0, 0]:.2f}m", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                depth_frame_bgr = cv2.putText(depth_frame_bgr, f"Y:{hand_loc_3d_positions[t, 0, 1]:.2f}m", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                depth_frame_bgr = cv2.putText(depth_frame_bgr, f"Z:{hand_loc_3d_positions[t, 0, 2]:.2f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # add on the image center X-Y axes
                cx = int(intrinsic_params[0, 2])
                cy = int(intrinsic_params[1, 2])

                arrow_len = 20
                thickness = 2
                tip_length = 0.3

                # X-axis arrow (red, pointing right)
                depth_frame_bgr = cv2.arrowedLine(
                    depth_frame_bgr,
                    (cx, cy),
                    (cx + arrow_len, cy),
                    (0, 0, 255),  # Red (BGR)
                    thickness,
                    tipLength=tip_length
                )

                # Y-axis arrow (green, pointing down)
                depth_frame_bgr = cv2.arrowedLine(
                    depth_frame_bgr,
                    (cx, cy),
                    (cx, cy + arrow_len),
                    (0, 255, 0),  # Green (BGR)
                    thickness,
                    tipLength=tip_length
                )

                os.makedirs('debug_depth_frames_{}'.format(camera_name), exist_ok=True)
                cv2.imwrite(f'debug_depth_frames_{camera_name}/depth_frame_{t:04d}.png', depth_frame_bgr)
        
        
            # Bring positions with respect to new image size
            fx = intrinsic_params[0, 0]
            fy = intrinsic_params[1, 1]
            cx = intrinsic_params[0, 2]
            cy = intrinsic_params[1, 2]
            if 'camera_front' == camera_name:    
                H_new = img_size[0]
                W_new = img_size[1]
                crop_h = H_orig - crop_params[0] - crop_params[1]
                crop_w = W_orig - crop_params[2] - crop_params[3]
                left = crop_params[2]
                top = crop_params[0]
                fx_new = fx * (W_new / crop_w)
                fy_new = fy * (H_new / crop_h)
                cx_new = (cx - left) * (W_new / crop_w)
                cy_new = (cy - top) * (H_new / crop_h)
                
                # crop and resize
                # Step 1: project to original pixels
                u_orig = fx * X / Z + cx
                v_orig = fy * Y / Z + cy
                # Step 2: apply crop
                u_crop = u_orig - crop_params[2]  # left
                v_crop = v_orig - crop_params[0]  # top
                # Step 3: resize
                u_new = u_crop * (img_size[1] / (W_orig - crop_params[2] - crop_params[3]))
                v_new = v_crop * (img_size[0] / (H_orig - crop_params[0] - crop_params[1]))
                # Step 4: back-project using new intrinsics
                X_new = (u_new - cx_new) * Z / fx_new
                Y_new = (v_new - cy_new) * Z / fy_new
            else:
                # just resize
                fx_new = fx * (img_size[1] / W_orig)
                fy_new = fy * (img_size[0] / H_orig)
                cx_new = cx * (img_size[1] / W_orig)
                cy_new = cy * (img_size[0] / H_orig)

                # Pixel coordinates in resized image
                u_new = (fx * X / Z + cx) * (img_size[1] / W_orig)
                v_new = (fy * Y / Z + cy) * (img_size[0] / H_orig)

                # Back-project using new intrinsics
                X_new = (u_new - cx_new) * Z / fx_new
                Y_new = (v_new - cy_new) * Z / fy_new

            # Update 3D positions
            # if hand_px_loc[0] == 0 and hand_px_loc[1] == 0:
            #     print("Warning: hand pixel location is (0,0), setting 3D position to inf")
            
            
            hand_loc_3d_positions[t, 0, 0] = X_new
            hand_loc_3d_positions[t, 0, 1] = Y_new
            hand_loc_3d_positions[t, 0, 2] = Z if hand_px_loc[0] !=0 and hand_px_loc[1] !=0 else 0.0

            # -----------------------------
            # Debug plot on final image
            # -----------------------------
            if plot:
                # Crop + resize depth frame for visualization
                if camera_name == 'camera_front':
                    depth_vis = depth_frame[crop_params[0]:H_orig-crop_params[1], crop_params[2]:W_orig-crop_params[3]]
                else:
                    depth_vis = depth_frame.copy()

                depth_vis = cv2.resize(depth_vis, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
                depth_vis = ((depth_vis - min_depth) / (max_depth - min_depth + 1e-8) * 255.0).astype(np.uint8)
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

                # Draw circle at new location
                depth_vis = cv2.circle(depth_vis, (int(u_new), int(v_new)), 5, (0, 255, 0), -1)

                # Overlay X,Y,Z on the final image
                depth_vis = cv2.putText(depth_vis, f"X:{X:.2f}m", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                depth_vis = cv2.putText(depth_vis, f"Y:{Y:.2f}m", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                depth_vis = cv2.putText(depth_vis, f"Z:{Z:.2f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Draw camera principal axes
                cx_new_vis = int((cx - crop_params[2]) * (img_size[1] / (W_orig - crop_params[2] - crop_params[3]))) if camera_name == 'camera_front' else int(cx * (img_size[1] / W_orig))
                cy_new_vis = int((cy - crop_params[0]) * (img_size[0] / (H_orig - crop_params[0] - crop_params[1]))) if camera_name == 'camera_front' else int(cy * (img_size[0] / H_orig))
                arrow_len = 20
                depth_vis = cv2.arrowedLine(depth_vis, (cx_new_vis, cy_new_vis), (cx_new_vis + arrow_len, cy_new_vis), (0, 0, 255), 2, tipLength=0.3)
                depth_vis = cv2.arrowedLine(depth_vis, (cx_new_vis, cy_new_vis), (cx_new_vis, cy_new_vis + arrow_len), (0, 255, 0), 2, tipLength=0.3)

                os.makedirs(f'debug_depth_frames_{camera_name}', exist_ok=True)
                cv2.imwrite(f'debug_depth_frames_{camera_name}/depth_frame_{t:04d}_final.png', depth_vis)
            
        
        else:
            hand_loc_3d_positions[t, 0, 0] = np.inf
            hand_loc_3d_positions[t, 0, 1] = np.inf
            hand_loc_3d_positions[t, 0, 2] = np.inf

    
    # # find inf values and replace with the first previous non-inf value
    # for t in range(hand_locs_px.shape[0]):
    #     if np.isinf(hand_loc_3d_positions[t, 0, 0]):
    #         if t == 0:
    #             # set to top left corner with Z = 0
    #             # convert the px (0,0) to 3D position
    #             hand_loc_3d_positions[t, 0, 0] = (0 - intrinsic_params[0, 2]) * min_depth / intrinsic_params[0, 0]
    #             hand_loc_3d_positions[t, 0, 1] = (0 - intrinsic_params[1, 2]) * min_depth / intrinsic_params[1, 1]
    #             hand_loc_3d_positions[t, 0, 2] = 0.0
                
    #             # # check t+1 and t+2 for non-inf
    #             # for future_t in range(t+1, hand_locs_px.shape[0]):
    #             #     if not np.isinf(hand_loc_3d_positions[future_t, 0, 0]):
    #             #         hand_loc_3d_positions[t, 0, 0] = hand_loc_3d_positions[future_t, 0, 0]
    #             #         hand_loc_3d_positions[t, 0, 1] = hand_loc_3d_positions[future_t, 0, 1]
    #             #         hand_loc_3d_positions[t, 0, 2] = hand_loc_3d_positions[future_t, 0, 2]
    #             #         break
    #         else:
    #             # check previous value
    #             if not np.isinf(hand_loc_3d_positions[t-1, 0, 0]):
    #                 hand_loc_3d_positions[t, 0, 0] = hand_loc_3d_positions[t-1, 0, 0]
    #                 hand_loc_3d_positions[t, 0, 1] = hand_loc_3d_positions[t-1, 0, 1]
    #                 hand_loc_3d_positions[t, 0, 2] = hand_loc_3d_positions[t-1, 0, 2]
                
    
    return hand_loc_3d_positions
       

if __name__ == '__main__':

    args = parse_args()

    if args.debug:    
        debugpy.listen(('0.0.0.0',5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    # print('Called with args:')
    # print(args)

    # hdf5/data/demo_0/obs/[front_image_0, front_image_1, ]
    # demo.keys():  <KeysViewHDF5 ['actions', 'dones', 'interventions', 'next_obs', 'obs', 'policy_acting', 'rewards', 'states', 'user_acting']>
    # obs.keys():  <KeysViewHDF5 ['ee_pose', 'front_image_1', 'front_image_2', 'gripper_position', 'hand_loc', 'joint_positions', 'joint_velocities', 'wrist_image']>

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    np.random.seed(cfg.RNG_SEED)

    # load model
    model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
    if not os.path.exists(model_dir):
        raise Exception('There is no input directory for loading network from ' + model_dir)
    load_name = os.path.join(model_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]']

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    
    task_folders = glob.glob(os.path.join(args.dataset_path, 'task_*'))
    camera_names = args.camera_names.split(',')
    use_depth = args.use_depth
    
    for task_folder in task_folders:
        print(f'Processing {task_folder} ...')
        task_name = task_folder.split('/')[-1]
        out_folder = os.path.join(args.output_path, task_name)
        os.makedirs(out_folder, exist_ok=True)
        
        video_files = glob.glob(os.path.join(task_folder, 'videos', '*.mp4'))
        video_files.sort(key=lambda x: int(x.split('/')[-1].split('_')[0].split('traj')[-1]))  # sort by demo index
        
        # group video files by demo index
        demo_dict = {}
        for video_file in video_files:
            traj_name = video_file.split('/')[-1].split('_')[0]
            if traj_name not in demo_dict:
                demo_dict[traj_name] = []
            demo_dict[traj_name].append(video_file)
        
        
    
        for traj_name, _ in demo_dict.items():
            # create a single hdf5 file for all demos in this task
            out_hdf5_path = os.path.join(out_folder, traj_name + '.hdf5')
            
            # check if the hdf5 file already exists
            # if os.path.exists(out_hdf5_path):
            #     print(f'HDF5 file {out_hdf5_path} already exists, skip processing.')
            #     continue
            
            new_f_out = h5py.File(out_hdf5_path, "w")
            demo_name = 'demo_0'
            val_demo_name = 'demo_1'
            release_frame = None
                        
            for front_image_index, camera_name in enumerate(camera_names):
                
                input_video_path = task_folder + '/videos/{}_{}_image.mp4'.format(traj_name, camera_name)
                print('Processing video: ', input_video_path)
                mpy_frame_list = []
                image = extractImages(input_video_path)
                # if (len(image) == 0):
                #     print('No image found in {}, skip this demo.'.format(input_video_path))
                #     continue
                image = np.asarray(image)
                image = image[::args.downsample_rate]
                img_size = 120.        
                
                with torch.no_grad():
                    if args.cuda > 0:
                        cfg.CUDA = True

                    if args.cuda > 0:
                        fasterRCNN.cuda()

                    fasterRCNN.eval()

                    start = time.time()
                    max_per_image = 100
                    thresh_hand = args.thresh_hand
                    thresh_obj = args.thresh_obj
                    vis = args.vis

                    num_images = len(image)
                    num_samples = 0 # num_images
                    imglist = ['{:06d}.png'.format(i) for i in range(num_images)]
            
                    print('Loaded Photo: {} images.'.format(num_images))
                    hand_det_result = []
                    while (num_images > 0):
                        total_tic = time.time()
                        num_images -= 1

                        im_in = image[num_images]
                        
                        # crop image
                        if args.crop and front_image_index == 0:
                            crop_params = [0, 30, 120, 120]
                            top, left = crop_params[0], crop_params[2]
                            img_height, img_width = im_in.shape[0], im_in.shape[1]
                            box_h, box_w = img_height - top - \
                            crop_params[1], img_width - left - crop_params[3]
                            im_in = im_in[top:top + box_h, left:left + box_w]
                            
                            # save cropped image
                            # cv2.imwrite('cropped_img.png', im_in)
                                
                        im_in = cv2.cvtColor(im_in, cv2.COLOR_RGB2BGR)
                        # bgr
                        im = im_in
                        im = cv2.resize(im, [int(img_size), int(img_size)])
                        mpy_frame_list.append(im)
                        blobs, im_scales = _get_image_blob(im)
                        assert len(im_scales) == 1, "Only single-image batch implemented"
                        im_blob = blobs
                        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
    
                        im_data_pt = torch.from_numpy(im_blob)
                        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
                        im_info_pt = torch.from_numpy(im_info_np)

                        with torch.no_grad():
                            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                            gt_boxes.resize_(1, 1, 5).zero_()
                            num_boxes.resize_(1).zero_()
                            box_info.resize_(1, 1, 5).zero_()

                        # pdb.set_trace()
                        det_tic = time.time()

                        rois, cls_prob, bbox_pred, \
                        rpn_loss_cls, rpn_loss_box, \
                        RCNN_loss_cls, RCNN_loss_bbox, \
                        rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

                        scores = cls_prob.data
                        boxes = rois.data[:, :, 1:5]

                        # extact predicted params
                        contact_vector = loss_list[0][0]  # hand contact state info
                        offset_vector = loss_list[1][0].detach()  # offset vector (factored into a unit vector and a magnitude)
                        lr_vector = loss_list[2][0].detach()  # hand side info (left/right)

                        # get hand contact
                        _, contact_indices = torch.max(contact_vector, 2)
                        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

                        # get hand side
                        lr = torch.sigmoid(lr_vector) > 0.5
                        lr = lr.squeeze(0).float()

                        if cfg.TEST.BBOX_REG:
                            # Apply bounding-box regression deltas
                            box_deltas = bbox_pred.data
                            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                                # Optionally normalize targets by a precomputed mean and stdev
                                if args.class_agnostic:
                                    if args.cuda > 0:
                                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                            cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                                    else:
                                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                                    box_deltas = box_deltas.view(1, -1, 4)
                                else:
                                    if args.cuda > 0:
                                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                            cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                                    else:
                                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

                            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                        else:
                            # Simply repeat the boxes, once for each class
                            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                        pred_boxes /= im_scales[0]

                        scores = scores.squeeze()
                        pred_boxes = pred_boxes.squeeze()
                        det_toc = time.time()
                        detect_time = det_toc - det_tic
                        misc_tic = time.time()
                        if vis:
                            im2show = np.copy(im)
                        obj_dets, hand_dets = None, None
                        for j in xrange(1, len(pascal_classes)):
                            # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
                            if pascal_classes[j] == 'hand':
                                inds = torch.nonzero(scores[:, j] > thresh_hand).view(-1)
                                
                                # get only the indices of left hands
                                lr_indices = torch.nonzero(lr.view(-1)[inds] == 1).view(-1)
                                inds = inds[lr_indices]
                                
                            elif pascal_classes[j] == 'targetobject':
                                inds = torch.nonzero(scores[:, j] > thresh_obj).view(-1)

                            # if there is det
                            if inds.numel() > 0:
                                cls_scores = scores[:, j][inds]
                                _, order = torch.sort(cls_scores, 0, True)
                                if args.class_agnostic:
                                    cls_boxes = pred_boxes[inds, :]
                                else:
                                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds],
                                                    offset_vector.squeeze(0)[inds], lr[inds]), 1)
                                cls_dets = cls_dets[order]
                                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                                cls_dets = cls_dets[keep.view(-1).long()]
                                if pascal_classes[j] == 'targetobject':
                                    obj_dets = cls_dets.cpu().numpy()
                                if pascal_classes[j] == 'hand':
                                    hand_dets = cls_dets.cpu().numpy()
                                    # print('hand_dets: ', hand_dets.shape)   # (1, 10)
                        
                        if hand_dets is not None:
                            save_hand_dets = hand_dets[0:1]
                            save_hand_dets = np.concatenate((save_hand_dets, np.array(calculate_center(save_hand_dets[0, :4]))[None]), axis=1)
                            save_hand_dets[:, :4] = save_hand_dets[:, :4].astype(np.float32) / img_size
                            save_hand_dets[:, -2:] = save_hand_dets[:, -2:].astype(np.float32) / img_size
                            hand_det_result.append(copy.deepcopy(save_hand_dets))
                        else:
                            save_hand_dets = np.zeros((1, 10))
                            save_hand_dets = np.concatenate((save_hand_dets, np.array(calculate_center(save_hand_dets[0, :4]))[None]), axis=1)
                            save_hand_dets[:, :4] = save_hand_dets[:, :4].astype(np.float32) / img_size
                            save_hand_dets[:, -2:] = save_hand_dets[:, -2:].astype(np.float32) / img_size
                            hand_det_result.append(copy.deepcopy(save_hand_dets))

                        if vis:
                            # visualization
                            # print(num_images, 'save_hand_dets', save_hand_dets)
                            im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)
                            im2show.save('detection.png')

                        misc_toc = time.time()
                        nms_time = misc_toc - misc_tic

                        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                                        .format(num_images + 1, len(imglist), detect_time, nms_time))
                        sys.stdout.flush()


                    print('Writing HDF5 file to {}'.format(out_hdf5_path))
                    hand_loc = np.asarray(hand_det_result)[::-1]
                    x = copy.deepcopy(hand_loc[:, :, -2])
                    y = copy.deepcopy(hand_loc[:, :, -1])
                    hand_loc[:, :, -2] = copy.deepcopy(y)
                    hand_loc[:, :, -1] = copy.deepcopy(x)
                    
                    # localize the frame where the object is released
                    prev_hand_cord = np.array([[0., 0.]])
                    if camera_name == 'camera_front':
                        for t, hand_sequence in enumerate(hand_loc):
                            hand_cord = hand_sequence[:, -2:]*img_size
                            print(f"Frame {t}, hand_cord: ", hand_cord)
                            # hand is moving forward
                            if hand_cord[0][0] > 88 and hand_cord[0][0] > prev_hand_cord[0][0]:
                                release_frame = t
                                prev_hand_cord = hand_cord
                            elif hand_cord[0][0] <= prev_hand_cord[0][0]-10:
                                release_frame = t
                                break
                        print(f"Release frame localized at: {release_frame}") 
                        
                    
                    
                    hand_loc = hand_loc[:release_frame, :, :]
                    image = image[:release_frame, : , :, :]
                    # print('hand_loc.shape: ', hand_loc.shape) # hand_loc.shape:  (145, 1, 12)
                    
                    shifted_hand_loc_1_array = np.concatenate((hand_loc, hand_loc[-1:]), axis=0)
                    hand_act = shifted_hand_loc_1_array[1:] - shifted_hand_loc_1_array[:-1]
                    # print('hand_act.shape: ', hand_act.shape)  # hand_act.shape:  (145, 1, 12)
                    obs_path = 'data/'+demo_name+'/obs'
                    new_f_out.create_dataset(obs_path+'/hand_loc_{}'.format(front_image_index), data=hand_loc)
                    new_f_out.create_dataset(obs_path + '/hand_act_{}'.format(front_image_index), data=hand_act)
                    new_f_out.create_dataset(obs_path + '/front_image_{}'.format(front_image_index), data=image)
                    
                    if front_image_index == 0:
                        new_f_out.create_dataset(obs_path + '/agentview_image'.format(front_image_index), data=image)
                    else:
                        new_f_out.create_dataset(obs_path + '/agentview_image_{}'.format(front_image_index), data=image)
                    
                    print('new_f_out[obs_path].keys(): ', new_f_out[obs_path].keys())
                    # new_f_out[obs_path].keys():  <KeysViewHDF5 ['front_image_1', 'front_image_2', 'hand_act_1', 'hand_act_2', 'hand_loc_1', 'hand_loc_2']>
                    if front_image_index == 0:
                        hand_loc_1 = copy.deepcopy(hand_loc)
                        
                        
                        print('Using depth information ...')
                        depth_video_path = task_folder + '/videos/{}_{}_depth.mp4'.format(traj_name, camera_name)
                        
                        robot0_eef_pos_3D = convert_from_px_to_3d_position(
                                                        hand_loc_1[:, :, -2:], 
                                                        camera_name, 
                                                        depth_video_path,
                                                        ZED_MINI_INTRINSIC_PARAMS,
                                                        MIN_DEPTH,
                                                        MAX_DEPTH,
                                                        plot=args.debug)
                                                        
                        new_f_out.create_dataset(obs_path + '/robot0_eef_pos_3D_{}'.format(front_image_index), data=robot0_eef_pos_3D)
                        
                        if len(camera_names) > 1:
                            all_hand_loc = copy.deepcopy(hand_loc_1[:, :, -2:])
                            
                    elif front_image_index < len(camera_names)-1:
                        # concatenate hand loc from different views
                        all_hand_loc = np.concatenate((all_hand_loc, hand_loc[:, :, -2:]), axis=2)
                        
                        # convert to 3D position
                        depth_video_path = task_folder + '/videos/{}_{}_depth.mp4'.format(traj_name, camera_name)
                        robot0_eef_pos_3D = convert_from_px_to_3d_position(
                                                        hand_loc[:, :, -2:], 
                                                        camera_name, 
                                                        depth_video_path,
                                                        ZED_MINI_INTRINSIC_PARAMS,
                                                        MIN_DEPTH,
                                                        MAX_DEPTH,
                                                        plot=args.debug)
                        new_f_out.create_dataset(obs_path + '/robot0_eef_pos_3D_{}'.format(front_image_index), data=robot0_eef_pos_3D)
                    
                    if front_image_index == len(camera_names)-1 or len(camera_names) == 1:
                        if len(camera_names) != 1:
                            all_hand_loc = np.concatenate((all_hand_loc, hand_loc[:, :, -2:]), axis=2)
                        else:
                            all_hand_loc = copy.deepcopy(hand_loc_1[:, :, -2:])
                        
                        depth_video_path = task_folder + '/videos/{}_{}_depth.mp4'.format(traj_name, camera_name)
                        robot0_eef_pos_3D = convert_from_px_to_3d_position(
                                                        hand_loc[:, :, -2:], 
                                                        camera_name, 
                                                        depth_video_path,
                                                        ZED_MINI_INTRINSIC_PARAMS,
                                                        MIN_DEPTH,
                                                        MAX_DEPTH,
                                                        plot=args.debug)
                        new_f_out.create_dataset(obs_path + '/robot0_eef_pos_3D_{}'.format(front_image_index), data=robot0_eef_pos_3D)
                        
                        new_f_out.create_dataset(obs_path + '/hand_loc', data=all_hand_loc)
                        all_skip_hand_loc = []
                        num_future_frame = 10
                        skip_len = 2
                        T = all_hand_loc.shape[0]
                        num_samples += T-1
                        for i in range(all_hand_loc.shape[0]):
                            each_skip_hand_loc = []
                            for j in range(num_future_frame):
                                if i + j * skip_len >= all_hand_loc.shape[0]:
                                    each_skip_hand_loc.append(all_hand_loc[-1])
                                else:
                                    each_skip_hand_loc.append(all_hand_loc[i + j * skip_len])
                            all_skip_hand_loc.append(each_skip_hand_loc)
                        all_skip_hand_loc = np.asarray(all_skip_hand_loc)
                        # all_skip_hand_loc.shape:  (145, 10, 1, 4)
                        all_skip_hand_loc = all_skip_hand_loc.reshape(T, -1)
                        new_f_out.create_dataset('data/'+demo_name+'/actions', data=all_skip_hand_loc)

                        new_f_out.create_dataset(obs_path + '/robot0_eef_pos'.format(front_image_index), data=all_hand_loc)
                        new_f_out.create_dataset(obs_path + '/robot0_eef_pos_future_traj'.format(front_image_index), data=all_skip_hand_loc)

                        # 'actions', 'dones', 'interventions', 'next_obs', 'obs', 'policy_acting', 'rewards', 'states', 'user_acting'
                        new_f_out.create_dataset('data/' + demo_name + '/dones', data=np.zeros((T-1)))
                        new_f_out.create_dataset('data/' + demo_name + '/interventions', data=np.zeros((T, 1)))
                        new_f_out.create_dataset('data/' + demo_name + '/policy_acting', data=np.zeros((T)))
                        new_f_out.create_dataset('data/' + demo_name + '/rewards', data=np.zeros((T-1)))
                        new_f_out.create_dataset('data/' + demo_name + '/states', data=np.zeros((0)))
                        new_f_out.create_dataset('data/' + demo_name + '/user_acting', data=np.zeros((T, 1)))
                        new_f_out['data/{}'.format(demo_name)].attrs["num_samples"] = T - 1
                        new_f_out.create_dataset('mask/train', data=[demo_name])
                        new_f_out.create_dataset('mask/valid', data=[val_demo_name])
                        print(f"Keys in new_f_out['data/']: {new_f_out['data'].keys()}")
                        print(f"Train - Key in obs: {new_f_out[obs_path].keys()}")
                        
                    # validation demo name
                    obs_path = 'data/' + val_demo_name + '/obs'
                    new_f_out.create_dataset(obs_path + '/hand_loc_{}'.format(front_image_index), data=hand_loc)
                    new_f_out.create_dataset(obs_path + '/hand_act_{}'.format(front_image_index), data=hand_act)
                    new_f_out.create_dataset(obs_path + '/front_image_{}'.format(front_image_index), data=image)
                    
                    if front_image_index != len(camera_names)-1:
                        new_f_out.create_dataset(obs_path + '/agentview_image_{}'.format(front_image_index), data=image)
                    elif front_image_index == len(camera_names)-1:
                        new_f_out.create_dataset(obs_path + '/agentview_image'.format(front_image_index), data=image)
                    
                    if front_image_index == 0:
                        val_hand_loc_1 = copy.deepcopy(hand_loc)
                        if len(camera_names) > 1:
                            val_all_hand_loc = copy.deepcopy(val_hand_loc_1[:, :, -2:])
                        
                        # convert to 3D position
                        val_depth_video_path = task_folder + '/videos/{}_{}_depth.mp4'.format(traj_name, camera_name)
                        val_robot0_eef_pos_3D = convert_from_px_to_3d_position(
                                                        val_hand_loc_1[:, :, -2:], 
                                                        camera_name, 
                                                        val_depth_video_path,
                                                        ZED_MINI_INTRINSIC_PARAMS,
                                                        MIN_DEPTH,
                                                        MAX_DEPTH,
                                                        plot=args.debug)
                        new_f_out.create_dataset(obs_path + '/robot0_eef_pos_3D_{}'.format(front_image_index), data=val_robot0_eef_pos_3D)
                            
                        
                            
                    elif front_image_index < len(camera_names)-1:
                        # concatenate hand loc from different views
                        val_all_hand_loc = np.concatenate((val_all_hand_loc, hand_loc[:, :, -2:]), axis=2)
                        
                        # convert to 3D position
                        val_depth_video_path = task_folder + '/videos/{}_{}_depth.mp4'.format(traj_name, camera_name)
                        val_robot0_eef_pos_3D = convert_from_px_to_3d_position(
                                                        hand_loc[:, :, -2:], 
                                                        camera_name, 
                                                        val_depth_video_path,
                                                        ZED_MINI_INTRINSIC_PARAMS,
                                                        MIN_DEPTH,
                                                        MAX_DEPTH,
                                                        plot=args.debug)
                        new_f_out.create_dataset(obs_path + '/robot0_eef_pos_3D_{}'.format(front_image_index), data=val_robot0_eef_pos_3D)
                    
                    if front_image_index == len(camera_names)-1 or len(camera_names) == 1:
                        if len(camera_names) != 1:
                            val_all_hand_loc = np.concatenate((val_all_hand_loc, hand_loc[:, :, -2:]), axis=2)
                            
                            val_depth_video_path = task_folder + '/videos/{}_{}_depth.mp4'.format(traj_name, camera_name)
                            val_robot0_eef_pos_3D = convert_from_px_to_3d_position(
                                                            hand_loc[:, :, -2:], 
                                                            camera_name, 
                                                            val_depth_video_path,ZED_MINI_INTRINSIC_PARAMS,
                                                            MIN_DEPTH,
                                                            MAX_DEPTH,
                                                            plot=args.debug)
                            new_f_out.create_dataset(obs_path + '/robot0_eef_pos_3D_{}'.format(front_image_index), data=val_robot0_eef_pos_3D)
                            
                            
                        else:
                            val_all_hand_loc = copy.deepcopy(val_hand_loc_1[:, :, -2:])
                        
                        new_f_out.create_dataset(obs_path + '/hand_loc', data=val_all_hand_loc)
                        # print('all_hand_loc.shape: ', all_hand_loc.shape)  # hand_act.shape:  (145, 1, 4)
                        all_skip_hand_loc = []
                        num_future_frame = 10
                        skip_len = 2
                        T = val_all_hand_loc.shape[0]
                        for i in range(val_all_hand_loc.shape[0]):
                            each_skip_hand_loc = []
                            for j in range(num_future_frame):
                                if i + j * skip_len >= val_all_hand_loc.shape[0]:
                                    each_skip_hand_loc.append(val_all_hand_loc[-1])
                                else:
                                    each_skip_hand_loc.append(val_all_hand_loc[i + j * skip_len])
                            all_skip_hand_loc.append(each_skip_hand_loc)
                        all_skip_hand_loc = np.asarray(all_skip_hand_loc)
                        # all_skip_hand_loc.shape:  (145, 10, 1, 4)
                        all_skip_hand_loc = all_skip_hand_loc.reshape(T, -1)
                        new_f_out.create_dataset('data/' + val_demo_name + '/actions', data=all_skip_hand_loc)

                        new_f_out.create_dataset(obs_path + '/robot0_eef_pos'.format(front_image_index), data=all_hand_loc)
                        new_f_out.create_dataset(obs_path + '/robot0_eef_pos_future_traj'.format(front_image_index), data=all_skip_hand_loc)
                        # 'actions', 'dones', 'interventions', 'next_obs', 'obs', 'policy_acting', 'rewards', 'states', 'user_acting'
                        new_f_out.create_dataset('data/' + val_demo_name + '/dones', data=np.zeros((T - 1)))
                        new_f_out.create_dataset('data/' + val_demo_name + '/interventions', data=np.zeros((T, 1)))
                        new_f_out.create_dataset('data/' + val_demo_name + '/policy_acting', data=np.zeros((T)))
                        new_f_out.create_dataset('data/' + val_demo_name + '/rewards', data=np.zeros((T - 1)))
                        new_f_out.create_dataset('data/' + val_demo_name + '/states', data=np.zeros((0)))
                        new_f_out.create_dataset('data/' + val_demo_name + '/user_acting', data=np.zeros((T, 1)))
                        new_f_out['data/{}'.format(val_demo_name)].attrs["num_samples"] = T - 1
                        print(f"Val obs keys: {new_f_out[obs_path].keys()}")
                        
            # save and close hdf5 file
            # print shape info
                
            data = new_f_out['data']
            data.attrs['total'] = num_samples # T - 1  # num_samples + num_samples
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
            data.attrs['env_args'] = json.dumps(env_meta, indent=4)
            print('Save to {}'.format(out_hdf5_path))
            new_f_out.close()
            