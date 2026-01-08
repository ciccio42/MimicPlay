# MimicPlay

## Hand-Object Detector
Follow the instruction in the [original directory](https://github.com/j96w/MimicPlay).

## 1. Create human-video dataset

```bash
# Set the correspondig paths
cd mimicplay/scripts/human_playdata_process/hand_object_detector
python human_pick_place_to_hdf5.py
cd mimicplay/scripts/human_playdata_process
python generate_merged_dataset.py
```

## 2. Create robot-video dataset
```bash
# Set the correspondig paths
cd mimicplay/scripts/human_playdata_process/
python create_real_world_robot_dataset.py # real-world robot
python generate_merged_dataset.py
python create_robot_dataset.py # sim robot
python generate_merged_dataset.py
```

**NOTE**
For human you need to start from *.mp4 video


## 3. Train Latent Planner
```bash
python scripts/train.py --config configs/highlevel_human.json --dataset /user/frosa/multi_task_lfd/datasets/pick_place/human_rgb_pick_place/hdf5/merged_dataset/human_pick_place_all_demos.hdf5
```

## 4. Test Latent Planner
```bash
python test_highlevel.py --config /user/frosa/Multi-Task-LFD-Framework/repo/mimic-play/MimicPlay/trained_models_highlevel_human_pick_place_all_demos_augmented/test/20251113182127/config.json  --checkpoint /user/frosa/Multi-Task-LFD-Framework/repo/mimic-play/MimicPlay/trained_models_highlevel_human_pick_place_all_demos_augmented/test/20251113182127/models/model_epoch_173_best_validation_-45.142740631103514.pth  --video_prompt "/user/frosa/multi_task_lfd/datasets/pick_place/human_rgb_pick_place/hdf5/merged_dataset/human_pick_place_all_demos.hdf5" --agent_path "/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/real_new_ur5e_pick_place/hdf5_files/merged_dataset/real_ur5e_all_demos.hdf5"
```

## 5. Train Low Level
```bash
# Robosuite example
python scripts/train.py --config configs/lowlevel.json --dataset 'datasets/image_demo_local.hdf5' --bddl_file 'scripts/bddl_files/KITCHEN_SCENE9_eval-task-1_turn_on_stove_put_pan_on_stove_put_bowl_on_shelf.bddl' --video_prompt 'datasets/eval-task-1_turn_on_stove_put_pan_on_stove_put_bowl_on_shelf/image_demo.hdf5'

# Low-Level Policy UR5e-SIM
python scripts/create_robot_dataset.py --data_dir /user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/ur5e_pick_place/ --output_path /user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/ur5e_pick_place/hdf5_files
python scripts/generate_merged_dataset.py --task_folder /user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/ur5e_pick_place/hdf5_files/ --output_folder /user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/ur5e_pick_place/hdf5_files/merged_dataset --robot_name ur5e --dataset_type all_demos
# create mapping between agent and demo files
python scripts/low_level_demo_task_id_mapping.py

# train
# Note: Remember to change trained_highlevel_planner and demo_path in configs/new_lowlevel.json
# robot0_eef_pos_3d_world: used for low-level policy
# robot0_eef_pos_3d_camera: used for high-level controller
python scripts/train.py --config configs/new_lowlevel.json --dataset '/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/ur5e_pick_place/hdf5_files/merged_dataset/ur5e_all_demos.hdf5'

```

## 6. Test Low Level