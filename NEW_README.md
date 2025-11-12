# MimicPlay

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