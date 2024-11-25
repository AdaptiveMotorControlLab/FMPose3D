#!/bin/bash

# Path to your video file
# VIDEO_PATH="/home/ti_wang/Ti_workspace/PrimatePose/data/macaque_monkey.mp4"
# VIDEO_PATH="/home/ti_wang/Ti_workspace/PrimatePose/data/demo/demo_8s/8s_3840_2160_25fps.mp4"
VIDEO_PATH=/home/ti_wang/Ti_workspace/PrimatePose/data/demo/monkey_2/monkey_2_3840_2160_25fps.mp4
# VIDEO_PATH=/home/ti_wang/Ti_workspace/PrimatePose/data/demo/monkey_single_1/6575012-uhd_3840_2160_25fps.mp4
# VIDEO_PATH=/home/ti_wang/Ti_workspace/PrimatePose/data/demo/single_monkey_2/1508571-uhd_3840_2160_25fps.mp4
# Number of animals to detect in each frame
NUM_ANIMALS=2

# Path to model configuration file
MODEL_CONFIG=/home/ti_wang/Ti_workspace/PrimatePose/project/pfm_pose_hrnet_train/train/pytorch_config.yaml
# Path to trained model snapshot
SNAPSHOT_PATH=/home/ti_wang/Ti_workspace/PrimatePose/project/pfm_pose_hrnet_train/train/snapshot-best-056.pt

# MODEL_CONFIG="/home/ti_wang/Ti_workspace/PrimatePose/project/split/mp_pose_hrnet_train/train/pytorch_config.yaml"
# Path to trained model snapshot
# SNAPSHOT_PATH="/home/ti_wang/Ti_workspace/PrimatePose/project/split/mp_pose_hrnet_train/train/snapshot-best-145.pt"

# Optional: Path to detector model if using top-down approach
# DETECTOR_PATH="/home/ti_wang/Ti_workspace/PrimatePose/project/split/oms_detector_fasterrcnn_train/train/snapshot-detector-best-021.pt"
# DETECTOR_PATH="/home/ti_wang/Ti_workspace/PrimatePose/project/split/mp_detector_fasterrcnn_train/train/snapshot-detector-best-024.pt"
DETECTOR_PATH=/home/ti_wang/Ti_workspace/PrimatePose/project/pfm_merged_checked_detector_fasterrcnn_train/train/snapshot-detector-best-083.pt

# Run video analysis
python analyze_video.py \
    "$VIDEO_PATH" \
    "$MODEL_CONFIG" \
    "$SNAPSHOT_PATH" \
    --detector_path "$DETECTOR_PATH" \
    --num_animals "$NUM_ANIMALS"
    
# # Run video analysis
# python analyze_video.py \
#     --video-path "$VIDEO_PATH" \
#     --model-config "$MODEL_CONFIG" \
#     --snapshot-path "$SNAPSHOT_PATH" \
#     --detector-path "$DETECTOR_PATH" \
#     --num-animals "$NUM_ANIMALS"
