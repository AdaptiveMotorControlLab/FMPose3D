#!/bin/bash

# Bash script to run the animal pose visualization

# Change to the FMPose_animals directory
cd /home/xiaohang/Ti_workspace/projects/FMPose_animals

root_path="./dataset/control_animal3dlatest/"

model_path='model/model_G_P_Attn_animal3d.py'
saved_model_path='./checkpoint/GPA_TrainBoth_TestCtrlAni3D_L4_lr1e-3_B32_20251111_140955/CFM_162_4620_best.pth'

test_dataset_paths=(
  "./dataset/control_animal3dlatest/test.json"
)

# Run the visualization script with example arguments
# Adjust the paths and arguments as needed
python visualize_animal_poses.py \
    --model_path $model_path \
    --root_path $root_path \
    --dataset animal3d \
    --saved_model_path $saved_model_path \
    --test_dataset_path ${test_dataset_paths[0]} \
    --gpu 0