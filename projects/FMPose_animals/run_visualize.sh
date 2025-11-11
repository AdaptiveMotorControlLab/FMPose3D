#!/bin/bash

# Bash script to run the animal pose visualization

# Change to the FMPose_animals directory
cd /home/xiaohang/Ti_workspace/projects/FMPose_animals

root_path="./dataset/animal3d/"

model_path='model/model_G_P_Attn_animal3d.py'
saved_model_path='./checkpoint/GPA_TrainBoth_TestAnimal3d_L4_lr1e-3_B32_20251111_134054/CFM_108_6413_best.pth'

test_dataset_paths=(
  "./dataset/animal3d/test.json"
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