#!/bin/bash

# Training script for 3D Pose Estimation (Global coordinate system)
# This script automatically copies source files and saves parameters for reproducibility

CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset_name "oms" \
    --train_json /home/ti_wang/Ti_workspace/PrimatePose/Pose3D/data/oms_train.json \
    --val_json /home/ti_wang/Ti_workspace/PrimatePose/Pose3D/data/oms_test.json \
    --epochs 200 \
    --batch_size 128 \
    --lr 5e-4 \
    --gpu 1 \
    --max_checkpoints 5 \
    --save_freq 5 \
    --seed 1