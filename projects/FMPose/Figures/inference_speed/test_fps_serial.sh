#!/bin/bash
# FPS Testing Script for FMPose (Serial Version - for comparison)

layers=5
lr=1e-3
decay=0.95
gpu_id=0
eval_multi_steps=3
batch_size=1
hypothesis_num=5

sh_file='FMPose_fps_test_serial.sh'

model_path='pretrained_model/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/250916_1953_32_model_GAMLP.py'
saved_model_path='pretrained_model/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/CFM_36_4972_best.pth'

echo "=================================="
echo "FMPose FPS Testing (Serial)"
echo "=================================="
echo "GPU: ${gpu_id}"
echo "Batch size: ${batch_size}"
echo "Hypothesis number: ${hypothesis_num}"
echo "Sampling steps: ${eval_multi_steps}"
echo "=================================="

# Test serial version (for-loop)
python3 FMPose_fps_test_serial.py \
--reload \
--saved_model_path "${saved_model_path}" \
--model_path "${model_path}" \
--eval_sample_steps ${eval_multi_steps} \
--test_augmentation False \
--batch_size ${batch_size} \
--layers ${layers} \
--gpu ${gpu_id} \
--hypothesis_num ${hypothesis_num} \
--sh_file ${sh_file}

echo ""
echo "Serial FPS test completed!"

