#!/bin/bash
# Train Rat7M with default split: train=['s1d1'], test=['s5d1']
# Model: model/model_attn.py (Conditional Flow Matching with Attention)

layers=5
batch_size=256
lr=1e-4
gpu_id=1
eval_sample_steps=3
num_saved_models=3
frames=1
epochs=100
model_path='model/model_attn.py'
# full datasets
# train_list='s1d1 s2d1 s2d2 s3d1 s4d1'
# test_list='s5d1 s5d2'
# train_views='0 1 2 3 4 5'
# test_views='0 1 2 3 4 5'
# using small subjects and views for debug
train_list='s1d1'
test_list='s5d1'
train_views='0'
test_views='0'
folder_name="Rat7M_data_L${layers}_lr${lr}_B${batch_size}_$(date +%Y%m%d_%H%M%S)"
sh_file='train_rat7m.sh'

python main_CFM_rat7m.py \
  --root_path "Rat7M_data/" \
  --train \
  --test 1 \
  --batch_size 256 \
  --lr 1e-4 \
  --model_path ${model_path} \
  --folder_name ${folder_name} \
  --layers ${layers} \
  --lr ${lr} \
  --gpu ${gpu_id} \
  --eval_sample_steps ${eval_sample_steps} \
  --num_saved_models ${num_saved_models} \
  --frames ${frames} \
  --train_list ${train_list} \
  --test_list ${test_list} \
  --train_views ${train_views} \
  --test_views ${test_views} \
  --sh_file ${sh_file} \
  --nepoch ${epochs}