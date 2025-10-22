#!/bin/bash
# Train Rat7M with default split: train=['s1d1'], test=['s5d1']
# Model: model/model_attn.py (Conditional Flow Matching with Attention)

layers=5
batch_size=1024
lr=1e-3
gpu_id=0
eval_sample_steps=3
num_saved_models=3
frames=1
epochs=100
# model_path='model/model_G_P_Attn_rat.py'
model_path='model/model_attn.py'

# root path denotes the path to the original dataset
root_path="/home/xiaohang/Ti_workspace/projects/FMPose_animals/dataset/animal3d/"
folder_name="Rat7M_data_GCN_L${layers}_lr${lr}_B${batch_size}_$(date +%Y%m%d_%H%M%S)"
sh_file='train_animal3d.sh'

python main_CFM_animal3d.py \
  --root_path ${root_path} \
  --dataset rat7m \
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
  --sh_file ${sh_file} \
  --nepoch ${epochs}