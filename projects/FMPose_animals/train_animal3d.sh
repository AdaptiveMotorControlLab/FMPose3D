#!/bin/bash
# Train Rat7M with default split: train=['s1d1'], test=['s5d1']
# Model: model/model_attn.py (Conditional Flow Matching with Attention)

layers=4
batch_size=32
lr=1e-3
gpu_id=0
eval_sample_steps=3
num_saved_models=3
frames=1
large_decay_epoch=15
lr_decay_large=0.7
n_joints=26
out_joints=26
epochs=400
# model_path='model/model_G_P_Attn_rat.py'
model_path='model/model_attn.py'

# root path denotes the path to the original dataset
root_path="/home/xiaohang/Ti_workspace/projects/FMPose_animals/dataset/"
folder_name="Rat7M_data_GCN_L${layers}_lr${lr}_B${batch_size}_$(date +%Y%m%d_%H%M%S)"
sh_file='train_animal3d.sh'

python main_CFM_animal3d.py \
  --root_path ${root_path} \
  --dataset rat7m \
  --train \
  --test 1 \
  --batch_size ${batch_size} \
  --lr ${lr} \
  --model_path ${model_path} \
  --folder_name ${folder_name} \
  --layers ${layers} \
  --gpu ${gpu_id} \
  --eval_sample_steps ${eval_sample_steps} \
  --num_saved_models ${num_saved_models} \
  --sh_file ${sh_file} \
  --nepoch ${epochs} \
  --large_decay_epoch ${large_decay_epoch} \
  --lr_decay_large ${lr_decay_large} 
  # --n_joints ${n_joints} \
  # --out_joints ${out_joints} 