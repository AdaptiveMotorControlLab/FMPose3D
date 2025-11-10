#Train CFM
layers=5
lr=1e-3
decay=0.98
gpu_id=1
eval_sample_steps=3
batch_size=256
large_decay_epoch=5
lr_decay_large=0.8
epochs=80
num_saved_models=3
frames=1
channel_dim=512
model_path='./model/model_G_P_Attn.py'
sh_file='run_FM_noiseX.sh'
folder_name=FM_GPA_TestP_deleteclone_inGCN_1GCN_P_Attn_layers${layers}_lr${lr}_decay${decay}_lr_decay_large_e${large_decay_epoch}_${lr_decay_large}_B${batch_size}_$(date +%Y%m%d_%H%M%S)

#--keypoints gt \
# training
python3 main_CFM_noise_pose.py \
  --train \
  --dataset h36m \
  --frames ${frames} \
  --model_path ${model_path} \
  --gpu ${gpu_id} \
  --batch_size ${batch_size} \
  --layers ${layers} \
  --lr ${lr} \
  --lr_decay ${decay} \
  --nepoch ${epochs} \
  --eval_sample_steps ${eval_sample_steps} \
  --folder_name ${folder_name} \
  --large_decay_epoch ${large_decay_epoch} \
  --lr_decay_large ${lr_decay_large} \
  --num_saved_models ${num_saved_models} \
  --sh_file ${sh_file} \
  --channel ${channel_dim}

# Testing CFM
# model_path='./pretrained_model/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/250916_1953_32_model_GAMLP.py'
# saved_model_path='./pretrained_model/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/CFM_36_4972_best.pth'

# python3 main_CFM_noise_pose_test.py \
# --reload \
# --saved_model_path ${saved_model_path} \
# --model_path ${model_path} \
# --gpu ${gpu_id} \
# --batch_size ${batch_size} \
# --layers ${layers} \
# --nepoch ${epochs} \
# --eval_sample_steps ${eval_sample_steps} \
# --num_saved_models ${num_saved_models} \
# --sh_file ${sh_file}