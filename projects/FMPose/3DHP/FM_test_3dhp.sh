#Train CFM
layers=5
lr=1e-3
gpu_id=0
eval_sample_steps=1,2
batch_size=1
epochs=100
folder_name=FM_GAMLP_noisePose_layers${layers}_1GCNParallelAttnMLP_attnD_0.1_projD_0.1_lr${lr}_decay${decay}_lr_decay_large_e${large_decay_epoch}_${lr_decay_large}_B${batch_size}_$(date +%Y%m%d_%H%M)

# 49.72
saved_model_path='./pretrained_models/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/CFM_36_4972_best.pth'
model_path='./pretrained_models/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/250916_1953_32_model_GAMLP.py'

# 49.69
# saved_model_path='./pretrained_models/FM_GPA_Noise_1GCN_P_Attn_layers5_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20251012_163120/CFM_23_4969_best.pth'
# model_path='./pretrained_models/FM_GPA_Noise_1GCN_P_Attn_layers5_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20251012_163120/251012_1631_22_model_G_P_Attn.py'

python FM_main_3dhp_noise_pose_camera.py \
    --gpu ${gpu_id} \
    --nepoch 1 \
    --batch_size ${batch_size} \
    --test \
    --token_dim 256 \
    --frames 1 \
    --layers ${layers} \
    --channel 512 \
    --d_hid 1024 \
    --eval_sample_steps ${eval_sample_steps} \
    --model_path "${model_path}" \
    --dataset '3dhp_valid' \
    --keypoints 'gt_17_univ' \
    --saved_model_path "${saved_model_path}"