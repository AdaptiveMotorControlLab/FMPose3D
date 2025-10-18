#Train CFM
layers=5
lr=1e-3
decay=0.98
gpu_id=0
eval_sample_steps=3
batch_size=256
large_decay_epoch=5
lr_decay_large=0.8
epochs=100
num_saved_models=5
model_name=model_GAMLP
folder_name=FM_GAMLP_noisePose_layers${layers}_1GCNParallelAttnMLP_attnD_0.1_projD_0.1_lr${lr}_decay${decay}_lr_decay_large_e${large_decay_epoch}_${lr_decay_large}_B${batch_size}_$(date +%Y%m%d_%H%M)

saved_model_path='./pretrained_models/FM_GPA_Noise_1GCN_P_Attn_layers5_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20251012_163120/CFM_23_4969_best.pth'
model_path='./pretrained_models/FM_GPA_Noise_1GCN_P_Attn_layers5_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20251012_163120/251012_163120_32_model_GAMLP.py'

python FM_main_3dhp_noise_pose.py \
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
    --model ${model_name} \
    --model_path "${model_path}" \
    --dataset '3dhp_valid' \
    --keypoints 'gt_17_univ' \
    --saved_model_path "${saved_model_path}"