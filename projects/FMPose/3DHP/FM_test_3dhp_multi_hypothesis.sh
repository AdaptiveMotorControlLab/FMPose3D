#Test CFM with Multi-Hypothesis
layers=5
lr=1e-3
gpu_id=1
eval_sample_steps=2
batch_size=1024
epochs=100

# Multi-hypothesis parameters
weight_softmax_tau=1.0
num_hypothesis_list=10
topk=5
mode='exp'
exp_temp=0.005

# Generate folder name with timestamp
folder_name=3DHP_MultiHyp_s${eval_sample_steps}_Top${topk}_${mode}_temp${exp_temp}_h${num_hypothesis_list}_noGS_$(date +%Y%m%d_%H%M%S)

# Model paths - 49.72
saved_model_path='./pretrained_models/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/CFM_36_4972_best.pth'
model_path='./pretrained_models/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/250916_1953_32_model_GAMLP.py'

# Alternative model - 49.69
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
    --saved_model_path "${saved_model_path}" \
    --num_hypothesis_list "${num_hypothesis_list}" \
    --topk ${topk} \
    --exp_temp ${exp_temp} \
    --weight_softmax_tau ${weight_softmax_tau} \
    --folder_name "${folder_name}" \
    --test_augmentation True \
    --test_augmentation_flip_hypothesis True