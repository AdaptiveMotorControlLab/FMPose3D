#Visualize CFM on 3DHP dataset
layers=5
lr=1e-3
decay=0.95
gpu_id=1
eval_sample_steps=3
batch_size=1
large_decay_epoch=5
lr_decay_large=0.9
epochs=100

# Multi-hypothesis parameters
hypothesis_num=10
topk=8
exp_temp=0.005

folder_name=noGS_$(date +%Y%m%d_%H%M%S)
# Test subjects - all test subjects
# subjects_test=TS1,TS2,TS3,TS4,TS5,TS6
# GS only
# subjects_test=TS1,TS2
# no GS only
subjects_test=TS3,TS4
# Outdoor only
# subjects_test=TS5,TS6

sh_file='vis_3dhp.sh'

# Model paths - same as H36M (49.72mm MPJPE)
model_path='./pretrained_models/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/250916_1953_32_model_GAMLP.py'
saved_model_path='./pretrained_models/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/CFM_36_4972_best.pth'

# Alternative model - 49.69mm
# model_path='./pretrained_models/FM_GPA_Noise_1GCN_P_Attn_layers5_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20251012_163120/251012_1631_22_model_G_P_Attn.py'
# saved_model_path='./pretrained_models/FM_GPA_Noise_1GCN_P_Attn_layers5_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20251012_163120/CFM_23_4969_best.pth'

#Visualize 3DHP
python vis_FM_3dhp_compareWithGT.py \
--reload \
--saved_model_path "${saved_model_path}" \
--model_path "${model_path}" \
--eval_sample_steps ${eval_sample_steps} \
--test_augmentation True \
--batch_size ${batch_size} \
--layers ${layers} \
--gpu ${gpu_id} \
--hypothesis_num "${hypothesis_num}" \
--topk ${topk} \
--exp_temp ${exp_temp} \
--dataset '3dhp_valid' \
--keypoints 'gt_17_univ' \
--subjects_test "${subjects_test}" \
--create_file 1 \
--sh_file ${sh_file} \
--folder_name ${folder_name}

