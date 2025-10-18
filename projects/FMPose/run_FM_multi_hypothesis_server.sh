#Test CFM with multi-hypothesis
layers=5
batch_size=1024
epochs=100
sh_file='run_FM_multi_hypothesis_server.sh'

weight_softmax_tau=1.0
num_hypothesis_list=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
eval_multi_steps=3
topk=8
gpu_id=0
mode='exp'
exp_temp=0.005
flipaug=False
folder_name=Cam_noaug_s${eval_multi_steps}_Top${topk}_${mode}_temp${exp_temp}_h${num_hypothesis_list}_noflipforNoise_test_results_$(date +%Y%m%d_%H%M%S)
job_name=fm-multi-hyp-19

# model_path='pretrained_model/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/250916_1953_32_model_GAMLP.py'
# saved_model_path='pretrained_model/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/CFM_36_4972_best.pth'
model_path='pretrained_model/FM_GPA_Noise_1GCN_P_Attn_layers5_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20251012_163120/251012_1631_22_model_G_P_Attn.py'
saved_model_path='pretrained_model/FM_GPA_Noise_1GCN_P_Attn_layers5_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20251012_163120/CFM_23_4969_best.pth'

# Build the Python command
PYTHON_CMD="cd /data/ti/Ti_workspace/projects/FMPose && python3 main_CFM_noise_pose_multiHypothesis_test.py \
--reload \
--topk ${topk} \
--exp_temp ${exp_temp} \
--weight_softmax_tau ${weight_softmax_tau} \
--folder_name ${folder_name} \
--saved_model_path ${saved_model_path} \
--model_path ${model_path} \
--eval_sample_steps ${eval_multi_steps} \
--test_augmentation True \
--test_augmentation_flip_hypothesis ${flipaug} \
--batch_size ${batch_size} \
--layers ${layers} \
--gpu ${gpu_id} \
--num_hypothesis_list ${num_hypothesis_list} \
--sh_file ${sh_file}"

runai submit --gpu 1 --node-pools h100 --name ${job_name} \
--image registry.rcp.epfl.ch/pfm_ti/fm_pose:v0.3 --backoff-limit 0 --large-shm \
--pvc home:${HOME} --pvc upmwmathis-scratch:/data -e HOME=${HOME} -p upmwmathis-wang3 \
--command -- /bin/bash -c "${PYTHON_CMD}"