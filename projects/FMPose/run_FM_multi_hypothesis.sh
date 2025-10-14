#Train CFM
layers=5
lr=1e-3
decay=0.95
batch_size=1024
large_decay_epoch=5
lr_decay_large=0.9
epochs=100
# Pass multiple values correctly to argparse using an array
# num_hypothesis_list=1,2,3,4,5,6,7,8,9,10

model_name=model_GUMLP

sh_file='run_FM_multi_hypothesis.sh'

# Read WANDB_API_KEY from file if not provided via env
key_file="$(dirname "$0")/wandb_api_key.txt"
if [ -z "$WANDB_API_KEY" ] && [ -f "$key_file" ]; then
  WANDB_API_KEY="$(head -n1 "$key_file" | tr -d ' \n\r')"
fi

if [ -n "$WANDB_API_KEY" ]; then
  wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

weight_softmax_tau=1.0
num_hypothesis_list=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
eval_multi_steps=3
topk=10
gpu_id=0
mode='exp'
exp_temp=0.01
# python3 main_CFM_noise_pose.py --train --model ${model_name} --gpu ${gpu_id} --batch_size ${batch_size} --layers ${layers} --lr ${lr} --lr_decay ${decay} --nepoch ${epochs} --eval_multi_steps --eval_sample_steps ${eval_multi_steps} --folder_name $folder_name --large_decay_epoch ${large_decay_epoch} --lr_decay_large ${lr_decay_large}
folder_name=exp_FlipHAug_s${eval_multi_steps}_Top${topk}_${mode}_temp${exp_temp}_h${num_hypothesis_list}_noflipforNoise_test_results_$(date +%Y%m%d_%H%M%S)
model_path='pretrained_model/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/250916_1953_32_model_GAMLP.py'
saved_model_path='pretrained_model/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/CFM_36_4972_best.pth'

#Test CFM
python3 main_CFM_noise_pose_multiHypothesis_test.py \
--reload \
--topk ${topk} \
--exp_temp ${exp_temp} \
--weight_softmax_tau ${weight_softmax_tau} \
--folder_name ${folder_name} \
--saved_model_path "${saved_model_path}" \
--model_path "${model_path}" \
--eval_sample_steps ${eval_multi_steps} \
--test_augmentation True \
--test_augmentation_flip_hypothesis True \
--batch_size ${batch_size} \
--layers ${layers} \
--gpu ${gpu_id} \
--num_hypothesis_list ${num_hypothesis_list} \
--sh_file ${sh_file}