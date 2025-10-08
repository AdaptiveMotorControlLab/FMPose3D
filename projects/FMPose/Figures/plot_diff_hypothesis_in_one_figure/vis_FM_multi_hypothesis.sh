#Train CFM
layers=5
lr=1e-3
decay=0.95
gpu_id=0
eval_multi_steps=1
batch_size=1
large_decay_epoch=5
lr_decay_large=0.9
epochs=100
# Pass multiple values correctly to argparse using an array
# num_hypothesis_list=1,2,3,4,5,6,7,8,9,10
num_hypothesis_list=8
hypothesis_num=5
# sweep values for k (uncertainty-aware threshold). Comma-separated list like "0.8 0.9 1.0 1.2".

sh_file='vis_FM_multi_hypothesis.sh'

# Read WANDB_API_KEY from file if not provided via env
key_file="$(dirname "$0")/wandb_api_key.txt"
if [ -z "$WANDB_API_KEY" ] && [ -f "$key_file" ]; then
  WANDB_API_KEY="$(head -n1 "$key_file" | tr -d ' \n\r')"
fi

if [ -n "$WANDB_API_KEY" ]; then
  wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

model_path='pretrained_model/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/250916_1953_32_model_GAMLP.py'
saved_model_path='pretrained_model/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/CFM_36_4972_best.pth'

#Test CFM
python3 vis_FMPose_hypothesis.py \
--reload \
--saved_model_path "${saved_model_path}" \
--model_path "${model_path}" \
--eval_sample_steps ${eval_multi_steps} \
--test_augmentation True \
--batch_size ${batch_size} \
--layers ${layers} \
--gpu ${gpu_id} \
--num_hypothesis_list ${num_hypothesis_list} \
--hypothesis_num ${hypothesis_num} \
--sh_file ${sh_file}