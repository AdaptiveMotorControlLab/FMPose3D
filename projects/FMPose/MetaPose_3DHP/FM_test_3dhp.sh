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

# Read WANDB_API_KEY from file if not provided via env
key_file="$(dirname "$0")/wandb_api_key.txt"
if [ -z "$WANDB_API_KEY" ] && [ -f "$key_file" ]; then
  WANDB_API_KEY="$(head -n1 "$key_file" | tr -d ' \n\r')"
fi

if [ -n "$WANDB_API_KEY" ]; then
  wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

python FM_main_3dhp_noise_pose.py --gpu ${gpu_id} --nepoch 1 --batch_size ${batch_size} --test --token_dim 256 \
--frames 1 --layers ${layers} --channel 512 --d_hid 1024 --eval_sample_steps ${eval_sample_steps} \
--model ${model_name} \
--dataset '3dhp_valid' --keypoints 'gt_17_univ'\
--saved_model_path '/home/ti_wang/Ti_workspace/projects/FMPose/checkpoint/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_lr1e-3_decay0.95_lr_decay_large_e5_0.8_B256_20250915_2211/CFM_24_4975.pth'