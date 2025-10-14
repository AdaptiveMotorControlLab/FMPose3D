#Train CFM
layers=5
lr=1e-3
decay=0.98
gpu_id=1
eval_sample_steps=3
batch_size=256
large_decay_epoch=5
lr_decay_large=0.8
epochs=100
num_saved_models=3
frames=1
model_path='model/model_G_P_Attn.py'
sh_file='run_FM_noiseX.sh'
folder_name=FM_GPA_Noise_1GCN_P_Attn_layers${layers}_lr${lr}_decay${decay}_lr_decay_large_e${large_decay_epoch}_${lr_decay_large}_B${batch_size}_$(date +%Y%m%d_%H%M%S)

# Read WANDB_API_KEY from file if not provided via env
key_file="$(dirname "$0")/wandb_api_key.txt"
if [ -z "$WANDB_API_KEY" ] && [ -f "$key_file" ]; then
  WANDB_API_KEY="$(head -n1 "$key_file" | tr -d ' \n\r')"
fi

if [ -n "$WANDB_API_KEY" ]; then
  wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

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
  --sh_file ${sh_file}

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

#Test CFM
# python3 main_CFM.py --reload --previous_dir "./debug/250908_1418_45" --model model_GUMLP --sample_steps 3 --test_augmentation True --layers 5 --gpu 0
# python3 main_CFM.py --reload --previous_dir "./checkpoint/FM_x0_noise_layers4_lr1e-3_decay0.98_sample3_20250911" --model ${model_name} --sample_steps 3 --test_augmentation True --batch_size ${batch_size} --layers ${layers} --gpu ${gpu_id}
# python3 main_CFM_T_pose.py --reload --previous_dir "./checkpoint/FM_GMLP_Tpose_noise_layers5_lr1e-3_decay0.95_B1024_20250913_1555" --model ${model_name} --eval_multi_steps --eval_sample_steps ${eval_multi_steps} --batch_size ${batch_size} --test_augmentation True --layers ${layers} --gpu ${gpu_id}