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
num_saved_models=3
frames=1
model_path='model/model_G_P_Attn_normal.py'
sh_file='run_FM_noiseX.sh'
folder_name=FM_GPA_1GCN_P_Attn_layers${layers}_lr${lr}_decay${decay}_lr_decay_large_e${large_decay_epoch}_${lr_decay_large}_B${batch_size}_$(date +%Y%m%d_%H%M%S)

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
python3 main_pose.py \
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