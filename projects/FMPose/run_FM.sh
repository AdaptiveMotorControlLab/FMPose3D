#Train CFM
layers=10
lr=1e-3
decay=0.98
gpu_id=1
model_name=model_CrossAttention
folder_name=FM_CrossAttention_x0_T_pose_noise_layers${layers}_lr${lr}_decay${decay}_$(date +%Y%m%d_%H%M)

# Read WANDB_API_KEY from file if not provided via env
key_file="$(dirname "$0")/wandb_api_key.txt"
if [ -z "$WANDB_API_KEY" ] && [ -f "$key_file" ]; then
  WANDB_API_KEY="$(head -n1 "$key_file" | tr -d ' \n\r')"
fi

if [ -n "$WANDB_API_KEY" ]; then
  wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi
python3 main_CFM_T_pose.py --train --model ${model_name} --gpu ${gpu_id} --layers ${layers} --lr ${lr} --lr_decay ${decay} --nepoch 100 --eval_multi_steps --eval_sample_steps 1,3,5,7,9 --folder_name $folder_name 
#Test CFM
# python3 main_CFM.py --reload --previous_dir "./debug/250908_1418_45" --model model_GUMLP --sample_steps 3 --test_augmentation True --layers 5 --gpu 0
# python3 main_CFM_test_vis.py --reload --previous_dir "./debug/250906_2235_46" --model model_GUMLP --sample_steps 19 --batch_size 1 --test_augmentation True --layers 4 --gpu 0