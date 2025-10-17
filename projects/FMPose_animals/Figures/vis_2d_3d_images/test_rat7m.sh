# Model parameters
layers=5
batch_size=1
eval_sample_steps=3
# Script name for logging
sh_file='test_rat7m.sh'
# Paths to pretrained model and model definition
model_path='./model/model_G_P_Attn_rat.py'
saved_model_path='./pretrained_models/test_GCN_model/CFM_4_8221_best.pth'

gpu_id=0

# Using small subjects and views for quick test
train_list='s1d1'
test_list='s1d1'
train_views='0'
test_views='0'
# Root path to dataset
root_path="/media/ti/datasets/Rat7M_data"

# Test pretrained model
python3 vis_main_CFM_rat7m.py \
  --reload \
  --saved_model_path "${saved_model_path}" \
  --model_path "${model_path}" \
  --dataset rat7m \
  --root_path ${root_path} \
  --eval_sample_steps ${eval_sample_steps} \
  --batch_size ${batch_size} \
  --layers ${layers} \
  --gpu ${gpu_id} \
  --train_list ${train_list} \
  --test_list ${test_list} \
  --train_views ${train_views} \
  --test_views ${test_views} \
  --sh_file ${sh_file}