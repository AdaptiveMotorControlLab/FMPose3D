# Test pretrained CFM model
layers=5
lr=1e-3
decay=0.95
gpu_id=1
batch_size=4
large_decay_epoch=5
lr_decay_large=0.9
epochs=100
eval_multi_steps=3
folder_name=debug_FM_projection
model_path='pretrained_model/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/250916_1953_32_model_GAMLP.py'
saved_model_path='pretrained_model/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/CFM_36_4972_best.pth'
sh_file='run_FM_debug.sh'

python3 main_CFM_noise_pose_debug.py \
  --reload \
  --test 1 \
  --model_path ${model_path} \
  --saved_model_path ${saved_model_path} \
  --gpu ${gpu_id} \
  --batch_size ${batch_size} \
  --layers ${layers} \
  --lr ${lr} \
  --lr_decay ${decay} \
  --nepoch 100 \
  --eval_sample_steps ${eval_multi_steps} \
  --test_augmentation True \
  --debug \
  --folder_name ${folder_name} \
  --sh_file ${sh_file}