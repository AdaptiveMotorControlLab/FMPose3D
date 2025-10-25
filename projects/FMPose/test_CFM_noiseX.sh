#Train CFM
layers=5
lr=1e-3
decay=0.98
gpu_id=0
eval_sample_steps=3
batch_size=256
large_decay_epoch=5
lr_decay_large=0.8
epochs=80
num_saved_models=3
frames=1
channel_dim=512
model_path='./model/model_G_P_Attn.py'
sh_file='test_CFM_noiseX.sh'

# Testing CFM with Classifier-Free Guidance

model_path='./pretrained_model/CFM_GPA_Noise_layers5_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20251025_002506/251025_0025_08_model_G_P_Attn.py'
saved_model_path='./pretrained_model/CFM_GPA_Noise_layers5_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20251025_002506/CFM_16_4983_best.pth'

guidance_scale=1.2  # 1.0 = no guidance, 1.5-3.0 = typical range, higher = stronger conditioning

python3 main_CFM_noise_pose_test.py \
--reload \
--saved_model_path ${saved_model_path} \
--model_path ${model_path} \
--gpu ${gpu_id} \
--batch_size ${batch_size} \
--layers ${layers} \
--nepoch ${epochs} \
--eval_sample_steps ${eval_sample_steps} \
--num_saved_models ${num_saved_models} \
--guidance_scale ${guidance_scale} \
--sh_file ${sh_file}

#Test CFM
# python3 main_CFM.py --reload --previous_dir "./debug/250908_1418_45" --model model_GUMLP --sample_steps 3 --test_augmentation True --layers 5 --gpu 0
# python3 main_CFM.py --reload --previous_dir "./checkpoint/FM_x0_noise_layers4_lr1e-3_decay0.98_sample3_20250911" --model ${model_name} --sample_steps 3 --test_augmentation True --batch_size ${batch_size} --layers ${layers} --gpu ${gpu_id}
# python3 main_CFM_T_pose.py --reload --previous_dir "./checkpoint/FM_GMLP_Tpose_noise_layers5_lr1e-3_decay0.95_B1024_20250913_1555" --model ${model_name} --eval_multi_steps --eval_sample_steps ${eval_multi_steps} --batch_size ${batch_size} --test_augmentation True --layers ${layers} --gpu ${gpu_id}