#Train CFM
layers=5
gpu_id=1
sample_steps=3
batch_size=1
# Pass multiple values correctly to argparse using an array
# num_hypothesis_list=1,2,3,4,5,6,7,8,9,10
# sweep values for k (uncertainty-aware threshold). Comma-separated list like "0.8 0.9 1.0 1.2".

sh_file='vis_in_the_wild.sh'

model_path='pretrained_model/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/250916_1953_32_model_GAMLP.py'
saved_model_path='pretrained_model/FM_GAMLP_noisePose_layers5_1GCNParallelAttnMLP_attnD_0.2_projD_0.25_lr1e-3_decay0.98_lr_decay_large_e5_0.8_B256_20250916_1953/CFM_36_4972_best.pth'

path='./images/image_00068.jpg'

python3 vis_in_the_wild.py \
 --type 'image' \
 --path ${path} \
 --saved_model_path "${saved_model_path}" \
 --model_path "${model_path}" \
 --sample_steps ${sample_steps} \
 --batch_size ${batch_size} \
 --layers ${layers} \
 --gpu ${gpu_id} \
 --sh_file ${sh_file}