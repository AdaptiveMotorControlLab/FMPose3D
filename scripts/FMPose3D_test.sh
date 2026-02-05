#inference
layers=5
batch_size=1024
sh_file='scripts/FMPose3D_test.sh'
weight_softmax_tau=1.0
num_hypothesis_list=1
eval_multi_steps=3
topk=8
gpu_id=0
mode='exp'
exp_temp=0.005
folder_name=test_s${eval_multi_steps}_${mode}_h${num_hypothesis_list}_$(date +%Y%m%d_%H%M%S)

model_path='../pre_trained_models/fmpose3d_h36m/model_GAMLP.py'
model_weights_path='../pre_trained_models/fmpose3d_h36m/FMpose3D_pretrained_weights.pth'

#Test
python3 scripts/FMPose3D_main.py \
--reload \
--topk ${topk} \
--exp_temp ${exp_temp} \
--weight_softmax_tau ${weight_softmax_tau} \
--folder_name ${folder_name} \
--model_weights_path "${model_weights_path}" \
--model_path "${model_path}" \
--eval_sample_steps ${eval_multi_steps} \
--test_augmentation True \
--batch_size ${batch_size} \
--layers ${layers} \
--gpu ${gpu_id} \
--num_hypothesis_list ${num_hypothesis_list} \
--sh_file ${sh_file}