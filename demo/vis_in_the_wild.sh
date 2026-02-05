#Test
layers=5
gpu_id=0
sample_steps=3
batch_size=1
sh_file='vis_in_the_wild.sh'

model_path='../pre_trained_models/fmpose3d_h36m/model_GAMLP.py'
model_weights_path='../pre_trained_models/fmpose3d_h36m/FMpose3D_pretrained_weights.pth'

target_path='./images/'  # folder containing multiple images
# target_path='./images/xx.png'  # single image
# target_path='./videos/xxx.mp4' # video path

python3 vis_in_the_wild.py \
 --type 'image' \
 --path ${target_path} \
 --model_weights_path "${model_weights_path}" \
 --model_path "${model_path}" \
 --sample_steps ${sample_steps} \
 --batch_size ${batch_size} \
 --layers ${layers} \
 --gpu ${gpu_id} \
 --sh_file ${sh_file}