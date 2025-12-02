#Test
layers=5
gpu_id=1
sample_steps=3
batch_size=1
sh_file='vis_in_the_wild.sh'

model_path='../pre_trained_models/fmpose_detected2d/model_GAMLP.py'
saved_model_path='../pre_trained_models/fmpose_detected2d/FMpose_36_4972_best.pth'

# path='./images/image_00068.jpg'  # single image
input_images_folder='./images/'  # folder containing multiple images

python3 vis_in_the_wild.py \
 --type 'image' \
 --path ${input_images_folder} \
 --saved_model_path "${saved_model_path}" \
 --model_path "${model_path}" \
 --sample_steps ${sample_steps} \
 --batch_size ${batch_size} \
 --layers ${layers} \
 --gpu ${gpu_id} \
 --sh_file ${sh_file}