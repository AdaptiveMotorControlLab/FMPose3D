#Test
layers=5
gpu_id=1
sample_steps=3
batch_size=1
sh_file='vis_animals.sh'
# n_joints=26
# out_joints=26

model_path='../../pre_trained_models/animal3d_pretrained_weights/model_animal3d.py'
saved_model_path='../../pre_trained_models/animal3d_pretrained_weights/CFM_154_4403_best.pth'

# path='./images/image_00068.jpg'  # single image
input_images_folder='./images/'  # folder containing multiple images

python3 vis_animals.py \
 --type 'image' \
 --path ${input_images_folder} \
 --saved_model_path "${saved_model_path}" \
 --model_path "${model_path}" \
 --sample_steps ${sample_steps} \
 --batch_size ${batch_size} \
 --layers ${layers} \
 --dataset animal3d \
 --gpu ${gpu_id} \
 --sh_file ${sh_file}