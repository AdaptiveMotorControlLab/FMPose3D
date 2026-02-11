layers=5
batch_size=13
lr=1e-3
gpu_id=0
eval_sample_steps=5
num_saved_models=3
frames=1
large_decay_epoch=15
lr_decay_large=0.75
n_joints=26
out_joints=26
epochs=300
model_type='fmpose3d_animals'
# model_path='' # set to a local file path to override the registry
saved_model_path='./pre_trained_models/fmpose3d_animals/fmpose3d_animals_pretrained_weights.pth'

# root path denotes the path to the original dataset
root_path="./dataset/"
train_dataset_paths=(
  "./dataset/animal3d/train.json"
  "./dataset/control_animal3dlatest/train.json"
)
test_dataset_paths=(
  "./dataset/control_animal3dlatest/test.json"
)

folder_name="TestCtrlAni3D_L${layers}_lr${lr}_B${batch_size}_$(date +%Y%m%d_%H%M%S)"
sh_file='scripts/test_animal3d.sh'

python ./scripts/main_animal3d.py \
  --root_path ${root_path} \
  --reload \
  --dataset animal3d \
  --test 1 \
  --batch_size ${batch_size} \
  --lr ${lr} \
  --model_type "${model_type}" \
  ${model_path:+--model_path "$model_path"} \
  --folder_name ${folder_name} \
  --layers ${layers} \
  --gpu ${gpu_id} \
  --eval_sample_steps ${eval_sample_steps} \
  --num_saved_models ${num_saved_models} \
  --sh_file ${sh_file} \
  --nepoch ${epochs} \
  --large_decay_epoch ${large_decay_epoch} \
  --lr_decay_large ${lr_decay_large} \
  --train_dataset_path ${train_dataset_paths[@]} \
  --test_dataset_path ${test_dataset_paths[@]} \
  --saved_model_path ${saved_model_path}