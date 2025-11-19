# bash train_animal3d_ti.sh
layers=5
batch_size=13
lr=1e-3
gpu_id=0
eval_sample_steps=3
num_saved_models=3
frames=1
large_decay_epoch=15
lr_decay_large=0.75
n_joints=26
out_joints=26
epochs=300
# model_path='model/model_attn.py'
model_path='model/model_G_P_Attn_animal3d.py'
saved_model_path='/home/xiaohang/Ti_workspace/projects/FMPose_animals/checkpoint/GPA_TrainBoth_TestCtrlAni3D_L5_lr1e-3_B13_20251113_232613/CFM_154_4403_best.pth' # for control_animal3dlatest

# root path denotes the path to the original dataset
root_path="./dataset/"
train_dataset_paths=(
  "./dataset/animal3d/train.json"
  "./dataset/control_animal3dlatest/train.json"
)
test_dataset_paths=(
  "./dataset/animal3d/test.json"
)

folder_name="GPA_TrainBoth_TestCtrlAni3D_L${layers}_lr${lr}_B${batch_size}_$(date +%Y%m%d_%H%M%S)"
sh_file='test_animal3d_ti.sh'

python main_CFM_animal3d_ti.py \
  --root_path ${root_path} \
  --reload \
  --dataset animal3d \
  --test 1 \
  --batch_size ${batch_size} \
  --lr ${lr} \
  --model_path ${model_path} \
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
  --saved_model_path ${saved_model_path} \
  # --n_joints ${n_joints} \
  # --out_joints ${out_joints} 