data_path_prefix="/home/ti_wang/data"
data_root=${data_path_prefix}"/SuperAnimal/TopViewMouse5K_20240913"
project_root=$(dirname $(realpath $0))

debug=0
gpu_id="1"
mode="train"

OOD_dataset_name=openfield-Pranav-2018-08-20

# file_name=SA-TVM_pose_rtmpose_s_OOD_${OOD_dataset_name}_20250515
file_name=SA-TVM_pose_hrnetw32_MergedTrainAndTest_OOD_${OOD_dataset_name}_20250611

# train and test json files
train_json="${data_root}/annotations/${mode}_IID_wo_${OOD_dataset_name}.json"
# test_json="${data_root}/annotations/test_IID_wo_${OOD_dataset_name}.json"
test_json="${data_root}/annotations/test_OOD_${OOD_dataset_name}.json"

pytorch_config_path=${project_root}/experiments/${file_name}/train/pytorch_config.yaml
# snapshot_path=${project_root}/experiments/${file_name}/train/snapshot-400.pt
snapshot_path=${project_root}/experiments/${file_name}/train/snapshot-200.pt
# snapshot_path=${project_root}/experiments/${file_name}/train/snapshot-best-014.pt

oks_sigma_TVM="0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1"

python evaluation.py \
    $data_root \
    --pytorch_config_path $pytorch_config_path \
    --snapshot_path $snapshot_path \
    --train_file $train_json \
    --test_file $test_json \
    --oks_sigma $oks_sigma_TVM