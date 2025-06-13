data_path_prefix="/home/ti_wang/data"
data_root=${data_path_prefix}"/SuperAnimal/Quadruped80K"
project_root=$(dirname $(realpath $0))

debug=0
gpu_id="0"
mode="train"

OOD_dataset_name=AP-10K

file_name="SAQ_pose_hrnetw32_MergedTrainAndTest_OOD_AP-10K_20250611"

# train and test json files
train_json="${data_root}/annotations/${mode}_IID_wo_${OOD_dataset_name}.json"
test_json="${data_root}/annotations/test_OOD_${OOD_dataset_name}.json"

pytorch_config_path=${project_root}/experiments/${file_name}/train/pytorch_config.yaml
# snapshot_path=${project_root}/experiments/${file_name}/train/snapshot-400.pt
snapshot_path=${project_root}/experiments/${file_name}/train/snapshot-172.pt
# snapshot_path=${project_root}/experiments/${file_name}/train/snapshot-best-301.pt

oks_sigma_Quadruped="0.026,0.067,0.067,0.067,0.067,0.025,0.067,0.067,0.067,0.067,0.025,0.067,0.067,0.067,0.067,0.035,0.067,0.035,0.067,0.067,0.035,0.067,0.035,0.067,0.079,0.072,0.062,0.079,0.072,0.062,0.089,0.107,0.107,0.087,0.087,0.089,0.067,0.067,0.067"

python evaluation.py \
    $data_root \
    --pytorch_config_path $pytorch_config_path \
    --snapshot_path $snapshot_path \
    --train_file $train_json \
    --test_file $test_json \
    --oks_sigma $oks_sigma_Quadruped