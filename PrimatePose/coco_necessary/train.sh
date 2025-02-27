# server 7 8 
project_root=$(dirname $(dirname $(realpath $0)))
data_path_prefix="/mnt/data/tiwang"

data_root=${data_path_prefix}"/v8_coco"

debug=0
gpu_id="1"
name=pfm_goodpose_merged
# name=pfm_merged_checked
# file=${name}_detector_fasterrcnn
# train_detector=1
file=${name}_pose_hrnet_V2.1
train_pose=1
dataset_file=${name}

mode="train"
version="v8"
# Generate run name based on configuration
run_name="${file}_lr5e-5_B32"

# for splitted datasets
train_file="${data_path_prefix}/primate_data/${dataset_file}_${mode}_${version}.json"
test_file="${data_path_prefix}/primate_data/${dataset_file}_test_${version}.json"

train_json="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/pfm_train_goodpose_merged.json"
test_json="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/pfm_test_goodpose_merged.json"

# snapshot_path="/home/ti_wang/Ti_workspace/PrimatePose/project/pfm_pose_hrnet_train/train/snapshot-018.pt"
        # --snapshot_path $snapshot_path

if [ "$debug" -eq 1 ]; then
    pytorch_config=${project_root}/project/Debug/${file}_${mode}/train/pytorch_config.yaml
    echo "Debug mode is ON, using debug pytorch_config: $pytorch_config"
    CUDA_VISIBLE_DEVICES=$gpu_id python3 train.py --debug \
        $([ "$train_detector" -eq 1 ] && echo "--train-detector") \
        $([ "$train_pose" -eq 1 ] && echo "--train-pose") \
        --project_root $data_root --pytorch_config $pytorch_config \
        --train_file $train_file --test_file $test_file \
        --device cuda --gpus 0 --run-name $run_name
else
    pytorch_config=${project_root}/project/${file}_${mode}/train/pytorch_config.yaml
    echo "Debug mode is OFF, using default pytorch_config: $pytorch_config"
    CUDA_VISIBLE_DEVICES=$gpu_id python3 train.py \
        $([ "$train_detector" -eq 1 ] && echo "--train-detector") \
        $([ "$train_pose" -eq 1 ] && echo "--train-pose") \
        --project_root $data_root --pytorch_config $pytorch_config \
        --train_file $train_json --test_file $test_json \
        --device cuda --gpus 0 --run-name $run_name
fi