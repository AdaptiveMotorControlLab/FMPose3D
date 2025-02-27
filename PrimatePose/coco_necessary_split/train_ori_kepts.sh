# server 7
project_root=$(dirname $(dirname $(realpath $0)))
data_path_prefix="/mnt/data/tiwang"
# server 8
# data_ph_prefix="/mnt/ti_wang"
# server amgm0
# data_path_prefix="/media/data/ti/data"

data_root=${data_path_prefix}"/v8_coco"

debug=0
gpu_id="1"
name=riken
# file=${name}_detector_fasterrcnn
# file=${name}_pose_reset
# file=${name}_ori_kepts_bbox_pose_hrnet
file=${name}_ori_detector_pose_hrnet
train_pose=1
train_detector=1

dataset_name=${name}
mode="train"
# Generate run name based on configuration
run_name="${file}"
# for splitted datasets
train_json="${data_path_prefix}/primate_data/data_v8.1/splitted_${mode}_datasets/${dataset_name}_${mode}.json"
test_json="${data_path_prefix}/primate_data/data_v8.1/splitted_test_datasets/${dataset_name}_test.json"

# train_file=/mnt/tiwang/primate_data/splitted_val_datasets/${file}_sampled_500.json
# test_file=/mnt/tiwang/primate_data/splitted_val_datasets/${file}_sampled_500.json
# for the pfm dataset
# train_file=${data_path_prefix}/primate_data/${file}.json
# test_file=${data_path_prefix}/primate_data/${file}.json

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
    pytorch_config=${project_root}/project/split/${file}_${mode}/train/pytorch_config.yaml
    echo "Debug mode is OFF, using default pytorch_config: $pytorch_config"
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        $([ "$train_detector" -eq 1 ] && echo "--train-detector") \
        $([ "$train_pose" -eq 1 ] && echo "--train-pose") \
        --project_root $data_root --pytorch_config $pytorch_config \
        --train_file $train_json --test_file $test_json \
        --device cuda --gpus 0 --run-name $run_name
fi