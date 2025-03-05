project_root=$(dirname $(dirname $(realpath $0)))
# server 8
# data_path_prefix="/mnt/data/tiwang"
data_path_prefix="/home/ti_wang/data/tiwang"
data_root=${data_path_prefix}"/v8_coco"
# server amgm0
# data_path_prefix="/media/data/ti/data"
debug=0
gpu_id="0"
name=riken
# file=${name}_detector_fasterrcnn
# train_detector=1
# file=${name}_pose_reset
# file=${name}_pose_hrnet_20250225
file=${name}_hrnet_only_pose_no_single_20250227
train_pose=1

dataset_name=${name}
mode="train"
# Generate run name based on configuration
run_name="${file}"

# for splitted datasets V8.2
train_json="${data_path_prefix}/primate_data/PFM_V8.2/samples/${dataset_name}_${mode}_pose_no_single.json"
test_json="${data_path_prefix}/primate_data/PFM_V8.2/samples/${dataset_name}_test_pose_no_single.json"

# train_json="${data_path_prefix}/primate_data/PFM_V8.2/splitted_${mode}_datasets/${dataset_name}_${mode}.json"
# test_json="${data_path_prefix}/primate_data/PFM_V8.2/splitted_test_datasets/${dataset_name}_test.json"

# for splitted datasets V8.0
# train_json="${data_path_prefix}/primate_data/splitted_${mode}_datasets/${dataset_name}_${mode}.json"
# test_json="${data_path_prefix}/primate_data/splitted_test_datasets/${dataset_name}_test.json"

# for dataset without testset
# train_json="${data_path_prefix}/primate_data/splitted_${mode}_datasets/${dataset_name}_train_${mode}.json"
# test_json="${data_path_prefix}/primate_data/splitted_test_datasets/${dataset_name}_train_test.json"

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