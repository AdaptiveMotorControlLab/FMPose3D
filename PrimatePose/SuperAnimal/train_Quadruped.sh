data_path_prefix="/home/ti_wang/data"
data_root=${data_path_prefix}"/SuperAnimal/Quadruped80K"
project_root=$(dirname $(realpath $0))

debug=0
gpu_id="0"
OOD_dataset_name=AP-10K

file_name=SAQ_pose_rtmpose_s_OOD_${OOD_dataset_name}_20250514

train_pose=1
train_detector=0

# Memory optimization parameters
batch_size=32  # Reduced from default 32
dataloader_workers=4  # Reduced from default 16

mode="train"

# Generate run name based on configuration
run_name="${file_name}"

# train and test json files
train_json="${data_root}/annotations/${mode}_IID_wo_${OOD_dataset_name}.json"
test_json="${data_root}/annotations/test_IID_wo_${OOD_dataset_name}.json"

if [ "$debug" -eq 1 ]; then
    pytorch_config=${project_root}/experiments/Debugs/${file_name}/train/pytorch_config.yaml
    echo "Debug mode is ON, using debug pytorch_config: $pytorch_config"
    CUDA_VISIBLE_DEVICES=$gpu_id python3 train.py --debug \
        $([ "$train_detector" -eq 1 ] && echo "--train-detector") \
        $([ "$train_pose" -eq 1 ] && echo "--train-pose") \
        --project_root $data_root --pytorch_config $pytorch_config \
        --train_file $train_file --test_file $test_file \
        --device cuda --gpus 0 --run-name $run_name \
        --batch-size $batch_size --dataloader-workers $dataloader_workers
else
    pytorch_config=${project_root}/experiments/${file_name}/train/pytorch_config.yaml
    echo "Debug mode is OFF, using default pytorch_config: $pytorch_config"
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        $([ "$train_detector" -eq 1 ] && echo "--train-detector") \
        $([ "$train_pose" -eq 1 ] && echo "--train-pose") \
        --project_root $data_root --pytorch_config $pytorch_config \
        --train_file $train_json --test_file $test_json \
        --device cuda --gpus 0 --run-name $run_name \
        --batch-size $batch_size --dataloader-workers $dataloader_workers
fi