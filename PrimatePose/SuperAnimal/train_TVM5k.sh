data_path_prefix="/home/ti_wang/data"
data_root=${data_path_prefix}"/SuperAnimal/TopViewMouse5K_20240913"
project_root=$(dirname $(realpath $0))

debug=0
gpu_id="1"
OOD_dataset_name=openfield-Pranav-2018-08-20

file_name=SA-TVM_pose_rtmpose_s_OOD_${OOD_dataset_name}_20250515


train_pose=1
train_detector=0

# Memory optimization parameters
batch_size=32  # Reduced from default 32
dataloader_workers=16  # Reduced from default 16

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
    
    # Copy this script to the experiment folder
    experiment_dir=${project_root}/experiments/${file_name}
    echo "Copying train_TVM5k.sh and train.py to ${experiment_dir}"
    cp "$0" "$experiment_dir/"
    cp "${project_root}/train.py" "$experiment_dir/"

    # Copy train.py to the experiment folder
    echo "Copying create_coco_project_Quadruped.sh and make_config.py to ${experiment_dir}"
    cp "${project_root}/create_coco_project_TVM5k.sh" "$experiment_dir/"
    cp "${project_root}/make_config.py" "$experiment_dir/"

    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        $([ "$train_detector" -eq 1 ] && echo "--train-detector") \
        $([ "$train_pose" -eq 1 ] && echo "--train-pose") \
        --project_root $data_root --pytorch_config $pytorch_config \
        --train_file $train_json --test_file $test_json \
        --device cuda --gpus 0 --run-name $run_name \
        --batch-size $batch_size --dataloader-workers $dataloader_workers
fi