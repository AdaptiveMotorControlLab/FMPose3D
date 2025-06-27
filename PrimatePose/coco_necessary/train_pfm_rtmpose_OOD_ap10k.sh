data_path_prefix="/home/ti_wang/data/tiwang"
data_root=${data_path_prefix}"/v8_coco"
project_root=$(dirname $(dirname $(realpath $0)))

debug=0
gpu_id="1"
OOD_dataset_name=ap10k

file_name=pfm_pose_rtmpose_s_OOD_${OOD_dataset_name}_V83_20250626

train_pose=1
train_detector=0

# Memory optimization parameters
batch_size=64  # Reduced from default 32
dataloader_workers=8  # Reduced from default 16

mode="train"

# Generate run name based on configuration
run_name="${file_name}"

# PFM V83 OOD ap10k; 
train_json="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/pfm_train_OOD_ap10k.json"
test_json="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_test_datasets/ap10k_test.json"

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
    echo "Copying train_pfm_rtmpose_OOD_ap10k.sh and train.py to ${experiment_dir}"
    cp "$0" "$experiment_dir/"
    cp "${project_root}/coco_necessary/train.py" "$experiment_dir/"

    # Copy train.py to the experiment folder
    echo "Copying create_coco_project_pfm_rtmpose_OOD_ap10k.sh and make_config.py to ${experiment_dir}"
    cp "${project_root}/coco_necessary/create_coco_project_pfm_rtmpose_OOD_ap10k.sh" "$experiment_dir/"
    cp "${project_root}/coco_necessary/make_config.py" "$experiment_dir/"

    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        $([ "$train_detector" -eq 1 ] && echo "--train-detector") \
        $([ "$train_pose" -eq 1 ] && echo "--train-pose") \
        --project_root $data_root --pytorch_config $pytorch_config \
        --train_file $train_json --test_file $test_json \
        --device cuda --gpus 0 --run-name $run_name \
        --batch-size $batch_size --dataloader-workers $dataloader_workers
fi