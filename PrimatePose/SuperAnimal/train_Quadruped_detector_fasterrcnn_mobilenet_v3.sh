data_path_prefix="/home/ti_wang/data"
data_root=${data_path_prefix}"/SuperAnimal/Quadruped80K"
project_root=$(dirname $(realpath $0))

debug=0
gpu_id="1"
OOD_dataset_name=AP-10K

file_name=SAQ_detector_fasterrcnn_mobilenet_v3_OOD_${OOD_dataset_name}_20250613

train_pose=0
train_detector=1

# Memory optimization parameters
batch_size=32  # Reduced from default 32
dataloader_workers=4  # Reduced from default 16

detector_batch_size=32
detector_dataloader_workers=4

mode="train"

# Generate run name based on configuration
run_name="${file_name}"

# train and test json files
train_json="${data_root}/annotations/merged_train_and_test_wo_AP-10K.json"
test_json="${data_root}/annotations/test_OOD_${OOD_dataset_name}.json"

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
    echo "Copying train_Quadruped.sh and train.py to ${experiment_dir}"
    cp "$0" "$experiment_dir/"
    cp "${project_root}/train.py" "$experiment_dir/"

    # Copy train.py to the experiment folder
    echo "Copying create_coco_project_Quadruped.sh and make_config.py to ${experiment_dir}"
    cp "${project_root}/create_coco_project_Quadruped.sh" "$experiment_dir/"
    cp "${project_root}/make_config.py" "$experiment_dir/"

    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        $([ "$train_detector" -eq 1 ] && echo "--train-detector") \
        $([ "$train_pose" -eq 1 ] && echo "--train-pose") \
        --project_root $data_root --pytorch_config $pytorch_config \
        --train_file $train_json --test_file $test_json \
        --device cuda --gpus 0 --run-name $run_name \
        --batch-size $batch_size --dataloader-workers $dataloader_workers
fi