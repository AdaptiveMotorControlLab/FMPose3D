project_root=$(dirname $(dirname $(realpath $0)))
data_path_prefix="/home/ti_wang/data/tiwang"
data_root=${data_path_prefix}"/v8_coco"

debug=0
gpu_id="1"

dataset_name=mp
file_name=${dataset_name}_pose_rtmpose_s_V82_20250516

train_pose=1
mode="train"

# for splitted datasets V8.2
train_json="${data_path_prefix}/primate_data/PFM_V8.2/splitted_${mode}_datasets/${dataset_name}_${mode}.json"
test_json="${data_path_prefix}/primate_data/PFM_V8.2/splitted_test_datasets/${dataset_name}_test.json"

# Pose model training configuration
batch_size=64
dataloader_workers=4

# wandb configuration
wandb_project_name="primatepose"
wandb_run_name="${file_name}"
wandb_group="RTMPose_split_datasets_v82"
wandb_tag="train"

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
    pytorch_config=${project_root}/experiments/pfm_rtmpose/split/${file_name}/train/pytorch_config.yaml
    echo "Debug mode is OFF, using default pytorch_config: $pytorch_config"
    experiment_dir=${project_root}/experiments/pfm_rtmpose/split/${file_name}
    echo "Copying training scripts to ${experiment_dir}"
    cp "$0" "$experiment_dir/"
    cp "${project_root}/coco_necessary_split/train.py" "$experiment_dir/"
    cp "${project_root}/coco_necessary_split/build_coco_project_rtmpose_subdatasets.sh" "$experiment_dir/"
    cp "${project_root}/coco_necessary_split/make_config.py" "$experiment_dir/"

    CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
        $([ "$train_detector" -eq 1 ] && echo "--train-detector") \
        $([ "$train_pose" -eq 1 ] && echo "--train-pose") \
        --project_root $data_root --pytorch_config $pytorch_config \
        --train_file $train_json --test_file $test_json \
        --device cuda --gpus 0 \
        --batch-size $batch_size --dataloader-workers $dataloader_workers \
        --wandb-project-name $wandb_project_name \
        --wandb-run-name $wandb_run_name \
        --wandb-group $wandb_group \
        --wandb-tag $wandb_tag
fi