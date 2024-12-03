#!/bin/bash

# Define the list of dataset names
# DATASETS=("ak" "lote" "oms" "other_dataset1" "other_dataset2") # Add all your dataset names here
# Only include "ak" dataset for testing
DATASETS="ak"
# Set common variables
pfm_root=$(dirname $(dirname $(realpath $0)))
data_path_prefix="/mnt/data/tiwang"
data_root="${data_path_prefix}/v8_coco"
project_root="${pfm_root}"
gpu_id="0"
mode="train"
snapshot_path="/home/ti_wang/Ti_workspace/PrimatePose/project/pfm_pose_hrnet_train/train/snapshot-best-056.pt"
model_arch="top_down_hrnet_w32"
debug=0

# Loop through each dataset
for name in "${DATASETS[@]}"; do
    echo "Processing dataset: $name"

    # Define file and dataset paths
    file="${name}_pose_hrnet"
    dataset_file="${name}"
    train_file="${data_path_prefix}/primate_data/splitted_train_datasets/${dataset_file}_train.json"
    test_file="${data_path_prefix}/primate_data/splitted_test_datasets/${dataset_file}_test.json"
    pytorch_config_path="${pfm_root}/project/split/${file}_train/train/pytorch_config.yaml"
    output_folder="${project_root}/project/split/${file}_${mode}"

    # Check if the folder exists
    if [ ! -d "$output_folder" ]; then
        echo "Output folder $output_folder does not exist. Creating it..."
        if [ "$debug" -eq 1 ]; then
            python make_config.py --debug "$data_root" "$output_folder" "$model_arch" --train_file "$train_file" --multi_animal
        else
            python make_config.py "$data_root" "$output_folder" "$model_arch" --train_file "$train_file" --multi_animal
        fi
    else
        echo "Output folder $output_folder already exists."
    fi

    # Run the evaluation script for the current dataset
    echo "Starting evaluation for dataset: $name"
    python evaluation.py \
        --project_root "$data_root" \
        --pytorch_config_path "$pytorch_config_path" \
        --snapshot_path "$snapshot_path" \
        --train_file "$train_file" \
        --test_file "$test_file"
        
    echo "Completed evaluation for dataset: $name"
    echo "------------------------------------"
done