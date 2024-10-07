# for amgm0
# proj_root=/home/ti/projects/PrimatePose/ti_data/data/v8_coco
# train_file=/mnt/tiwang/primate_data/primate_test_1.1.json
# test_file=/mnt/tiwang/primate_data/primate_test_1.1.json
# for mml
debug=0
gpu_id=1
# file=ap10k_val
file=pfm_val_v8
proj_root=/mnt/tiwang/v8_coco
# for splitted datasets
# train_file=/mnt/tiwang/primate_data/splitted_val_datasets/${file}.json
# test_file=/mnt/tiwang/primate_data/splitted_val_datasets/${file}.json

# train_file=/mnt/tiwang/primate_data/splitted_val_datasets/${file}_sampled_500.json
# test_file=/mnt/tiwang/primate_data/splitted_val_datasets/${file}_sampled_500.json

train_file=/mnt/tiwang/primate_data/${file}.json
test_file=/mnt/tiwang/primate_data/${file}.json

if [ "$debug" -eq 1 ]; then
    pytorch_config=/home/ti_wang/Ti_workspace/PrimatePose/project/Debug/${file}/train/pytorch_config.yaml
    echo "Debug mode is ON, using debug pytorch_config: $pytorch_config"
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py --debug $proj_root $pytorch_config --train_file $train_file --test_file $test_file --device cuda --gpus 0
else
    pytorch_config=/home/ti_wang/Ti_workspace/PrimatePose/project/${file}/train/pytorch_config.yaml
    echo "Debug mode is OFF, using default pytorch_config: $pytorch_config"
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py --debug $proj_root $pytorch_config --train_file $train_file --test_file $test_file --device cuda --gpus 0
fi