# server 7
data_path_prefix="/mnt/data/tiwang"
# server amgm0
# data_path_prefix="/media/data/ti/data"

proj_root=${data_path_prefix}"/v8_coco"

debug=0
gpu_id="1"
dataset_file=oap
file=oap_hrnet
# file=chimpact_hrnet_onlyPose
mode="train"
# for splitted datasets
train_file=${data_path_prefix}/primate_data/splitted_train_datasets/${dataset_file}_train.json
test_file=${data_path_prefix}/primate_data/splitted_test_datasets/${dataset_file}_test.json
# train_file=/mnt/tiwang/primate_data/splitted_val_datasets/${file}_sampled_500.json
# test_file=/mnt/tiwang/primate_data/splitted_val_datasets/${file}_sampled_500.json

# train_file=/mnt/data/tiwang/primate_data/${file}.json
# test_file=/mnt/data/tiwang/primate_data/${file}.json

if [ "$debug" -eq 1 ]; then
    pytorch_config=/app/project/Debug/${file}_${mode}/train/pytorch_config.yaml
    echo "Debug mode is ON, using debug pytorch_config: $pytorch_config"
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py --debug --project_root $proj_root --pytorch_config $pytorch_config --train_file $train_file --test_file $test_file --device cuda --gpus 0
else
    pytorch_config=/app/project/split/${file}_${mode}/train/pytorch_config.yaml
    echo "Debug mode is OFF, using default pytorch_config: $pytorch_config"
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py --project_root $proj_root --pytorch_config $pytorch_config --train_file $train_file --test_file $test_file --device cuda --gpus 0
fi