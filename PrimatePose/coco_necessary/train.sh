# for amgm0
# proj_root=/home/ti/projects/PrimatePose/ti_data/data/v8_coco
# train_file=/mnt/tiwang/primate_data/primate_test_1.1.json
# test_file=/mnt/tiwang/primate_data/primate_test_1.1.json
# server 7
data_path_prefix="/mnt/data/tiwang"
# server amgm0
# data_path_prefix="/media/data/ti/data"

proj_root="${data_path_prefix}/v8_coco"

debug=0
gpu_id="0"
file=pfm
mode="train"
version="v8"
# for splitted datasets
train_file="/${data_path_prefix}/primate_data/${file}_${mode}_${version}.json"
test_file="/${data_path_prefix}/primate_data/${file}_test_${version}.json"

if [ "$debug" -eq 1 ]; then
    pytorch_config=/app/project/Debug/${file}_${mode}/train/pytorch_config.yaml
    echo "Debug mode is ON, using debug pytorch_config: $pytorch_config"
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py --debug $proj_root $pytorch_config --train_file $train_file --test_file $test_file --device cuda --gpus 0
else
    pytorch_config=/app/project/${file}_${mode}/train/pytorch_config.yaml
    echo "Debug mode is OFF, using default pytorch_config: $pytorch_config"
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py $proj_root $pytorch_config --train_file $train_file --test_file $test_file --device cuda --gpus 0
fi