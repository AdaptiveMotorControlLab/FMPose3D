# server 7
data_path_prefix="/mnt/data/tiwang"
# server amgm0
# data_path_prefix="/media/data/ti/data"

proj_root="${data_path_prefix}/v8_coco"
debug=0
gpu_id="0"
file=oms
dataset_file=oms
mode="train"
# for splitted datasets
train_file=${data_path_prefix}/primate_data/splitted_train_datasets/${dataset_file}_train.json
test_file=${data_path_prefix}/primate_data/splitted_test_datasets/${dataset_file}_test.json

pytorch_config_path=/app/project/split/${file}_train/train/pytorch_config.yaml
snapshot_path=/app/project/split/${file}_train/train/snapshot-025.pt
detector_snapshot_path=/app/project/split/${file}_train/train/snapshot-detector-020.pt

python evaluation.py $proj_root --pytorch_config_path $pytorch_config_path --snapshot_path $snapshot_path --train_file $train_file --test_file $test_file  --detector_path $detector_snapshot_path