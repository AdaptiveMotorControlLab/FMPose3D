project_root=$(dirname $(dirname $(realpath $0)))
data_path_prefix="/mnt/data/tiwang"
data_root=${data_path_prefix}"/v8_coco"

debug=0
gpu_id="1"
file=oms
dataset_file=oms
mode="train"
# for splitted datasets
train_file=${data_path_prefix}/primate_data/splitted_train_datasets/${dataset_file}_train.json

test_file=${data_path_prefix}/primate_data/splitted_test_datasets/${dataset_file}_test.json
# test_file=${data_path_prefix}/primate_data/splitted_test_datasets/ak_test.json

pytorch_config_path=${project_root}/project/split/${file}_train/train/pytorch_config.yaml
snapshot_path=${project_root}/project/split/${file}_train/train/snapshot-100.pt
detector_snapshot_path=${project_root}/project/split/${file}_train/train/snapshot-detector-020.pt

python3 evaluation.py $data_root --pytorch_config_path $pytorch_config_path --snapshot_path $snapshot_path --train_file $train_file --test_file $test_file  --detector_path $detector_snapshot_path
