data_path_prefix="/home/ti_wang/data"
data_root=${data_path_prefix}"/SuperAnimal/Quadruped80K"
project_root=$(dirname $(dirname $(realpath $0)))

debug=0
dataset_name=ap10k

file_name=${dataset_name}_pose_rtmpose_s_20250514

mode="train"

# train and test json files
train_json="${data_root}/annotations/${mode}_OOD_${dataset_name}.json"
test_json="${data_root}/annotations/${mode}_IID_wo_${dataset_name}.json"

# model_arch="top_down_resnet_50"
# model_arch="top_down_hrnet_w32"
model_arch="rtmpose_s"

if [ "$debug" -eq 1 ]; then
    out_name="${project_root}/project/Debug/${file_name}"
    python make_config.py --debug $data_root $out_name $model_arch --train_file $train_json --multi_animal 
else
    out_name="${project_root}/SuperAnimal/experiments/${file_name}"
    python make_config.py $data_root $out_name $model_arch --train_file $train_json --test_file $test_json --multi_animal
fi