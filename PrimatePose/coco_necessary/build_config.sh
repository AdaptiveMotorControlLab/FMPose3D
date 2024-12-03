# server 78
data_path_prefix="/mnt/data/tiwang"
data_root=${data_path_prefix}"/v8_coco"

project_root=$(dirname $(dirname $(realpath $0)))

# name=pfm_merged_checked
name=pfm_goodpose_merged
# file=${name}_detector_fasterrcnn
file=${name}_pose_hrnet
dataset_file=${name}
mode="train"
debug=0
version="v8"
# for whole datasets
# train_json="${data_path_prefix}/primate_data/${dataset_file}_${mode}_${version}.json"
# train_json="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/pfm_train_merged.json"
train_json="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/pfm_train_goodpose_merged.json"

# model_arch="top_down_resnet_50"
model_arch="top_down_hrnet_w32"

if [ "$debug" -eq 1 ]
then
    out_name="${project_root}/project/${file}_${mode}"
    python make_config.py --debug $data_root $out_name $model_arch --train_file $train_json --multi_animal 
else
    out_name="${project_root}/project/${file}_${mode}"
    python make_config.py $data_root $out_name $model_arch --train_file $train_json --multi_animal
fi