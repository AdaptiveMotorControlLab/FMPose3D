# server 7 8
# data_path_prefix="/mnt/data/tiwang"
data_path_prefix="/home/ti_wang/data/tiwang"
data_root=${data_path_prefix}"/v8_coco"

project_root=$(dirname $(dirname $(realpath $0)))

# name=pfm_merged_checked
name=pfm
# file=${name}_detector_fasterrcnn
file=${name}_pose_V82_wo_riken_chimpact_20250304
dataset_file=${name}
mode="train"
debug=0

# version="v8"
# for whole datasets
# train_json="${data_path_prefix}/primate_data/${dataset_file}_${mode}_${version}.json"
# train_json="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/pfm_train_merged.json"

# PFM V82 only pose
# train_json="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/pfm_train_pose_V82_no_wrong_bbox.json"


# PFM V82 wo riken and chimpact; only pose
train_json="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/pfm_train_wo_riken_chimpact_V82.json"

# model_arch="top_down_resnet_50"
model_arch="top_down_hrnet_w32"

if [ "$debug" -eq 1 ]
then
    out_name="${project_root}/project/${file}"
    python make_config.py --debug $data_root $out_name $model_arch --train_file $train_json --multi_animal 
else
    out_name="${project_root}/project/${file}"
    python make_config.py $data_root $out_name $model_arch --train_file $train_json --multi_animal
fi