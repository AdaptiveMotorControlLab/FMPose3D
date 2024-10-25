# server 7
data_path_prefix="/mnt/data/tiwang"
# server amgm0
# data_path_prefix="/media/data/ti/data"

proj_root=${data_path_prefix}+"/v8_coco"

dataset_file=pfm
file=pfm
mode="train"
debug=0
version="v8"
# for whole datasets
train_json="/${data_path_prefix}/primate_data/${dataset_file}_${mode}_${version}.json"
model_arch="top_down_resnet_50"

# out_name="/app/Ti_workspace/PrimatePose/project/${file}"

if [ "$debug" -eq 1 ]; then
    out_name="/app/project/${file}_${mode}"
    python make_config.py --debug $proj_root $out_name $model_arch --train_file $train_json --multi_animal 
else
    out_name="/app/project/${file}_${mode}"
    python make_config.py $proj_root $out_name $model_arch --train_file $train_json --multi_animal
fi