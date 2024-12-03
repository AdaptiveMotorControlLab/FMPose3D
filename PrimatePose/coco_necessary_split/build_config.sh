# server 78
data_path_prefix="/mnt/data/tiwang"
data_root=${data_path_prefix}"/v8_coco"

project_root=$(dirname $(dirname $(realpath $0)))

debug=0
name=anipose
# file=${name}_detector_fasterrcnn
file=${name}_pose_hrnet
# file=${name}_pose_hrnet

dataset_file=${name}
mode="train"
# for splitted datasets
train_json="${data_path_prefix}/primate_data/splitted_${mode}_datasets/${dataset_file}_${mode}.json"
# for whole dataset
# train_json="/mnt/data/tiwang/primate_data/${file}.json"
# model_arch="top_down_resnet_50"
model_arch="top_down_hrnet_w32"

if [ "$debug" -eq 1 ]; then
    out_name="${project_root}/project/${file}_${mode}"
    python make_config.py --debug $data_root $out_name $model_arch --train_file $train_json --multi_animal 
else
    out_name="${project_root}/project/split/${file}_${mode}"
    python make_config.py $data_root $out_name $model_arch --train_file $train_json --multi_animal
fi
