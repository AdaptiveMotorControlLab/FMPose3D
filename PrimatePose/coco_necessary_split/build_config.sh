# server 7
data_path_prefix="/mnt/data/tiwang"
# server amgm0
# data_path_prefix="/media/data/ti/data"

proj_root=${data_path_prefix}+"/v8_coco"

dataset_file=chimpact
file=chimpact
mode="train"
debug=0
# for splitted datasets
train_json="/${data_path_prefix}/primate_data/splitted_${mode}_datasets/${dataset_file}_${mode}.json"
# for whole dataset
# train_json="/mnt/data/tiwang/primate_data/${file}.json"
model_arch="top_down_resnet_50"

if [ "$debug" -eq 1 ]; then
    out_name="/app/project/${file}_${mode}"
    python make_config.py --debug $proj_root $out_name $model_arch --train_file $train_json --multi_animal 
else
    out_name="/app/project/split/${file}_${mode}"
    python make_config.py $proj_root $out_name $model_arch --train_file $train_json --multi_animal
fi