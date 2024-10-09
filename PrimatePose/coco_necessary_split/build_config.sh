proj_root=/mnt/tiwang/v8_coco
# file=ap10k_val
file=pfm_val_v8

debug=0
# for splitted datasets
train_json="/mnt/tiwang/primate_data/splitted_val_datasets/${file}.json"
# for whole dataset
# train_json="/mnt/data/tiwang/primate_data/${file}.json"

model_arch="top_down_resnet_50"
out_name="/app/Ti_workspace/PrimatePose/project/${file}_2"
if [ "$debug" -eq 1 ]; then
    python make_config.py --debug $proj_root $out_name $model_arch --train_file $train_json --multi_animal 
else
    python make_config.py $proj_root $out_name $model_arch --train_file $train_json --multi_animal
fi