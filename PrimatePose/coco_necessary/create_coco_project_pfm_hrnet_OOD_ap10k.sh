data_path_prefix="/home/ti_wang/data/tiwang"
data_root=${data_path_prefix}"/v8_coco"

project_root=$(dirname $(dirname $(realpath $0)))

OOD_dataset_name=ap10k
file_name=pfm_pose_hrnetw32_OOD_${OOD_dataset_name}_V83_20250626

mode="train"
debug=0

# PFM V83 OOD ap10k;
train_json="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/pfm_train_OOD_ap10k.json"
test_json="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_test_datasets/ap10k_test.json"

model_arch="top_down_hrnet_w32"
detector_arch="fasterrcnn_mobilenet_v3_large_fpn"

if [ "$debug" -eq 1 ]; then
    out_name="${project_root}/experiments/Debugs/${file_name}"
    python make_config.py --debug $data_root $out_name $model_arch --train_file $train_json --multi_animal 
else
    out_name="${project_root}/experiments/${file_name}"
    python make_config.py $data_root $out_name --model_arch $model_arch --detector_arch $detector_arch --train_file $train_json --test_file $test_json --multi_animal
fi