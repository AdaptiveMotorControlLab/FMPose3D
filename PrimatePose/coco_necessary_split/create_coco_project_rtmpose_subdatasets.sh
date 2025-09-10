data_path_prefix="/home/ti_wang/data/tiwang"
data_root=${data_path_prefix}"/v8_coco"
project_root=$(dirname $(dirname $(realpath $0)))

debug=0

dataset_name=deepwild
file_name=${dataset_name}_pose_rtmpose_s_V83_20250701

mode="train"
# for splitted datasets V8.3
train_json="${data_path_prefix}/primate_data/PFM_V8.3/splitted_${mode}_datasets/${dataset_name}_${mode}.json"
test_json="${data_path_prefix}/primate_data/PFM_V8.3/splitted_test_datasets/${dataset_name}_test.json"

model_arch="rtmpose_s"
detector_arch="fasterrcnn_mobilenet_v3_large_fpn"

if [ "$debug" -eq 1 ]; then
    out_name="${project_root}/experiments/Debug/pfm_rtmpose/split/${file_name}"
    python make_config.py --debug $data_root $out_name $model_arch --train_file $train_json --multi_animal 
else
    out_name="${project_root}/experiments/pfm_rtmpose/split/${file_name}"
    python make_config.py $data_root $out_name --model_arch $model_arch --detector_arch $detector_arch --train_file $train_json --test_file $test_json --multi_animal
fi