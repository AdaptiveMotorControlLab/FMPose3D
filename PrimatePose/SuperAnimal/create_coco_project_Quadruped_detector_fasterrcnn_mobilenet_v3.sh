data_path_prefix="/home/ti_wang/data"
data_root=${data_path_prefix}"/SuperAnimal/Quadruped80K"
project_root=$(dirname $(realpath $0))

debug=0
OOD_dataset_name=AP-10K

file_name=SAQ_detector_fasterrcnn_mobilenet_v3_allData_${OOD_dataset_name}_20250616

mode="train"

# train and test json files
# train_json="${data_root}/annotations/${mode}_IID_wo_${OOD_dataset_name}.json"
# test_json="${data_root}/annotations/test_IID_wo_${OOD_dataset_name}.json"

train_json="${data_root}/annotations/merged_train_and_test.json"
test_json="${data_root}/annotations/test_OOD_${OOD_dataset_name}.json"

model_arch="rtmpose_s"
detector_arch="fasterrcnn_mobilenet_v3_large_fpn"

if [ "$debug" -eq 1 ]; then
    out_name="${project_root}/experiments/Debugs/${file_name}"
    python make_config.py --debug $data_root $out_name $model_arch --train_file $train_json --multi_animal 
else
    out_name="${project_root}/experiments/${file_name}"
    python make_config.py $data_root $out_name --model_arch $model_arch --detector_arch $detector_arch --train_file $train_json --test_file $test_json --multi_animal
fi