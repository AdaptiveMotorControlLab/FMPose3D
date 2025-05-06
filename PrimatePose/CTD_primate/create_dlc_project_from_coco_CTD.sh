
species="riken"
task_name="riken_bandy"
image_folder_coco_project="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/Marmosets-Banty-2022-07-01_coco"

#task_name=ome/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/splitted_train_datasets/${species}_train.json"
# test_json_path="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/splitted_test_datasets/${species}_test.json"

train_json_path="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/Marmosets-Banty-2022-07-01_coco/annotations/train.json"
test_json_path="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/Marmosets-Banty-2022-07-01_coco/annotations/test.json"

python create_dlc_project_from_coco_CTD_symlink.py \
--output_dir "/home/ti_wang/Ti_workspace/PrimatePose/CTD_primate" \
--task $task_name \
--experimenter "ti" \
--coco_project $image_folder_coco_project \
--train_file $train_json_path \
--test_file $test_json_path
