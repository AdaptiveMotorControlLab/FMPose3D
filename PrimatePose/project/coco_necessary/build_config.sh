# proj_root=/home/ti_wang/Ti_workspace/PrimatePose/data/data/v8_1_coco
# train_json=/mnt/tiwang/primate_data/primate_train_1.1.json
proj_root=/mnt/tiwang/v8_coco
train_json=/home/ti_wang/Ti_workspace/PrimatePose/data/primate_test_1.2.json
model_arch="top_down_resnet_50"
out_name="/home/ti_wang/Ti_workspace/PrimatePose/project/debug5_nan"
python make_config.py $proj_root $out_name $model_arch --train_file $train_json --multi_animal