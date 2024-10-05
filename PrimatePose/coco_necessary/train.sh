# for amgm0
# proj_root=/home/ti/projects/PrimatePose/ti_data/data/v8_coco
# train_file=/mnt/tiwang/primate_data/primate_test_1.1.json
# test_file=/mnt/tiwang/primate_data/primate_test_1.1.json
# for mml
file=pfm_val_v8
proj_root=/mnt/tiwang/v8_coco
# train_file=/mnt/tiwang/primate_data/splitted_val_datasets/${file}.json
# test_file=/mnt/tiwang/primate_data/splitted_val_datasets/${file}.json

# train_file=/mnt/tiwang/primate_data/splitted_val_datasets/${file}_sampled_500.json
# test_file=/mnt/tiwang/primate_data/splitted_val_datasets/${file}_sampled_500.json

train_file=/mnt/tiwang/primate_data/${file}.json
test_file=/mnt/tiwang/primate_data/${file}.json
# train_file=//home/ti/projects/PrimatePose/ti_data/data/pfm_test_8_items.json
# test_file=/home/ti/projects/PrimatePose/ti_data/data/pfm_test_8_items.json
pytorch_config=/home/ti_wang/Ti_workspace/PrimatePose/project/debug_${file}/train/pytorch_config.yaml
# pytorch_config=/home/ti_wang/Ti_workspace/PrimatePose/project/debug5_nan/train/pytorch_config.yaml

CUDA_VISIBLE_DEVICES=1 python train.py $proj_root $pytorch_config --train_file $train_file --test_file $test_file --device cuda --gpus 0