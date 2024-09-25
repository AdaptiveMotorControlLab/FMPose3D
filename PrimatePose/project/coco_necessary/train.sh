# for amgm0
# proj_root=/home/ti/projects/PrimatePose/ti_data/data/v8_coco
train_file=/mnt/tiwang/primate_data/primate_test_1.1.json
test_file=/mnt/tiwang/primate_data/primate_test_1.1.json
# for mml 
proj_root=/mnt/tiwang/v8_coco
# train_file=/mnt/tiwang/primate_data/splitted_val_datasets/ap10k_val.json
# test_file=/mnt/tiwang/primate_data/splitted_val_datasets/ap10k_val.json
# train_file=/mnt/tiwang/primate_data/splitted_val_datasets/anipose_val.json
# test_file=/mnt/tiwang/primate_data/splitted_val_datasets/anipose_val.json
# train_file=//home/ti/projects/PrimatePose/ti_data/data/pfm_test_8_items.json
# test_file=/home/ti/projects/PrimatePose/ti_data/data/pfm_test_8_items.json
pytorch_config=/home/ti_wang/Ti_workspace/PrimatePose/project/debug5_nan/train/pytorch_config.yaml
CUDA_VISIBLE_DEVICES=0 python train.py $proj_root $pytorch_config --train_file $train_file --test_file $test_file --device cuda --gpus 0