CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset_name "oms" \
    --train_json /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_train_datasets/oms_train.json \
    --val_json /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_test_datasets/oms_test.json \
    --epochs 200 \
    --batch_size 128 \
    --lr 1e-4 \
    --seed 1