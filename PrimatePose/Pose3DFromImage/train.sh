
# wandb configuration
wandb_project_name="Pose3D"
wandb_group="oms_pose3d_from_image"
wandb_run_name="oms_pose3d_from_image_20250820_vit_s16_dino"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name "oms" \
    --train_json /home/ti_wang/Ti_workspace/PrimatePose/Pose3D/data/oms_train.json \
    --val_json /home/ti_wang/Ti_workspace/PrimatePose/Pose3D/data/oms_test.json \
    --epochs 200 \
    --batch_size 128 \
    --backbone vit_s16_dino \
    --lr 1e-4 \
    --seed 1 \
    --wandb \
    --wandb_project_name $wandb_project_name \
    --wandb_run_name $wandb_run_name \
    --wandb_group $wandb_group