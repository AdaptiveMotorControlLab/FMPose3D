# fine-tuning sapiens 0.3b on coco data
CUDA_VISIBLE_DEVICES=1 python pose/tools/custom_train.py pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768_human.py

# fine-tuning sapiens 0.3b on primate data
# CUDA_VISIBLE_DEVICES=1 python pose/tools/custom_train.py pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768_pfm.py

#  vis tensorboard

# tensorboard --logdir work_dirs/sapiens_0.3b-210e_coco-1024x768_pfm/20250426_181234/vis_data/
