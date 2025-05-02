
# fine-tuning sapiens 0.3b on primate data 
CUDA_VISIBLE_DEVICES=0 python pose/tools/custom_train.py pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768_pfm_cluster.py