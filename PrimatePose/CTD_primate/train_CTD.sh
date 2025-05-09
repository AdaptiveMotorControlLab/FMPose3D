# BU training
CUDA_VISIBLE_DEVICES=1 python CTD_primate.py \
--config "./projects/pfm-ti-2025-05-03/config.yaml" \
--bu_shuffle 1 \
--batch_size 16 \
--bu_net_type "resnet_50" \
--train_BU True \
--evaluate_BU True
# --create_BU_dataset True
# --ctd_net_type "ctd_prenet_rtmpose_m" \
# --train_CTD True \
# --create_CTD_dataset True \
# --evaluate_CTD True

# CTD training
# CUDA_VISIBLE_DEVICES=0 python CTD_primate.py \
# --config "./projects/riken_bandy-ti-2025-05-06/config.yaml" \
# --bu_shuffle 1 \
# --ctd_shuffle 3 \
# --batch_size 64 \
# --ctd_net_type "ctd_prenet_rtmpose_m" \
# --train_CTD True \
# --evaluate_CTD True \
# --create_CTD_dataset True 
# --create_BU_dataset True \
# --train_BU True \
# --evaluate_BU True