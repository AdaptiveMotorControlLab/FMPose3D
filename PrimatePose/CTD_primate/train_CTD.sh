python CTD_primate.py \
--config "./pfm_oms_small-dlc-2025-05-02/config.yaml" \
--bu_shuffle 1 \
--ctd_shuffle 10 \
--batch_size 32 \
--ctd_net_type "ctd_prenet_rtmpose_m" \
--create_BU_dataset True \
--create_CTD_dataset True \
--train_CTD True 