# CTD training
CUDA_VISIBLE_DEVICES=0 python CTD_primate.py \
--config "./projects/riken_bandy-ti-2025-05-06/config.yaml" \
--bu_shuffle 1 \
--ctd_shuffle 2 \
--evaluate_CTD True

# CUDA_VISIBLE_DEVICES=0 python CTD_primate.py \
# --config "./projects/pfm_oms_small-dlc-2025-05-02/config.yaml" \
# --bu_shuffle 1 \
# --ctd_shuffle 2 \
# --evaluate_CTD True