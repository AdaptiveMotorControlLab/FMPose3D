#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

layers=5
gpu_id=1
eval_sample_steps=3
batch_size=1024
saved_model_path="${SCRIPT_DIR}/pretrained/fmpose3d_h36m/FMpose3D_pretrained_weights.pth"

num_hypothesis_list=1
topk=6
subjects_test=TS1,TS2,TS3,TS4,TS5,TS6

folder_name=s_${eval_sample_steps}_S${subjects_test}_h${num_hypothesis_list}_$(date +%Y%m%d_%H%M%S)

python3 "${SCRIPT_DIR}/infer_3dhp.py" \
    --gpu "${gpu_id}" \
    --batch-size "${batch_size}" \
    --frames 1 \
    --layers "${layers}" \
    --channel 512 \
    --d-hid 1024 \
    --token-dim 256 \
    --eval-sample-steps "${eval_sample_steps}" \
    --dataset-path "${SCRIPT_DIR}/dataset/data_test_3dhp.npz" \
    --saved-model-path "${saved_model_path}" \
    --num-hypothesis-list "${num_hypothesis_list}" \
    --topk "${topk}" \
    --folder-name "${folder_name}" \
    --test-augmentation True \
    --test-augmentation-flip-hypothesis True \
    --subjects-test "${subjects_test}" \
    --results-dir "${SCRIPT_DIR}/results" \
    "$@"
