#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

python_bin="${PYTHON_BIN:-/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python}"
base="${ITOP_BASE:-/home/workspace/lrh/RESULTS/Tpami/ITOP/reviewer_factorial_3844f99/itop_final_n512/seed_42}"
output="${OUTPUT_ROOT:-/home/workspace/lrh/RESULTS/Tpami/ITOP/reviewer_controls_matched_20260816}"
data="${ITOP_DATA:-/home/workspace/lrh/DATA/Tpami/ITOP}"

run_seed() {
  local seed="$1"
  "${python_bin}" -m scripts.train_itop \
    --model shuffled_graph_student_t \
    --phase frozen_head \
    --backbone_checkpoint "${base}/deterministic/best_model.pt" \
    --feature_cache "${base}/frozen_features" \
    --data_dir "${data}" \
    --run_dir "${output}/seed_${seed}/shuffled_graph_student_t" \
    --student_t_dof 5 \
    --hidden_dim 64 \
    --lmax 2 \
    --num_layers 2 \
    --num_basis 8 \
    --num_points 512 \
    --num_neighbors 16 \
    --batch_size 16 \
    --num_epochs 60 \
    --patience 5 \
    --lr 0.0005 \
    --weight_decay 0.00001 \
    --num_workers 8 \
    --prefetch_factor 2 \
    --seed "${seed}" \
    --split_seed "${seed}" \
    --backbone_precision bf16 \
    --tp_backend e3nn \
    --cueq_method naive \
    --device cuda:0
}

run_seed 43
run_seed 44
