#!/usr/bin/env bash
export MASTER_PORT=$((12000 + RANDOM % 20000))

CONFIG=${1:-configs/ablations/dense_adamw.yaml}

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node 8 \
    --master_port "${MASTER_PORT}" \
    -m ortho_llm.scripts.train \
    --config "${CONFIG}"
