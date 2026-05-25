#!/usr/bin/env bash
export MASTER_PORT=$((12000 + RANDOM % 20000))

ORTH_TYPE=${1:-all}
NUM_SUBMATRICES=${2:-64}


OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node 8 \
    --master_port "${MASTER_PORT}" \
    train.py \
    --data-dir ./data/C4-50B/ \
    --num-layers 24 \
    --hidden-size 1280 \
    --num-heads 16 \
    --batch-size 8 \
    --global-batch-size 512 \
    --seq-length 2048 \
    --lr 1.2e-3 \
    --min-lr 1.2e-5 \
    --so-lr 1.0 \
    --num-steps 50_000 \
    --orthogonal-type ${ORTH_TYPE} \
    --num-submatrices ${NUM_SUBMATRICES}
