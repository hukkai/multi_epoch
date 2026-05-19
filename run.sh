#!/usr/bin/env bash
export MASTER_PORT=$((12000 + RANDOM % 20000))

ORTH_TYPE=${1:-all}
WEIGHT_DECAY=${2:-0.1}
SUB_MATRIX=${3:-96}
SO_LR=${4:-1.0}


OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node 8 \
    --master_port "${MASTER_PORT}" \
    train.py \
    --data-dir ./data/C4-50B/ \
    --num-layers 18 \
    --hidden-size 1536 \
    --num-heads 24 \
    --batch-size 16 \
    --global-batch-size 512 \
    --seq-length 2048 \
    --lr 1.2e-3 \
    --min-lr 1.2e-5 \
    --so-lr ${SO_LR} \
    --num-steps 50_000 \
    --orthogonal-type "${ORTH_TYPE}" \
    --sub-matrix ${SUB_MATRIX} \
    --weight-decay ${WEIGHT_DECAY}
