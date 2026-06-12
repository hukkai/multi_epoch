#!/usr/bin/env bash
export MASTER_PORT=$((12000 + RANDOM % 20000))

CONFIG=${1:-configs/adamw.yaml}
CONFIG_NAME=${CONFIG##*/}
TRAIN_SCRIPT=train.py

if [[ "${CONFIG_NAME}" == *muon* ]]; then
    TRAIN_SCRIPT=train_muon.py
fi

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node 8 \
    --master_port "${MASTER_PORT}" \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG}"
