#!/usr/bin/env bash
# torchrun / environment vars
NNODES=1
NPROC_PER_NODE=8
MASTER_ADDR=127.0.0.1
MASTER_PORT=12345
torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  src/train.py --config configs/dev.json
