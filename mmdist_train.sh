#!/bin/bash
export NCCL_SOCKET_IFNAME=ib0
NUM_PROC=$1
NUM_NODE=$2
NODE_RANK=$3
IP=$4
PORT=$5
shift
shift
shift
shift
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --nnodes=$NUM_NODE --node_rank=$NODE_RANK --master_addr="$IP" --master_port=$PORT main.py "$@"