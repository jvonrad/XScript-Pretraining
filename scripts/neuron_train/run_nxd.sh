#!/usr/bin/env bash
# Launch nxd_test.py under torchrun.  usage: run_nxd.sh <world> [port]
#
# torchrun is FINE here (unlike our own trainer): it forces rank i -> core i, and
# when we take the whole box that is exactly the mapping we want.  The pinning
# problem only bit us when we needed a core SUBSET.
set -euo pipefail
WORLD=${1:-32}
PORT=${2:-41100}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; export PATH="$HOME/.local/bin:$PATH"
source ~/neuron_venv/bin/activate
export PJRT_DEVICE=NEURON
export NEURON_CC_FLAGS="--cache_dir=/mnt/scratch/xscript_nxd/neuron-cache --optlevel=1"
export NEURON_RT_ROOT_COMM_ID="127.0.0.1:$PORT"
export MASTER_ADDR=localhost MASTER_PORT=$((PORT + 100))
export PYTHONUNBUFFERED=1
export NEURON_RT_EXEC_TIMEOUT=600
mkdir -p /mnt/scratch/xscript_nxd/neuron-cache

exec torchrun --nproc_per_node="$WORLD" --master_port="$MASTER_PORT" \
  /home/ubuntu/xscript_prod/nxd_test.py
