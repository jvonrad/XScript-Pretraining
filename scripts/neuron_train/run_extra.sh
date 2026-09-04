#!/bin/bash
# Launch ONE corroboration-tokenizer training run on a pinned core block via
# xmp.spawn (NOT torchrun: torch_neuronx overwrites NEURON_RT_VISIBLE_CORES
# under torchrun, see NEURON.md 9).
# args: $1=MODEL (run name)  $2=CORES (comma-separated PHYSICAL core ids)  $3=comm_port
export XSCRIPT_SCRATCH=${XSCRIPT_SCRATCH:-/mnt/scratch/xscript}
export WORK=${WORK:-/mnt/scratch/xscript_train}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; export PATH="$HOME/.local/bin:/opt/aws/neuron/bin:$PATH"
source ~/neuron_venv/bin/activate 2>/dev/null
export PJRT_DEVICE=NEURON
mkdir -p "$WORK/neuron-cache"
export NEURON_CC_FLAGS="--cache_dir=$WORK/neuron-cache --optlevel=1"
export NEURON_RT_VISIBLE_CORES="$2"
export NEURONCORE_NUM_DEVICES=$(awk -F',' '{print NF}' <<<"$2")
export NEURON_RT_ROOT_COMM_ID="127.0.0.1:$3"
# Distinct rendezvous port per concurrent job (default 12355 collides -> hang).
export MASTER_ADDR=localhost
export MASTER_PORT=$(( $3 + 100 ))
export PROD_MODEL="$1"
export PYTHONUNBUFFERED=1
# optional wandb creds (kept out of the repo); trainer degrades gracefully if absent.
[ -f /home/ubuntu/xscript_prod/wandb_env.sh ] && source /home/ubuntu/xscript_prod/wandb_env.sh
cd /home/ubuntu/XScript-Pretraining
exec python3 scripts/neuron_train/prod_extra.py
