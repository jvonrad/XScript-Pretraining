#!/bin/bash
# Launch ONE production training model on a pinned core block via xmp.spawn.
# args: $1=MODEL  $2=CORES(comma phys ids)  $3=comm_port  $4=WARM_PATH(optional)
export XSCRIPT_SCRATCH=/mnt/scratch/xscript
export WORK=/mnt/scratch/xscript_train
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; export PATH="$HOME/.local/bin:$PATH"
source ~/neuron_venv/bin/activate 2>/dev/null
export PJRT_DEVICE=NEURON
export NEURON_CC_FLAGS="--cache_dir=$WORK/neuron-cache --optlevel=1"
export NEURON_RT_VISIBLE_CORES="$2"
export NEURONCORE_NUM_DEVICES=$(awk -F',' '{print NF}' <<<"$2")
export NEURON_RT_ROOT_COMM_ID="127.0.0.1:$3"
# Distinct rendezvous port for dist.init_process_group("xla", init_method="xla://").
# Without this, concurrent models collide on the default port (12355) and the
# 2nd job HANGS at process-group init -- the coexistence hang we hit.
export MASTER_ADDR=localhost
export MASTER_PORT=$(( $3 + 100 ))
export PROD_MODEL="$1"
export PROD_WARM="${4:-}"
export PROD_TARGET_B="${PROD_TARGET_B:-30}"   # stop-at token target (B), from env
export PYTHONUNBUFFERED=1   # flush the trainer's prints (steps/loss) to the log live
# wandb creds (kept out of the repo); trainer degrades gracefully if absent.
[ -f /home/ubuntu/xscript_prod/wandb_env.sh ] && source /home/ubuntu/xscript_prod/wandb_env.sh
cd /home/ubuntu/XScript-Pretraining
exec python3 /home/ubuntu/xscript_prod/prod_train.py
