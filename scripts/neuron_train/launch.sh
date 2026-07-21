#!/usr/bin/env bash
# Launch one XScript pretraining run on AWS Trainium (trn2.48xlarge, Neuron).
#
# Data-parallel over the box's logical NeuronCores via torchrun + PJRT. Unlike
# the eval fan-out (15 independent single-model jobs, one per core-pair), a
# *training* run is ONE job whose replicas span all cores. Resumable: re-run
# the same RUN and it continues from the latest checkpoint (deterministic,
# world-size-independent loader).
#
# Usage:
#   RUN=zh__unigram_destarved bash scripts/neuron_train/launch.sh
#   RUN=de__unigram_starved NPROC=32 bash scripts/neuron_train/launch.sh
#
# The three unfinished models from CLAUDE.md are:
#   de__unigram_starved   (de-starved)
#   zh__unigram_destarved (zh-fair)
#   zh__unigram_starved   (zh-starved)
set -euo pipefail

: "${RUN:?set RUN=<run name>, e.g. RUN=zh__unigram_destarved}"
BASE="${BASE:-configs/base_main.yaml}"
FLAVOR="${FLAVOR:-unigram}"
ONLY_30B="${ONLY_30B:-1}"                 # these are self-contained 30B cells
# trn2.48xlarge: 16 devices x logical-neuroncore-config 2 = 32 logical cores.
NPROC="${NPROC:-32}"
# Big scratch for checkpoints + compile cache (root vol is tiny on 48xlarge).
WORK="${WORK:-/home/ubuntu/xscript_train}"
export XSCRIPT_SCRATCH="${XSCRIPT_SCRATCH:-$WORK/scratch}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# --- Neuron env (mirrors CLAUDE.md's activation guard) ---
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH="$HOME/.local/bin:$PATH"
# shellcheck disable=SC1091
source ~/neuron_venv/bin/activate          # sets PJRT_DEVICE=NEURON
export PJRT_DEVICE=NEURON
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

# Persistent compile cache so a resumed/second run skips first-compile.
# One training-step graph compiles once (fixed shapes) and is reused forever.
mkdir -p "$WORK/neuron-cache" "$XSCRIPT_SCRATCH"
# --optlevel=1: the full ~1B-param model's fwd+bwd+optimizer-step fused into
# ONE graph (this trainer's whole design -- one xm.mark_step() per optimizer
# step) exceeds the compiler's default 10M-instruction budget at the default
# optlevel (observed: 53M instructions, NCC_EVRF007) with a real 16-layer/
# dim-2048 model; the smoke tests never caught this because they only ever
# used a toy 2-layer/dim-64 model. --optlevel=1 was the compiler's own
# suggested first remedy.
export NEURON_CC_FLAGS="--cache_dir=$WORK/neuron-cache --optlevel=1 ${NEURON_CC_FLAGS:-}"
# bf16 matmul on the tensor engine (matches the CUDA bf16 autocast path).
export XLA_DOWNCAST_BF16="${XLA_DOWNCAST_BF16:-0}"   # we use torch.autocast instead

ONLY=()
[ "$ONLY_30B" = 1 ] && ONLY+=(--only-30b)

# torchrun defaults to port 29500 for its c10d rendezvous; unset, that
# collides when multiple runs launch concurrently on this shared host (as
# they do here -- 3 independent training jobs at once). Derive a distinct
# port per RUN name so repeat launches of the same run are stable but two
# different runs never collide.
MASTER_PORT="${MASTER_PORT:-$((20000 + $(cksum <<<"$RUN" | cut -d' ' -f1) % 30000))}"

echo "[launch] RUN=$RUN nproc=$NPROC flavor=$FLAVOR base=$BASE only_30b=$ONLY_30B port=$MASTER_PORT"
echo "[launch] scratch=$XSCRIPT_SCRATCH cache=$WORK/neuron-cache"

exec torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    scripts/neuron_train/run_train.py "$RUN" \
    --base "$BASE" --flavor "$FLAVOR" "${ONLY[@]}"
