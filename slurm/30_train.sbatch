#!/bin/bash
#SBATCH --job-name=xs-train
#SBATCH --ntasks-per-node=1
# Four loader/training processes share each node. Sixteen host cores are ample
# and avoid exhausting the association's CPU-minute quota while GPUs are the
# actual bottleneck.
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/train-%x-%j.out
# Multi-node pretraining of one run of the matrix (4 GH200 GPUs/node).
#
# Usage:
#   sbatch --account=brics.u6jh --partition=workq --nodes=2 \
#          --job-name=en-ar__unigram_destarved \
#          --export=ALL,RUN=en-ar__unigram_destarved,FLAVOR=unigram,BASE=configs/base_main.yaml \
#          slurm/30_train.sbatch
#
# Resumable: re-submitting the same RUN continues from the last checkpoint
# (the loader is deterministic and world-size-independent, so the node count
# may even change between resubmissions).
# Login shells export TMPDIR=/local/user/$UID.  That directory is not present
# on every node of a multi-node allocation, and the module automatically adds
# TMPDIR to APPTAINER_BINDPATH.  Select a path guaranteed to exist everywhere
# before loading the module.
export TMPDIR=/tmp
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/tmp/apptainer-$UID}"
module load brics/apptainer-multi-node
# Slurm copies this script to a private spooldir, so "$(dirname "$0")" is NOT
# the repo. Resolve env.sh from the submit dir (where `sbatch` was run).
XS_DIR="${SLURM_SUBMIT_DIR:-$PWD}"; [ -f "$XS_DIR/slurm/env.sh" ] && XS_DIR="$XS_DIR/slurm"
source "$XS_DIR/env.sh"
mkdir -p logs

[ -f "$XSCRIPT_RUNTIME/.complete" ] || {
  echo "missing $XSCRIPT_RUNTIME/.complete; submit slurm/02_setup_runtime.sbatch first" >&2
  exit 1
}

: "${RUN:?set RUN=<run name> (see: xscript runs)}"
BASE="${BASE:-configs/base_main.yaml}"
FLAVOR="${FLAVOR:-unigram}"
ONLY_30B="${ONLY_30B:-0}"
RESTART_DELAY="${RESTART_DELAY:-30}"
# Do not begin another torchrun when too little allocation time remains for a
# useful attempt.  This does not replace checkpointing; it avoids a restart
# racing Slurm's hard walltime.
RESTART_GUARD="${RESTART_GUARD:-600}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
NPROC=4
TRAIN_MODE=()
[ "$ONLY_30B" = 1 ] && TRAIN_MODE+=(--only-30b)
TRAIN_OVERRIDES=()
[ -n "${OUTPUT_NAME:-}" ] && TRAIN_OVERRIDES+=(--output-name "$OUTPUT_NAME")
[ -n "${SEED:-}" ] && TRAIN_OVERRIDES+=(--seed "$SEED")
[ -n "${DATA_SEED:-}" ] && TRAIN_OVERRIDES+=(--data-seed "$DATA_SEED")
[ -n "${WANDB_RUN_ID:-}" ] && TRAIN_OVERRIDES+=(--wandb-id "$WANDB_RUN_ID")

# SLURM_JOB_END_TIME is normally exported as an epoch timestamp.  Retain a
# scheduler-query fallback so a site configuration that omits it cannot make
# the supervisor itself fail under `set -u`.
if [ -z "${SLURM_JOB_END_TIME:-}" ]; then
  job_end_iso=$(squeue -h -j "$SLURM_JOB_ID" -o '%e')
  SLURM_JOB_END_TIME=$(date -d "$job_end_iso" +%s)
fi

echo "RUN=$RUN nodes=$SLURM_NNODES gpus=$((SLURM_NNODES*NPROC)) master=$MASTER_ADDR only_30b=$ONLY_30B"

attempt=0
while true; do
  now=$(date +%s)
  remaining=$((SLURM_JOB_END_TIME - now))
  if (( remaining <= RESTART_GUARD )); then
    echo "[supervisor] only ${remaining}s remain; leaving allocation before hard walltime"
    exit 75
  fi

  attempt=$((attempt + 1))
  # A fresh rendezvous port prevents a failed process group from poisoning the
  # next attempt while all attempts remain inside this same allocation.
  export MASTER_PORT=$((20000 + (SLURM_JOB_ID + attempt) % 30000))
  echo "[supervisor] attempt=$attempt remaining=${remaining}s port=$MASTER_PORT start=$(date --iso-8601=seconds)"

  if srun --kill-on-bad-exit=1 --ntasks-per-node=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    apptainer exec --nv --bind "$APPTAINER_BINDS" "$CONTAINER" /host/adapt.sh bash -lc "
      cd '$REPO_DIR'
      export PYTHONPATH='$REPO_DIR/src:$XSCRIPT_RUNTIME'
      export XSCRIPT_SCRATCH='$XSCRIPT_SCRATCH' HF_HOME='$HF_HOME'
      export TRITON_LIBCUDA_PATH='$TRITON_LIBCUDA_PATH'
      # adapt.sh appends the host CUDA 12.6 path; NGC 25.05's cuSPARSE requires
      # its bundled nvJitLink 12.9, so keep the container CUDA directory first.
      export LD_LIBRARY_PATH='$TRITON_LIBCUDA_PATH':/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
      export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
      export CURL_CA_BUNDLE=\$SSL_CERT_FILE REQUESTS_CA_BUNDLE=\$SSL_CERT_FILE
      torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$NPROC \
        --node_rank=\$SLURM_PROCID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        -m xscript.cli train '$RUN' --base '$BASE' --flavor '$FLAVOR' ${TRAIN_MODE[*]} ${TRAIN_OVERRIDES[*]}
    "; then
    echo "[supervisor] training completed successfully on attempt=$attempt"
    exit 0
  else
    rc=$?
    echo "[supervisor] attempt=$attempt failed rc=$rc at $(date --iso-8601=seconds); allocation retained"
  fi

  now=$(date +%s)
  remaining=$((SLURM_JOB_END_TIME - now))
  if (( remaining <= RESTART_GUARD + RESTART_DELAY )); then
    echo "[supervisor] insufficient time for another attempt (${remaining}s remain)"
    exit 75
  fi
  echo "[supervisor] restarting from latest checkpoint in ${RESTART_DELAY}s"
  sleep "$RESTART_DELAY"
done
