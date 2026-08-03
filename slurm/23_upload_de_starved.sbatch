#!/bin/bash
#SBATCH --job-name=xs-upload-de
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/upload-de-starved-%j.out
# Upload the de__unigram_starved checkpoints to jvonrad/xscript-eval, in the
# same `runs/<friendly-name>/checkpoints/final.pt.partNNN` layout as the other
# 109 models.
#
# WHY A JOB. scripts/external_bench/upload_chunked.py exists because "this
# login node kills any single process after a few minutes of sustained work".
# That is the same systemd Linger=no teardown documented in §6h — a compute
# node has no such limit, so this runs there instead. It still writes 900MB
# parts, because that is what every existing model dir looks like and what
# n_parts.txt records; `fetch_checkpoint` would also accept a whole final.pt
# (it checks that branch first), but mixing conventions across the repo buys
# nothing.
#
# Idempotent: skips any part already present at the right size, so a re-run
# after an interruption uploads only what is missing.
#
# Usage:
#   sbatch --account=brics.u6sg --partition=workq slurm/23_upload_de_starved.sbatch
set -euo pipefail

XS_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export XSCRIPT_SCRATCH="${XSCRIPT_SCRATCH:-/scratch/u6sg/jvonrad.u6sg/xscript}"
export HF_HOME="${HF_HOME:-$XSCRIPT_SCRATCH/hf_cache}"
cd "$XS_DIR"
mkdir -p logs
source "$XSCRIPT_SCRATCH/venv/bin/activate"

echo "host=$(hostname) start=$(date --iso-8601=seconds)"
python -u scripts/external_bench/upload_de_starved.py \
    --repo jvonrad/xscript-eval \
    --run-dir "$XSCRIPT_SCRATCH/runs/de__unigram_starved/checkpoints" \
    --staging "$XSCRIPT_SCRATCH/_hf_parts_de_starved" \
    ${UPLOAD_ARGS:-}
echo "end=$(date --iso-8601=seconds)"
