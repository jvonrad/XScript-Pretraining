#!/bin/bash
#SBATCH --job-name=xs-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/bench-%x-%j.out
# Downstream evaluation for one completed checkpoint via lm-eval-harness.
# Kept separate from 40_eval.sbatch because Global-MMLU + Belebele + XNLI
# across five languages is substantially more expensive than BPB/BTS/MEXA.
#
# Usage:
#   sbatch --account=brics.u6jh --partition=workq \
#     --job-name=bench-en__unigram_destarved \
#     --export=ALL,RUN=en__unigram_destarved,TOK=unigram_destarved \
#     slurm/41_bench.sbatch
module load brics/apptainer-multi-node
XS_DIR="${SLURM_SUBMIT_DIR:-$PWD}"; [ -f "$XS_DIR/slurm/env.sh" ] && XS_DIR="$XS_DIR/slurm"
source "$XS_DIR/env.sh"
mkdir -p logs

: "${RUN:?set RUN=<completed run name>}"
: "${TOK:?set TOK=<checkpoint tokenizer name>}"
TAG="${TAG:-final}"
BATCH_SIZE="${BATCH_SIZE:-4}"

# lm_eval must match the container's Python 3.12, not the data-prep venv's
# Python 3.11. Run slurm/02_setup_runtime.sbatch once before GPU jobs.
[ -f "$XSCRIPT_RUNTIME/.complete" ] || {
  echo "missing $XSCRIPT_RUNTIME/.complete; submit slurm/02_setup_runtime.sbatch first" >&2
  exit 1
}
in_container "
  export XSCRIPT_SCRATCH='$XSCRIPT_SCRATCH' HF_HOME='$HF_HOME'
  xscript eval-bench '$RUN' --tok '$TOK' --tag '$TAG' --batch-size '$BATCH_SIZE'
"
