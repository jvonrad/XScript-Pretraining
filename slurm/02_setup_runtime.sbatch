#!/bin/bash
#SBATCH --job-name=xs-runtime-setup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=logs/runtime-setup-%j.out
# One-time installation of packages missing from the NGC container's Python
# 3.12. Do not reuse the data-prep venv: it is Python 3.11 and ABI-incompatible.
# Torch itself remains the optimized copy supplied by the NGC container.
#
# Usage:
#   sbatch --account=brics.u6jh --partition=workq slurm/02_setup_runtime.sbatch
module load brics/apptainer-multi-node
XS_DIR="${SLURM_SUBMIT_DIR:-$PWD}"; [ -f "$XS_DIR/slurm/env.sh" ] && XS_DIR="$XS_DIR/slurm"
source "$XS_DIR/env.sh"
mkdir -p logs

if [ -f "$XSCRIPT_RUNTIME/.complete" ]; then
  echo "container runtime already installed at $XSCRIPT_RUNTIME"
  exit 0
fi
mkdir -p "$XSCRIPT_RUNTIME"

in_container "
  set -e
  python -m pip install --target '$XSCRIPT_RUNTIME' \
    'sentencepiece>=0.2.0' 'tokenizers>=0.19' zstandard wandb \
    'lm_eval==0.4.12' 'huggingface_hub<1'
  PYTHONPATH='$XSCRIPT_RUNTIME' python -c \"import importlib.metadata as m, sentencepiece, tokenizers, wandb, zstandard; assert m.version('lm_eval') == '0.4.12'; assert int(m.version('huggingface_hub').split('.')[0]) < 1\"
  touch '$XSCRIPT_RUNTIME/.complete'
"
echo "container runtime ready: $XSCRIPT_RUNTIME"
