#!/bin/bash
#SBATCH --job-name=xs-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval-%j.out
# Single-GPU evaluation: re-score BPB and MEXA alignment for every finished run
# of a flavor, then compute the BTS interaction table. Runs inside the container
# (needs torch) but on one GPU.
#
# Usage: sbatch --account=brics.u6jh --partition=workq \
#          --export=ALL,FLAVOR=unigram slurm/40_eval.sbatch
# The module binds /host/adapt.sh into the container (in_container needs it),
# even though this single-GPU job never touches the NCCL plugin itself.
module load brics/apptainer-multi-node
# Slurm copies this script to a private spooldir, so "$(dirname "$0")" is NOT
# the repo. Resolve env.sh from the submit dir (where `sbatch` was run).
XS_DIR="${SLURM_SUBMIT_DIR:-$PWD}"; [ -f "$XS_DIR/slurm/env.sh" ] && XS_DIR="$XS_DIR/slurm"
source "$XS_DIR/env.sh"
mkdir -p logs
FLAVOR="${FLAVOR:-unigram}"

[ -f "$XSCRIPT_RUNTIME/.complete" ] || {
  echo "missing $XSCRIPT_RUNTIME/.complete; submit slurm/02_setup_runtime.sbatch first" >&2
  exit 1
}

in_container "
  set -e
  export XSCRIPT_SCRATCH='$XSCRIPT_SCRATCH' HF_HOME='$HF_HOME'
  for RUN in \$(xscript runs --flavor $FLAVOR); do
    case \"\$RUN\" in *__trunk) continue;; esac        # trunks are not deliverables
    TOK=\$(echo \"\$RUN\" | grep -oE '${FLAVOR}_(starved|destarved)')  # tok from run name
    CK='$XSCRIPT_SCRATCH/runs/'\"\$RUN\"'/checkpoints/final.pt'
    [ -f \"\$CK\" ] || { echo \"skip \$RUN (no final.pt)\"; continue; }
    echo \"== eval \$RUN (tok=\$TOK) ==\"
    xscript eval-bpb \"\$RUN\" --tok \"\$TOK\" --tag final || true
    xscript eval-align \"\$RUN\" --tok \"\$TOK\" || true
  done
  echo '== BTS interaction =='
  xscript bts --flavor $FLAVOR --source flores
  xscript bts --flavor $FLAVOR --source holdout
"
