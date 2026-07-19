#!/bin/bash
#SBATCH --job-name=xs-pack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=logs/pack-%j.out
# Tokenize the FineWeb pools into uint16 shards. CPU-only, no internet needed
# (pools are already on scratch). Packs every language for the chosen tokenizer
# condition(s). Run once per flavor you decided to train models with.
#
# Usage: sbatch --account=brics.u6jh --partition=workq \
#          --export=ALL,FLAVOR=unigram slurm/20_pack.sbatch
# FLAVOR defaults to unigram -- the only flavor we pretrain models with
# (bpe/pa are tokenizer-analysis comparators, not packed/trained).
# Slurm copies this script to a private spooldir, so "$(dirname "$0")" is NOT
# the repo. Resolve env.sh from the submit dir (where `sbatch` was run).
XS_DIR="${SLURM_SUBMIT_DIR:-$PWD}"; [ -f "$XS_DIR/slurm/env.sh" ] && XS_DIR="$XS_DIR/slurm"
source "$XS_DIR/env.sh"
source "$XSCRIPT_SCRATCH/venv/bin/activate"
mkdir -p logs

FLAVOR="${FLAVOR:-unigram}"
for COND in starved destarved; do
  TOK="${FLAVOR}_${COND}"
  for L in en de fr ar zh; do
    echo "== pack $L / $TOK =="
    xscript pack --lang "$L" --tok "$TOK" --workers 64
  done
done
echo "packing complete."
