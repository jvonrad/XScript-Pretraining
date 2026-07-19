#!/bin/bash
#SBATCH --job-name=xs-tok-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/tok-train-%j.out
# Trains all 5 tokenizers (unigram/bpe x {starved,destarved} + pa_destarved)
# and runs the fertility/allocation gate. Pure CPU, no internet needed --
# reads the corpora `10_prep_data.sh` already downloaded to scratch.
#
# Runs as a real Slurm allocation rather than on the login node: interactive
# login shells here are capped at MemoryMax=4GiB / TasksMax=500 (systemd
# session cgroup - see `systemctl show user-$(id -u).slice`), and
# SentencePiece Unigram / BPE merge-learning over multi-GB corpora needs far
# more than that. A workq node gets its full ~460GB instead.
#
# Usage: sbatch --account=brics.u6jh --partition=workq slurm/11_tok_train.sbatch
# Slurm copies this script to a private spooldir, so "$(dirname "$0")" is NOT
# the repo. Resolve env.sh from the submit dir (where `sbatch` was run).

# tokenizers saved at: /scratch/u6jh/jvonrad.u6jh/xscript/tokenizers/

XS_DIR="${SLURM_SUBMIT_DIR:-$PWD}"; [ -f "$XS_DIR/slurm/env.sh" ] && XS_DIR="$XS_DIR/slurm"
source "$XS_DIR/env.sh"
source "$XSCRIPT_SCRATCH/venv/bin/activate"
mkdir -p logs

echo "== train 5 tokenizers (unigram/bpe x conds + pa_destarved) =="
xscript tok-train --flavor all    # needs the [tok] extra (parity-aware-bpe)

echo "== tokenizer gate (inspect results/tok_analysis before training models) =="
xscript tok-analyze

echo
echo "Done. Inspect results/tok_analysis/{report.md,samples.md}, pick a"
echo "flavor, then pack with the chosen tokenizers (slurm/20_pack.sbatch)."
