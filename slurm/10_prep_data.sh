#!/bin/bash
# Data preparation that needs internet -> run on a login node (not a job).
# Idempotent: every step caches, so re-running resumes.
#
# Requires: HF_TOKEN exported with accepted terms for openlanguagedata/flores_plus.
# Usage: bash slurm/10_prep_data.sh
#
# Deliberately stops short of tok-train/tok-analyze: those need no internet
# but are CPU/memory-heavy (SentencePiece Unigram + BPE merge-learning over
# multi-GB corpora), which blows past this login node's interactive-session
# cgroup cap (MemoryMax=4GiB / TasksMax=500). Submit slurm/11_tok_train.sbatch
# as a real job for that step instead.
source "$(dirname "$0")/env.sh"
source "$XSCRIPT_SCRATCH/venv/bin/activate"

: "${HF_TOKEN:?export HF_TOKEN (with FLORES+ terms accepted) first}"

echo "== FLORES+ + byte premiums =="
xscript flores-download
xscript byte-premium

echo "== tokenizer corpora (raw FineWeb/FineWeb2, ATLAS-scale) =="
xscript tok-corpus both --gb 4

echo "== FineWeb(-2)-HQ pools for all 5 languages =="
for L in en de fr ar zh; do
  xscript pool --lang "$L"
done

echo
echo "Login-node prep done. Next: sbatch --account=brics.u6jh --partition=workq"
echo "slurm/11_tok_train.sbatch  (trains the 5 tokenizers + the fertility gate)."
