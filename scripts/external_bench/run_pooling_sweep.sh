#!/usr/bin/env bash
# Four-pooling alignment sweep over the 8 EN-anchored bilinguals.
#
#   bash run_pooling_sweep.sh <workdir> <out-subdir> [suffix]
#
# suffix "" -> the 30B finals (primary); "-2b" -> the 2B tier (secondary).
#
# One model at a time: each checkpoint is ~4 GB and is deleted after scoring
# (no --keep-checkpoints), so peak disk stays near one checkpoint plus the
# cached embeddings. Embeddings are kept because T4's length control has to
# re-run retrieval on a sentence subset, and that must be a pure-CPU
# re-derivation rather than another accelerator pass -- the same lesson as
# CLAUDE.md section 6b's cached embeddings and section 6e's raw sidecars.
set -euo pipefail

WORK=${1:?workdir}
SUBDIR=${2:?out-subdir}
SUFFIX=${3:-}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH="$HOME/.local/bin:$PATH"
# shellcheck disable=SC1091
source "$HOME/neuron_venv/bin/activate"
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
# Checkpoints are ~4.4 GB in 5 parts; the plain python downloader ran at ~2 MB/s
# here, which would make an 8-model sweep download-bound (~30 min/model against
# ~100 s of actual compute).
export HF_HUB_ENABLE_HF_TRANSFER=1

cd "$(dirname "$0")"

for p in de fr ar zh; do
  for t in fair starved; do
    RUN="en-${p}-${t}${SUFFIX}"
    echo "=============== ${RUN} ==============="
    python run_alignment.py \
      --repo jvonrad/xscript-eval --device xla \
      --workdir "$WORK" --runs "$RUN" \
      --langs en de fr ar zh --split both \
      --poolings mean mean_nobos weighted last \
      --emb-dir "$WORK/embeddings" \
      --out-subdir "$SUBDIR"
  done
done
echo "SWEEP COMPLETE"
