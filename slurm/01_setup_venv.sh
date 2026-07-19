#!/bin/bash
# One-time: CPU virtualenv for the data-prep steps that need internet and run on
# the login node (flores-download, byte-premium, tok-corpus, tok-train,
# tok-analyze, pool). These need no torch. Training/eval instead use the
# container's torch, with the repo added via PYTHONPATH.
source "$(dirname "$0")/env.sh"

module load cray-python/3.11.7 2>/dev/null || module load cray-python || true
VENV="$XSCRIPT_SCRATCH/venv"
if [[ ! -d "$VENV" ]]; then
  python -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -e "$REPO_DIR[tok]"     # base + byte-level/parity-aware learners (no torch)
echo "venv ready: $VENV"
echo "activate with: source $VENV/bin/activate"
