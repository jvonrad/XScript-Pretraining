#!/usr/bin/env bash
#
# Knowledge-neuron sweep over the 8 bilingual 30B finals (CLAUDE.md 6j).
#
#   bash run_kn_fleet.sh <cores> <workdir> <model> [model...]
#
# One worker per logical core-pair; launch two of these with disjoint core
# sets and disjoint model lists. All graph shapes are shared across models
# (width 64, fixed batches) and weight-independent, so once one worker has
# compiled them the cache serves everyone -- warm with a --limit-facts smoke
# or start the second worker after the first model completes.
#
# NOT -e: one failed model must not take the worker down (6f).
set -uo pipefail

CORES="${1:?usage: run_kn_fleet.sh <cores> <workdir> <model...>}"
WORK="${2:?}"
shift 2

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$WORK/logs"; mkdir -p "$LOGS"
LOG="$LOGS/kn_$CORES.log"

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/neuron_venv/bin/activate"
export NEURON_RT_VISIBLE_CORES="$CORES"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

say() { echo "[kn $CORES] $(date -u +%F' '%H:%M:%S) $*" >> "$LOG"; }
say "starting models: $*"

for m in "$@"; do
    if [ -f "$WORK/results/knowneurons/${m}_ablation.json" ]; then
        say "$m: already complete, skipping"; continue
    fi
    say "=== $m: start ==="
    t0=$(date +%s)
    python "$HERE/run_knowneurons.py" --device xla --workdir "$WORK" \
        --max-width 64 --runs "$m" >> "$LOGS/$m.log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then say "=== $m: done in ${dt}s ==="
    else say "=== $m: FAILED (rc=$rc) after ${dt}s ==="; fi
done
say "ALL DONE on cores $CORES"
