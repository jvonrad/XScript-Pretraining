#!/usr/bin/env bash
#
# Sweep the 68 low-budget checkpoints (1B-15B) over the six benchmark families
# that carry the trajectory curve, so the 41 calibrated finals stop being a
# single-budget snapshot (CLAUDE.md 6f, open item 3).
#
#   bash run_sweep68.sh <cores> <model-list-file> <workdir>
#
#     cores       NEURON_RT_VISIBLE_CORES value for this worker, e.g. 0-1
#     model-list  one friendly model name per line, already budget-interleaved
#
# Two things here are deliberate and should not be "simplified":
#
# 1. --keep-checkpoints is scoped to ONE MODEL, not the whole sweep.  Both
#    runners score the same checkpoint, so keeping it across the pair saves a
#    4 GB download; keeping it across all 68 would need ~272 GB against this
#    box's 190 GB root and would wedge the sweep around model 47.  The
#    checkpoint is therefore deleted explicitly before the next model starts.
#
# 2. The families are split across two batch sizes.  HellaSwag and BMLAMA
#    carry much longer prompts than SIB-200/ARC-Easy/StoryCloze/WinoGrande,
#    and _score_active_xla materializes a [batch, width, 65536] one-hot, so
#    batch 16 on a wide graph dies with NCC_EOOM002 (NEURON.md 5, CLAUDE.md 6e).
#
# Resumability is the runners' own: --require-raw (default on) treats a task
# scored before the raw sidecar existed as not done, so re-running this script
# backfills rather than redoing.  Killing a worker between models is safe;
# killing one mid-compile is not (NEURON.md 4 -- truncated compile-cache entry).
set -uo pipefail        # NOT -e: one failed model must not take the worker down

CORES="${1:?usage: run_sweep68.sh <cores> <model-list> <workdir>}"
LIST="${2:?usage: run_sweep68.sh <cores> <model-list> <workdir>}"
WORK="${3:-/home/ubuntu/xscript_bench}"

REPO=jvonrad/xscript-eval
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$WORK/logs"; mkdir -p "$LOGS"

SHORT_FAMILIES=(sib200 mub_arceasy mub_storycloze)
LONG_FAMILIES=(mub_hellaswag mub_bmlama)

export NEURON_RT_VISIBLE_CORES="$CORES"
# The host-side numpy/tokenizer phases will otherwise grab all 12 cores and
# starve the other worker (NEURON.md 6b-ops).
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

say() { echo "[sweep68 $CORES] $(date -u +%H:%M:%S) $*"; }

while read -r m; do
    [ -z "$m" ] && continue
    case "$m" in \#*) continue ;; esac

    marker="$WORK/logs/$m.done"
    if [ -f "$marker" ]; then say "$m: already complete, skipping"; continue; fi

    say "=== $m: start ==="
    t0=$(date +%s)

    python "$HERE/run_extra_bench.py" --repo "$REPO" --device xla \
        --batch-size 16 --workdir "$WORK" --own-langs --keep-checkpoints \
        --families "${SHORT_FAMILIES[@]}" --runs "$m" \
        >> "$LOGS/$m.log" 2>&1
    rc_short=$?

    python "$HERE/run_extra_bench.py" --repo "$REPO" --device xla \
        --batch-size 8 --workdir "$WORK" --own-langs --keep-checkpoints \
        --families "${LONG_FAMILIES[@]}" --runs "$m" \
        >> "$LOGS/$m.log" 2>&1
    rc_long=$?

    # --xnli-raw-all-langs is what makes en/de/fr calibratable; without it the
    # curve would mix acc_cal (ar/zh) with raw acc (en/de/fr), which is the
    # mixed-estimator failure 6e documents.
    python "$HERE/run_appendix_c5.py" --repo "$REPO" --device xla \
        --batch-size-short 16 --batch-size 8 --workdir "$WORK" --own-langs \
        --keep-checkpoints --only xnli --xnli-raw-all-langs --runs "$m" \
        >> "$LOGS/$m.log" 2>&1
    rc_xnli=$?

    # Drop the 4 GB checkpoint before the next model (see note 1 above).
    rm -f "$WORK/_assembled/runs/$m/checkpoints/final.pt"
    rm -f "$WORK/xscript/runs/$m/checkpoints/final.pt"
    rmdir -p "$WORK/_assembled/runs/$m/checkpoints" 2>/dev/null

    dt=$(( $(date +%s) - t0 ))
    if [ $rc_short -eq 0 ] && [ $rc_long -eq 0 ] && [ $rc_xnli -eq 0 ]; then
        touch "$marker"
        say "=== $m: done in ${dt}s ==="
    else
        say "=== $m: FAILED (short=$rc_short long=$rc_long xnli=$rc_xnli) after ${dt}s ==="
    fi
done < "$LIST"

say "worker finished its list"
