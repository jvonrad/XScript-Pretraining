#!/usr/bin/env bash
#
# Score the fifteen cooled 30B finals on PolyFact (jvonrad/PolyFact), own
# training languages only.
#
#   bash run_polyfact.sh <cores> <model-list> <workdir> [families...]
#
# PolyFact is a KNOWLEDGE instrument, not a capability one. Per CLAUDE.md 6f it
# must never be pooled into the cross-language capability aggregate: like the
# native MMLU exams it answers "what does this model know", and knowledge
# benchmarks are used for within-language contrasts only. Unlike translated
# MMLU it is not Anglocentric by construction -- every item is a Wikidata
# triple rendered natively per language, and the gold entity was verified
# identical across all five (c5_tasks/polyfact/utils.py).
#
# All fifteen models here are COOLED 30B finals at LR 3.0e-4, so they are
# mutually comparable but must not be paired against any mid-stable
# checkpoint (CLAUDE.md 6/6d, and 6g's 4-5x headroom-per-token measurement).
#
# Batch 16 is safe: prompts are one short question plus a localized cue, an
# order of magnitude below the Global-MMLU prompts that force batch 8.
set -uo pipefail        # NOT -e: one failed model must not take the worker down

CORES="${1:?usage: run_polyfact.sh <cores> <model-list> <workdir> [families...]}"
LIST="${2:?}"
WORK="${3:-/mnt/scratch/xscript_bench}"
shift 3 || true
FAMILIES=("${@:-polyfact}")

REPO=jvonrad/xscript-eval
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$WORK/logs"; mkdir -p "$LOGS"
LOG="$LOGS/polyfact_$CORES.log"

export NEURON_RT_VISIBLE_CORES="$CORES"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

say() { echo "[polyfact $CORES] $(date -u +%F' '%H:%M:%S) $*" >> "$LOG"; }

say "starting $(grep -cve '^\s*$' -e '^#' "$LIST") models, families: ${FAMILIES[*]}"

while read -r m; do
    [ -z "$m" ] && continue
    case "$m" in \#*) continue ;; esac
    marker="$LOGS/$m.polyfact.done"
    if [ -f "$marker" ]; then say "$m: already complete, skipping"; continue; fi

    say "=== $m: start ==="
    t0=$(date +%s)

    python "$HERE/run_extra_bench.py" --repo "$REPO" --device xla \
        --batch-size 16 --workdir "$WORK" --own-langs --keep-checkpoints \
        --families "${FAMILIES[@]}" --runs "$m" >> "$LOGS/$m.log" 2>&1
    rc=$?

    rm -f "$WORK/_assembled/runs/$m/checkpoints/final.pt"
    rm -f "$WORK/xscript/runs/$m/checkpoints/final.pt"
    rmdir -p "$WORK/_assembled/runs/$m/checkpoints" 2>/dev/null

    dt=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
        touch "$marker"; say "=== $m: done in ${dt}s ==="
    else
        say "=== $m: FAILED (rc=$rc) after ${dt}s ==="
    fi
done < "$LIST"

say "ALL DONE -- PolyFact pass finished on cores $CORES"
