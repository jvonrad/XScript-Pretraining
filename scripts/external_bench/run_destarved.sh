#!/usr/bin/env bash
#
# Score the seven `de-starved-*` checkpoints (CLAUDE.md 6h, retrained
# 2026-08-03) over the same six families as the 68-checkpoint trajectory sweep.
# These have NO benchmark scores of any kind -- they were uploaded to HF after
# every sweep in this project had already run -- so unlike the `*-15b` pass
# this one needs SIB-200 and XNLI as well as the four MuBench families.
#
#   bash run_destarved.sh <cores> <model-list> <workdir>
#
# Why it matters: 6g's same-vs-cross-script gap (+.014) is carried by FRENCH
# alone -- German sits with ar/zh -- and `de-starved` not existing is why the
# same-script side had only one language in both tokenizer conditions. Five of
# these seven are step-for-step identical to a `de-fair-Xb` checkpoint, so the
# contrast is LR-matched by construction rather than by interpolation.
#
# Mirrors run_sweep68.sh exactly (same families, same batch split, same
# per-model checkpoint deletion) so these rows are directly comparable to the
# de-fair ones. A separate file rather than a flag: bash reads a script by byte
# offset, so editing one a worker is executing can desync it (CLAUDE.md 6g).
set -uo pipefail        # NOT -e: one failed model must not take the worker down

CORES="${1:?usage: run_destarved.sh <cores> <model-list> <workdir>}"
LIST="${2:?}"
WORK="${3:-/mnt/scratch/xscript_bench}"

REPO=jvonrad/xscript-eval
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$WORK/logs"; mkdir -p "$LOGS"
LOG="$LOGS/destarved_$CORES.log"

SHORT_FAMILIES=(sib200 mub_arceasy mub_storycloze)
LONG_FAMILIES=(mub_hellaswag mub_bmlama)   # long prompts -> batch 8 (NCC_EOOM002)

export NEURON_RT_VISIBLE_CORES="$CORES"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

say() { echo "[destarved $CORES] $(date -u +%F' '%H:%M:%S) $*" >> "$LOG"; }

say "starting $(grep -cve '^\s*$' -e '^#' "$LIST") models on cores $CORES"

while read -r m; do
    [ -z "$m" ] && continue
    case "$m" in \#*) continue ;; esac
    marker="$LOGS/$m.done"
    if [ -f "$marker" ]; then say "$m: already complete, skipping"; continue; fi

    say "=== $m: start ==="
    t0=$(date +%s)

    python "$HERE/run_extra_bench.py" --repo "$REPO" --device xla \
        --batch-size 16 --workdir "$WORK" --own-langs --keep-checkpoints \
        --families "${SHORT_FAMILIES[@]}" --runs "$m" >> "$LOGS/$m.log" 2>&1
    rc_short=$?

    python "$HERE/run_extra_bench.py" --repo "$REPO" --device xla \
        --batch-size 8 --workdir "$WORK" --own-langs --keep-checkpoints \
        --families "${LONG_FAMILIES[@]}" --runs "$m" >> "$LOGS/$m.log" 2>&1
    rc_long=$?

    # --xnli-raw-all-langs is what makes de calibratable; without it this curve
    # would mix acc_cal (ar/zh) with raw acc (en/de/fr) -- the mixed-estimator
    # failure CLAUDE.md 6e documents.
    python "$HERE/run_appendix_c5.py" --repo "$REPO" --device xla \
        --batch-size-short 16 --batch-size 8 --workdir "$WORK" --own-langs \
        --keep-checkpoints --only xnli --xnli-raw-all-langs --runs "$m" \
        >> "$LOGS/$m.log" 2>&1
    rc_xnli=$?

    rm -f "$WORK/_assembled/runs/$m/checkpoints/final.pt"
    rm -f "$WORK/xscript/runs/$m/checkpoints/final.pt"
    rmdir -p "$WORK/_assembled/runs/$m/checkpoints" 2>/dev/null

    dt=$(( $(date +%s) - t0 ))
    if [ $rc_short -eq 0 ] && [ $rc_long -eq 0 ] && [ $rc_xnli -eq 0 ]; then
        touch "$marker"; say "=== $m: done in ${dt}s ==="
    else
        say "=== $m: FAILED (short=$rc_short long=$rc_long xnli=$rc_xnli) after ${dt}s ==="
    fi
done < "$LIST"

say "ALL DONE -- de-starved pass finished on cores $CORES"
