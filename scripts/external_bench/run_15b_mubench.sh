#!/usr/bin/env bash
#
# Score the nine `*-15b` monolinguals on the four MuBench families -- the one
# gap CLAUDE.md 6f (open item 2) and 6g both call the highest-value missing
# number.  The mono and bilingual budget rosters intersect only at 2B/5B/30B,
# so 15B is the ONLY trainable-matched tier that is also mid-stable; without
# it, 6g's -.044 English dilution cost at 30B cannot be separated from the
# cooldown.
#
#   bash run_15b_mubench.sh <cores> <model-list> <workdir>
#
# SIB-200, XNLI and Taxi-1500 are NOT re-run: all nine models already have them
# calibrated in results/recalibrated/ (committed).  Only their raw sidecars died
# with the old box; regenerating those is a separate, optional job.
#
# A new file rather than a flag on run_finals_mubench.sh on purpose: bash reads
# a script incrementally by byte offset, so editing a script another worker is
# executing can make it jump to a wrong offset mid-sweep (CLAUDE.md 6g).
set -uo pipefail

CORES="${1:?usage: run_15b_mubench.sh <cores> <model-list> <workdir>}"
LIST="${2:?}"
WORK="${3:-/mnt/scratch/xscript_bench}"

REPO=jvonrad/xscript-eval
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$WORK/logs"; mkdir -p "$LOGS"
LOG="$LOGS/mubench15b_$CORES.log"

SHORT=(mub_arceasy mub_storycloze)     # short prompts -> batch 16
LONG=(mub_hellaswag mub_bmlama)        # long prompts  -> batch 8 (NCC_EOOM002)

export NEURON_RT_VISIBLE_CORES="$CORES"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

say() { echo "[15b $CORES] $(date -u +%F' '%H:%M:%S) $*" >> "$LOG"; }

say "starting $(grep -cve '^\s*$' -e '^#' "$LIST") models on cores $CORES"

while read -r m; do
    [ -z "$m" ] && continue
    case "$m" in \#*) continue ;; esac
    marker="$LOGS/$m.mubench.done"
    if [ -f "$marker" ]; then say "$m: already complete, skipping"; continue; fi

    say "=== $m: start ==="
    t0=$(date +%s)

    # --keep-checkpoints is safe here only because the checkpoint is deleted
    # below, per model -- 9 x 4 GB would otherwise accumulate (CLAUDE.md 6g).
    python "$HERE/run_extra_bench.py" --repo "$REPO" --device xla \
        --batch-size 16 --workdir "$WORK" --own-langs --keep-checkpoints \
        --families "${SHORT[@]}" --runs "$m" >> "$LOGS/$m.log" 2>&1
    rc_s=$?

    python "$HERE/run_extra_bench.py" --repo "$REPO" --device xla \
        --batch-size 8 --workdir "$WORK" --own-langs --keep-checkpoints \
        --families "${LONG[@]}" --runs "$m" >> "$LOGS/$m.log" 2>&1
    rc_l=$?

    rm -f "$WORK/_assembled/runs/$m/checkpoints/final.pt"
    rm -f "$WORK/xscript/runs/$m/checkpoints/final.pt"
    rmdir -p "$WORK/_assembled/runs/$m/checkpoints" 2>/dev/null

    dt=$(( $(date +%s) - t0 ))
    if [ $rc_s -eq 0 ] && [ $rc_l -eq 0 ]; then
        touch "$marker"; say "=== $m: done in ${dt}s ==="
    else
        say "=== $m: FAILED (short=$rc_s long=$rc_l) after ${dt}s ==="
    fi
done < "$LIST"

say "ALL DONE -- 15b MuBench pass finished on cores $CORES"
