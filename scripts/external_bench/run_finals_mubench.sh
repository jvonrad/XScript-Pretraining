#!/usr/bin/env bash
#
# Queued follow-on: score the 12b/23b checkpoints on the four MuBench families,
# which have never been run on ANY checkpoint (CLAUDE.md 6f listed them as
# "sweep running" but that sweep died with the previous box).
#
#   bash run_finals_mubench.sh <cores> <model-list> <workdir>
#
# Waits for THIS core-pair's chain_worker.sh to exit, then takes over the same
# cores.  Deliberately self-contained rather than reusing run_sweep68.sh: bash
# reads a script incrementally by byte offset, so editing or re-parameterising
# a script that two workers are currently executing can make them jump to the
# wrong offset mid-sweep.  Nothing here touches a running file.
#
# SIB-200 and XNLI are NOT re-run: all 17 of these models already have them
# calibrated in results/recalibrated/ (committed to git).  Only the raw
# sidecars for those two were lost with the old box.  To regenerate those as
# well -- insurance so a future scoring change stays pure-CPU across the whole
# trajectory -- add sib200 to SHORT and set DO_XNLI=1; it roughly triples the
# runtime.
set -uo pipefail

CORES="${1:?usage: run_finals_mubench.sh <cores> <model-list> <workdir>}"
LIST="${2:?}"
WORK="${3:-/home/ubuntu/xscript_bench}"

REPO=jvonrad/xscript-eval
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$WORK/logs"; mkdir -p "$LOGS"
LOG="$LOGS/finals_$CORES.log"

SHORT=(mub_arceasy mub_storycloze)     # add sib200 here to regenerate its raw
LONG=(mub_hellaswag mub_bmlama)        # long prompts -> batch 8 (NCC_EOOM002)
DO_XNLI=0

export NEURON_RT_VISIBLE_CORES="$CORES"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

say() { echo "[finals $CORES] $(date -u +%F' '%H:%M:%S) $*" >> "$LOG"; }

# ---- wait for this core-pair's 68-sweep chain to finish -------------------
# Anchored on the exact core argument so it waits for ITS OWN worker only, and
# cannot match the shell running this line (NEURON.md 6f: an unanchored
# pattern matched the caller and deadlocked a chained sweep for 3.7h).
say "waiting for chain_worker.sh $CORES to finish before claiming these cores"
while pgrep -f "^bash $HERE/chain_worker.sh $CORES " > /dev/null; do
    sleep 120
done
say "cores $CORES free; starting $(wc -l < "$LIST") models"

while read -r m; do
    [ -z "$m" ] && continue
    case "$m" in \#*) continue ;; esac
    marker="$LOGS/$m.mubench.done"
    if [ -f "$marker" ]; then say "$m: already complete, skipping"; continue; fi

    say "=== $m: start ==="
    t0=$(date +%s)

    python "$HERE/run_extra_bench.py" --repo "$REPO" --device xla \
        --batch-size 16 --workdir "$WORK" --own-langs --keep-checkpoints \
        --families "${SHORT[@]}" --runs "$m" >> "$LOGS/$m.log" 2>&1
    rc_s=$?

    python "$HERE/run_extra_bench.py" --repo "$REPO" --device xla \
        --batch-size 8 --workdir "$WORK" --own-langs --keep-checkpoints \
        --families "${LONG[@]}" --runs "$m" >> "$LOGS/$m.log" 2>&1
    rc_l=$?

    rc_x=0
    if [ "$DO_XNLI" -eq 1 ]; then
        python "$HERE/run_appendix_c5.py" --repo "$REPO" --device xla \
            --batch-size-short 16 --batch-size 8 --workdir "$WORK" --own-langs \
            --keep-checkpoints --only xnli --xnli-raw-all-langs --runs "$m" \
            >> "$LOGS/$m.log" 2>&1
        rc_x=$?
    fi

    rm -f "$WORK/_assembled/runs/$m/checkpoints/final.pt"
    rm -f "$WORK/xscript/runs/$m/checkpoints/final.pt"
    rmdir -p "$WORK/_assembled/runs/$m/checkpoints" 2>/dev/null

    dt=$(( $(date +%s) - t0 ))
    if [ $rc_s -eq 0 ] && [ $rc_l -eq 0 ] && [ $rc_x -eq 0 ]; then
        touch "$marker"; say "=== $m: done in ${dt}s ==="
    else
        say "=== $m: FAILED (short=$rc_s long=$rc_l xnli=$rc_x) after ${dt}s ==="
    fi
done < "$LIST"

say "ALL DONE -- finals MuBench pass finished on cores $CORES"
