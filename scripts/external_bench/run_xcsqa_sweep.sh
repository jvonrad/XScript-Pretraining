#!/usr/bin/env bash
#
# Score checkpoints on X-CSQA (multilingual CommonsenseQA).
#
#   bash run_xcsqa_sweep.sh <cores> <model-list> [workdir]
#
# One process per logical core-pair; run several with disjoint <cores> and
# disjoint <model-list>s to fan out (NEURON.md 5).  Resumable: a model with a
# `.xcsqa.done` marker is skipped, so a killed sweep is restarted by re-running
# the same command.
#
# Deliberately a SEPARATE file rather than a parameterisation of
# run_finals_mubench.sh.  CLAUDE.md 6g: bash reads a script incrementally by
# byte offset, so editing or extending a script that a worker is currently
# executing can make it jump to a wrong offset mid-sweep.  Nothing here touches
# a running file.
#
# BEFORE THE FIRST RUN, once:
#     python verify_xcsqa.py --csqa      # pool-identity gate, pure CPU
# It is the check CLAUDE.md 6e makes standing practice, and X-CSQA needs it:
# the upstream `test` split is BLIND (every answerKey ""), which would score as
# garbage rather than erroring.
set -uo pipefail

CORES="${1:?usage: run_xcsqa_sweep.sh <cores> <model-list> [workdir]}"
LIST="${2:?usage: run_xcsqa_sweep.sh <cores> <model-list> [workdir]}"
WORK="${3:-/home/ubuntu/xscript_bench}"

REPO=jvonrad/xscript-eval
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$WORK/logs"; mkdir -p "$LOGS"
LOG="$LOGS/xcsqa_$CORES.log"

# Primary family only.  The two controls (xcsqa_enopt = English options, the
# ~14-point label-language effect; xcsqa_encue = English cue, which measured
# 0.000 on Global-MMLU) belong on a handful of models, not on all 116 --
# run them separately with --families xcsqa_enopt.
FAMILIES=(xcsqa)

# Batch 16 is safe here and needs no long/short split: X-CSQA prompts are the
# shortest in the suite (median 73 chars / max 279 for en, 24/91 for zh),
# well under SIB-200's FLORES sentences, and nowhere near the Global-MMLU
# cloze prompts whose task-wide max of 1088 tokens forces batch 8 to avoid
# NCC_EOOM002.
BATCH=16

export NEURON_RT_VISIBLE_CORES="$CORES"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

say() { echo "[xcsqa $CORES] $(date -u +%F' '%H:%M:%S) $*" >> "$LOG"; }

say "starting $(grep -cvE '^\s*(#|$)' "$LIST") models on cores $CORES"

while read -r m; do
    [ -z "$m" ] && continue
    case "$m" in \#*) continue ;; esac

    marker="$LOGS/$m.xcsqa.done"
    if [ -f "$marker" ]; then say "$m: already complete, skipping"; continue; fi

    say "=== $m: start ==="
    t0=$(date +%s)

    # --own-langs: only this model's training languages.  Every transfer cell
    # pairs trained-language scores, so nothing needed is lost; what is given
    # up is the zero-shot cross-lingual readout (CLAUDE.md 6f).
    python "$HERE/run_extra_bench.py" --repo "$REPO" --device xla \
        --batch-size "$BATCH" --workdir "$WORK" --own-langs --keep-checkpoints \
        --families "${FAMILIES[@]}" --runs "$m" >> "$LOGS/$m.log" 2>&1
    rc=$?

    # --keep-checkpoints above, then delete HERE, so the 4GB file lives only
    # for this model.  CLAUDE.md 6g: keeping them across a sweep is what put
    # 68 x 4GB against a 190GB root.
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

say "ALL DONE -- X-CSQA sweep finished on cores $CORES"
