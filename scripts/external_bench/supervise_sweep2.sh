#!/usr/bin/env bash
#
# Detached supervisor for the 68-checkpoint sweep.  Every INTERVAL it appends a
# liveness + plausibility snapshot to $WORK/logs/health.log, and when both
# workers are gone it writes the final report and exits.
#
#   bash supervise_sweep.sh <workdir>
#
# This is what replaces a human watching the sweep.  It does NOT restart
# anything (chain_worker.sh owns that); it only records, so that a sweep which
# silently starts producing constant predictions is visible at the end instead
# of six months later -- the failure mode CLAUDE.md 6/6d/6e documents five
# times over.
set -uo pipefail

WORK="${1:-/home/ubuntu/xscript_bench}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL=1200          # 20 min
HEALTH="$WORK/logs/health.log"

say() { echo "[supervise] $(date -u +%F' '%H:%M:%S) $*" >> "$HEALTH"; }

say "supervisor started; interval ${INTERVAL}s"

while true; do
    # Anchored patterns throughout (NEURON.md 6f).
    workers=$(pgrep -cf "^bash $HERE/run_sweep68.sh" || true)
    finals=$(pgrep -cf "^bash $HERE/run_finals_mubench.sh" || true)
    workers=$((workers + finals))
    chains=$(pgrep -cf "^bash $HERE/chain_worker.sh" || true)
    done_n=$(ls "$WORK"/logs/*.done 2>/dev/null | wc -l)
    free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')

    say "workers=$workers chains=$chains models_done=$done_n/68 free=${free_gb}G"

    if [ "${free_gb:-99}" -lt 15 ]; then
        say "WARNING: only ${free_gb}G free on / -- checkpoints may fail to assemble"
    fi

    {
        echo "----- health_check $(date -u +%F' '%H:%M:%S) -----"
        python "$HERE/health_check.py" "$WORK" --quiet 2>&1 | tail -40
    } >> "$HEALTH"

    if [ "$chains" -eq 0 ] && [ "$workers" -eq 0 ]; then
        say "no workers and no chains left -- sweep is over, writing final report"
        {
            echo "===================== FINAL ====================="
            python "$HERE/health_check.py" "$WORK" 2>&1
        } >> "$HEALTH"
        say "SUPERVISOR EXITING"
        exit 0
    fi
    sleep "$INTERVAL"
done
