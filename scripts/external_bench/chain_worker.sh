#!/usr/bin/env bash
#
# One detached sweep worker: waits for its gate, then drives run_sweep68.sh
# over its model list, restarting it if it dies.  Survives logout (launch with
# setsid nohup).
#
#   bash chain_worker.sh <cores> <model-list> <workdir> [gate-seconds]
#
# Restart policy: run_sweep68.sh skips any model with a .done marker and the
# runners themselves skip already-scored tasks, so re-invoking is a resume, not
# a redo.  Capped at MAX_TRIES so a model that fails deterministically cannot
# spin forever.
#
# The gate exists because of NEURON.md 9 hazard 3: launching two jobs that
# compile the same graph simultaneously can deadlock both.  Worker B waits for
# worker A's first model to finish, by which time the shared task-graph shapes
# are in /var/tmp/neuron-compile-cache and B's compiles are cache hits.
set -uo pipefail

CORES="${1:?usage: chain_worker.sh <cores> <model-list> <workdir> [gate-seconds]}"
LIST="${2:?}"
WORK="${3:-/home/ubuntu/xscript_bench}"
GATE="${4:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_TRIES=4
LOG="$WORK/logs/chain_$CORES.log"

say() { echo "[chain $CORES] $(date -u +%F' '%H:%M:%S) $*" >> "$LOG"; }

say "gate: waiting up to ${GATE}s for the warm-up model to finish compiling"
waited=0
while [ "$waited" -lt "$GATE" ]; do
    # Anchored (NEURON.md 6f): an unanchored pattern matches this very shell.
    if ! pgrep -f "^bash $HERE/run_sweep68.sh" > /dev/null; then
        say "gate: no warm-up worker running after ${waited}s, proceeding"
        break
    fi
    sleep 30
    waited=$((waited + 30))
done
[ "$waited" -ge "$GATE" ] && say "gate: timeout reached, proceeding anyway"

for try in $(seq 1 $MAX_TRIES); do
    say "attempt $try/$MAX_TRIES over $(wc -l < "$LIST") models"
    bash "$HERE/run_sweep68.sh" "$CORES" "$LIST" "$WORK" >> "$LOG" 2>&1

    remaining=0
    while read -r m; do
        [ -z "$m" ] && continue
        [ -f "$WORK/logs/$m.done" ] || remaining=$((remaining + 1))
    done < "$LIST"

    if [ "$remaining" -eq 0 ]; then
        say "ALL DONE -- every model in this list has a .done marker"
        exit 0
    fi
    say "attempt $try finished with $remaining model(s) incomplete; retrying"
    sleep 60
done

say "GIVING UP after $MAX_TRIES attempts; $remaining model(s) still incomplete"
