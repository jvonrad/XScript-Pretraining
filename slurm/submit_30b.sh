#!/bin/bash
# Submit independent, self-contained 30B runs.  Each allocation supervises its
# torchrun process and restarts it in-place after failures.
#
# All 30:
#   NODES=2 bash slurm/submit_30b.sh
# Selected runs (space-separated):
#   RUNS="en__unigram_starved en__unigram_destarved" bash slurm/submit_30b.sh
# Preview without submission:
#   DRY_RUN=1 bash slurm/submit_30b.sh
source "$(dirname "$0")/env.sh"

FLAVOR="${FLAVOR:-unigram}"
NODES="${NODES:-2}"
BASE="${BASE:-configs/base_main.yaml}"
DRY_RUN="${DRY_RUN:-0}"
XSCRIPT_CLI="${XSCRIPT_CLI:-$XSCRIPT_SCRATCH/venv/bin/xscript}"
[ -x "$XSCRIPT_CLI" ] || {
  echo "missing executable CLI: $XSCRIPT_CLI" >&2
  exit 1
}

if [ -n "${RUNS:-}" ]; then
  read -r -a selected <<< "$RUNS"
else
  mapfile -t selected < <("$XSCRIPT_CLI" runs --base "$BASE" --flavor "$FLAVOR" --only-30b)
fi

valid=$("$XSCRIPT_CLI" runs --base "$BASE" --flavor "$FLAVOR" --only-30b)
for run in "${selected[@]}"; do
  if ! grep -Fxq "$run" <<< "$valid"; then
    echo "invalid 30B run: $run" >&2
    exit 2
  fi
done

active_jobs=$(squeue --me -h -o '%j')
echo "submitting ${#selected[@]} independent 30B run(s), $NODES node(s) each"
for run in "${selected[@]}"; do
  if grep -Fxq "$run" <<< "$active_jobs"; then
    echo "$run -> skipped (already pending/running)"
    continue
  fi
  if [ -f "$XSCRIPT_SCRATCH/runs/$run/checkpoints/final.pt" ]; then
    echo "$run -> skipped (final checkpoint exists)"
    continue
  fi
  cmd=(sbatch --parsable --account="$XS_ACCOUNT" --partition="$XS_PARTITION"
       --nodes="$NODES" --job-name="$run"
       --export="ALL,RUN=$run,FLAVOR=$FLAVOR,BASE=$BASE,ONLY_30B=1"
       slurm/30_train.sbatch)
  if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY_RUN:'; printf ' %q' "${cmd[@]}"; printf '\n'
  else
    jid=$("${cmd[@]}")
    echo "$run -> job $jid"
  fi
done
