#!/usr/bin/env bash
# Drive probe_cultural_drift.py across all four EN-anchored fair bilinguals.
#
# Checkpoints are staged with run_bpb.py rather than a second downloader:
# `--source flores --limit 8` fetches + assembles + symlinks the checkpoint
# through the already-verified `fetch_checkpoint` path and scores 8 sentences
# (seconds) as a side effect. Duplicating that fetch logic here would be a
# second thing to keep correct.
#
# One language pair at a time, deleting after each, so peak disk is ~18GB
# rather than ~70GB.
set -uo pipefail
WORK=/home/ubuntu/xscript_bpb
SP="$1"                       # scratch dir for logs
BUDGETS="2b 5b 10b 23b"       # mid-stable only -- the 30B finals are COOLED
PY=/home/ubuntu/neuron_venv/bin/python3
export PYTHONPATH="$SP/py311libs"
export XSCRIPT_FLORES=$WORK/xscript/flores_plus

for pair in en-de en-fr en-ar en-zh; do
  runs=""
  for b in $BUDGETS; do runs="$runs ${pair}-fair-${b}"; done
  echo "=== staging $pair: $runs ==="
  $PY -u scripts/external_bench/run_bpb.py --repo jvonrad/xscript-eval \
      --runs $runs --langs en --split dev --limit 8 --device cpu \
      --keep-checkpoints --workdir $WORK >> "$SP/drift_stage.log" 2>&1
  echo "=== probing $pair ==="
  $PY -u scripts/external_bench/probe_cultural_drift.py --repo jvonrad/xscript-eval \
      --runs $runs --workdir $WORK --device cpu \
      --out results/cultural_drift/raw_${pair}.json >> "$SP/drift_probe.log" 2>&1
  for b in $BUDGETS; do
    rm -f $WORK/_assembled/runs/${pair}-fair-${b}/checkpoints/final.pt
    rm -f $WORK/xscript/runs/${pair}-fair-${b}/checkpoints/final.pt
  done
  echo "=== done $pair ==="
done
echo "ALL PAIRS DONE"
