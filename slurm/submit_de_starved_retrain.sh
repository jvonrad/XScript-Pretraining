#!/bin/bash
# Submit the de__unigram_starved retrain (the one missing monolingual).
#
#   bash slurm/submit_de_starved_retrain.sh            # submit (2 nodes)
#   DRY_RUN=1 bash slurm/submit_de_starved_retrain.sh  # print, don't submit
#   NODES=1 bash slurm/submit_de_starved_retrain.sh    # 1 node, ~2x wall-clock
#
# NODE COUNT. Throughput is ~218 M tokens/GPU-hour either way (measured over 7
# prior runs; 1 node = 248k tok/s, 2 nodes = 480k tok/s, i.e. ~61k/GPU both).
# One node is ~1.5% cheaper per GPU-hour (no inter-node collectives); two nodes
# halve the wall-clock and so surface a repeat of the 1-3B loss spike in ~1.7h
# instead of ~3.4h, which is worth more than 1.5% here. Either fits the budget.
#
# WALLTIME is a SECOND safety net, not the stopping rule. The run stops itself
# at `train.max_tokens` (16.1B) in the config. The --time below is sized just
# past that so a hang cannot quietly drain the grant.
set -euo pipefail
source "$(dirname "$0")/env.sh"

RUN=de__unigram_starved
BASE="${BASE:-configs/base_de_starved_retrain.yaml}"
NODES="${NODES:-2}"
DRY_RUN="${DRY_RUN:-0}"

# 16.1B / (NODES * 4 GPUs * ~61k tok/s) + 1h slack for startup, compile, evals
# and checkpoint writes.
if [ "$NODES" = 1 ]; then TIME="19:00:00"; else TIME="11:00:00"; fi

# --- preflight: fail loudly here rather than 10 minutes into an allocation ---
fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }
[ -f "$BASE" ] || fail "missing config $BASE"
[ -f "$CONTAINER" ] || fail "missing container $CONTAINER (run slurm/00_pull_container.sh)"
[ -d "$XSCRIPT_SCRATCH/tokenizers/unigram_starved" ] || fail "missing tokenizer unigram_starved"
[ -f "$XSCRIPT_SCRATCH/shards/de__unigram_starved/index.json" ] || \
  fail "missing packed shards (run slurm/22_pool_and_pack_de.sbatch)"

TOKENS=$(python3 -c "import json;print(sum(json.load(open('$XSCRIPT_SCRATCH/shards/de__unigram_starved/index.json')).values()))")
echo "packed tokens available: $(python3 -c "print(f'{$TOKENS/1e9:.2f}B')")"
python3 -c "
t=$TOKENS
assert t >= 16.1e9, f'only {t/1e9:.2f}B packed, need 16.1B -- pool/pack fell short'
print(f'  OK covers the 16.1B target with {(t-16.1e9)/1e9:.2f}B spare (no epoching)')
" || fail "insufficient packed tokens"

# FLORES+ is optional (train.py wraps it in try/except) but feeds the
# eval/flores_de_bpb curve that the ATLAS-BTS analysis reads. Warn, don't block.
[ -d "$XSCRIPT_SCRATCH/flores_plus" ] || \
  echo "WARNING: no FLORES+ on scratch -- eval/flores_de_bpb will be SKIPPED." \
       "Run 'xscript flores-download' with HF_TOKEN set to get the BTS curve."

cmd=(sbatch --parsable --account="$XS_ACCOUNT" --partition="$XS_PARTITION"
     --nodes="$NODES" --time="$TIME" --job-name="$RUN"
     --export="ALL,RUN=$RUN,FLAVOR=unigram,BASE=$BASE,ONLY_30B=1"
     slurm/30_train.sbatch)

echo
echo "run     : $RUN"
echo "config  : $BASE  (schedule identical to base_main; seeds changed; max_tokens=16.1B)"
echo "nodes   : $NODES  ($((NODES*4)) GH200)   walltime cap: $TIME"
echo "budget  : ~74 GPU-h of ~100 at the measured 218 Mtok/GPU-h"
echo
if [ "$DRY_RUN" = 1 ]; then
  printf 'DRY_RUN:'; printf ' %q' "${cmd[@]}"; printf '\n'
else
  jid=$("${cmd[@]}")
  echo "submitted job $jid"
  echo
  echo "WATCH THE 1-3B WINDOW. The previous attempt reached loss 2.68 at 0.73B,"
  echo "then spiked to 3.6-3.8 across 1.5-2.6B and never recovered. If loss is"
  echo "above ~3.2 at 2B, kill it and change data_seed again rather than paying"
  echo "for a run that cannot catch up:"
  echo "    python scripts/train_status.py"
fi
