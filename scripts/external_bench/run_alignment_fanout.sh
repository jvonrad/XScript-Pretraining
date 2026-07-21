#!/usr/bin/env bash
# Fan out run_alignment.py across FREE Neuron core-pairs, leaving any running
# training job untouched.
#
#   export HF_TOKEN=hf_...
#   bash run_alignment_fanout.sh [WORKDIR]
#
# Unlike CLAUDE.md section 5's benchmark fan-out this discovers free devices at
# runtime instead of assuming the box is idle: it reads neuron-ls for devices
# with no PID, so a concurrent training run (which pins itself via
# NEURON_RT_VISIBLE_CORES) is never scheduled over. Re-running skips models that
# already have a result JSON, so it is safe to resume after a partial sweep.
set -uo pipefail

WORK="${1:-/mnt/scratch/xscript_align}"
REPO="${REPO:-jvonrad/xscript-eval}"
# Host threads per job. Alignment's analysis phase is numpy-multithreaded and
# would otherwise grab all `nproc` cores per process; unbounded, N parallel jobs
# oversubscribe the box and starve a concurrent trainer's dataloaders even
# though the Neuron cores are disjoint. Keep JOBS * THREADS <= nproc.
THREADS="${THREADS:-8}"
MAX_JOBS="${MAX_JOBS:-12}"

[ -n "${HF_TOKEN:-}" ] || { echo "ERROR: export HF_TOKEN first (repo is private)"; exit 1; }
command -v neuron-ls >/dev/null || { echo "ERROR: activate ~/neuron_venv first"; exit 1; }

mkdir -p "$WORK"
RESULTS="$WORK/results/alignment"
EMB="${EMB_DIR:-$WORK/embeddings}"
mkdir -p "$RESULTS" "$EMB"

# --- free devices -> logical core-pairs ------------------------------------
# neuron-ls columns: | device | cores | ... | core-ids | ... | PID | ...
# A device with PID "NA" is free. Each device is 4 consecutive PHYSICAL core
# ids (0-63); under logical-neuroncore-config 2 that is two pinnable pairs.
mapfile -t FREE_PAIRS < <(
  neuron-ls | awk -F'|' '/^\| [0-9]+ /{
      gsub(/ /,"",$4); gsub(/ /,"",$8);
      if ($8 == "NA") { split($4, r, "-"); print r[1]"-"(r[1]+1); print (r[1]+2)"-"(r[1]+3); }
  }'
)
if [ "${#FREE_PAIRS[@]}" -eq 0 ]; then echo "ERROR: no free Neuron devices"; exit 1; fi
echo "[fanout] ${#FREE_PAIRS[@]} free core-pairs: ${FREE_PAIRS[*]}"
echo "[fanout] busy devices are left alone (training pins itself via NEURON_RT_VISIBLE_CORES)"

MODELS=(ar-fair ar-fair-15b ar-starved ar-starved-15b de-fair de-fair-15b
        en-ar-fair en-ar-starved en-de-fair en-de-starved en-fair en-fair-15b
        en-fr-fair en-fr-starved en-starved en-starved-15b en-zh-fair
        en-zh-fair-23b en-zh-starved en-zh-starved-23b fr-fair fr-fair-15b
        fr-starved fr-starved-15b zh-fair-12b zh-starved-12b)

# --- warm the compile cache sequentially ------------------------------------
# Graph width depends on the TOKENIZER (starved and destarved tokenize to
# different max lengths), so one model per tokenizer covers every shape. Doing
# this first stops the parallel jobs racing on first-compile writes -- and a
# kill mid-compile leaves a .lock that hangs OTHER jobs, including training.
#
# These are FULL runs, deliberately not `--limit`ed. A limited run would defeat
# the purpose twice over: `fixed_width` is derived from the longest sentence in
# the set, so a subset compiles a DIFFERENT (narrower) graph than the real run
# and warms nothing; and it would leave a truncated result JSON that the
# resume-skip below then treats as done. Their output is a normal result and is
# reused, so this costs nothing beyond running those two models first.
for warm in en-fair en-starved; do
  if [ ! -f "$RESULTS/$warm.json" ] || [ ! -f "$EMB/$warm.npz" ]; then
    echo "[fanout] warming compile cache with $warm (full run) ..."
    NEURON_RT_VISIBLE_CORES="${FREE_PAIRS[0]}" OMP_NUM_THREADS=$THREADS \
    OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
      python run_alignment.py --repo "$REPO" --runs "$warm" --device xla \
        --workdir "$WORK" --emb-dir "$EMB" > "$WORK/warm_$warm.log" 2>&1 \
      || echo "[fanout] WARN: warm-up $warm failed, see $WORK/warm_$warm.log"
  fi
done

# --- fan out ----------------------------------------------------------------
i=0
for m in "${MODELS[@]}"; do
  [ -f "$RESULTS/$m.json" ] && [ -f "$EMB/$m.npz" ] && { echo "[fanout] skip $m (already done)"; continue; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 10; done
  pair="${FREE_PAIRS[$(( i % ${#FREE_PAIRS[@]} ))]}"
  echo "[fanout] $m -> cores $pair"
  NEURON_RT_VISIBLE_CORES="$pair" OMP_NUM_THREADS=$THREADS \
  OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    setsid nohup python run_alignment.py --repo "$REPO" --runs "$m" \
      --device xla --workdir "$WORK" --emb-dir "$EMB" > "$WORK/$m.log" 2>&1 < /dev/null &
  i=$(( i + 1 ))
  sleep 3   # stagger checkpoint downloads
done

# `wait` on setsid children is unreliable (CLAUDE.md section 5) -- poll instead.
until ! pgrep -f "run_alignment.py --repo $REPO" > /dev/null; do sleep 15; done

echo "[fanout] done. $(ls "$RESULTS"/*.json 2>/dev/null | wc -l)/${#MODELS[@]} results in $RESULTS"
missing=()
for m in "${MODELS[@]}"; do [ -f "$RESULTS/$m.json" ] || missing+=("$m"); done
[ "${#missing[@]}" -gt 0 ] && echo "[fanout] MISSING: ${missing[*]} (check $WORK/<model>.log)"
echo "[fanout] aggregate with: python analyze_alignment.py $RESULTS"
