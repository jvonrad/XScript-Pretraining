#!/usr/bin/env bash
# Fan out run_appendix_c5.py across FREE Neuron core-pairs, leaving any running
# training job untouched.
#
#   bash run_c5_fanout.sh [WORKDIR]
#
# Like run_alignment_fanout.sh this discovers free devices at runtime from
# neuron-ls (devices with no PID) rather than assuming an idle box, so a
# concurrent training run -- which pins itself via NEURON_RT_VISIBLE_CORES -- is
# never scheduled over.
#
# Unlike that script it uses a DYNAMIC work queue instead of round-robin
# assignment. With ~80 models of near-identical cost over 24 slots, a static
# split leaves some slots with 4 models and others with 3, so the sweep runs
# ~33% longer than the work requires. Workers pop from a shared queue under
# flock instead, so every slot stays busy until the queue drains.
#
# Assumes checkpoints were already fetched by prefetch_checkpoints.py; if one is
# missing the eval process downloads it itself, just more slowly.
set -uo pipefail

WORK="${1:-/mnt/scratch/xscript_c5}"
REPO="${REPO:-jvonrad/xscript-eval}"
BATCH="${BATCH:-8}"              # Belebele's long passages: keep <= 8 (CLAUDE.md s4)
BATCH_SHORT="${BATCH_SHORT:-32}"
THREADS="${THREADS:-4}"          # host threads per job; JOBS * THREADS <= nproc

command -v neuron-ls >/dev/null || { echo "ERROR: activate ~/neuron_venv first"; exit 1; }
cd "$(dirname "$0")" || exit 1

RESULTS="$WORK/results/appendix_c5"
LOGS="$WORK/logs"
mkdir -p "$RESULTS" "$LOGS"

# --- free devices -> logical core-pairs -------------------------------------
# neuron-ls columns: | device | cores | core-ids | memory | connected | bdf | PID
# A device with PID "NA" is free. Each device is 4 consecutive PHYSICAL core ids
# (0-63); under logical-neuroncore-config 2 that is two pinnable pairs.
mapfile -t FREE_PAIRS < <(
  neuron-ls | awk -F'|' '/^\| [0-9]+ /{
      gsub(/ /,"",$4); gsub(/ /,"",$8);
      if ($8 == "NA") { split($4, r, "-"); print r[1]"-"(r[1]+1); print (r[1]+2)"-"(r[1]+3); }
  }'
)
[ "${#FREE_PAIRS[@]}" -gt 0 ] || { echo "ERROR: no free Neuron devices"; exit 1; }
echo "[c5] ${#FREE_PAIRS[@]} free core-pairs: ${FREE_PAIRS[*]}"

# --- build the queue --------------------------------------------------------
# Skip runs that already have a SUCCESSFUL result. A failed eval also writes
# <run>_final.json, but with an "error" key -- those are requeued.
mapfile -t ALL < <(python - "$WORK" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
m = json.loads(Path(hf_hub_download("jvonrad/xscript-eval", "models.json")).read_text())
rdir = Path(sys.argv[1]) / "results/appendix_c5"
done = set()
if rdir.is_dir():
    for f in rdir.glob("*_final.json"):
        try:
            if "error" not in json.loads(f.read_text()):
                done.add(f.name.removesuffix("_final.json"))
        except Exception:
            pass
for r in sorted(m):
    if r not in done:
        print(r)
PY
)
[ "${#ALL[@]}" -gt 0 ] || { echo "[c5] nothing to do -- all models already evaluated"; exit 0; }
echo "[c5] ${#ALL[@]} model(s) to evaluate"

QUEUE="$WORK/.queue"
LOCK="$WORK/.queue.lock"
printf '%s\n' "${ALL[@]}" > "$QUEUE"
: > "$LOCK"

# Atomically pop the first line of the queue. flock serialises the read+rewrite
# so two workers can never claim the same model.
pop() {
  flock "$LOCK" bash -c '
    q="$1"
    [ -s "$q" ] || exit 1
    head -n1 "$q"
    tail -n +2 "$q" > "$q.new" && mv "$q.new" "$q"
  ' _ "$QUEUE"
}

worker() {
  local pair="$1" m
  while m="$(pop)"; do
    [ -n "$m" ] || break
    echo "[c5] $(date +%H:%M:%S) cores $pair -> $m"
    # HF_HUB_OFFLINE: every input is already local (checkpoints prefetched,
    # datasets + tokenizers cached), and lm_eval still HEADs the hub for each
    # benchmark dataset even when cached -- 24 of those at once is what 429ed
    # the first attempt. Offline makes the whole sweep network-free.
    #
    # --keep-checkpoints is REQUIRED alongside it: without it a failed eval
    # deletes its own checkpoint in a finally block, and offline mode can never
    # refetch it, so a retry is impossible. Disk is already sized for all 81.
    NEURON_RT_VISIBLE_CORES="$pair" \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
      python run_appendix_c5.py --repo "$REPO" --runs "$m" --device xla \
        --batch-size "$BATCH" --batch-size-short "$BATCH_SHORT" \
        --keep-checkpoints --workdir "$WORK" > "$LOGS/$m.log" 2>&1 \
      || echo "[c5] WARN: $m exited nonzero, see $LOGS/$m.log"
    # Exit code alone is NOT enough: run_appendix_c5.py catches a per-model
    # exception, writes {"error": ...} into the result JSON, and still exits 0.
    # Without this check a failed model is indistinguishable from a passing one
    # in the fan-out log (9 real failures once showed up as "WARN: 0").
    if grep -q '"error"' "$RESULTS/${m}_final.json" 2>/dev/null; then
      echo "[c5] WARN: $m FAILED -> $(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('error','')[:120])" "$RESULTS/${m}_final.json" 2>/dev/null)"
    elif [ ! -f "$RESULTS/${m}_final.json" ]; then
      echo "[c5] WARN: $m produced NO result JSON, see $LOGS/$m.log"
    fi
  done
  echo "[c5] cores $pair drained"
}

# Pre-warm the cached repo listing with ONE request. Every worker needs it at
# startup; if they all start cold they issue 24 simultaneous repo_info calls and
# the hub 429s the lot (this is what killed 47 jobs on the first attempt).
if [ ! -s "$WORK/_repo_files.json" ]; then
  echo "[c5] warming repo listing ..."
  python - "$REPO" "$WORK" <<'PY'
import json, os, sys
from huggingface_hub import HfApi
repo, work = sys.argv[1], sys.argv[2]
sizes = {s.rfilename: (s.size or 0)
         for s in HfApi().repo_info(repo, files_metadata=True).siblings}
tmp = os.path.join(work, "_repo_files.json.tmp")
open(tmp, "w").write(json.dumps(sizes))
os.replace(tmp, os.path.join(work, "_repo_files.json"))
print(f"[c5] cached listing: {len(sizes)} files")
PY
fi

# Stagger starts: even with the listing cached each worker still HEADs the src/
# and tokenizer files, so 24 at once is a burst the hub throttles. A few seconds
# apart costs nothing against ~3h of eval per model.
for pair in "${FREE_PAIRS[@]}"; do
  worker "$pair" &
  sleep 4
done

# `wait` on backgrounded children is unreliable when this script is itself
# detached (CLAUDE.md s5), so poll for the queue to drain AND the workers to go
# quiet rather than trusting it.
wait
echo "[c5] all workers drained at $(date)"
echo "[c5] results: $RESULTS ($(ls "$RESULTS"/*_final.json 2>/dev/null | wc -l) JSONs)"
