#!/usr/bin/env bash
# Fan out run_alignment.py across FREE Neuron core-pairs, dynamic work queue.
#
#   bash run_align_fanout2.sh [WORKDIR] [EMB_DIR]
#
# Differs from run_alignment_fanout.sh in three ways, all learned from the C.5
# sweep:
#   * model list comes from models.json, not a hardcoded 26-name array
#   * dynamic work queue instead of round-robin, so no slot idles at the tail
#   * runs fully offline: checkpoints, tokenizers and FLORES are already local,
#     and hitting the hub from N processes at once is what 429ed the C.5 launch
#
# Free devices are discovered from neuron-ls at runtime, so a concurrent
# training job (which pins itself via NEURON_RT_VISIBLE_CORES) is never
# scheduled over. Resumable: models with a result JSON are skipped.
set -uo pipefail

WORK="${1:-/mnt/scratch/xscript_c5}"
EMB="${2:-/mnt/scratch/xscript_align/embeddings}"
REPO="${REPO:-jvonrad/xscript-eval}"
# Host threads per job. The retrieval/CKA phase is numpy-multithreaded and will
# otherwise grab all `nproc` per process, starving a concurrent trainer's
# dataloaders even though the Neuron cores are disjoint. Keep JOBS*THREADS<=nproc.
THREADS="${THREADS:-8}"

command -v neuron-ls >/dev/null || { echo "ERROR: activate ~/neuron_venv first"; exit 1; }
cd "$(dirname "$0")" || exit 1
RESULTS="$WORK/results/alignment"; LOGS="$WORK/logs_align"
mkdir -p "$RESULTS" "$LOGS" "$EMB"

mapfile -t FREE_PAIRS < <(
  neuron-ls | awk -F'|' '/^\| [0-9]+ /{
      gsub(/ /,"",$4); gsub(/ /,"",$8);
      if ($8 == "NA") { split($4, r, "-"); print r[1]"-"(r[1]+1); print (r[1]+2)"-"(r[1]+3); }
  }'
)
[ "${#FREE_PAIRS[@]}" -gt 0 ] || { echo "ERROR: no free Neuron devices"; exit 1; }
echo "[align] ${#FREE_PAIRS[@]} free core-pairs: ${FREE_PAIRS[*]}"

mapfile -t TODO < <(python3 - "$WORK" <<'PY'
import json, sys, os, glob
work = sys.argv[1]
models = sorted(json.load(open(os.path.join(work, "_repo", "models.json"))))
done = {os.path.basename(f).removesuffix(".json")
        for f in glob.glob(os.path.join(work, "results/alignment", "*.json"))}
for m in models:
    if m not in done:
        print(m)
PY
)
[ "${#TODO[@]}" -gt 0 ] || { echo "[align] nothing to do"; exit 0; }
echo "[align] ${#TODO[@]} model(s) to embed"

QUEUE="$WORK/.aqueue"; LOCK="$WORK/.aqueue.lock"
printf '%s\n' "${TODO[@]}" > "$QUEUE"; : > "$LOCK"

pop() {
  flock "$LOCK" bash -c '
    q="$1"; [ -s "$q" ] || exit 1
    head -n1 "$q"; tail -n +2 "$q" > "$q.new" && mv "$q.new" "$q"' _ "$QUEUE"
}

worker() {
  local pair="$1" m
  while m="$(pop)"; do
    [ -n "$m" ] || break
    echo "[align] $(date +%H:%M:%S) cores $pair -> $m"
    NEURON_RT_VISIBLE_CORES="$pair" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
      python run_alignment.py --repo "$REPO" --runs "$m" --device xla \
        --workdir "$WORK" --emb-dir "$EMB" > "$LOGS/$m.log" 2>&1 \
      || echo "[align] WARN: $m exited nonzero, see $LOGS/$m.log"
    # exit code is not sufficient on its own -- check the artifact actually landed
    [ -f "$RESULTS/$m.json" ] || echo "[align] WARN: $m produced NO result JSON"
  done
  echo "[align] cores $pair drained"
}

for pair in "${FREE_PAIRS[@]}"; do worker "$pair" & sleep 2; done
wait
echo "[align] all workers drained at $(date)"
echo "[align] results: $(ls "$RESULTS"/*.json 2>/dev/null | wc -l) JSONs"
