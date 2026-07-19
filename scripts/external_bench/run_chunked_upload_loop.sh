#!/usr/bin/env bash
# Drives the chunked upload: launches a FRESH short-lived process per chunk
# (see upload_one_part.py for why -- the login node kills a process after
# ~2.5-3GB of cumulative transfer *within that process*, not by wall-clock).
# This outer loop does no data transfer itself, so it can run indefinitely.
#
# Idempotent/resumable: re-running just skips whatever's already committed.
#
#   cd scripts/external_bench
#   setsid nohup bash run_chunked_upload_loop.sh \
#       > /scratch/u6jh/jvonrad.u6jh/xscript/chunked_upload.log 2>&1 < /dev/null &
#   disown
#   tail -f /scratch/u6jh/jvonrad.u6jh/xscript/chunked_upload.log
set -u
cd "$(dirname "$0")"
source ../../slurm/env.sh
command -v apptainer >/dev/null 2>&1 || module load brics/apptainer-multi-node 2>/dev/null || true

REPO=jvonrad/xscript-eval
MAX_PARTS=8   # generous upper bound (4.36GB / 900MB ~= 5 parts); extra indices are cheap no-ops

run_part() {
  local model="$1" idx="$2"
  apptainer exec --bind "$APPTAINER_BINDS" "$CONTAINER" bash -lc "
    export PYTHONPATH='$XSCRIPT_RUNTIME' HF_TOKEN='$HF_TOKEN'
    cd '$PWD'
    python -u upload_one_part.py --repo $REPO --model $model --index $idx
  "
}

models=$(ls "$XSCRIPT_SCRATCH/_hf_staging/runs")
echo "models: $models"

for model in $models; do
  for idx in $(seq 0 $((MAX_PARTS - 1))); do
    tries=0
    until run_part "$model" "$idx"; do
      tries=$((tries + 1))
      if [ "$tries" -ge 5 ]; then
        echo ">>> $model part $idx failed $tries times, giving up on it for now"
        break
      fi
      echo ">>> $model part $idx: retry $tries in 5s..."
      sleep 5
    done
  done
done

echo ">>> chunk pass complete. Run verify_upload.py to confirm everything landed."
