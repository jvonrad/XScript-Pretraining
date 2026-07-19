#!/bin/bash
# Recover the 16 live 2026-07-15 allocations whose original supervisors
# inherited a login-only /local/user/$UID Apptainer bind.  Run detached from a
# login node; each replacement srun stays inside its already-allocated job.
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

module unload brics/apptainer-multi-node 2>/dev/null || true
unset APPTAINER_BINDPATH SINGULARITY_BINDPATH
unset APPTAINERENV_TMPDIR SINGULARITYENV_TMPDIR
export TMPDIR=/tmp
export APPTAINER_CACHEDIR=/tmp/apptainer-$UID
module load brics/apptainer-multi-node
source slurm/env.sh
# env.sh enables errexit; this controller must inspect failed srun statuses and
# retry them rather than exiting on the first non-zero status.
set +e

jobs=(
  "5656655 en-ar__unigram_destarved"
  "5656654 en-ar__unigram_starved"
  "5656653 en-de__unigram_destarved"
  "5656649 en-de__unigram_starved"
  "5657088 de__unigram_destarved"
  "5657089 de__unigram_starved"
  "5657090 en-fr__unigram_destarved"
  "5657091 en-fr__unigram_starved"
  "5657092 en-zh__unigram_destarved"
  "5657093 en-zh__unigram_starved"
  "5657094 en__unigram_destarved"
  "5657095 en__unigram_starved"
  "5657096 fr__unigram_destarved"
  "5657097 fr__unigram_starved"
  "5657098 zh__unigram_destarved"
  "5657099 zh__unigram_starved"
)

run_one() {
  local jid=$1 run=$2 attempt=0 rc remaining master end_iso end_epoch port
  local log="logs/recover-${run}-${jid}.out"
  echo "[$(date --iso-8601=seconds)] attach $run to allocation $jid" >> "$log"

  while true; do
    if [ "$(squeue -h -j "$jid" -o '%T')" != RUNNING ]; then
      echo "[$(date --iso-8601=seconds)] allocation no longer running" >> "$log"
      return 1
    fi
    end_iso=$(squeue -h -j "$jid" -o '%e')
    end_epoch=$(date -d "$end_iso" +%s)
    remaining=$((end_epoch - $(date +%s)))
    if (( remaining <= 600 )); then
      echo "[$(date --iso-8601=seconds)] only ${remaining}s remain" >> "$log"
      return 75
    fi

    master=$(scontrol show hostnames "$(squeue -h -j "$jid" -o '%N')" | head -n1)
    attempt=$((attempt + 1))
    port=$((20000 + (jid + 1000 + attempt) % 30000))
    echo "[$(date --iso-8601=seconds)] attempt=$attempt remaining=${remaining}s master=$master port=$port" >> "$log"

    srun --jobid="$jid" --overlap --kill-on-bad-exit=1 \
      --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=72 \
      apptainer exec --nv --bind "$APPTAINER_BINDS" "$CONTAINER" /host/adapt.sh bash -lc "
        cd '$REPO_DIR'
        export PYTHONPATH='$REPO_DIR/src:$XSCRIPT_RUNTIME'
        export XSCRIPT_SCRATCH='$XSCRIPT_SCRATCH' HF_HOME='$HF_HOME'
        export TRITON_LIBCUDA_PATH='$TRITON_LIBCUDA_PATH'
        export LD_LIBRARY_PATH='$TRITON_LIBCUDA_PATH':/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
        export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
        export CURL_CA_BUNDLE=\$SSL_CERT_FILE REQUESTS_CA_BUNDLE=\$SSL_CERT_FILE
        torchrun --nnodes=2 --nproc_per_node=4 --node_rank=\$SLURM_PROCID \
          --rdzv_backend=c10d --rdzv_endpoint=$master:$port \
          -m xscript.cli train '$run' --base configs/base_main.yaml \
          --flavor unigram --only-30b
      " >> "$log" 2>&1
    rc=$?
    if (( rc == 0 )); then
      echo "[$(date --iso-8601=seconds)] training complete" >> "$log"
      return 0
    fi
    echo "[$(date --iso-8601=seconds)] failed rc=$rc; retrying in 30s" >> "$log"
    sleep 30
  done
}

pids=()
for spec in "${jobs[@]}"; do
  read -r jid run <<< "$spec"
  run_one "$jid" "$run" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
exit "$rc"
