#!/usr/bin/env bash
# Resumable, self-verifying uploader loop for the HF export.
#
# Launch it detached from YOUR login shell so it survives logout (no tmux/screen
# on this host):
#
#   cd /home/u6jh/jvonrad.u6jh/XScript-Pretraining/scripts/external_bench
#   setsid nohup bash run_upload_loop.sh \
#       > /scratch/u6jh/jvonrad.u6jh/xscript/upload_loop.log 2>&1 < /dev/null &
#   disown
#   tail -f /scratch/u6jh/jvonrad.u6jh/xscript/upload_loop.log
#
# upload_to_hf.py is resumable (per-file state under _hf_staging/.cache) and
# exits non-zero until all 15 checkpoints are committed, so this loop keeps
# resuming through any interruption and stops only on true completion.
set -u
cd "$(dirname "$0")"
source ../../slurm/env.sh
if ! command -v apptainer >/dev/null 2>&1; then
  module load brics/apptainer-multi-node 2>/dev/null || true
fi
echo "launcher: apptainer=$(command -v apptainer || echo MISSING)"
echo "launcher: container=$CONTAINER"

attempt=0
until apptainer exec --bind "$APPTAINER_BINDS" "$CONTAINER" bash -lc "
    export PYTHONPATH='$XSCRIPT_RUNTIME' HF_TOKEN='$HF_TOKEN'
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt CURL_CA_BUNDLE=\$SSL_CERT_FILE
    cd '$PWD'
    python -u upload_to_hf.py --repo jvonrad/xscript-eval --workers 4
  "
do
  attempt=$((attempt + 1))
  echo ">>> upload incomplete (attempt $attempt); resuming in 10s..."
  sleep 10
done
echo ">>> UPLOAD COMPLETE -- all 15 checkpoints committed"
