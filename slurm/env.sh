#!/bin/bash
# Shared environment for all XScript jobs on Isambard-AI (HPE Cray EX, GH200).
# Source this at the top of every job script: `source "$(dirname "$0")/env.sh"`.
#
# NOTE: Slingshot/NCCL env comes from the brics/apptainer-multi-node module's
# /host/adapt.sh (see below); verified against module v0.3.2 on 2026-07-07.

set -euo pipefail

# --- project / accounting ---
export XS_ACCOUNT="${XS_ACCOUNT:-brics.u6jh}"
export XS_PARTITION="${XS_PARTITION:-workq}"

# --- storage (5TB Lustre scratch) ---
export XSCRIPT_SCRATCH="${XSCRIPT_SCRATCH:-/scratch/u6jh/jvonrad.u6jh/xscript}"
export REPO_DIR="${REPO_DIR:-$HOME/XScript-Pretraining}"
export HF_HOME="${HF_HOME:-$XSCRIPT_SCRATCH/hf_cache}"
export XSCRIPT_RUNTIME="${XSCRIPT_RUNTIME:-$XSCRIPT_SCRATCH/runtime_py312/xscript_lmeval_0.4.12_hf0}"
# Triton JIT links with `-lcuda`, while the NGC compatibility directory on
# Isambard exposes libcuda.so.1 but no unversioned libcuda.so.  Provide both
# names in a scratch-backed directory visible inside the container.
export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-$XSCRIPT_RUNTIME/libcuda}"
mkdir -p "$TRITON_LIBCUDA_PATH"
ln -sfn /usr/local/cuda/compat/lib/libcuda.so.1 "$TRITON_LIBCUDA_PATH/libcuda.so.1"
ln -sfn /usr/local/cuda/compat/lib/libcuda.so.1 "$TRITON_LIBCUDA_PATH/libcuda.so"
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$XSCRIPT_SCRATCH" "$HF_HOME"

# --- container (NGC PyTorch for aarch64/Grace-Hopper) ---
export NGC_IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/pytorch:25.05-py3}"
export CONTAINER="${CONTAINER:-$XSCRIPT_SCRATCH/containers/pytorch_25.05-py3.sif}"

# --- Slingshot / NCCL ---
# Do NOT set NCCL_*/FI_* here. `module load brics/apptainer-multi-node` (in the
# sbatch scripts) bind-mounts the admin-built NCCL 2.26 + aws-ofi-nccl plugin +
# Cray libfabric/CXI into the container at /host, and its /host/adapt.sh wrapper
# exports the authoritative tuned env (NCCL_NET="AWS Libfabric",
# NCCL_SOCKET_IFNAME=hsn, FI_CXI_*, ...) inside the container, overriding
# anything inherited from the host. Duplicating values here would only drift.

# --- apptainer bind set (repo + scratch visible inside the container) ---
export APPTAINER_BINDS="${APPTAINER_BINDS:-$XSCRIPT_SCRATCH:$XSCRIPT_SCRATCH,$REPO_DIR:$REPO_DIR}"

# Run a command inside the container with the repo on PYTHONPATH.
in_container() {
  apptainer exec --nv --bind "$APPTAINER_BINDS" "$CONTAINER" /host/adapt.sh \
    bash -lc "cd '$REPO_DIR' && \
      export PYTHONPATH='$REPO_DIR/src:$XSCRIPT_RUNTIME' && \
      export TRITON_LIBCUDA_PATH='$TRITON_LIBCUDA_PATH' && \
      export LD_LIBRARY_PATH='$TRITON_LIBCUDA_PATH':/usr/local/cuda/lib64:\$LD_LIBRARY_PATH && \
      export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt && \
      export CURL_CA_BUNDLE=\$SSL_CERT_FILE REQUESTS_CA_BUNDLE=\$SSL_CERT_FILE && \
      $*"
}
