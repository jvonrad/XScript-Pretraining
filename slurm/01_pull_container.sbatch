#!/bin/bash
#SBATCH --job-name=xs-container
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/container-%j.out
# Batch-job version of slurm/00_pull_container.sh.
#
# The .sh runs on a login node, where systemd kills it as soon as the last
# session closes (`loginctl show-user` -> Linger=no, enable-linger denied).
# A ~20min apptainer build reliably outlives an interactive session only by
# luck; one attempt on 2026-08-02 died mid-build. Compute nodes have internet
# (verified, job 5878501), so build there instead.
#
# The login-node script's mksquashfs throttling (-processors 4 -mem 1G) exists
# purely to fit the 4GiB/500-task interactive cgroup. A compute node has no
# such cap, so this uses the defaults and is faster.
#
# Usage:
#   sbatch --account=brics.u6sg --partition=workq slurm/01_pull_container.sbatch
set -euo pipefail

XS_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$XS_DIR"
mkdir -p logs
source slurm/env.sh

# Keep apptainer's layer cache off $HOME (small quota) and off /tmp (node-local,
# wiped between jobs -- fine here, the build is one shot).
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$XSCRIPT_SCRATCH/apptainer_cache}"
mkdir -p "$APPTAINER_CACHEDIR" "$(dirname "$CONTAINER")"

if [ -f "$CONTAINER" ]; then
  echo "container already present: $CONTAINER"
  exit 0
fi

echo "host=$(hostname) start=$(date --iso-8601=seconds)"
echo "building $NGC_IMAGE -> $CONTAINER"
apptainer build "$CONTAINER" "docker://${NGC_IMAGE}"

echo "end=$(date --iso-8601=seconds)"
ls -la "$CONTAINER"
