#!/bin/bash
# One-time: pull the NGC PyTorch image to a local .sif on scratch.
# Run on a login node (needs internet). ~15-25 min the first time.
source "$(dirname "$0")/env.sh"

mkdir -p "$(dirname "$CONTAINER")"
if [[ -f "$CONTAINER" ]]; then
  echo "container already present: $CONTAINER"
  exit 0
fi
echo "pulling $NGC_IMAGE -> $CONTAINER"
# Login-node sessions here run inside a systemd scope capped at TasksMax=500
# and MemoryMax=4GiB (`systemctl show user-$(id -u).slice`), regardless of
# the node's real 144 CPUs / 237GB RAM. mksquashfs (used internally to build
# the SIF) defaults to one thread per detected CPU with unlimited memory
# (`mksquashfs procs = 0`, `mksquashfs mem = Unlimited` in apptainer.conf,
# admin-set, not ours to change), so unconstrained it blows past both caps:
# first "FATAL ERROR: Failed to create thread" (TasksMax), then a silent
# cgroup OOM-kill ("signal: killed", not visible in dmesg - that's
# access-restricted for non-root here). Cap both via mksquashfs's own flags,
# leaving headroom under 4GB for apptainer's other work in the same cgroup.
MKSQUASHFS_PROCS="${MKSQUASHFS_PROCS:-4}"
MKSQUASHFS_MEM="${MKSQUASHFS_MEM:-1G}"
# `--mksquashfs-args` only exists on `apptainer build`, not `pull` (pull has
# no way to pass it through) - `build` accepts the same docker:// source and
# produces the same SIF, so it's a direct substitute here.
# apptainer 1.4.1's multi-layer progress-bar renderer panics
# ("index out of range" in progress_roundtrip.go) on images with enough
# concurrent layers, which NGC's PyTorch image has plenty of. `--quiet`
# (must precede the subcommand - it's a global flag) skips that renderer.
apptainer --quiet build --mksquashfs-args="-processors $MKSQUASHFS_PROCS -mem $MKSQUASHFS_MEM" "$CONTAINER" "docker://$NGC_IMAGE"
echo "done."
