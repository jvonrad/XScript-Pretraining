#!/bin/bash
#SBATCH --job-name=xs-hold
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/hold-%j.out
# Reserve one node (4 GH200) for 24h and do nothing, so you can srun
# interactive work into it from any login session. The reservation survives
# your ssh session dropping because the sleep runs on the compute node.
#
# Usage:
#   sbatch --account=brics.u6jh --partition=workq slurm/99_hold.sbatch
#
# Then, from any login node:
#   JOB=$(squeue --me --name=xs-hold -h -o %i)   # the hold job id
#   srun --jobid=$JOB --overlap --pty bash       # interactive shell on the node
#
# Free it when done (don't leave 4x GH200 idle):
#   scancel $JOB
mkdir -p logs
echo "hold: node=$SLURM_JOB_NODELIST job=$SLURM_JOB_ID until $(date -d '+24 hours')"
sleep 24h
