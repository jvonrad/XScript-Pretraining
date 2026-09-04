#!/bin/bash
# Launch ONE native-stack training run inside the `neuron-native` container.
# args: $1=RUN  $2=CORES (e.g. 0-15 = 4 chips)  $3=port  [$4=extra args]
# Host side: docker exec; the container has /repo (this repo) and /mnt/scratch.
RUN=$1; CORES=$2; PORT=$3; EXTRA=${4:-}
NPROC=$(( $(echo $CORES | cut -d- -f2) - $(echo $CORES | cut -d- -f1) + 1 ))
# Pin the job's ranks to the NUMA node its chips hang off (neuron-ls CPU AFFINITY on this
# trn2.48xlarge: devices 0-3 and 12-15 -> cpus 48-95,144-191; devices 4-11 -> 0-47,96-143)
# and bound host threads: unpinned, four concurrent 4-chip jobs fell from 133k to ~80k tok/s.
FIRST_CHIP=$(( $(echo $CORES | cut -d- -f1) / 4 ))
if [ $FIRST_CHIP -ge 4 ] && [ $FIRST_CHIP -le 11 ]; then CPUS=0-47,96-143; else CPUS=48-95,144-191; fi
OMP=${OMP_NUM_THREADS:-2}
WANDB=$(grep WANDB_API_KEY /home/ubuntu/xscript_prod/wandb_env.sh 2>/dev/null | cut -d= -f2)
exec sudo docker exec -e NEURON_RT_NUM_CORES=$NPROC -e NEURON_RT_VISIBLE_CORES=$CORES \
  -e NEURON_RT_ROOT_COMM_ID=127.0.0.1:$((PORT+1000)) \
  -e TORCH_NEURONX_SEGMENT_ALLOCATOR_CONFIG='kMaxSplitSizeBytes=268435456,kMaxNonSplitRoundingBytes=67108864' \
  -e NEURON_RT_DISABLE_EXECUTION_BARRIER=1 -e XSCRIPT_SCRATCH=/mnt/scratch/xscript \
  -e NEURON_COMPILE_CACHE_URL=/mnt/scratch/neff-cache \
  -e WANDB_API_KEY="$WANDB" -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=$OMP -e MKL_NUM_THREADS=$OMP -w /repo neuron-native \
  taskset -c $CPUS torchrun --nproc_per_node $NPROC --rdzv_backend c10d --rdzv_endpoint localhost:$PORT \
    scripts/neuron_native/train_native.py $RUN --only-30b $EXTRA
