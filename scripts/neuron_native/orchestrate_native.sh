#!/bin/bash
# Keep-alive orchestrator for native-stack runs: model i -> chips 4i..4i+3
# (cores 16i..16i+15, world=16). Staggered launch, restart on death/hang
# (resume from last.pt), kill by run-name pattern INSIDE the container.
set -u
export PATH="/opt/aws/neuron/bin:$PATH"
REPO=/home/ubuntu/XScript-Pretraining; LOGS=${LOGS:-/home/ubuntu/logs/train_native}
RUNS=${XSCRIPT_SCRATCH:-/mnt/scratch/xscript}/runs
HANG_SECS=${HANG_SECS:-1800}; STARTUP_SECS=${STARTUP_SECS:-2400}; CHIPS=${CHIPS:-4}
mkdir -p "$LOGS"; : "${MODELS:?}"; read -r -a MODELS <<<"$MODELS"
declare -A CORES PORT PID DONE LASTN LASTT
i=0; for m in "${MODELS[@]}"; do c0=$((16*i/4*CHIPS)); CORES[$m]="$((CHIPS*4*i))-$((CHIPS*4*i+CHIPS*4-1))"; PORT[$m]=$((29500+i)); DONE[$m]=0; i=$((i+1)); done
nsteps() { wc -l < "$RUNS/$1/train.jsonl" 2>/dev/null || echo 0; }
kill_run() { sudo docker exec neuron-native bash -c "pkill -9 -f 'train_native.py $1 ' ; pkill -9 -f 'rdzv_endpoint localhost:${PORT[$1]} '" 2>/dev/null; sleep 8; }
launch() { local m="$1" before start; kill_run "$m"; before=$(nsteps "$m")
  setsid bash "$REPO/scripts/neuron_native/run_native.sh" "$m" "${CORES[$m]}" "${PORT[$m]}" >> "$LOGS/$m.log" 2>&1 < /dev/null &
  PID[$m]=$!; echo "$(date '+%F %T') LAUNCH $m pid ${PID[$m]} cores ${CORES[$m]} port ${PORT[$m]}"; start=$(date +%s)
  while :; do sleep 20
    [ "$(nsteps "$m")" -gt "$before" ] && { echo "$(date '+%F %T') UP $m"; return 0; }
    kill -0 "${PID[$m]}" 2>/dev/null || { echo "$(date '+%F %T') $m exited during startup"; return 1; }
    [ $(( $(date +%s)-start )) -gt $STARTUP_SECS ] && { echo "$(date '+%F %T') $m slow startup, proceeding"; return 1; }
  done; }
echo "$(date '+%F %T') === native orchestrator: ${MODELS[*]} ==="
for m in "${MODELS[@]}"; do launch "$m"; done
for m in "${MODELS[@]}"; do LASTN[$m]=$(nsteps "$m"); LASTT[$m]=$(date +%s); done
while true; do sleep 120; alldone=1; now=$(date +%s)
  for m in "${MODELS[@]}"; do [ "${DONE[$m]}" = 1 ] && continue
    grep -q "DONE $m @" "$LOGS/$m.log" 2>/dev/null && { DONE[$m]=1; echo "$(date '+%F %T') DONE $m"; continue; }
    alldone=0; n=$(nsteps "$m"); [ "$n" -gt "${LASTN[$m]}" ] && { LASTN[$m]=$n; LASTT[$m]=$now; }
    dead=0; kill -0 "${PID[$m]}" 2>/dev/null || dead=1; hung=0; [ $((now-${LASTT[$m]})) -gt $HANG_SECS ] && hung=1
    if [ "$dead" = 1 ] || [ "$hung" = 1 ]; then echo "$(date '+%F %T') $([ $dead = 1 ] && echo DIED || echo HUNG) $m ($n steps) -- restarting"; launch "$m"; LASTN[$m]=$(nsteps "$m"); LASTT[$m]=$(date +%s); fi
  done; [ "$alldone" = 1 ] && { echo "$(date '+%F %T') === ALL DONE ==="; break; }
done
