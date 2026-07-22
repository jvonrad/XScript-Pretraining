#!/bin/bash
# Final push: train ONLY the two Chinese models to 15B tokens, world=8 each
# (world>8 fails the intra-node collective), then stop. Keep-alive restarts on
# death/hang (resume from last.pt; first ~1B is a warm-start from the 12B ckpt).
set -u
export PROD_TARGET_B=15
PROD=/home/ubuntu/xscript_prod
LOGS=$PROD/logs
RUNS=/mnt/scratch/xscript/runs
ZHFAIR=/home/ubuntu/xscript_bench_ci/xscript/runs/zh-fair-12b/checkpoints/final.pt
ZHSTARV=/home/ubuntu/xscript_bench_ci/xscript/runs/zh-starved-12b/checkpoints/final.pt
HANG_SECS=1200

MODELS=(zh__unigram_destarved zh__unigram_starved)
declare -A CORES PORT WARM DEVS PID DONE
CORES[zh__unigram_destarved]="8,9,10,11,12,13,14,15";  DEVS[zh__unigram_destarved]="2 3"
CORES[zh__unigram_starved]="16,17,18,19,20,21,22,23";  DEVS[zh__unigram_starved]="4 5"
PORT[zh__unigram_destarved]=48712; PORT[zh__unigram_starved]=48713
WARM[zh__unigram_destarved]="$ZHFAIR"; WARM[zh__unigram_starved]="$ZHSTARV"

jsonl() { echo "$RUNS/$1/train.jsonl"; }
nsteps() { wc -l < "$(jsonl "$1")" 2>/dev/null || echo 0; }
kill_devs() { local devs="$1" pid cmd; for d in $devs; do
    for pid in $(neuron-ls --show-all-procs 2>/dev/null | awk -F'|' -v d=" $d " '$2==d{f=1;next} /^\+/{f=0} f' | grep -oE '[0-9]{5,}' | sort -u); do
      cmd=$(ps -o cmd= -p "$pid" 2>/dev/null); case "$cmd" in *"-c from"*|*prod_train*) kill -9 "$pid" 2>/dev/null;; esac
    done; done; }
launch() { local m="$1" before start
  kill_devs "${DEVS[$m]}"; sleep 4
  find /mnt/scratch/xscript_train/neuron-cache -name "*.lock" -delete 2>/dev/null
  before=$(nsteps "$m")
  PROD_TARGET_B=15 setsid bash "$PROD/run_prod.sh" "$m" "${CORES[$m]}" "${PORT[$m]}" "${WARM[$m]}" > "$LOGS/$m.log" 2>&1 < /dev/null &
  PID[$m]=$!
  echo "$(date '+%F %T') LAUNCH $m pid ${PID[$m]} cores ${CORES[$m]} (staggered, target 15B)"
  start=$(date +%s)
  while :; do sleep 20
    [ "$(nsteps "$m")" -gt "$before" ] && { echo "$(date '+%F %T') UP $m"; return 0; }
    kill -0 "${PID[$m]}" 2>/dev/null || { echo "$(date '+%F %T') $m exited during startup"; return 1; }
    [ $(( $(date +%s)-start )) -gt 1500 ] && { echo "$(date '+%F %T') $m slow startup, proceeding"; return 1; }
  done; }

echo "$(date '+%F %T') === zh->15B orchestrator start ==="
for m in "${MODELS[@]}"; do DONE[$m]=0; launch "$m"; done
declare -A LASTN LASTT; for m in "${MODELS[@]}"; do LASTN[$m]=$(nsteps "$m"); LASTT[$m]=$(date +%s); done
while true; do sleep 120; alldone=1; now=$(date +%s)
  for m in "${MODELS[@]}"; do
    [ "${DONE[$m]}" = 1 ] && continue
    if grep -qE "PROD_${m}_DONE|DONE ${m} @" "$LOGS/$m.log" 2>/dev/null; then DONE[$m]=1; echo "$(date '+%F %T') DONE $m (15B reached)"; continue; fi
    alldone=0; n=$(nsteps "$m"); [ "$n" -gt "${LASTN[$m]}" ] && { LASTN[$m]=$n; LASTT[$m]=$now; }
    dead=0; kill -0 "${PID[$m]}" 2>/dev/null || dead=1; hung=0; [ $((now-${LASTT[$m]})) -gt $HANG_SECS ] && hung=1
    if [ "$dead" = 1 ] || [ "$hung" = 1 ]; then
      echo "$(date '+%F %T') $([ $dead = 1 ] && echo DIED || echo HUNG) $m (${n} steps) -- restarting"; launch "$m"; LASTN[$m]=$(nsteps "$m"); LASTT[$m]=$(date +%s)
    fi
  done
  [ "$alldone" = 1 ] && { echo "$(date '+%F %T') === both zh DONE @15B ==="; break; }
done
