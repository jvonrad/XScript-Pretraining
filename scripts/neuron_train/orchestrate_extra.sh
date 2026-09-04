#!/bin/bash
# Keep-alive orchestrator for the corroboration-tokenizer runs on a
# trn2.48xlarge: N models, each world=8 on its own pair of devices
# (physical cores 8k..8k+7), staggered launch (each model compiles alone so the
# shared NEFF cache is warm for the next), restart on death or hang (resume
# from last.pt with the ZeRO optimizer shards). Modeled on orchestrate_zh15.sh.
#
# Usage:  MODELS="en-de__unigram_50lang en-fr__unigram_50lang ..." \
#         bash scripts/neuron_train/orchestrate_extra.sh
# Models are assigned device pairs in order: model i -> cores 8i..8i+7.
set -u
export PATH="/opt/aws/neuron/bin:$PATH"
REPO=/home/ubuntu/XScript-Pretraining
LOGS=${LOGS:-/home/ubuntu/logs/train}
RUNS=${XSCRIPT_SCRATCH:-/mnt/scratch/xscript}/runs
CACHE=${WORK:-/mnt/scratch/xscript_train}/neuron-cache
HANG_SECS=${HANG_SECS:-1800}
STARTUP_SECS=${STARTUP_SECS:-2400}
mkdir -p "$LOGS"
: "${MODELS:?set MODELS=\"run1 run2 ...\"}"
read -r -a MODELS <<<"$MODELS"

declare -A CORES PORT DEVS PID DONE LASTN LASTT
i=0
for m in "${MODELS[@]}"; do
  c0=$((8*i)); CORES[$m]=$(seq -s, $c0 $((c0+7))); DEVS[$m]="$((2*i)) $((2*i+1))"
  PORT[$m]=$((48711+i)); DONE[$m]=0; i=$((i+1))
done

jsonl() { echo "$RUNS/$1/train.jsonl"; }
nsteps() { wc -l < "$(jsonl "$1")" 2>/dev/null || echo 0; }
kill_devs() { local devs="$1" pid cmd; for d in $devs; do
    for pid in $(neuron-ls --show-all-procs 2>/dev/null | awk -F'|' -v d=" $d " '$2==d{f=1;next} /^\+/{f=0} f' | grep -oE '[0-9]{4,}' | sort -u); do
      cmd=$(ps -o cmd= -p "$pid" 2>/dev/null); case "$cmd" in *"-c from"*|*prod_extra*) kill -9 "$pid" 2>/dev/null;; esac
    done; done; }
launch() { local m="$1" before start
  kill_devs "${DEVS[$m]}"; sleep 6
  find "$CACHE" -name "*.lock" -delete 2>/dev/null
  before=$(nsteps "$m")
  setsid bash "$REPO/scripts/neuron_train/run_extra.sh" "$m" "${CORES[$m]}" "${PORT[$m]}" >> "$LOGS/$m.log" 2>&1 < /dev/null &
  PID[$m]=$!
  echo "$(date '+%F %T') LAUNCH $m pid ${PID[$m]} cores ${CORES[$m]} port ${PORT[$m]}"
  start=$(date +%s)
  while :; do sleep 20
    [ "$(nsteps "$m")" -gt "$before" ] && { echo "$(date '+%F %T') UP $m ($(nsteps "$m") logged steps)"; return 0; }
    kill -0 "${PID[$m]}" 2>/dev/null || { echo "$(date '+%F %T') $m exited during startup (see $LOGS/$m.log)"; return 1; }
    [ $(( $(date +%s)-start )) -gt $STARTUP_SECS ] && { echo "$(date '+%F %T') $m slow startup, proceeding"; return 1; }
  done; }

echo "$(date '+%F %T') === orchestrator start: ${MODELS[*]} ==="
for m in "${MODELS[@]}"; do launch "$m"; done
for m in "${MODELS[@]}"; do LASTN[$m]=$(nsteps "$m"); LASTT[$m]=$(date +%s); done
while true; do sleep 120; alldone=1; now=$(date +%s)
  for m in "${MODELS[@]}"; do
    [ "${DONE[$m]}" = 1 ] && continue
    if grep -qE "PROD_${m}_DONE|DONE ${m} @" "$LOGS/$m.log" 2>/dev/null; then DONE[$m]=1; echo "$(date '+%F %T') DONE $m"; continue; fi
    alldone=0; n=$(nsteps "$m"); [ "$n" -gt "${LASTN[$m]}" ] && { LASTN[$m]=$n; LASTT[$m]=$now; }
    dead=0; kill -0 "${PID[$m]}" 2>/dev/null || dead=1; hung=0; [ $((now-${LASTT[$m]})) -gt $HANG_SECS ] && hung=1
    if [ "$dead" = 1 ] || [ "$hung" = 1 ]; then
      echo "$(date '+%F %T') $([ $dead = 1 ] && echo DIED || echo HUNG) $m (${n} steps) -- restarting"; launch "$m"; LASTN[$m]=$(nsteps "$m"); LASTT[$m]=$(date +%s)
    fi
  done
  [ "$alldone" = 1 ] && { echo "$(date '+%F %T') === ALL DONE ==="; break; }
done
