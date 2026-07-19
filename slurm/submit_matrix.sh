#!/bin/bash
# Submit the full training matrix. Standard 30B runs go straight to the queue;
# extended cells submit a trunk first and make their 30B/100B cooldown branches
# depend on it (afterok), since a cooldown reads the trunk's stable checkpoint.
#
# Usage: FLAVOR=unigram NODES=2 bash slurm/submit_matrix.sh
source "$(dirname "$0")/env.sh"
FLAVOR="${FLAVOR:-unigram}"
NODES="${NODES:-2}"
BASE="${BASE:-configs/base_main.yaml}"
SB="sbatch --account=$XS_ACCOUNT --partition=$XS_PARTITION --nodes=$NODES"

declare -A trunk_job
# trunks first
for RUN in $(xscript runs --base "$BASE" --flavor "$FLAVOR"); do
  case "$RUN" in
    *__trunk)
      jid=$($SB --job-name="$RUN" \
             --export=ALL,RUN="$RUN",FLAVOR="$FLAVOR",BASE="$BASE" \
             slurm/30_train.sbatch | awk '{print $NF}')
      trunk_job["$RUN"]=$jid
      echo "trunk  $RUN -> job $jid" ;;
  esac
done
# everything else
for RUN in $(xscript runs --base "$BASE" --flavor "$FLAVOR"); do
  case "$RUN" in
    *__trunk) continue ;;
  esac
  dep=""
  # extended deliverables (name has a matching __trunk) wait on that trunk
  mix_tok="${RUN%%__100b}"                       # strip 100b suffix if present
  base_mix_tok="${mix_tok}"
  tj="${trunk_job[${base_mix_tok}__trunk]:-}"
  [ -n "$tj" ] && dep="--dependency=afterok:$tj"
  jid=$($SB $dep --job-name="$RUN" \
         --export=ALL,RUN="$RUN",FLAVOR="$FLAVOR",BASE="$BASE" \
         slurm/30_train.sbatch | awk '{print $NF}')
  echo "run    $RUN -> job $jid ${dep}"
done
echo "submitted. monitor with: squeue -u \$USER"
