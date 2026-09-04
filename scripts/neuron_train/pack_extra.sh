#!/bin/bash
# Pack every (language, corroboration tokenizer) pair needed for the extra runs:
#   unigram_50lang  x {en,de,fr,ar,zh}   and   unigram_bi_X x {en,X}
# 13 packs, all concurrent (13 x WORKERS processes; size WORKERS for the box).
# Pools must be complete (stats.json text_bytes >= budget or exhausted) or the
# packer holds back the last shard -- re-run later to pick it up.
set -u
export XSCRIPT_SCRATCH=${XSCRIPT_SCRATCH:-/mnt/scratch/xscript}
WORKERS=${WORKERS:-14}
LOGS=${LOGS:-/home/ubuntu/logs/pack}; mkdir -p "$LOGS"
PY=${PY:-$HOME/dataprep_venv/bin/python}
cd /home/ubuntu/XScript-Pretraining
pairs=()
for L in en de fr ar zh; do pairs+=("$L unigram_50lang"); done
for X in de fr ar zh; do pairs+=("en unigram_bi_$X" "$X unigram_bi_$X"); done
for p in "${pairs[@]}"; do set -- $p
  nohup $PY -u -m xscript.cli pack --lang $1 --tok $2 --workers $WORKERS > "$LOGS/pack_$1__$2.log" 2>&1 &
  echo "pack $1 $2 pid $!"
done
wait
echo "PACK_ALL_DONE"
