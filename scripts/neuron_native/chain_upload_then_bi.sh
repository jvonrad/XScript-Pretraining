#!/bin/bash
# 1) upload the four finished en-X__unigram_20lang runs to HF (roster + full archive),
# 2) then launch the four bilingual-tokenizer runs (en-X__unigram_bi_X) on all 16 chips.
set -u
cd /home/ubuntu/XScript-Pretraining
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; source ~/neuron_venv/bin/activate   # NEURON.md 1 guard
export HF_HUB_ENABLE_HF_TRANSFER=1 XSCRIPT_SCRATCH=/mnt/scratch/xscript
echo "$(date '+%F %T') upload start"
python scripts/external_bench/upload_native_runs.py --repo jvonrad/xscript-eval \
  --runs en-de__unigram_20lang en-fr__unigram_20lang en-ar__unigram_20lang en-zh__unigram_20lang \
  --suffix 20lang --tok unigram_20lang --archive-all > /home/ubuntu/logs/upload_20lang.log 2>&1
echo "$(date '+%F %T') upload exit $? (see logs/upload_20lang.log)"
echo "$(date '+%F %T') launching bi_X runs"
MODELS="en-de__unigram_bi_de en-fr__unigram_bi_fr en-ar__unigram_bi_ar en-zh__unigram_bi_zh" \
  bash scripts/neuron_native/orchestrate_native.sh > /home/ubuntu/logs/orchestrate_bi.log 2>&1
echo "$(date '+%F %T') orchestrator exited"
