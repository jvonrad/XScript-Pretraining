#!/bin/bash
#SBATCH --job-name=xs-de-data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=08:00:00
#SBATCH --output=logs/de-data-%j.out
# German pool + pack for the de__unigram_starved retrain, in ONE job.
#
# WHY A JOB AND NOT THE LOGIN NODE. README.md says data prep runs on a login
# node "(internet)", implying compute nodes have none. Measured otherwise on
# 2026-08-02 (job 5878501, nid010882): a compute node listed all 570
# FineWeb2-HQ deu_Latn parquet files from HF. That matters because the login
# node CANNOT hold a long background job -- `loginctl show-user` reports
# Linger=no and enabling it is "Access denied", so systemd tears down every
# user process when the last session closes. Two prior attempts died exactly
# that way (pool at ~24GB, container mid-build), nohup/setsid/disown
# notwithstanding. A batch job is the only durable option here.
#
# The login node is still right for SHORT interactive prep (flores-download,
# the tokenizer fetch) -- just not for anything measured in hours.
#
# POOL SIZE. 72GB of German text. Measured 3.292 bytes/token for
# unigram_starved on real FineWeb2-HQ text (not the 3.350 FLORES figure), so
# 72GB -> ~21.9B tokens, covering the 16.1B target and the 21.8B ceiling that
# 100 GPU-hours could ever reach.
#
# WORKERS. fast_pool at 8, deliberately: its docstring records that HF's
# 5000-resolver-requests/5min account quota was tripped at 32 combined workers,
# after which every file-open 429'd and the run FALSELY reported "corpus
# exhausted" well short of budget. Packing is local CPU and takes all 64.
#
# AUTHENTICATION IS NOT OPTIONAL HERE, even though FineWeb2-HQ is public.
# The first attempt (job 5878535) ran anonymously and HF started returning 429
# on every worker after ~8GB, despite the documented-safe 8 workers -- the
# resolver quota for unauthenticated traffic is far lower. It had been
# sustaining 33.9MB/s until then. HF_HOME below points at a scratch token
# cache written with mode 600 (`$HF_HOME/token`), which huggingface_hub picks
# up automatically, so no credential appears in this script, in the
# environment, or in the job's stdout.
#
# Usage:
#   sbatch --account=brics.u6sg --partition=workq slurm/22_pool_and_pack_de.sbatch
set -euo pipefail

XS_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export XSCRIPT_SCRATCH="${XSCRIPT_SCRATCH:-/scratch/u6sg/jvonrad.u6sg/xscript}"
export HF_HOME="${HF_HOME:-$XSCRIPT_SCRATCH/hf_cache}"
cd "$XS_DIR"
mkdir -p logs
source "$XSCRIPT_SCRATCH/venv/bin/activate"

echo "host=$(hostname) start=$(date --iso-8601=seconds)"

# ---- 1. pool -------------------------------------------------------------
# `python -u`: fast_pool prints progress every 16 files, and block-buffered
# stdout makes a multi-hour job look hung for its first hour.
echo "== pool de (72GB) =="
python -u scripts/neuron_train/fast_pool.py --lang de --gb 72 --workers 8

# ---- 2. pack -------------------------------------------------------------
echo
echo "== pack de / unigram_starved =="
xscript pack --lang de --tok unigram_starved --workers 64

# ---- 3. verify -----------------------------------------------------------
echo
python - <<'PY'
import json, os
from pathlib import Path
S = Path(os.environ["XSCRIPT_SCRATCH"])
pool = json.loads((S / "pools" / "de" / "stats.json").read_text())
idx = json.loads((S / "shards" / "de__unigram_starved" / "index.json").read_text())
tot = sum(idx.values())
print(f"pool   : {pool['text_bytes']/1e9:.2f} GB text, {pool['docs']:,} docs, "
      f"exhausted={pool['exhausted']}")
print(f"shards : {len(idx)} files, {tot/1e9:.3f}B tokens "
      f"({pool['text_bytes']/tot:.3f} bytes/token)")
print()
targets = [("de-fair-8b  (token-matched)",  7.753e9),
           ("de-fair-12b (token-matched)", 11.754e9),
           ("de-fair-15b (token-matched)", 14.755e9),
           ("content-match de-fair-12b",   16.100e9)]
bad = [l for l, n in targets if tot < n]
for label, need in targets:
    print(f"  {'OK   ' if tot >= need else 'SHORT'} {label:30} needs {need/1e9:6.3f}B")
if bad:
    raise SystemExit(f"\nFAIL: pool too small for {len(bad)} target(s)")
print(f"\nOK: {(tot-16.1e9)/1e9:.2f}B spare above the 16.1B target (no epoching).")
PY
echo "end=$(date --iso-8601=seconds)"
