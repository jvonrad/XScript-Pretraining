#!/usr/bin/env bash
# Four-pooling alignment over the FULL bilingual checkpoint trajectory, each
# model scored on its OWN language pair.
#
#   bash run_pooling_trajectory.sh <workdir> <out-subdir> [budgets...]
#
# Default budgets are the ones `run_pooling_sweep.sh` did not cover (2b and the
# 30B final are already done), giving 48 bilingual checkpoints in total:
# en-{ar,de,fr,zh}-{fair,starved}-{2b,5b,10b,15b,23b} plus the 30B finals.
#
# ⛔ ALL FIVE LANGUAGES ARE EMBEDDED, even though only the own pair is reported.
# Restricting to `--langs en <partner>` was tried first and REJECTED on
# evidence. It looks free -- causal attention means right padding never reaches
# a real position -- but `_fixed_width` then shrinks (en-ar 90 vs 112, en-zh 90,
# en-de 100; en-fr is the ONE pair that keeps 112, so testing on en-fr is
# vacuous and passed bit-for-bit for the wrong reason). At a different width the
# compiler tiles the matmuls differently, and the fp accumulation order changes:
#
#   centered/mutual_nn   0/68 layer-cells differ   0.000e+00   <- what we report
#   centered/dprime     60/68                      4.7e-07     <- under tolerance
#   raw/mutual_nn        4/68                      5.0e-04     <- 1 flipped retrieval
#
# So the headline metric is provably unaffected, but `raw` flips the occasional
# near-tie. Across a TRAJECTORY that is exactly how a spurious wiggle gets
# manufactured, and the 2b/30B points already exist at 5 languages. Keeping one
# graph width for every checkpoint costs ~2x wall clock and buys internal
# consistency with those points and with results/alignment_v2_107.
#
# One difference from run_pooling_sweep.sh:
#  1. **The downloaded checkpoint PARTS are deleted after every model.**
#     `run_alignment.py`'s own cleanup removes the assembled `final.pt` but not
#     the five `final.pt.part00*` files it was built from, so a sweep silently
#     accumulates ~4.4 GB per model -- 66 GB had piled up before this was
#     noticed, and 32 more models would have needed 140 GB.
#
# No --emb-dir: 4 poolings x 2 languages x 32 models would be ~72 GB of
# embeddings, and the trajectory question is answered from the per-layer
# metrics in the JSONs. Re-run one model with --emb-dir if a subset analysis
# (T4-style) is ever wanted on a specific checkpoint.
set -euo pipefail

WORK=${1:?workdir}
SUBDIR=${2:?out-subdir}
shift 2
BUDGETS=("$@")
[ ${#BUDGETS[@]} -eq 0 ] && BUDGETS=(-5b -10b -15b -23b)

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH="$HOME/.local/bin:$PATH"
# shellcheck disable=SC1091
source "$HOME/neuron_venv/bin/activate"
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
export HF_HUB_ENABLE_HF_TRANSFER=1

cd "$(dirname "$0")"

for b in "${BUDGETS[@]}"; do
  for p in de fr ar zh; do
    for t in fair starved; do
      RUN="en-${p}-${t}${b}"
      if [ -f "$WORK/results/$SUBDIR/$RUN.json" ]; then
        echo "=============== ${RUN} (already done, skipping) ==============="
        continue
      fi
      echo "=============== ${RUN} ==============="
      python run_alignment.py \
        --repo jvonrad/xscript-eval --device xla \
        --workdir "$WORK" --runs "$RUN" \
        --langs en de fr ar zh --split both \
        --poolings mean mean_nobos weighted last \
        --out-subdir "$SUBDIR"
      # See header note 2: drop the downloaded parts, not just the assembly.
      rm -rf "$WORK/_repo/runs/$RUN" "$WORK/_assembled"
      df -h / | tail -1
    done
  done
done
echo "TRAJECTORY SWEEP COMPLETE"
