# Dense BPB-vs-tokens curves (cached from W&B)

`bpb_curves.csv` — long format: `run, step, tokens, metric, value`, 1,952 points
across 25 runs. `metric` is `eval/{flores,holdout}_{en,de,fr,ar,zh}_bpb`.
`runs_meta.json` carries per-run state, `tokens_per_step` and final `tokens_b`.

Cached from `jonathan-von-rad/XScript-Pretraining` because this is the **only**
fine-grained performance-over-training-tokens source in the repo — the 107
checkpoint evals (`../appendix_c5`, `../alignment_v2_107`) are much coarser, and
**holdout** BPB shards are not on the eval box at all. Regenerate with
`scripts/external_bench/bts_from_wandb.py`.

## Extraction gotcha (already handled here)

The trainer logs `tokens_b` and the `eval/*_bpb` metrics in **separate**
`wandb.log()` calls, so they usually land on different steps. Any pull that
requires both on one row silently drops most eval points. Tokens are therefore
reconstructed as `step x tokens_per_step` (exactly linear; median ratio over
rows that do have both).

## Runs you must NOT use

- **Non-English-anchor bilinguals** (`de-ar`, `de-fr`, `de-zh`, `fr-ar`) appear
  here with eval points but **never actually ran** — see CLAUDE.md §6. Excluded
  in `bts_from_wandb.py`'s `load()`; drop them.
- ⛔ **`de__unigram_starved` is ONE W&B id holding TWO trainings.** It is no
  longer in `EXCLUDE_RUNS` — the 2026-08-03 retrain has landed (CLAUDE.md §6h)
  — but it must be cut at the seam:

  | steps | tokens | what it is | BPB |
  |---|---|---|---|
  | 273–7361 | 0.25–6.75B | **original, diverged** at the warmup/peak-LR seam | 1.546 → 1.280 → 1.740 → 1.301 |
  | 8451–16081 | 7.75–14.75B | **retrain** | 1.0803 → 1.0472, monotone |

  The retrain resumed the same id, so its early points collided with existing
  steps and W&B dropped them; only its post-7361 history survived. Across the
  seam BPB falls **1.3012 → 1.0803 in one eval interval** — a 0.221 drop that
  is not learning but the model changing identity. Interpolated together the
  two produce a curve belonging to no model that ever existed.

  This is cut in **two** places, deliberately: `pull_wandb_curves.py`'s
  `RUN_MIN_STEP` (at the source) and `bts_from_wandb.py`'s `RUN_MIN_TOKENS_B`
  (at the analysis). The second is what protects you if someone re-pulls with
  a different tool. **The data in this directory is already cut** — the CSV
  holds only the 8 retrain points.

  ⚠️ Consequence: de/starved has **no curve below 7.75B**, against 1–22B for
  every other run. Its ATLAS-BTS anchor is therefore forced into a different
  region of the loss curve than de/destarved's, and the two are not directly
  comparable (FLORES gives 0.845, holdout 0.498, for the same cell). The
  repo-style BTS at ~11.15B/lang *is* comparable, since that budget is inside
  every cell's window. Scoring the existing `de-starved-{1,2,5}b` checkpoints
  with `run_bpb.py` would fill 1–7.75B and remove this asymmetry.

## Analysis window

For cooldown-clean comparisons restrict to the **stable-LR window (1B-24B
tokens)**: `base_main.yaml` is WSD with decay starting at 24B, so anything past
that mixes a cooled model with mid-stable ones. This is the confound that
invalidated the original checkpoint-based BTS numbers (CLAUDE.md §6).
