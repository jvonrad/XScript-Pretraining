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
- **`de__unigram_starved` collapsed mid-run** (anchor BPB ~1.72 vs the destarved
  twin's ~1.06). It is in `EXCLUDE_RUNS`.

## Analysis window

For cooldown-clean comparisons restrict to the **stable-LR window (1B-24B
tokens)**: `base_main.yaml` is WSD with decay starting at 24B, so anything past
that mixes a cooled model with mid-stable ones. This is the confound that
invalidated the original checkpoint-based BTS numbers (CLAUDE.md §6).
