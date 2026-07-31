# Recalibrated benchmark results (CLAUDE.md §6e)

Re-scored 2026-07-31 across 41 checkpoints after the fifth format artifact was
found: SIB-200, Taxi-1500 and XNLI(en/de/fr) were being ranked by estimators
whose argmax is decided by a per-candidate constant rather than by the
document. See CLAUDE.md §6e for what that retracts.

**These do NOT replace `results/extra_bench/` and `results/appendix_c5/`** —
they sit alongside. Two differences:

* they were produced with `--own-langs`, so each model was scored only on its
  own training languages. The transfer tables need nothing else (every cell
  pairs trained-language scores), but the out-of-domain cells behind §6d's
  per-model all-language tables and the untrained arm of its label-language
  control are **only** in the older directories.
* `correct[lang][task]` here carries the full estimator family — `acc`,
  `acc_norm`, `acc_tokennorm`, `acc_pmi`, `acc_cal`, `acc_cal_loo`,
  `acc_cal_pmi` — backfilled by `backfill_calibrated.py`. Select `acc_cal` for
  SIB-200 / Taxi-1500 / XNLI. `analyze_extra_bench.py` and
  `bootstrap_transfer.py` work unchanged apart from that selection.

## The raw loglikelihoods are NOT committed here

86 MB of per-candidate loglikelihoods (`raw/<run>_raw.json`, 57 MB for
extra_bench + 29 MB for appendix_c5) are what make every estimator a pure-CPU
re-derivation, so a future scoring change never costs another accelerator
pass. They are too large for git — same call as §6b's embeddings, which went
to HF for the same reason.

They currently live on the eval box at `/home/ubuntu/xscript_bench/results/`.
**That box is ephemeral: copy them off, or upload them next to
`jvonrad/xscript-embeddings`, before it is torn down.** Regenerating costs a
full re-run (~14h on a trn2.3xlarge's two core-pairs, ~1h on a 48xlarge).

Reproduce the reports with:

    python scripts/external_bench/analyze_raw_scores.py <dir> --report variants degeneracy trajectory
    python scripts/external_bench/analyze_raw_scores.py <dir> --report transfer --variant acc_cal --pairs pairs.json
