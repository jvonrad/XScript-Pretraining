# Dense BPB-vs-tokens curves (cached from W&B)

`bpb_curves.csv` — long format: `run, name, wandb_id, step, tokens, metric,
value`, **1,938 points across 31 roster curves**. `metric` is
`eval/{flores,holdout}_{en,de,fr,ar,zh}_bpb`.

`bpb_curves_other.csv` / `histories_other.json` hold everything that is **not**
the thesis roster — 612 points across 58 curves: the 56 `__100b` attempts
(a later experiment, added to the project after this cache was first written),
the `probe_*` / `ctrl-scratch__*` / `*_scratch` probes, and the
`train.max_tokens` unit tests. They are pulled rather than dropped, but kept
out of the roster artifacts so an analysis that globs `bpb_curves.csv` cannot
pick up a curve from a different experiment. The rule is
`pull_wandb_curves.py`'s `is_roster()` — `{mix}__unigram_{starved,destarved}`
and nothing else, the same shape `bts_from_wandb.py`'s `load()` accepts, so a
future `__100b-v14` sorts itself with no list to maintain.

`run` is the **curve id**, not the display name — a W&B display name is not
unique (four of them cover 7–8 separate `__100b` attempts each, and a resume
shares its parent's name), so keying on it would merge unrelated trainings
silently. `wandb_id` gives the source run of each individual row, which is what
makes a deliberately merged curve auditable.

`runs_meta.json` carries one entry per **W&B run** (not per curve): state,
`tokens_per_step`, final `tokens_b`, eval-point counts, and the seam/merge
bookkeeping. `eval_final_bpb.json` carries the end-of-training
`eval_final/*_bpb` block per run with the `_step` and token position it was
logged at.

Cached from `jonathan-von-rad/XScript-Pretraining` because this is the **only**
fine-grained performance-over-training-tokens source in the repo — the 107
checkpoint evals (`../appendix_c5`, `../alignment_v2_107`) are much coarser, and
**holdout** BPB shards are not on the eval box at all. Everything here is
produced by `scripts/external_bench/pull_wandb_curves.py`; read it with
`scripts/external_bench/bts_from_wandb.py`.

## Extraction gotcha (already handled here)

The trainer logs `tokens_b` and the `eval/*_bpb` metrics in **separate**
`wandb.log()` calls, so they usually land on different steps. Any pull that
requires both on one row silently drops most eval points. Tokens are therefore
reconstructed as `step x tokens_per_step` (exactly linear; median ratio over
rows that do have both).

A second one, fixed 2026-09-01: the eval key names used to be read from the run
**summary**, and a run whose summary happened to carry no `eval/*_bpb` key was
skipped without its history ever being scanned — it just showed up as
`n_eval: 0`. Keys now come from the history rows themselves. Three runs were
being dropped that way; see the last bullet below for what they actually
contain. `runs_meta.json` also now records `tokens_per_step_delta` /
`affine_fit_max_err_b`, which is how far a run's real step→token map departs
from `step x tokens_per_step` — 0 for most runs, **0.77B for
`en__unigram_starved`**, which mixed world sizes mid-run (CLAUDE.md §6h).

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

- ⛔ **`__100b` / `probe_*` / `ctrl-scratch__*` / `*_scratch` / `*__capped` /
  `*__uncapped`** — split out to `bpb_curves_other.csv` at the source (see
  above). Four `__100b` display names cover 7–8 separate trainings each, so
  these are also the reason the CSV keys on a curve id rather than a name.
- ⛔ **The three `__neuron` runs contain NO eval data at all.** They were
  invisible until the summary-key bug above was fixed, and what the fix
  reveals is that the bug was hiding an absence, not data:

  | run id | what it is | eval rows |
  |---|---|---|
  | `zh__unigram_starved__neuron` | zh 12B→15B resume, `_step` 12820–16100 = 11.763–14.987B, finished | **0** |
  | `zh__unigram_destarved__neuron` | same, for the destarved twin | **0** |
  | `de__unigram_starved__neuron` | crashed 1-node attempt, `_step` 20–520 = 0.020–0.511B | **0** |

  All three log only `loss/lr/mix.*/step/tok_per_s/tokens/tokens_b` — the
  in-loop eval never ran. **So the last ~3B of both Chinese runs does not exist
  in W&B and cannot be recovered by any puller**; the zh curves in
  `bpb_curves.csv` still stop at the parents' 11.75B / 12.75B. That gap is
  filled instead by scoring the 15B zh checkpoints — see
  `bpb_curves_ckpt.csv` below.

  The two zh resumes are nonetheless declared in `pull_wandb_curves.py`'s
  `RUN_MERGE`, because they *are* the second leg of those trainings and share
  their parents' display name. `_step` continues across the resume (12820 >
  the parents' last eval mark at 12811) but tokens/step changes 917,504 →
  983,040 (2 nodes → 1), so their token axis needs a fitted offset of
  −0.8396B; the fit is exact and the seam gap is +0.0089B. Declaring the merge
  keeps it explicit and asserted rather than something that would have
  happened by name collision.

  ⚠️ `zh__unigram_destarved__neuron` **overlaps its parent by 0.99B**: it
  restarted from the 11.754B checkpoint while the parent had already reached
  12.754B, so `_step` 12820–13900 holds two different trainings. Harmless only
  because the resume has no eval rows (`parent_overlap_b` in
  `runs_meta.json`); do not interleave the two blindly if that ever changes.
- `de__unigram_starved__neuron` is **not** the 2026-08-03 retrain. The retrain
  resumed the `de__unigram_starved` id itself (see above) and reached 16.09B;
  this is a separate crashed 0.5B attempt that merely reports the same display
  name.

## `bpb_curves_ckpt.csv` — the zh 12→15B gap, filled from checkpoints

Not from W&B. Produced by `scripts/external_bench/run_bpb.py` +
`bpb_fill_from_checkpoints.py`, because the two zh resumes logged no eval at
all (above) and `zh-{fair,starved}-15b` are `step15865_14756M`, taken from
inside that unlogged stretch. Kept in a **separate file** on purpose:
`bpb_curves.csv` means "what W&B holds, pulled reproducibly", and mixing a
locally-computed point into it would destroy that. Join on `run` to plot them
together.

It is the **same quantity**, not a proxy — both are FLORES+ **dev** (n=997,
`bpb.py`'s `flores.load_parallel(langs, "dev")`), same NLL/bytes definition.
Note `run_bpb.py` defaults to `--split both` (n=2009); the fill must use
`--split dev` or it is not on this axis.

Validated rather than assumed: `zh-{fair,starved}-12b` ARE the checkpoints
behind the logged step-12811 points, so scoring them reproduces a number W&B
already holds. `bpb_fill_from_checkpoints.py` asserts this before writing.

| model | step | scored | W&B logged | Δ |
|---|---|---|---|---|
| zh-fair-12b (control) | 12811 | 1.314353 | 1.314361 | **−8.3e−06** |
| zh-starved-12b (control) | 12811 | 1.349123 | 1.349130 | **−6.8e−06** |
| **zh-fair-15b** | 15865 | **1.312721** | *(none — unlogged)* | — |
| **zh-starved-15b** | 15865 | **1.343951** | *(none — unlogged)* | — |

So the 15B points splice on with no calibration offset. The fair−starved gap
holds across the fill (+0.0348 at 11.754B, +0.0312 at 14.756B).

⛔ **FLORES only — the holdout half of the gap CANNOT be filled this way.**
`eval/holdout_*_bpb` reads 500 docs from the reserved FineWeb2-HQ holdout
shard (`bpb.load_holdout`); those shards are not on the eval box, so
recovering it needs the language's pool rebuilt (CLAUDE.md §6h did that for
German at 72.6GB). `bts_from_wandb.py --source holdout` still stops where W&B
does. `bts_from_wandb.py` does not read this file yet.

Cost, for reference: **2m02s** for 997 sentences on plain CPU (12 cores, 9.9GB
RSS) — no accelerator needed — against ~6 min to download each 4.4GB
checkpoint. The per-sentence `(nll_nats, bytes)` `run_bpb.py` writes are kept,
so `bts_matched.py` can bootstrap over sentences from these too.

## Analysis window

For cooldown-clean comparisons restrict to the **stable-LR window (1B-24B
tokens)**: `base_main.yaml` is WSD with decay starting at 24B, so anything past
that mixes a cooled model with mid-stable ones. This is the confound that
invalidated the original checkpoint-based BTS numbers (CLAUDE.md §6).
