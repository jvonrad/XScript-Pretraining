# Dense BPB-vs-tokens curves (cached from W&B)

`bpb_curves.csv` — long format: `run, name, wandb_id, leg, step, tokens,
metric, value`, **1,944 points across 31 roster curves**. `leg` is empty except
where one W&B id holds more than one training (only `de__unigram_starved`
today — see below). `metric` is
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
the **holdout** BPB shards died with the training allocation (they are
reconstructible per language — see the holdout section below — but only `de`
has been rebuilt and controlled). Everything here is
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
- ⛔ **`de__unigram_starved` is ONE W&B id holding TWO trainings**, and the
  first of them is itself part healthy and part wreckage. It is no longer in
  `EXCLUDE_RUNS` — the 2026-08-03 retrain has landed (CLAUDE.md §6h) — but it
  has to be cut into legs. `pull_wandb_curves.py`'s `RUN_SEGMENTS` does that,
  and the CSV's **`leg` column** carries the result per row (the two trainings
  share one `wandb_id`, so nothing else distinguishes them):

  | steps | tokens | leg | what it is | in the CSV? |
  |---|---|---|---|---|
  | 273–819 | 0.25–0.75B | `orig-prespike` | **original, still healthy** — flores 1.5461 → 1.3728 → **1.2804**, holdout 1.5642 → 1.3498 → **1.2508**, both falling | **yes** |
  | 1092–8380 | 1.00–7.69B | — | **the divergence and its incomplete recovery** | **no** |
  | 8451–16081 | 7.75–14.75B | `retrain` | **retrain** (seed 1) | yes |

  **Where it diverged: 1.00B, not ~7B.** Both eval metrics turn upward at
  step 1092 = 1.002B — exactly where warmup ends and the LR pins at peak
  3.0e-3 — and train loss turns at ~0.92B off a floor of 2.6806 @0.734B. The
  run's *last* logged token is 7.689B, which is the number that invites the
  ~7B intuition; that is where it was abandoned, not where it broke.

  **It did recover — in shape, not in level, which is why the tail is still
  unusable.** BPB peaks at 1.7403 / 1.6929 @2.75B (+0.676 / +0.719 vs
  de-fair), then descends monotonically from 3.25B on. But at its last eval
  (6.754B) it is 1.3012 / 1.2937 — **still 0.021 above its OWN 0.751B floor
  after 6B further tokens**, and still +0.291 / +0.374 behind de-fair, against
  the retrain's +0.074 / +0.054 at 7.754B. Train loss says the same: 2.7108 at
  7.689B against its 2.6806 floor. Nothing after the spike is a de/starved
  point at its nominal budget.

  Across the leg seam BPB falls **1.3012 → 1.0803 in one eval interval** — a
  0.221 drop that is not learning but the model changing identity. The retrain
  resumed the same id, so its own early points collided with existing steps
  and W&B dropped them; only its post-8380 history survived (the leg boundary
  is an 18.5-day wall-clock gap between steps 8380 and 8400 — there is no step
  gap to find it by).

  ⚠️ **The two kept legs are different SEEDS** of a byte-identical config
  (only `seed`/`data_seed` changed, §6h), so this curve is a seed mixture
  below 1B. Use the `leg` column before treating it as one training.

  ℹ️ **The restored prefix cannot move any BTS number**, so it is for plotting
  and completeness only: all three points are at 0.25–0.75B, i.e. inside
  warmup, and `bts_from_wandb.py`'s `stable_window()` keeps only 1B–24B —
  independently of its `RUN_MIN_TOKENS_B` floor. That is not a coincidence:
  the run diverged *at* the warmup/peak-LR seam, so everything healthy is by
  construction pre-1B. Verified: the BTS table is byte-identical with and
  without the prefix.

  ⚠️ Consequence, unchanged: de/starved has **no usable curve between 0.75B and
  7.75B**, against 1–22B for every other run. Its ATLAS-BTS anchor is
  therefore forced into a different region of the loss curve than
  de/destarved's, and the two are not directly comparable (FLORES gives 0.845,
  holdout 0.498, for the same cell). The repo-style BTS at ~11.15B/lang *is*
  comparable, since that budget is inside every cell's window. Scoring the
  existing `de-starved-{1,2,5}b` checkpoints with `run_bpb.py` would fill
  1–7.75B with real seed-1 points and remove this asymmetry — the same route
  `bpb_curves_ckpt.csv` already took for zh.

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

`bts_from_wandb.py` reads this file automatically (`--ckpt-csv` defaults to
it) and merges its points **exempt from `RUN_MIN_TOKENS_B`** — that floor
exists to cut the diverged original, and a fill point is never the diverged
original. Pass `--ckpt-csv ''` to disable.

ℹ️ A **control** point survives the merge as a near-duplicate of its W&B twin
rather than deduplicating away, because the two sit on marginally different
x: this file takes tokens from the checkpoint NAME (`step8451_7753M` →
7.753000B, the trainer's own count) while the W&B curve reconstructs
`step × tokens_per_step` → 7.753826B. That is a **0.011%** x-offset carrying
a **5–9e−06** y-difference (the CPU-vs-Neuron rescoring gap), i.e. three
orders of magnitude below the ±0.008–0.019 checkpoint-to-checkpoint noise
CLAUDE.md §6 documents. Harmless for interpolation and root-finding, but it
does mean a cell's point count is one higher than "W&B points + new points"
would suggest.

Cost, for reference: **2m02s** for 997 FLORES sentences on plain CPU (12
cores, 9.9GB RSS) — no accelerator needed — against ~6 min to download each
4.4GB checkpoint. The per-sentence `(nll_nats, bytes)` `run_bpb.py` writes
are kept, so `bts_matched.py` can bootstrap over sentences from these too.

### Holdout (2026-09-03): reconstructible, and reconstructed for `de`

The `eval/holdout_*_bpb` half was long treated as unfillable, because
`bpb.load_holdout` reads 500 docs from a reserved FineWeb2-HQ shard that died
with the Isambard-AI allocation. It is not lost, though — it is a
**deterministic function of the public corpus**, and every step is still in
`xscript.data.fineweb`: `_list_parquets()` sorts the manifest, `files[0]` is
reserved and never enters the pool, and the holdout is its first
`HOLDOUT_BYTES` (30 MiB) of text in file order. `rebuild_holdout.py` re-runs
those same functions rather than reimplementing the recipe.

For German: source `epfml/FineWeb2-HQ/deu_Latn`, 570 files, reserved file
`deu_Latn/000_00000.parquet` → **31.46 MB / 7,391 docs**, of which
`load_holdout("de", 500)` takes the same first 500 the trainer took.

⛔ Two things this depends on, both easy to get wrong:

1. **Score it with `bpb.score_texts`, not the FLORES path.** Holdout docs are
   full web pages that exceed the 2048-token context. The trainer scores them
   in sliding non-overlapping windows; `run_bpb.py`'s fixed-shape FLORES
   adapter would **truncate** to one window instead. Different numbers.
   `run_bpb.py --source holdout` calls the trainer's own function.
2. **The control is per language.** Each language has its own manifest, and
   `fr`/`ar` fall back to a second repo once the primary is exhausted
   (`FALLBACK_SOURCES`). Only `de` has been rebuilt and controlled:

   | model | scored | W&B logged | Δ |
   |---|---|---|---|
   | `de-starved-8b` (step 8451) | 0.9699196558 | 0.9699254960 | **−5.84e−06** |

   `zh`/`fr`/`ar`/`en` are **not attempted** — rebuild and re-control each
   before filling it. Scoring cost is ~35 min/checkpoint on CPU (500 web
   docs ≈ 1.92 MB, batch-1 windows), against ~2 min for FLORES.

## Analysis window

For cooldown-clean comparisons restrict to the **stable-LR window (1B-24B
tokens)**: `base_main.yaml` is WSD with decay starting at 24B, so anything past
that mixes a cooled model with mid-stable ones. This is the confound that
invalidated the original checkpoint-based BTS numbers (CLAUDE.md §6).
