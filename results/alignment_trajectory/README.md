# Bilingual alignment trajectory: every checkpoint x every layer x four poolings

All **48** EN-anchored bilingual checkpoints — `en-{ar,de,fr,zh}-{fair,starved}`
at **2B / 5B / 10B / 15B / 23B / 30B** — each scored on its **own** language
pair, at every layer, under all four sentence poolings. This is the trajectory
counterpart to `results/alignment_pooling/`, which had only the 2B and 30B ends.

**Budgets are TOTAL tokens.** The bilinguals mix 50/50, so a 23B run has seen
~11.4B of each language. Everything at 23B and below is mid-stable at peak LR
3.0e-3 (decay starts at 24B), so **those five columns are LR-matched by
construction**; the **30B column is cooled** and is not on the same curve
(CLAUDE.md section 6).

## Headline

1. **The depth-of-emergence delay is the robust result.** Starved reaches
   `mutual_nn >= 0.90` deeper than fair in **all 44 measurable cells** (6
   budgets x 4 pairs x 2 poolings, excluding cells where neither arm reaches
   the threshold). It holds at every budget, in both tokenizer conditions, and
   under `mean` and `weighted` alike.
2. **⛔ A fixed layer band is NOT a safe trajectory summary.** Past ~10B every
   model develops a **mid-stack trough**, and the trough *migrates* with budget
   and sits at a *different layer* for fair vs starved. An L5-8 band therefore
   measures where each model's trough happens to fall, not how well it aligns —
   which is how `en-de` at 23B produces a **negative** gap (−0.023) between two
   otherwise unremarkable checkpoints.
3. **Peak-layer gaps are ceilinged nearly everywhere** (`SAT` in the tables):
   at 5B and above, both arms exceed 0.95 on all four pairs under mean-family
   pooling. Only the 2B ar/zh cells are unsaturated — the same finding as
   `results/alignment_pooling/` T6, now confirmed across the full grid.

## The trough, and why it invalidates band summaries

Per-layer alignment is **not monotone in depth**. `mean` pooling, trough layer
and depth below the best pre-trough layer (`L2(0.00)` = no trough yet):

| pair | tok | 2B | 5B | 10B | 15B | 23B | 30B |
|---|---|---|---|---|---|---|---|
| en-de | fair | L2(0.00) | L2(0.00) | L2(0.00) | L6(0.17) | **L2(0.36)** | L2(0.44) |
| en-de | starved | L2(0.00) | L2(0.00) | L2(0.00) | L8(0.42) | **L8(0.38)** | L5(0.40) |
| en-fr | fair | L3(0.25) | L2(0.11) | L2(0.09) | L2(0.23) | L2(0.40) | L2(0.39) |
| en-fr | starved | L2(0.05) | L2(0.00) | L2(0.00) | L6(0.52) | L6(0.49) | L6(0.39) |
| en-ar | fair | L3(0.00) | L2(0.00) | L2(0.00) | L5(0.33) | L5(0.32) | L5(0.34) |
| en-ar | starved | L6(0.01) | L2(0.00) | L2(0.00) | L5(0.02) | L5(0.03) | L8(0.24) |
| en-zh | fair | L3(0.00) | L2(0.00) | L2(0.00) | L7(0.09) | L5(0.46) | L5(0.41) |
| en-zh | starved | L2(0.11) | L2(0.06) | L5(0.19) | L5(0.57) | **L5(0.67)** | L5(0.64) |

Two facts the single-budget tables could not have shown:

* **The trough does not exist below ~10B.** Depths are ~0.00 at 2B/5B/10B and
  0.2–0.67 from 15B on. It is something training *creates*, not a property of
  the architecture. This is the same feature CLAUDE.md section 6b diagnosed for
  `fr-starved` (CKA and d' collapsing and recovering together) — that was one
  checkpoint of a phenomenon the whole roster develops.
* **Its layer differs by tokenizer.** For de and fr the fair model troughs
  early (L2) while starved troughs mid-stack (L6–L8) — i.e. *inside* the L5-8
  band. Much of the "L5-8 fair−starved gap" in `results/alignment_pooling/` is
  therefore the starved model's trough sitting in the measurement window while
  the fair model's sits outside it.

`en-de` at 23B is the clean demonstration: fair troughs at L2 (0.36 deep) and
starved at L8, so the band straddles starved's *recovery* and fair's *trough*,
and the sign flips. Nothing is wrong with either checkpoint — the estimator is.

This is the same class of error as the fixed-layer probe section 6b already
retracted: **a fixed depth is only comparable when the feature being measured
sits at the same depth in both arms.** It does not here.

## What survives: depth to `mutual_nn >= 0.90` (fair / starved)

| pooling | pair | 2B | 5B | 10B | 15B | 23B | 30B |
|---|---|---|---|---|---|---|---|
| mean | en-de | 4/11 | 2/9 | 2/10 | 2/11 | 8/10 | 8/10 |
| mean | en-fr | 8/9 | 1/8 | 1/9 | 1/10 | 1/5 | 1/9 |
| mean | en-ar | –/– | 9/11 | 7/11 | 8/11 | 10/11 | 8/11 |
| mean | en-zh | –/– | 7/12 | 5/10 | 6/12 | 8/11 | 8/11 |
| weighted | en-de | 5/11 | 3/9 | 2/8 | 2/11 | 10/11 | 9/11 |
| weighted | en-fr | 4/9 | 2/8 | 2/7 | 1/11 | 1/5 | 1/5 |
| weighted | en-ar | –/– | 8/11 | 7/10 | 7/9 | 6/9 | 6/9 |
| weighted | en-zh | –/– | 7/10 | 6/8 | 6/13 | 6/13 | 7/13 |

**Starved is deeper in 44/44 measurable cells** for `mean` and `weighted`.
Widening to all four poolings gives **68/71** (25 further cells are unmeasurable
because one arm never reaches 0.90); the three exceptions are
`mean_nobos`/en-de/23B (−1, the trough artifact above) and `last`/en-fr at 5B
(−4) and 10B (−1), where fair's last-token curve is still near-flat and the
threshold crossing is noise. The delay is largest in the
5B–15B range (+7 to +9 layers on de/fr) and compresses at 23B/30B as both arms
saturate. Note `en-fr` fair reaching 0.90 at **L1** from 5B onward is the
lexical-overlap leakage documented in `results/alignment_pooling/` (layer 0-1 is
a bag of token embeddings, `last` gets 0.004 there) — read the fr row as
"starved needs 5–10 layers, fair needs none because L1 already leaks", not as a
+8-layer representational delay.

## Mid-stack gap by budget (for completeness — see the caveat above)

L5-8 fair − starved, `mean` / `weighted`:

| pair | 2B | 5B | 10B | 15B | 23B | 30B |
|---|---|---|---|---|---|---|
| en-de | +.305/+.321 | +.210/+.239 | +.208/+.180 | +.178/+.266 | **−.023/−.035** | +.300/+.338 |
| en-fr | +.116/+.218 | +.087/+.127 | +.103/+.113 | +.325/+.476 | +.146/+.218 | +.145/+.229 |
| en-ar | +.197/+.234 | +.605/+.606 | +.642/+.621 | +.418/+.550 | +.384/+.365 | +.456/+.429 |
| en-zh | +.254/+.193 | +.468/+.391 | +.487/+.221 | +.703/+.772 | +.412/+.668 | +.362/+.580 |

Positive in 47 of 48 cells, and the sole exception (`en-de` 23B) is the trough
artifact above. But the trajectory is **not monotone** and should not be fitted:
the `en-fr` 15B spike (+0.325) and the `en-de` 23B dip are both trough
migration, not training effects.

## Files

- `per_model/en-{ar,de,fr,zh}-{fair,starved}[-2b|-5b|-10b|-15b|-23b].json` —
  48 files, all four poolings, same schema as `results/alignment_v2_107/` with a
  `poolings` level inserted. Per-query bootstrap arrays stripped (27 MB rather
  than ~600 MB); the untrimmed files are on the eval box only.
- `trajectory_{mean,mean_nobos,weighted,last}.md` — per-layer x per-budget
  grids, fair−starved gaps with `SAT` flags, and the trough table.
- `trajectory_summary.md` / `.json` — peak, mid-gap, depth, saturation per
  (pooling, pair, budget).
- Regenerate: `scripts/external_bench/run_pooling_trajectory.sh <work> <subdir>`
  then `analyze_pooling_trajectory.py <results> --out-dir <dir>`.

## Method notes

**All five languages are embedded even though only the own pair is reported.**
Restricting to `--langs en <partner>` was tried and rejected on evidence: it
shrinks `fixed_width` (en-ar 90 vs 112), the compiler then tiles the matmuls
differently, and the fp accumulation order changes. Measured effect —
`centered/mutual_nn` 0/68 layer-cells differ (0.000e+00), `centered/dprime`
4.7e-07, but `raw/mutual_nn` flips one retrieval in 2009. The reported metric is
provably unaffected, but across a trajectory that is exactly how a spurious
wiggle appears, and the 2B/30B points already existed at 5 languages. One graph
width for all 48 costs ~2x wall clock and buys internal consistency.
(`en-fr` is the one pair whose width is unchanged at 112, so testing the
restriction on `en-fr` alone passes vacuously — it must be tested on ar or zh.)

**Guards.** `mean` reproduces `results/alignment_v2_107/` bit-for-bit
(0.000e+00 over 340 pair x variant x layer cells), both alone and inside the
4-pooling run; the cached embeddings re-derive the in-run metrics at 0.000e+00.

## Limits

One training run per cell — no error bars on the runs. `centered`/`mutual_nn`
only (`raw` is in the JSONs, unanalysed). The 30B column is cooled and not
comparable to the others as a trend point. The trough is characterised
descriptively here; *why* training creates it, and why its depth differs by
tokenizer, is not established.
