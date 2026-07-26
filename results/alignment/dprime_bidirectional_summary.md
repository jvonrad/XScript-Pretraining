# Bidirectional d': is the asymmetry material?

**Change made.** `_discriminability` in `src/xscript/eval/alignment.py` now returns
`dprime_a2b` (the original row-wise A→B statistic, unchanged), `dprime_b2a` (the
same computation on the transpose -- B retrieving A, variance recomputed on the
column, not reused from the row), and `dprime_sym` (their mean). `_retrieval_sim`
exposes all three plus the untouched `dprime` (== `dprime_a2b`, kept for every
downstream consumer). `export_alignment.py`'s `PER_LAYER_METRICS` now includes the
three new columns for the next full pipeline run.

**Correctness checks (both passed):**
1. `dprime_b2a` via `_discriminability_1d(sim.T)` matches an independently-written
   column-wise formula to 1e-9 on a random matrix.
2. Recomputing `dprime_a2b` from the cached HF embeddings for all 8 EN-anchored
   bilingual models x 17 layers x 2 variants (272 cells) matches the **pre-existing
   committed** `results/alignment/per_layer.csv` `dprime` column to 1e-3 -- the old
   unidirectional numbers are exactly reproduced; nothing silently changed.

## Point 4: is the asymmetry large enough to matter?

**It depends entirely on variant, and the practically relevant answer is no.**

| variant | mean \|asym\| | mean rel. asym (\|a2b-b2a\|/\|sym\|) | max rel. asym | cells >10% rel. asym |
|---|---|---|---|---|
| raw | 0.387 | 19.96% | 90.3% | 89/136 (65%) |
| centered | 0.031 | 0.79% | 4.6% | 0/136 (0%) |

- **Raw**: the asymmetry is large and frequent -- 65% of cells exceed the 10%
  threshold, and one cell (en-ar-starved, L5) reaches 90% of `dprime_sym`. Raw-variant
  d' should not be treated as direction-agnostic.
- **Centered**: asymmetry is uniformly negligible -- max 4.6%, zero cells over the
  10% flag. The bottom row of `fig_dprime_bidirectional.png` shows the solid (A→B)
  and dashed (B→A) lines essentially overlapping for every one of the 8 cells.

The repo's own headline numbers (`align_v2.txt`, the README's suggested plots) are
all **centered**, not raw -- centering is the variant used for reporting throughout.
So: **for the numbers that are actually quoted, this is a one-line robustness
footnote, not a reanalysis.** The existing unidirectional `dprime` stands. Raw-variant
readers should be told the number is directional; that's the only actionable change.

This also explains itself mechanically: the module's own docstring already notes
that raw embeddings are dominated by a per-language centroid direction that
"swamps the translation signal." That same centroid difference between languages
is exactly what makes the row-wise and column-wise non-match distributions differ
in raw space -- centering removes the centroid and the two directions converge.

## Point 5: is the (raw) asymmetry systematic, or a bug?

**Systematic, not a bug -- confirmed by an independent metric already in `per_layer.csv`.**
The sign of the new `dprime_a2b - dprime_b2a` agrees with the sign of the
pre-existing `top1_a2b - top1_b2a` gap in 88-91% of cells with non-trivial asymmetry
(Pearson r = 0.74 across all 272 raw+centered cells). Two independently-computed
statistics -- one continuous or one thresholded, one built new here and one already
committed -- pointing the same direction rules out a transpose/indexing bug.

The direction also has a describable pattern, not noise: in raw space, **early-to-mid
layers (roughly L1-8) tend to favor EN→partner (a2b > b2a)**, i.e. English queries
retrieve their translation better than the reverse; **layers 9+ often flip toward
partner→EN**. This flip is visible in all four pairs' raw-variant panels in the
figure (solid above dashed early, crossing over and dashed pulling ahead by L9-11).
It is consistent with English dominating the FineWeb2 pretraining mixture regardless
of the bilingual pair (EN row statistics are less noisy / more consistently spaced
early on), while later layers -- closer to the LM head -- start reflecting each
model's own generation-side asymmetry (which language it was more recently trained
to *produce*). No pair/script pattern beyond that: it is not confined to cross-script
(ar/zh) or same-script (de/fr) alone -- both groups show the early/late flip in raw
space, and both groups converge to ~0 asymmetry once centered.

## Bottom line

Use `dprime_sym` (or just keep `dprime`/`dprime_a2b`, its unidirectional predecessor)
for centered-variant reporting -- the asymmetry there is noise-level and the existing
committed numbers need no revision. If raw-variant d' is ever quoted on its own
(outside of "raw vs centered" comparisons), report `dprime_sym` or both directions
explicitly, since raw asymmetry is large enough (up to 90% of the symmetric value)
to change qualitative conclusions in individual layer/pair cells.
