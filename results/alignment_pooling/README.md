# Is the fair-vs-starved alignment gap a property of the models, or of mean pooling?

**Verdict: the depth-of-emergence claim SURVIVES, and is strengthened — but one
attached claim dies.** The early/mid-layer fair-vs-starved gap is not a pooling
artifact: it persists when BOS is removed and *grows* under MEXA's own
position-weighted pooling. What does not survive is the companion statement
that the tokenizer "does not increase maximum alignment at late layers" — that
is a **saturation artifact** of mean pooling. The 2B tier (T6) settles it: the
peak gap is ~0 in exactly the six model-pairs where both conditions are
ceilinged and **+0.13** in exactly the two where they are not. By extension
§6b's "peak-layer effect decays to ~0 by 23B" is confounded with the models
crossing the metric's ceiling, and is not established either.

## Why this was tested

`src/xscript/eval/alignment.py` pools sentences with an **unweighted mean over
all non-pad tokens, BOS included** (`_encode` passes `bos=True`). MEXA
(arXiv 2410.05873) uses a **position-weighted** average, `w_t = t / sum_k k`,
and compares it only against **last-token** pooling. Plain unweighted mean is
evaluated nowhere in that paper, so our estimator was unvalidated on exactly
the axis that matters here.

That axis matters because the starved tokenizer emits **1.14-1.32x more tokens**
for the same text (en 1.14, de 1.24, fr 1.22, ar 1.32, zh 1.18). Under mean
pooling the weight on BOS is `1/T` and half the mass sits on the first half of
the sentence, where representations are least contextualised and most
token-identity-driven. Under MEXA's weighting BOS carries `~2/T^2` and the
early half ~25%; under last-token both are 0. So a fair-vs-starved difference
could in principle be manufactured by the pooling rule interacting with
fertility, with no difference in the trained models.

## What was run

Four poolings from **one** forward pass (`model.layer_reps` already returns the
full `(n_layers+1, B, T, d)` stack, so the extra poolings are free):

| pooling | weight on token t of a length-L sentence | BOS weight |
|---|---|---|
| `mean` | `1/L` — **the historical estimator** | `1/L` |
| `mean_nobos` | `1/(L-1)`, BOS masked out | 0 |
| `weighted` | `t / (L(L+1)/2)` — **MEXA's default** | `~2/L^2` |
| `last` | 1 at `t = L`, else 0 — MEXA's baseline | 0 |

Models: the 8 EN-anchored bilinguals at 30B, `en-{ar,de,fr,zh}-{fair,starved}`,
and the same 8 at **`-2b`** (T6). FLORES+ dev+devtest (n=2009), all 5 languages
embedded, each scored on its **own trained pair**, `centered` variant, metric
`mutual_nn`.

### Guard (mandatory, passed before anything below was read)

A re-derivation is not trusted here until it reproduces the original — the
standing rule from `rawscores.check_reproduces()` (CLAUDE.md section 6e). Run in
two stages, so that "my environment is wrong" and "the patch perturbs numerics"
could not be confused:

| stage | what it proves | result |
|---|---|---|
| **A**: `--poolings mean` alone vs `results/alignment_v2_107/en-fr-fair.json` | data / tokenizer / checkpoint / harness identity | **0.000e+00** over 340 (pair x variant x layer) cells |
| **B**: `--poolings mean mean_nobos weighted last`, compare its `mean` | adding poolings changes the XLA graph but not the numbers | **0.000e+00** over the same 340 cells |
| C: cached embeddings re-derive the in-run metrics | T4 can be pure-CPU | **0.000e+00** |

Not "within tolerance" — **bit-for-bit**. Note `mutual_nn`'s quantum is
`1/2009 = 4.98e-4`, ~500x the 1e-6 tolerance, so a single flipped retrieval
would have been visible. `mean` is bit-exact *by construction*: its weight
matrix **is** the original 0/1 length mask, so the pooling expression is
character-identical to the pre-patch line.

Independently, the `mean` rows of T1 below reproduce the pre-registered claim
table exactly for all four pairs.

---

## T1. fair - starved by layer band, all four poolings

`mutual_nn`, `centered`, own trained pair. **fair minus starved.**

| pooling | pair | L0-4 | L5-8 | L9-12 | L13-16 | peak |
|---|---|---|---|---|---|---|
| mean | en-de | -0.024 | **+0.300** | +0.070 | +0.002 | +0.001 |
| mean | en-fr | -0.017 | **+0.145** | +0.016 | +0.000 | -0.000 |
| mean | en-ar | +0.443 | **+0.456** | +0.125 | +0.004 | +0.001 |
| mean | en-zh | +0.178 | **+0.362** | +0.053 | +0.011 | +0.006 |
| mean_nobos | en-de | -0.012 | **+0.293** | +0.049 | +0.001 | +0.002 |
| mean_nobos | en-fr | -0.005 | **+0.095** | +0.014 | +0.001 | +0.000 |
| mean_nobos | en-ar | +0.453 | **+0.493** | -0.015 | +0.004 | -0.004 |
| mean_nobos | en-zh | +0.164 | **+0.612** | +0.085 | +0.007 | +0.027 |
| weighted | en-de | +0.004 | **+0.338** | +0.088 | +0.004 | +0.001 |
| weighted | en-fr | +0.033 | **+0.229** | +0.101 | +0.006 | +0.000 |
| weighted | en-ar | +0.439 | **+0.429** | -0.027 | -0.001 | -0.011 |
| weighted | en-zh | +0.116 | **+0.580** | +0.164 | +0.045 | +0.025 |
| last | en-de | -0.038 | +0.017 | +0.001 | +0.075 | +0.027 |
| last | en-fr | -0.043 | +0.025 | +0.027 | +0.274 | +0.377 |
| last | en-ar | +0.113 | +0.130 | -0.648 | -0.102 | -0.005 |
| last | en-zh | +0.059 | +0.496 | +0.810 | +0.574 | +0.210 |

**The L5-8 gap is not a mean-pooling artifact.** It survives removing BOS
(`mean_nobos`, 4/4 pairs) and it is **larger** under MEXA's own default
(`weighted`) in 3 of 4 pairs — de +0.300→+0.338, fr +0.145→+0.229,
zh +0.362→+0.580, with ar essentially unchanged (+0.456→+0.429). If the
effect were manufactured by mean pooling's over-weighting of early,
token-identity-driven positions, down-weighting exactly those positions would
have shrunk it. It grows.

`last` is the exception and behaves differently everywhere; see T3.

## T2. depth to `mutual_nn >= 0.90`

`-` = never reaches 0.90 at any layer.

| pooling | pair | fair | starved | delay | fair_max | starved_max |
|---|---|---|---|---|---|---|
| mean | en-de | 8 | 10 | **+2** | 0.996 | 0.995 |
| mean | en-fr | 1 | 9 | **+8** | 0.997 | 0.997 |
| mean | en-ar | 8 | 11 | **+3** | 0.985 | 0.983 |
| mean | en-zh | 8 | 11 | **+3** | 0.973 | 0.966 |
| mean_nobos | en-de | 1 | 10 | **+9** | 0.997 | 0.995 |
| mean_nobos | en-fr | 1 | 4 | **+3** | 0.999 | 0.998 |
| mean_nobos | en-ar | 5 | 9 | **+4** | 0.991 | 0.996 |
| mean_nobos | en-zh | 5 | 10 | **+5** | 0.997 | 0.970 |
| weighted | en-de | 9 | 11 | **+2** | 0.995 | 0.994 |
| weighted | en-fr | 1 | 5 | **+4** | 0.996 | 0.996 |
| weighted | en-ar | 6 | 9 | **+3** | 0.986 | 0.997 |
| weighted | en-zh | 7 | 13 | **+6** | 0.992 | 0.967 |
| last | en-de | - | - | n/a | 0.863 | 0.836 |
| last | en-fr | 16 | - | n/a | 0.925 | 0.549 |
| last | en-ar | - | 11 | n/a | 0.899 | 0.904 |
| last | en-zh | 10 | - | n/a | 0.946 | 0.736 |

**The delay survives: positive in all 12 mean-family cells (+2 to +9).** The
`mean` column reproduces the pre-registered fair 8/1/8/8 vs starved 10/9/11/11
exactly. Under `last` the threshold is not reachable in most cells, so
depth-to-0.90 is simply undefined there — not evidence either way.

### The `en-fr-fair` L1 anomaly is confirmed as lexical leakage — do not quote early layers

It was flagged in advance as suspicious. It is real and it is leakage:

| pair | `mean` L0 | `weighted` L0 | **`last` L0** | model-free TF-IDF floor |
|---|---|---|---|---|
| en-de | 0.509 | 0.414 | **0.003** | 0.397 |
| en-fr | 0.639 | 0.573 | **0.004** | 0.533 |
| en-ar | 0.352 | 0.321 | **0.007** | 0.121 |
| en-zh | 0.367 | 0.311 | **0.004** | 0.175 |

Layer 0 is the **embedding** output, before any transformer block. Mean-pooled,
it retrieves en-fr translations at 0.639 — right beside the model-free TF-IDF
token-overlap floor of 0.533, and it cannot be anything else, because a mean
over token embeddings is a bag of tokens. The last token of the same layer
retrieves at **0.004**. So the low-layer "alignment" that mean pooling reports
carries essentially **no contextual information**; it is shared digits, dates
and Latin-script named entities, exactly the leakage the module docstring
already warns about for the lexical floor. `en-fr-fair` "reaching 0.90 at L1"
is that, not alignment. Same for `en-de-fair` at L1 under `mean_nobos`.

## T3. Does the L13-16 gap stay ~0 under every pooling? **No — and that is the finding**

| pooling | en-de | en-fr | en-ar | en-zh |
|---|---|---|---|---|
| mean | +0.002 | +0.000 | +0.004 | +0.011 |
| mean_nobos | +0.001 | +0.001 | +0.004 | +0.007 |
| weighted | +0.004 | +0.006 | -0.001 | +0.045 |
| **last** | **+0.075** | **+0.274** | **-0.102** | **+0.574** |

T3 holds for the three mean-family poolings and fails badly for `last`. The
cause is not a bug — it is **saturation**, which CLAUDE.md section 6b already
identifies as this metric's core weakness:

| pair | `mean` L13-16 range, fair | `mean` L13-16 range, starved |
|---|---|---|
| en-de | 0.993-0.996 | 0.992-0.995 |
| en-fr | 0.994-0.997 | 0.993-0.997 |
| en-ar | 0.967-0.985 | 0.960-0.983 |
| en-zh | 0.959-0.973 | 0.944-0.966 |

Both conditions sit at 0.94-1.00. A zero gap between two ceilinged numbers is
**not** evidence that the conditions are equal — it is evidence that the
estimator has run out of range. `last` does not ceiling (`starved_max` is 0.549
for fr and 0.736 for zh against fair's 0.925 / 0.946) and reports a large
top-layer difference in the same direction as everywhere else.

So the clause *"the destarved tokenizer does not increase maximum alignment at
late layers"* is **not established**. It is what a saturated estimator must
report regardless of the truth.

Read `last`'s magnitudes with care, though: its per-layer curves are
non-monotonic (en-ar dips to 0.10-0.22 across L8-12 before recovering to 0.90
at L15-16, which is the whole of the -0.648 at L9-12), it is a
single-token readout rather than a sentence one, and MEXA itself prefers
`weighted` over it. It is strong enough to **falsify** the no-gap-at-the-top
claim, not to pin down the size of the real gap.

## T4. Length control — the gap is not driven by fertility

L5-8 gap, recomputed on tertiles of the per-sentence `starved - fair`
token-count difference (summed over the pair's two languages). Retrieval is
re-run from scratch on each subset, pure-CPU from the cached embeddings.

| pair | pooling | all | smallest | mid | largest | `gap_at_0` | retained |
|---|---|---|---|---|---|---|---|
| en-de | mean | +0.300 | +0.255 | +0.229 | +0.195 | +0.293 | 98% |
| en-de | mean_nobos | +0.293 | +0.251 | +0.219 | +0.173 | +0.301 | 103% |
| en-de | weighted | +0.338 | +0.286 | +0.335 | +0.312 | +0.286 | 85% |
| en-fr | mean | +0.145 | +0.125 | +0.078 | +0.093 | +0.132 | 92% |
| en-fr | mean_nobos | +0.095 | +0.081 | +0.047 | +0.076 | +0.070 | 74% |
| en-fr | weighted | +0.229 | +0.183 | +0.241 | +0.280 | +0.121 | 53% |
| en-ar | mean | +0.456 | +0.389 | +0.393 | +0.382 | +0.397 | 87% |
| en-ar | mean_nobos | +0.493 | +0.452 | +0.434 | +0.372 | +0.519 | 105% |
| en-ar | weighted | +0.429 | +0.368 | +0.361 | +0.351 | +0.382 | 89% |
| en-zh | mean | +0.362 | +0.328 | +0.337 | +0.341 | +0.320 | 88% |
| en-zh | mean_nobos | +0.612 | +0.577 | +0.539 | +0.490 | +0.642 | 105% |
| en-zh | weighted | +0.580 | +0.542 | +0.558 | +0.592 | +0.502 | 87% |

(`gap_at_0` linearly extrapolates the three tertile gaps to a zero length
difference; `retained` = `gap_at_0 / all`. `last` rows are omitted here: its
`all` gap is ~0 for de/fr, so the ratio is a near-zero-denominator artifact
(-0.9, -3.1) rather than a measurement.)

**The gap barely moves with fertility.** Within a pair it is nearly flat across
tertiles (en-ar `mean`: .389 / .393 / .382; en-zh `mean`: .328 / .337 / .341),
and extrapolating to an identical-length corpus retains **85-105%** of it in 10
of 12 mean-family cells. If length were the mechanism, the gap would fall
towards zero as the tokenizers converge on the same token count. It does not.

Caveat: even the smallest tertile still has a median difference of +9 to +12
tokens, so this is an extrapolation, not a zero-difference control. Note also
that the tertile gaps are *smaller* than the full-set gap for every pooling —
part of that is the subset being an easier retrieval task (669 candidates
instead of 2009), which is why only the trend across tertiles, not the
absolute level, is interpreted.

## T5. Does `mean_nobos` move the gap toward `weighted`/`last`? **No**

L5-8 gap, `mean` -> `mean_nobos`:

| pair | mean | mean_nobos | direction | `last` |
|---|---|---|---|---|
| en-de | +0.300 | +0.293 | flat | +0.017 |
| en-fr | +0.145 | +0.095 | down 34% | +0.025 |
| en-ar | +0.456 | +0.493 | **up 8%** | +0.130 |
| en-zh | +0.362 | +0.612 | **up 69%** | +0.496 |

Masking BOS moves the gap *away* from `last` for the two cross-script pairs and
leaves de essentially unchanged; only fr shrinks. So **BOS / attention-sink
dilution is not the mechanism**, even though it was the most mechanically
plausible one (its weight is `1/T`, and `T` is exactly what the tokenizer
changes). Whatever drives the L5-8 gap is distributed across the sentence, not
concentrated in the first position.

---

## T6. The 2B tier settles T3: the top-layer gap vanishes **iff** the metric ceilings

The secondary tier (`en-{ar,de,fr,zh}-{fair,starved}-2b`, same four poolings) was
run because at 2B the cross-script pairs have not yet saturated. That turns the
saturation argument above from an inference into a measurement — a natural
experiment with saturated and unsaturated cells side by side under the *same*
estimator.

`mean` pooling, own pair, best layer:

| tier | pair | fair_max | starved_max | **peak gap** | saturated? |
|---|---|---|---|---|---|
| 30B | en-de | 0.996 | 0.995 | +0.001 | yes |
| 30B | en-fr | 0.997 | 0.997 | -0.000 | yes |
| 30B | en-ar | 0.985 | 0.983 | +0.001 | yes |
| 30B | en-zh | 0.973 | 0.966 | +0.006 | yes |
| 2B | en-de | 0.998 | 0.992 | +0.005 | yes |
| 2B | en-fr | 0.996 | 0.993 | +0.003 | yes |
| **2B** | **en-ar** | **0.693** | **0.563** | **+0.130** | **no** |
| **2B** | **en-zh** | **0.592** | **0.462** | **+0.129** | **no** |

**The peak gap is ~0 in exactly the six cells where both conditions are
ceilinged, and ~+0.13 in exactly the two where they are not.** Saturation is
not merely a plausible explanation for the vanishing top-layer gap; it is a
perfect predictor of it across two token budgets and four language pairs.
(+0.130 / +0.129 also reproduce the pre-registered 2B peak-layer values
exactly, a third independent check on the harness.)

At 2B the top-layer gap is large under **every** pooling, so T3's premise fails
here for the mean family too, not just for `last`:

| pooling | en-ar L13-16 | en-ar peak | en-zh L13-16 | en-zh peak |
|---|---|---|---|---|
| mean | +0.144 | +0.130 | +0.122 | +0.129 |
| mean_nobos | +0.147 | +0.129 | +0.130 | +0.177 |
| weighted | +0.144 | +0.144 | +0.133 | +0.148 |
| last | +0.110 | +0.104 | +0.184 | +0.103 |

**Consequence for §6b's "the peak-layer effect decays to ~0 by 23B".** That
decay is measured with an estimator whose ceiling the models cross somewhere
between 2B and 30B: ar/zh `fair_max` goes 0.693 → 0.985 and 0.592 → 0.973. A
statistic that cannot exceed 1.0 must show a shrinking gap as both arms
approach it, whether or not the underlying difference shrinks. **The decay is
therefore not established** — it is confounded with saturation, and separating
the two needs a non-ceilinged statistic (`cka`, `dprime`, `cosine_margin`) or
`last`, on the intermediate budgets.

The 2B tier also reproduces the main conclusions independently: L5-8 gap
positive in all 4 pairs under all 4 poolings (mean +0.305/+0.116/+0.197/+0.254,
weighted +0.321/+0.218/+0.234/+0.193), and the depth-to-0.90 delay positive in
every cell where the threshold is reachable (+1 to +7).

---

## What to quote

- **Quote `weighted` (MEXA's default) or `mean`, in the mid-stack.** They agree
  on sign and rank everywhere; `weighted` is the estimator with a published
  validation behind it, and it is the more conservative choice for early layers
  because it down-weights the leakage below.
- **Never quote layers 0-2 from any mean-family pooling.** They are at or near
  the model-free TF-IDF floor and `last` puts them at ~0.004: that is lexical
  overlap, not representation alignment.
- **Never quote a top-layer or peak-layer gap from a mean-family pooling
  without first checking whether the cell is ceilinged.** Print `fair_max` and
  `starved_max`: if both exceed ~0.95 the gap is uninformative regardless of
  its value (T6). Report `last`, or a non-saturating statistic (`cka`,
  `dprime`, `cosine_margin`), or say explicitly that the cell is ceilinged.
  This applies to any *trend* across token budgets too, since the models cross
  the ceiling somewhere inside the 2B-30B range.
- The depth-of-emergence ordering (**starved emerges deeper**) is robust: 12/12
  mean-family cells, both tokenizer conditions, all four pairs, and it is not
  explained by fertility (T4) or by BOS (T5).

## Files

- `en-{ar,de,fr,zh}-{fair,starved}[-2b].json` — 16 per-model files (30B finals
  and the 2B tier), all four poolings, same schema as
  `results/alignment_v2_107/` with a `poolings` level inserted, plus
  `token_counts` per language for T4. The per-query bootstrap arrays
  (`hits`, `dprime_*_q`) are stripped here to keep the set at 9 MB instead of
  ~200 MB; the untrimmed files and the 4-pooling embedding cache (~85 GB) exist
  only on the eval box, as in section 6b.
- `tables.json` / `tables_2b.json` — T1/T2/T4 as data, per tier.
- Regenerate: `scripts/external_bench/run_pooling_sweep.sh`, then
  `scripts/external_bench/analyze_pooling.py <results> --emb-dir <emb>`.
  Guard: `scripts/external_bench/verify_pooling_guard.py`.

## Limits

One training run per cell, so these gaps have no error bars (the *deltas* are
over a fixed 2009-sentence pool, but the runs are n=1) — the same limit as
every other single-run contrast in this repo. `centered`/`mutual_nn` only;
`raw` is in the JSONs but was not analysed. Only the 2B and 30B tiers were run,
so T6 brackets the saturation crossover without locating it — the intermediate
budgets (5b/8b/10b/15b/23b) would, and are now a ~5 min/model job. T4's
smallest tertile still carries a median +9 to +12 token difference, so its
zero-difference figure is an extrapolation.
