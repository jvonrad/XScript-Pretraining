# Draft addendum for CLAUDE.md §6b

Paste as a new subsection at the end of §6b (before §6c). Written in §6b's
voice. Two lines elsewhere also need editing — see "Edits outside §6b" at the
bottom.

---

### Pooling sensitivity: the depth-of-emergence finding survives, the "no gap at the top" clause does not

Our pooling is an **unweighted mean over all non-pad tokens, BOS included**
(`_encode` passes `bos=True`). MEXA (arXiv 2410.05873) uses a
**position-weighted** average, `w_t = t / Σ_k k`, and only ever compares that
against **last-token** pooling — plain unweighted mean appears nowhere in the
paper. So the estimator every number in §6b rests on was unvalidated on the one
axis this project is *about*: the starved tokenizer emits **1.14–1.32× more
tokens** for the same text, and mean pooling's weight on BOS is `1/T` with half
its mass on the first half of the sentence. MEXA's weighting puts ~25% there and
`~2/T²` on BOS; last-token puts 0 on both. A fair-vs-starved difference could
therefore have been manufactured by the pooling rule interacting with fertility.

`_embed` now emits **four poolings from one forward pass** (`mean`,
`mean_nobos`, `weighted`, `last`) — `layer_reps` already returns the full
`(n_layers+1, B, T, d)` stack, so the extra poolings cost no accelerator time.
Run over the 8 EN-anchored bilinguals at 30B **and the same 8 at `-2b`**, own
trained pair, `centered`, `mutual_nn`. Full tables:
`results/alignment_pooling/README.md`. The whole sweep is ~5 min/model on a
`trn2.3xlarge` core-pair, so re-running it on another budget tier is cheap.

**Guarded first, in two stages** (this is the `rawscores.check_reproduces()`
rule, and the two stages exist so "wrong environment" and "patch perturbs
numerics" could not be confused): `mean` alone, then `mean` inside the
4-pooling run, both against `results/alignment_v2_107/en-fr-fair.json` — **max
|Δ| = 0.000e+00 on mutual_nn and dprime across all 340 (pair × variant × layer)
cells**, plus the same for metrics re-derived from the cached embeddings.
Bit-for-bit, not "within tolerance". `mean` is exact *by construction*: its
weight matrix **is** the original 0/1 length mask, so the pooling expression is
character-identical to the pre-patch line.

**✅ The mid-stack gap is NOT a pooling artifact.** L5-8, fair − starved:

| pair | mean | mean_nobos | **weighted** (MEXA) | last |
|---|---|---|---|---|
| en-de | +0.300 | +0.293 | **+0.338** | +0.017 |
| en-fr | +0.145 | +0.095 | **+0.229** | +0.025 |
| en-ar | +0.456 | +0.493 | **+0.429** | +0.130 |
| en-zh | +0.362 | +0.612 | **+0.580** | +0.496 |

Under MEXA's own default the gap is **larger** in 3 of 4 pairs. Down-weighting
exactly the early, token-identity-driven positions that the artifact story
blames should have shrunk it; it grows. The depth-to-0.90 delay is likewise
positive in **all 12** mean-family cells (+2 to +9 layers). Two further
controls, both negative for the artifact story: restricting to the tertile of
sentences where the two tokenizers most nearly agree on length leaves the gap
essentially flat (en-ar 0.389/0.393/0.382 across tertiles), and extrapolating
to an identical-length corpus retains **85–105%** of it in 10 of 12 cells; and
masking BOS moves the gap *away* from last-token pooling for ar/zh rather than
toward it, so **BOS/attention-sink dilution is not the mechanism** either.

**⛔ RETRACTED: "the tokenizer does not increase maximum alignment at late
layers" — and with it "the peak-layer effect decays to ~0 by 23B".** Both are
**saturation artifacts**, i.e. the same weakness this section already documents
for top-1 retrieval, one level down. At L13-16 mean pooling puts *both*
conditions at 0.94–1.00 (en-fr 0.994–0.997 fair vs 0.993–0.997 starved), so the
~0 gap is the estimator running out of range, not the conditions being equal.

Running the **2B tier** turns that from an inference into a measurement, because
at 2B the cross-script pairs have not yet ceilinged — saturated and unsaturated
cells side by side under the *same* estimator:

| tier | pair | fair_max | starved_max | peak gap | ceilinged? |
|---|---|---|---|---|---|
| 30B | en-de / en-fr / en-ar / en-zh | .996/.997/.985/.973 | .995/.997/.983/.966 | +.001/−.000/+.001/+.006 | yes |
| 2B | en-de / en-fr | .998/.996 | .992/.993 | +.005/+.003 | yes |
| **2B** | **en-ar** | **0.693** | **0.563** | **+0.130** | **no** |
| **2B** | **en-zh** | **0.592** | **0.462** | **+0.129** | **no** |

**The peak gap is ~0 in exactly the six ceilinged cells and ~+0.13 in exactly
the two that are not** — across two budgets and four pairs, and at 2B it is
large under *all four* poolings (ar/zh L13-16: mean +.144/+.122, weighted
+.144/+.133, last +.110/+.184). So the reported decay to ~0 by 23B is
confounded with `fair_max` climbing 0.693 → 0.985 (ar) and 0.592 → 0.973 (zh):
a statistic bounded by 1.0 *must* show a shrinking gap as both arms approach it.
Separating real decay from ceiling needs a non-saturating statistic (`cka`,
`dprime`, `cosine_margin`) or `last` on the intermediate budgets — none of
which has been run.

`last` corroborates at 30B (`starved_max` **0.549** fr, **0.736** zh against
fair's 0.925/0.946), but read its *magnitudes* cautiously: its curves are
non-monotonic (en-ar dips to 0.10–0.22 across L8-12 before recovering, which is
the whole of its −0.648 there) and MEXA itself prefers `weighted`.

**⛔ Layers 0-2 of any mean-family pooling are lexical overlap, not alignment.**
The suspicious `en-fr-fair` "reaches 0.90 at L1" is confirmed leakage. Layer 0
is the **embedding** output, before any block; mean-pooled it retrieves en-fr at
**0.639**, beside the model-free TF-IDF floor of 0.533 — and it can be nothing
else, since a mean over token embeddings is a bag of tokens. The last token of
that same layer retrieves at **0.004**:

| pair | mean L0 | weighted L0 | last L0 | TF-IDF floor |
|---|---|---|---|---|
| en-de | 0.509 | 0.414 | **0.003** | 0.397 |
| en-fr | 0.639 | 0.573 | **0.004** | 0.533 |
| en-ar | 0.352 | 0.321 | **0.007** | 0.121 |
| en-zh | 0.367 | 0.311 | **0.004** | 0.175 |

This is the **eighth** time in this project an uncontrolled measurement choice
turned out to be measuring the benchmark rather than the training — and the
second inside §6b alone, after the fixed-layer probe. The pattern is identical:
the number was not wrong, the estimator was undefined on the axis being varied.

**What to quote now.** `weighted` (published validation, more conservative in
early layers) or `mean` — they agree on sign and rank everywhere — and **only
in the mid-stack**. Never layers 0-2 from a mean-family pooling. Never a
top-layer or peak-layer gap from one either: report `last`, or a
non-saturating statistic (`cka`, `dprime`, `cosine_margin`), or state that the
cell is ceilinged. The depth-of-emergence ordering (**starved emerges deeper**)
stands, and is now controlled for fertility and for BOS.

Caveats: one run per cell, so no error bars on the runs. Only 2B and 30B were
run, so the saturation crossover is bracketed but not located — the
intermediate tiers (5b/8b/10b/15b/23b) would locate it, and that is now the
obvious next step. T4's smallest tertile still carries a median +9 to +12 token
difference, so its zero-difference number is an extrapolation, not a control.
`centered`/`mutual_nn` only.

---

## Edits outside §6b

1. **§6b, "Resolution: alignment transfer, corrected for layer-selection bias"**
   — the peak-layer Δ table is computed on mean pooling at each model's argmax
   layer, i.e. squarely inside the saturated region. Add a pointer: *"⚠️ peak
   layer is ceilinged under mean pooling (both conditions 0.94–1.00 at 30B) —
   see the pooling addendum; the ordering is unaffected but the magnitudes are
   compressed, and any decay-with-budget read off this table is confounded with
   the models crossing the metric's ceiling."*
   The same warning applies to the **"Sweep result … top-1 retrieval is
   SATURATED and unusable"** paragraph, which already reaches this conclusion
   for top-1 and for the `SAT`-flagged delta rows: the pooling sweep shows the
   *same* ceiling silently governs `mutual_nn` peak/late-layer gaps, which are
   not `SAT`-flagged anywhere.

2. **CLAUDE.md TL;DR** — the line "Seven times now an evaluation number turned
   out to be measuring the benchmark rather than the training" becomes **eight**,
   adding: *mean-pooling's saturation at the top of the stack and its
   bag-of-tokens leakage at the bottom*.
