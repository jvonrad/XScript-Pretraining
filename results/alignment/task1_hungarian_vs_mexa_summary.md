# Task 1: does Limisiewicz et al.'s retrieval protocol null out on our models?

**Setup.** Raw (uncentered), mean-pooled FLORES+ embeddings, final checkpoint, all 17
layers, for the 4 EN-anchored bilingual pairs x 2 tokenizer conditions (8 models).
Three matching rules — Hungarian optimal assignment (`scipy.optimize.linear_sum_assignment`
on the negated cosine matrix), MEXA mutual-NN (argmax both directions agree), plain
one-directional top-1 argmax — each at n=1000 (5 random subsamples, seeds 0-4, averaged)
and n=2009 (full pool).

## Headline

**No — Hungarian matching shrinks the fair-minus-starved gap a lot, but does not null
it on the cross-script pairs.** Sample size (1000 vs 2009) changes almost nothing.
The disagreement with Limisiewicz's published null is attributable to the **matching
rule**, not to n — but even under their exact rule, at their exact scale, a real
allocation-driven gap survives on our decoder models.

## Mean gap across layers >= 1 (excludes the layer-0 byte-fragmentation artifact)

| pair  | rule       | gap @ n=1000 | gap @ n=2009 |
|-------|------------|-------------:|-------------:|
| en-ar | hungarian  | 0.172        | 0.192        |
| en-ar | mexa_mutual| 0.290        | 0.280        |
| en-zh | hungarian  | 0.064        | 0.070        |
| en-zh | mexa_mutual| 0.199        | 0.198        |
| en-de | hungarian  | 0.002        | 0.010        |
| en-de | mexa_mutual| 0.179        | 0.185        |
| en-fr | hungarian  | -0.0003      | 0.004        |
| en-fr | mexa_mutual| 0.072        | 0.078        |

(en-de/en-fr are same-script Latin-Latin pairs, where the fair/starved tokenizers barely
differ in Latin allocation to begin with — 29,038 vs 30,561 pieces, ~5%. Their near-zero
Hungarian gap is a sanity floor, not evidence the rule nulls a real cross-script effect.)

## 2x2 decomposition (rule x n) on the actual cross-script pairs

- **n effect**: trivial. 1000 -> 2009 moves every cell by <=0.02 absolute, an order of
  magnitude smaller than the rule effect. Tatoeba's n<=1000 is *not* why they saw a null.
- **rule effect**: large. Hungarian cuts the gap by ~35% on en-ar (0.19 vs MEXA's 0.28)
  and ~65% on en-zh (0.07 vs MEXA's 0.20), because global optimal assignment resolves
  most of the hubness/argmax errors that sink MEXA's mutual-NN score in early-to-mid
  layers (see `fig_hungarian_vs_mexa_vs_top1.png`: solid Hungarian lines reach >0.9 by
  layer 2-4 for `fair`, while dashed MEXA lines don't clear 0.9 until layer 8-11).
- **Absolute-vs-gap**: by layer 9-11 essentially every condition/pair saturates near
  1.0 under Hungarian (matches the brief's mu_Max~0.963≈0.963 washout at the single best
  layer). If Limisiewicz-style reporting focuses on a best/near-ceiling layer rather than
  a layer-mean, that ceiling effect *plus* the more forgiving matching rule together would
  produce something close to their reported null. Averaged over layers (our mean-layer
  view, matching how we report MEXA), the gap survives: starved models take measurably
  more layers to reach that ceiling (en-ar starved lags fair by ~5-7 layers before both
  saturate), which the peak-layer/best-layer style of reporting hides entirely.

## Answer to the acceptance question

The disagreement between our MEXA numbers and Limisiewicz's null is **driven by the
matching rule (and implicitly, peak-vs-mean layer reporting), not by sample size**.
Reproducing their exact protocol (Hungarian, n<=1000) on our models does narrow the
gap substantially but does not eliminate it on en-ar/en-zh — so the contradiction is
partly a methodological artifact (their rule is far more forgiving) and partly real:
our decoder-only, cross-script bilingual models still show an allocation-driven
alignment penalty that a purely encoder/Tatoeba-style evaluation would undersell.
