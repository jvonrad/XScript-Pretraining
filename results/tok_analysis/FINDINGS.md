# Tokenizer allocation vs. the benchmark-gain ordering

Does vocabulary allocation explain why de-starving the tokenizer helps
`DE < FR < AR < ZH`? **Partly — the cross-script/same-script split is
robustly explained; the four-way ordering is not, because the four-way
ordering is not established in the data.**

Inputs: `unigram_starved` / `unigram_destarved` (65536 pieces, 4 specials,
256 byte-fallback atoms, verified against `meta.json`); FLORES dev+devtest
n=2009; the 107 cached `results/appendix_c5/*_final.json` runs.

---

## 0. Two data-provenance issues found first

**The FLORES on the eval box is a smoke-test fixture, not FLORES.**
`/mnt/scratch/smoke_neuron/flores_plus/` contains synthetic gibberish
(random letter strings for English, ~20 recycled Han characters for
Chinese). Running `analyze.py` against it produced plausible-looking but
entirely fictitious tables — including a *reversed* fertility gate
(`de=0.859` instead of `1.371`). The tokenizers alongside it are 1024-piece
smoke models, not the 65536-piece real ones. Anything computed from that
tree is void.

Real FLORES+ is gated and no HF token exists on this box, so the analysis
uses the public FLORES-200 tarball
(`dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`), validated against
the committed `results/byte_premium/byte_premiums.json`: n=997/1012 exact,
en/de byte-identical, fr/ar/zh within 0.06% (FLORES+ made a handful of
character-level corrections). Recovered byte premiums are
`de 1.1861 / fr 1.2378 / ar 1.6002 / zh 0.9249` against documented
`1.186 / 1.237 / 1.600 / 0.925`.

**Confirmation the pipeline is correct:** the regenerated gate row
reproduces CLAUDE.md's documented fertility ratios exactly —
`en=1.200, de=1.371, fr=1.301, ar=1.476, zh=1.304`.

---

## 1. The `DE < FR < AR < ZH` ordering is a single-checkpoint reading

The supplied gains reproduce **exactly**, and only, from one checkpoint per
language — bilingual `en-X-*-23b` and monolingual `X-*-12b` (both ~11.5B
tokens/language, so the matched-exposure choice is principled):

| lang | bilingual `en-X-23b` | given | monolingual `X-12b` | given |
|---|---|---|---|---|
| de | +0.38 | +0.4 | n/a | n/a |
| fr | +0.72 | +0.7 | +1.27 | +1.3 |
| ar | +1.33 | +1.3 | +1.37 | +1.4 |
| zh | +1.84 | +1.8 | +2.17 | +2.2 |

But the same quantity across the rest of the stable-LR window is unstable,
and DE's 23b point is an outlier among its own five:

| lang | 2b | 5b | 10b | 15b | **23b** | mean | sd |
|---|---|---|---|---|---|---|---|
| de | +1.58 | +1.70 | +1.97 | +2.14 | **+0.38** | +1.55 | 0.69 |
| fr | +0.50 | +1.28 | +0.52 | +3.59 | **+0.72** | +1.32 | 1.31 |
| ar | +0.56 | +1.69 | +0.69 | +0.50 | **+1.33** | +0.95 | 0.53 |
| zh | +0.44 | +1.68 | +0.67 | +0.96 | **+1.84** | +1.12 | 0.62 |

Within-language checkpoint sd is 0.53–1.31pp against a between-language
range of 1.46pp. **Averaged over the window the ordering reverses to
`AR < ZH < FR < DE`.**

Paired bootstrap (B=2000, per-example hit lists, resampling doc indices)
on the quoted 23b cells:

| lang | Δ [95% CI] | | adjacent rank | P |
|---|---|---|---|---|
| de | +0.38 [−0.66, +1.40] | | P(DE<FR) | 0.68 — not resolved |
| fr | +0.72 [−1.78, +3.34] | | P(FR<AR) | 0.76 — not resolved |
| ar | +1.33 [+0.43, +2.21] | | P(AR<ZH) | **0.999** |
| zh | +1.84 [+0.58, +3.05] | | P(DE<ZH) | **1.000** |

And that CI reflects eval-sentence sampling only — it does not include the
checkpoint noise above, so it understates the real uncertainty (same caveat
CLAUDE.md §6 makes about the BTS CIs).

**What is actually established: `ZH > AR`, and cross-script > DE at the
matched budget. The DE-vs-FR rank is not resolvable and flips with the
aggregation choice.** This is the fourth instance of the project's recurring
pattern — a headline ordering that is mostly reading one noisy draw.

---

## 2. Allocation: the primary hypothesis is confirmed

Script-level allocation (`allocation_detail.json`):

| script | cond | pieces | %64k | %single-char | **multi-char** | mean len | median |
|---|---|---|---|---|---|---|---|
| Latin | starved | 29038 | 44.31 | 2.47 | 28322 | 4.31 | 4 |
| Latin | destarved | 30561 | 46.63 | 1.31 | 30161 | 6.22 | 6 |
| Latin | **ratio d/s** | 1.05 | | 0.53 | **1.065** | 1.44 | |
| Arabic | starved | 3718 | 5.67 | 5.22 | 3524 | 2.94 | 3 |
| Arabic | destarved | 13132 | 20.04 | 1.00 | 13001 | 4.69 | 5 |
| Arabic | **ratio d/s** | 3.53 | | 0.19 | **3.689** | 1.59 | |
| Han | starved | 9093 | 13.87 | **97.77** | **203** | 1.02 | 1 |
| Han | destarved | 17625 | 26.89 | 55.30 | **7878** | 1.57 | 1 |
| Han | **ratio d/s** | 1.94 | | 0.57 | **38.8** | 1.53 | |

Piece-length distribution (chars, 8 = 8+):

| script | cond | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---|---|---|---|---|---|---|---|---|
| Latin | starved | 716 | 3097 | 6811 | 7277 | 4669 | 2934 | 1794 | 1740 |
| Latin | destarved | 400 | 1455 | 3131 | 4026 | 4233 | 4389 | 3842 | 9085 |
| Arabic | starved | 194 | 1224 | 1401 | 571 | 230 | 62 | 22 | 14 |
| Arabic | destarved | 131 | 924 | 2626 | 2860 | 2607 | 1867 | 1308 | 809 |
| Han | starved | **8890** | 197 | 5 | 1 | 0 | 0 | 0 | 0 |
| Han | destarved | 9747 | 6377 | 1044 | 387 | 34 | 20 | 5 | 11 |

Emitted tokens on FLORES:

| lang | cond | %1-char | %byte | %multi | mean len | uniq used | uniq own-script multi-char | %tok own-script multi-char |
|---|---|---|---|---|---|---|---|---|
| en | starved | 17.91 | 0.000 | 81.57 | 3.07 | 5767 | 5321 | 80.35 |
| en | destarved | 17.80 | 0.000 | 81.58 | 3.68 | 8119 | 7628 | 80.15 |
| de | starved | 16.44 | 0.007 | 83.03 | 2.85 | 5274 | 4812 | 82.07 |
| de | destarved | 16.80 | 0.009 | 82.46 | 3.91 | 9282 | 8784 | 81.18 |
| de | **ratio** | 1.021 | | **0.993** | 1.371 | 1.760 | **1.825** | **0.989** |
| fr | starved | 22.35 | 0.000 | 76.97 | 2.74 | 5285 | 4870 | 76.16 |
| fr | destarved | 23.86 | 0.000 | 75.16 | 3.57 | 8685 | 8247 | 74.06 |
| fr | **ratio** | 1.067 | | **0.976** | 1.301 | 1.643 | **1.693** | **0.972** |
| ar | starved | 25.37 | 0.000 | 72.20 | 2.12 | 2431 | 1836 | 71.16 |
| ar | destarved | 17.40 | 0.000 | 80.69 | 3.13 | 8943 | 8327 | 79.08 |
| ar | **ratio** | 0.686 | | **1.118** | 1.476 | 3.679 | **4.535** | **1.111** |
| zh | starved | **87.22** | 0.062 | **9.04** | 1.07 | 3912 | **154** | **6.17** |
| zh | destarved | 58.63 | 0.000 | 37.11 | 1.40 | 7311 | 3762 | 33.19 |
| zh | **ratio** | 0.672 | | **4.103** | 1.304 | 1.869 | **24.4** | **5.377** |

**The hypothesis's central prediction holds precisely.** ZH has the largest
change in multi-character availability (Han multi-char pieces 203 → 7878,
**38.8x**; share of emitted Chinese tokens that are multi-character Han
6.17% → 33.19%, **5.4x**) despite the *second-smallest* fertility change
(1.304). DE has essentially **no** change in multi-character availability
(Latin multi-char pieces 1.065x; emitted multi-char share 83.03% → 82.46%,
ratio 0.993 — i.e. none) despite the *largest* fertility change (1.371).

### The mechanism: character-inventory cost

| lang | distinct chars on FLORES | own-script pieces (starved) | of which single-char |
|---|---|---|---|
| en/de/fr | 70 / 73 / 84 | 29038 | 716 |
| ar | 91 | 3718 | 194 |
| zh | **2513** | 9093 | **8890** |

Han must spend ~9k slots on bare character coverage before a single word
piece can exist. Under starved it receives 9093 and spends 8890 of them on
single characters, leaving **203** for everything else — Chinese has an
alphabet and essentially no vocabulary. De-starving is what first buys
Chinese a multi-character vocabulary at all.

This is also why BPB barely moves for ZH: a single-character Han token
already carries ~3 bytes, so bits-*per-byte* is nearly unaffected by having
no word-level pieces, while the downstream benefit of having them is real.
BPB penalty tracks fertility (mechanically); multi-character availability is
a different axis that BPB is close to blind to.

---

## 3. Falsifiers

**(a) "If ZH's single-char share is similar across conditions, the
hypothesis is dead." — DOES NOT HOLD (hypothesis survives).** Han
single-char share 97.77% → 55.30%; emitted single-char Chinese tokens
87.22% → 58.63%. The largest change of any language on both measures.

**(b) "If DE's multi-char availability changes as much as ZH's, the
hypothesis is dead." — DOES NOT HOLD (hypothesis survives).** Latin
multi-char pieces change by 1.065x vs Han's 38.8x, a factor of ~36
difference; DE's emitted multi-char share actually *falls* slightly
(ratio 0.993). Note DE and FR share the Latin bucket entirely, so
script-level allocation cannot distinguish them even in principle.

**(c) "Does ANY single allocation statistic reproduce `DE<FR<AR<ZH`?"**
26 candidates tested; **all are listed** in the table below (none dropped).
Four reproduce the quoted ordering, but three of them tie DE with FR
(both Latin) and the tie is broken arbitrarily. **Exactly one statistic
orders all four distinctly and matches: `E8 = % of emitted tokens that are
single-character under the starved tokenizer`** (de 16.4 < fr 22.4 <
ar 25.4 < zh 87.2, Spearman +1.00 vs the quoted gains, −0.80 vs BPB penalty).

**That single match is exactly what fishing produces and should not be
quoted as a result.** With n=4, a random statistic reproduces a specific
4-way ordering with probability 1/24 ≈ 4.2%; over 26 candidates the expected
number of chance matches is ~1.1. Finding one is the null expectation.

The honest reading of the table is not the winner but the *pattern*: every
operationalization of the mono-vs-multi-character construct (A2, A8, E1, E2,
E6, E7, E8, E11, E12) agrees on `{DE,FR} << AR << ZH`, and they disagree
only on the DE/FR tiebreak — where most **usage-based** statistics put FR
*below* DE, the reverse of the quoted ordering. Since the DE-vs-FR benchmark
gap is itself unresolvable (§1, P=0.68), there is nothing there to explain.

| statistic | DE | FR | AR | ZH | ordering | ρ vs quoted gain | ρ vs window-mean gain | ρ vs BPB penalty |
|---|---|---|---|---|---|---|---|---|
| A1 script pieces, ratio d/s | 1.052 | 1.052 | 3.532 | 1.938 | DE=FR<ZH<AR | 0.74 | −0.95 | −0.21 |
| A2 script multi-char pieces, ratio d/s | 1.065 | 1.065 | 3.689 | 38.81 | DE=FR<AR<ZH | 0.95 | −0.74 | −0.63 |
| A3 script %single-char, ratio d/s | 0.531 | 0.531 | 0.191 | 0.566 | AR<DE=FR<ZH | 0.32 | 0.32 | −0.63 |
| A4 script mean piece len, ratio d/s | 1.442 | 1.442 | 1.594 | 1.532 | DE=FR<ZH<AR | 0.74 | −0.95 | −0.21 |
| A5 script median piece len, ratio d/s | 1.500 | 1.500 | 1.667 | 1.000 | ZH<DE=FR<AR | −0.32 | −0.32 | 0.63 |
| A6 script multi-char pieces (starved) | 28322 | 28322 | 3524 | 203 | ZH<AR<DE=FR | −0.95 | 0.74 | 0.63 |
| A7 script multi-char pieces (destarved) | 30161 | 30161 | 13001 | 7878 | ZH<AR<DE=FR | −0.95 | 0.74 | 0.63 |
| A8 script %single-char (starved) | 2.466 | 2.466 | 5.218 | 97.77 | DE=FR<AR<ZH | 0.95 | −0.74 | −0.63 |
| A9 script pct_of_vocab (starved) | 44.31 | 44.31 | 5.673 | 13.88 | AR<ZH<DE=FR | −0.74 | 0.95 | 0.21 |
| A10 script pct_of_vocab, ratio d/s | 1.052 | 1.052 | 3.532 | 1.938 | DE=FR<ZH<AR | 0.74 | −0.95 | −0.21 |
| E1 %emitted single-char, ratio d/s | 1.021 | 1.067 | 0.686 | 0.672 | ZH<AR<DE<FR | −0.80 | 0.60 | 0.40 |
| E2 %emitted multi-char, ratio d/s | 0.993 | 0.976 | 1.118 | 4.103 | FR<DE<AR<ZH | 0.80 | −0.60 | −0.40 |
| E3 mean emitted piece len, ratio d/s | 1.371 | 1.301 | 1.476 | 1.304 | FR<ZH<DE<AR | 0.00 | −0.40 | 0.60 |
| E4 unique pieces used, ratio d/s | 1.760 | 1.643 | 3.679 | 1.869 | FR<DE<ZH<AR | 0.60 | −0.80 | 0.00 |
| E5 unique own-script used, ratio d/s | 1.806 | 1.675 | 4.404 | 2.312 | FR<DE<ZH<AR | 0.60 | −0.80 | 0.00 |
| E6 uniq own-script multi-char, ratio d/s | 1.825 | 1.693 | 4.535 | 24.43 | FR<DE<AR<ZH | 0.80 | −0.60 | −0.40 |
| E7 %tok own-script multi-char, ratio d/s | 0.989 | 0.972 | 1.111 | 5.377 | FR<DE<AR<ZH | 0.80 | −0.60 | −0.40 |
| **E8 %emitted single-char (starved)** | **16.44** | **22.35** | **25.37** | **87.22** | **DE<FR<AR<ZH** | **1.00** | −0.80 | −0.80 |
| E9 %emitted multi-char (starved) | 83.03 | 76.97 | 72.20 | 9.04 | ZH<AR<FR<DE | −1.00 | 0.80 | 0.80 |
| E10 %tok own-script multi-char (starved) | 82.07 | 76.16 | 71.16 | 6.17 | ZH<AR<FR<DE | −1.00 | 0.80 | 0.80 |
| E11 abs gain in %emitted multi-char | −0.57 | −1.82 | 8.49 | 28.07 | FR<DE<AR<ZH | 0.80 | −0.60 | −0.40 |
| E12 abs gain in %tok own-scr multi-char | −0.89 | −2.11 | 7.92 | 27.02 | FR<DE<AR<ZH | 0.80 | −0.60 | −0.40 |
| X1 fertility ratio starved/fair | 1.371 | 1.301 | 1.476 | 1.304 | FR<ZH<DE<AR | 0.00 | −0.40 | 0.60 |
| X2 byte premium vs EN | 1.186 | 1.237 | 1.600 | 0.925 | ZH<DE<FR<AR | −0.20 | −0.40 | 0.40 |
| X3 n competing langs in script | 289 | 289 | 21 | 4 | ZH<AR<DE=FR | −0.95 | 0.74 | 0.63 |
| X4 script isolation (−X3) | −289 | −289 | −21 | −4 | DE=FR<AR<ZH | 0.95 | −0.74 | −0.63 |

All ρ are Spearman over **n=4** (n=3 vs the monolingual target, DE absent) —
reported for sign structure only; none is statistically meaningful.

---

## 4. Is it a format / answer-matching artifact?

Three independent checks, all pointing the same way: **no.**

**(i) Untrained-language control (the decisive one).** Every C.5 model is
scored on every language, so the fair−starved gain can be measured on
languages a model *never trained on*, where there is no competence to
differ and any gain must be scoring/format. Averaged over the stable window:

| eval lang | gain, models trained on it | gain, models never trained on it |
|---|---|---|
| de | **+1.55** | −0.54 |
| fr | **+1.30** | −0.15 |
| ar | **+0.96** | −0.27 |
| zh | **+1.26** | −0.38 |
| en | +1.65 | +1.33 ⚠ |

De-starving does **not** help on languages the model never learned — it
mildly hurts. The gain requires competence, so it is not a format effect.
(English is the exception, but English is not a clean control: CLAUDE.md
§6b documents heavy incidental English in every pool, so "untrained on
English" models have substantial English exposure. Treat the English row
as uninformative rather than as evidence of an artifact.)

**(ii) Scoring rule split.** Unnormalized `acc` (XNLI, XStoryCloze,
XWinograd) is biased by option token count; `acc_norm` (Belebele, ARC,
HellaSwag) is length-normalized. If the gain were option-length bias it
would concentrate in the former. It does not — trained-language gains are
de +0.78/+1.81, fr +1.49/+1.18, ar +1.10/+0.87, zh +1.60/+0.75
(acc-only / acc_norm). Present in both.

**(iii) Direction of the length-bias channel.** Unnormalized scoring's bias
scales with the *spread* of token counts across options. The destarved
tokenizer has **higher** dispersion of tokens/byte than starved in every
language (CV ratio d/s: en 1.29, de 1.78, fr 1.57, ar 1.88, zh 2.31) —
because its pieces are longer and more variable, while starved is close to
uniform near-character segmentation. So the better tokenizer should, if
anything, suffer *more* option-length bias, yet it scores higher. The
artifact channel runs opposite to the observed effect.

Not tested directly: per-option token counts on the actual benchmark answer
strings (the datasets are not on this box). The XNLI connective asymmetry
already documented in CLAUDE.md §6 is a real instance of this class of bug
for `fr`, so the channel is not hypothetical — it is just not what produces
the ZH>AR>{FR,DE} pattern here.

---

## 5. Corpus composition — allocation is downstream of the mixture

| condition | n files | composition |
|---|---|---|
| destarved | 5 | `ar, de, en, fr, zh` (no script tags) |
| starved | 419 | Latn 289 (69.0%), Cyrl 49 (11.7%), **Arab 21 (5.0%)**, Deva 13, **Hani 4 (1.0%)**, Mymr 4, Beng/Grek/Hebr 3 each, 15 scripts with 1–2 files |

Competing-language counts per script are `Latin 289 : Arabic 21 : Han 4`,
and pieces-per-competing-language run `Latin 100 : Arabic 177 : Han 2273`.
So Han is *not* starved of slots per language — it is starved of slots
*relative to its character inventory*. The mixture sets the slot budget; the
character inventory decides whether those slots can buy words. Both are
needed, and neither alone orders the four languages.

The `X3/X4` rows in §3 show the competing-language count correlates with the
quoted gains about as well as the allocation statistics do (ρ = ±0.95) while
being a *property of the training mixture, not of the vocabulary*. With n=4
and DE/FR tied, the data cannot separate "allocation" from "mixture" as the
explanatory variable — they are collinear here.

---

## 6. Bottom line

- **Explicable, robustly:** the same-script / cross-script split, and
  `ZH > AR`. Han's vocabulary under starvation is 97.8% single characters
  with 203 multi-character pieces total; de-starving multiplies that by 38.8
  while leaving Latin essentially untouched (1.065x). The primary hypothesis
  is confirmed on both of its predictions, and it explains the BPB/benchmark
  divergence: BPB tracks fertility, multi-character availability is a
  near-orthogonal axis that bits-per-byte cannot see.
- **Not explicable, because not established:** the DE-vs-FR rank. The quoted
  `DE<FR<AR<ZH` is one checkpoint pair; P(DE<FR)=0.68, P(FR<AR)=0.76, and
  the ordering reverses to `AR<ZH<FR<DE` when averaged over the stable
  window. No allocation statistic should be fit to it.
- **Not an evaluation artifact:** the gain vanishes on untrained languages,
  appears in length-normalized and unnormalized tasks alike, and the
  option-length bias channel runs the wrong way.

### Caveats that weaken specific claims

- `de-starved` monolingual does not exist (collapsed at 7.7B), so every DE
  statement rests on the bilingual `en-de` family alone, with 4 benchmark
  tasks rather than 5.
- ZH monolinguals stop at 12B, so the ZH monolingual gain has no larger
  budget to check against.
- DE and FR share the Latin allocation bucket exactly; script-level
  allocation cannot order them even in principle.
- Fertility and allocation are measured on FLORES as a proxy for the
  training pools (CLAUDE.md §7 validates this for de/zh to ~2%).
- n=4 languages throughout. All rank correlations are sign structure only.
