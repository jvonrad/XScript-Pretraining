# Vocabulary overlap between languages, fair vs starved

`scripts/external_bench/vocab_overlap.py`, FLORES+ dev+devtest (n=2009
**parallel** sentences, so all five languages carry identical content),
real 65536-piece `unigram_destarved` (fair) / `unigram_starved` (starved)
from `/mnt/scratch/xscript/tokenizers`. Tables: `overlap.md`, raw
`overlap.json`.

## 1. Cross-script overlap is structurally ZERO — everything measured is floor

Not one Arabic-script piece is shared with any Latin language, not one Han
piece, and `ar ∩ zh` contains no native piece of either script. Every
cross-script "shared" type is punctuation/digits or a Latin string embedded
in the non-Latin text (names, acronyms, URLs):

| pair | shared types (fair) | of which sym/num | of which Latin | native-script |
|---|---|---|---|---|
| en-ar | 479 | 302 | 177 | **0** |
| en-zh | 1123 | 291 | 832 | **0** |
| ar-zh | 399 | 271 | 128 | **0** |

Restricted to genuine subwords (letter-bearing, ≥3 chars) `en ∩ ar` is
**87 pieces, 100% Latin** (`▁to`, `▁of`, `▁Google`) out of ~7.9k on each
side. This is the input-layer version of §6b's lexical floor, and it means
**neither tokenizer can produce cross-script subword sharing at all** — the
fair/starved contrast across scripts is a contrast between two floors.

## 2. Raw overlap says starved ≫ fair — but that is mostly a set-size artifact

| pair | Jaccard fair | Jaccard starved | Δ |
|---|---|---|---|
| en-de | 0.195 | 0.385 | −0.190 |
| en-fr | 0.255 | 0.478 | −0.223 |
| de-fr | 0.161 | 0.359 | −0.198 |
| en-zh | 0.078 | 0.133 | −0.054 |
| en-ar | 0.029 | 0.062 | −0.033 |

The fair tokenizer emits **1.6–3.7× more distinct types** for the same text
(ar 2431 → 8943, zh 3912 → 7311), and Jaccard falls mechanically as the sets
grow. Controlling for it — top-K most frequent types per language, K equal
across languages *and* tokenizers — the ordering survives but shrinks:

| pair | top-2000 shared, fair | starved | Δ |
|---|---|---|---|
| en-de | 398 | 643 | −245 |
| en-fr | 492 | 836 | −344 |
| en-zh | 79 | 169 | −90 |
| en-ar | 49 | 79 | −30 |

**So the starved tokenizer really does force more vocabulary sharing**, by
~1.6× at matched size, and the effect is largest same-script.

## 3. The mechanism is piece length, and it costs the non-Latin languages most

Mean emitted piece length (chars) fair → starved: en 4.41→3.67, de
4.55→3.32, fr 4.25→3.27, **ar 3.77→2.55**, **zh 1.47→1.13**. Short pieces
are generic and get reused across languages, which is the entire source of
the higher starved overlap. Count of own-script pieces ≥3 chars:

| lang | fair | starved | ratio |
|---|---|---|---|
| en / de / fr (Latin) | 7281 / 8377 / 7883 | 4945 / 4344 / 4431 | 0.68 / 0.52 / 0.56 |
| ar (Arabic script) | 7814 | 1386 | **0.18** |
| zh (Han) | 580 | **4** | **0.007** |

Starvation leaves Chinese with **four** Han pieces of ≥3 characters, and
Arabic with a fifth of its vocabulary. It buys higher overlap by deleting
non-Latin vocabulary, not by building shared vocabulary.

⚠️ **The `en-zh` subword overlap coefficient (0.888 starved vs 0.452 fair)
is an artifact of exactly this and must not be quoted.** 621 of starved zh's
625 ≥3-char pieces *are Latin*, so "almost all of Chinese's long pieces are
shared with English" is a restatement of "Chinese has no long pieces".

## 4. Vocabulary sharing and parametric sharing move in OPPOSITE directions

§6j: the fair tokenizer increases parametric knowledge sharing (fair ≥
starved in all 8 cells, Arabic-led, transfer rate .334 vs .098). Here the
fair tokenizer *decreases* vocabulary overlap in every one of the 10 pairs,
under every estimator. So forcing languages onto a shared, shorter piece
inventory at the input does **not** produce shared internal representations
— it produces fragmentation, and the model then stores the same fact in
disjoint neurons. Sharing embeddings is not sharing knowledge.

## Caveats

* Usage-based sets, so they depend on the measurement corpus (FLORES,
  news register, 2009 sentences); the Zipf tail is truncated at the same
  content for all five languages, which is what makes the comparison fair,
  but "types used" is not "vocabulary size".
* One deterministic measurement per cell — no CIs (the sets are essentially
  saturated at n=2009 for the frequent pieces that carry the mass).
* Script bucketing is `tok/analyze.py`'s coarse codepoint map; `mixed`
  pieces are counted by majority letter script.
