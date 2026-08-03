# CLAUDE.md

**Scientific findings** for XScript-Pretraining (cross-script pretraining penalty
× tokenizer starvation): what the checkpoints actually show on the intrinsic
(BPB→BTS), downstream (XNLI, Appendix-C.5 suite) and representational
(MEXA-style alignment) metrics — including, importantly, **which headline
numbers did not survive scrutiny**.

See [README.md](README.md) for the experiment design, and
**[NEURON.md](NEURON.md)** for everything about *running* this on AWS Trainium:
environment setup, dependency pins, the XLA scoring adaptation and its silent
traps, the eval fan-out, and the training port. Section numbers are preserved
across both files, so cross-references still resolve — **§3, §6, §6b, §6c, §6d,
§8 are here; §1, §2, §4, §5, §7, §9 are in NEURON.md**.

---

## TL;DR

- ✅ **`de-starved` EXISTS now (§6h)** — the missing monolingual every other
  section had to work around. 16.10B tokens, 9h36m, 76.8 of ~100 GPU-hours.
  The original did **not** collapse: it diverged at the warmup/peak-LR seam
  (BPB floor 1.2804 @0.75B, then monotonic rise as the LR pinned at 3.0e-3).
  Fixed by changing **only** `seed`/`data_seed` — the LR schedule is untouched,
  because softening it would de-match the run from de-fair and defeat its
  purpose. Checkpoints at **7.753B / 11.754B / 14.755B** land *step-for-step*
  on de-fair-8b/-12b/-15b, plus **16.100B** as the content-match of de-fair-12b
  (x1.371 fertility). The same-script starved penalty is a **flat ~+0.05 BPB**
  across all three matched budgets (not shrinking — the 7.753B point is an
  outlier). Also in §6h: **218 M tokens/GPU-hour** on GH200 (so 30B ≈ 138
  GPU-h — the number to budget with), a `train.max_tokens` cap that stops a run
  early *without* perturbing the LR, and ⛔ its `final.pt` is **mid-stable, not
  cooled** — upload it as `de-starved-16b`, never bare `de-starved`.
- ⛔ **READ §6g FIRST** (then §6e). The trajectory sweep is **done**: 100
  checkpoints x 1B-30B on SIB-200, XNLI and four MuBench families, 792 cells,
  0 anomalies. It overturns two of §6f's own rules. (1) **`acc_norm` is not
  the right estimator for MuBench** — plain `acc` discriminates better and is
  more monotone on ARC-Easy and BMLAMA; use the per-family table in §6g.
  (2) **`acc_tokennorm` must never be used** — it is algebraically guaranteed
  to favour the more fragmented candidate, and the starved tokenizer fragments
  1.14-1.32x more, so it would manufacture a fair-vs-starved effect from
  arithmetic alone. Headline science: the same-vs-cross-script transfer gap is
  **+0.014** and stable across all four LR-matched budget tiers, but it is
  carried by **French**, not by script (German sits with ar/zh) — the mirror
  image of §6e's "cross-script is really Arabic".
- ⛔ **READ §6e FIRST.** A fifth format artifact was found and fixed: SIB-200,
  Taxi-1500 and XNLI(en/de/fr) were scored by estimators whose argmax is
  decided by a per-candidate constant rather than by the document (SIB-200's
  majority class has **median recall 0.000** under `acc`). Re-scored with prior
  calibration across all 41 checkpoints, §6d's same-vs-cross-script gap falls
  +0.053 → **+0.016**, no same-script cell stays significant, the
  cross-script × starvation interaction disappears, Taxi-1500 stops
  corroborating, the fair-vs-starved effect reverses from English to the
  partner languages, and Global-MMLU turns out **not** to be at chance
  (+0.059 ≈ 15σ in cloze format at n=14,042). Runners now persist raw
  per-candidate loglikelihoods, so any future scoring change is pure CPU.
- **Most of the suite discriminates once scored correctly — "only XNLI" was
  wrong.** Own-language, all five languages, over each cell's own empirical
  null: HellaSwag **13–23σ** (the strongest), Taxi-1500 11–17σ, XNLI 5–13σ,
  SIB-200, Belebele-cloze 2.4–3.9σ. The earlier "only XNLI" verdict came from
  reading tables that pool every model over languages it never trained on,
  which drags own-language signal into the noise. `xnli_ar`/`xnli_zh` remain
  debiased automatically inside `bench.py`.
- **Cross-script costs nothing in attained capability, and the transfer gap is
  much smaller than §6d reports.** Calibrated, own-language SIB-200 is flat
  across scripts (ar .734, fr .738, en .761, de .694) — Arabic is the
  strongest non-English model. The same-vs-cross-script transfer gap falls
  from +0.053 to **+0.016**, no same-script cell remains significant, and the
  residual is **Arabic-specific, not cross-script**: zh is +0.011, identical
  to de's +0.010. There is **no cross-script × starvation interaction**
  (ar/fair −0.040 vs ar/starved −0.037). XNLI does not reproduce even the
  Arabic effect (+0.018, opposite sign), so it is not established. See §6e.
- **The `*-12b` vs `*-23b` pairing is LR-matched for free** (both mid-stable at
  peak LR; decay starts at 24B), giving 7 of 8 transfer cells clean with no new
  training — §6's matched-token table had only 2 of 7 (§6d).
- **zh HellaSwag was never missing** — the okapi data has it; 4 malformed rows
  make the split unloadable, so lm-eval ships no task for it. Repaired in §6d;
  every zh checkpoint scores .350–.411 acc_norm against .250 chance.
- **The BPB→BTS headline numbers did not survive recomputation.** They are
  confounded by LR state (cooled finals vs mid-stable intermediates) and swamped
  by checkpoint noise; the interaction is **not established** at any LR-matched
  budget (§6). The cooldown-clean W&B-curve version *does* put same-script above
  the 0.5 dilution null and cross-script below it — but the separation shrinks
  with scale, so quote the curve, not a single number.
- **Content-matched (fertility-corrected) BTS is what makes the interaction
  reproducible**: +0.0058 (FLORES) / +0.0056 (holdout), vs a sign-flipping
  token-matched version (§6).
- **Matched-token downstream transfer deltas are the best-powered evidence** —
  but ⚠️ 5 of 7 cells carry the same LR-state confound; only the `zh` cells are
  clean (§6).
- **Representation alignment: only the *ordering* is robust** (cross-script >
  same-script, the reverse of the downstream ordering). The absolute deltas
  depend on an unresolved layer-selection rule (§6b).
- **Seven times now** an evaluation number turned out to be measuring the
  benchmark rather than the training: XNLI connectives, the Belebele letter
  format, the alignment fixed-layer probe, SIB-200's `acc`/`acc_norm` length
  degeneracy, Global-MMLU's underpowered n=200, lm-eval's hardcoded English
  scaffolding, and **ARC comparing English ARC-Easy against non-English
  ARC-Challenge**. Six of the seven were invisible in the accuracy number
  alone. Control before quoting — and check *dataset identity* before any
  cross-language comparison (§6e).

---

## 3. The models & sharded checkpoints

`jvonrad/xscript-eval` (private) holds **116 entries** in `models.json` — the
15 headline models plus their token-budget intermediates:

- mono: `en-{fair,starved}`, `fr-{fair,starved}`, `ar-{fair,starved}`, `de-fair`,
  and 🆕 **`de-starved-{1,2,5,8,12,15,16}b`** (§6h retrain, 2026-08-03)
- bilingual: `en-{ar,de,fr,zh}-{fair,starved}`

`fair` = `unigram_destarved` tokenizer, `starved` = `unigram_starved`. Each model
maps to its tokenizer + training languages in `models.json`.

⛔ **There is deliberately no bare `de-starved`.** That run has no cooled 30B
final — it stops at 16.1B mid-stable — so the name would silently pair a cooled
30B de-fair against an uncooled 16B de-starved. Its largest checkpoint is
`de-starved-16b`. See §6h.

**Checkpoints are uploaded split into 5 parts** (`final.pt.part000..004` +
`n_parts.txt`) because they couldn't be pushed whole from the training cluster.
`run_benchmarks.py`'s `fetch_checkpoint()` **reassembles them transparently** and
validates the count against `n_parts.txt`. No manual reassembly needed.

---

## 6. Scientific findings

### BPB → BTS: recomputed, and the headline numbers do not survive

`results/bts/*` (committed in the first commit, never revisited) is **not
usable**. It was computed on the training cluster from each run's
`train.jsonl`, which is unreproducible here (`RUNS` points at a cluster path),
and it contradicts itself: its two variants disagree in *sign* on the headline
penalty in all four (source × tokenizer) cells — `matched_total` says
−0.023..−0.035 (cross-script transfers *better*, i.e. no penalty),
`matched_lang` says +0.001..+0.013 (penalty exists). Four separate defects:

1. **Token dilution** — `matched_total` compares a 30B mono against a 30B
   bilingual whose per-language share is only 15B.
2. **Silent degeneration** — `matched_lang` picks the mono checkpoint *nearest*
   `total*mix_prob`; for zh that returned the final checkpoint, so zh's
   "matched_lang" equals its "matched_total" exactly (shift 0.0000) while every
   other partner shifted +0.04..+0.10. zh is also the only partner with a
   positive `matched_total`, which is what drives the negative penalty there.
3. **Non-like-for-like partner sets** — `de-starved` mono does not exist, so
   `penalty(starved)` averaged same-script over `{fr}` while
   `penalty(destarved)` used `{de,fr}`; recomputing like-for-like removes
   65–76% of the reported interaction (and flips its sign in one cell).
4. **LR-state mismatch** — the decisive one, below.

**Recompute** (`scripts/external_bench/run_bpb.py` + `bts_matched.py`,
FLORES+ dev+devtest n=2009, per-sentence NLL/bytes, paired bootstrap over
sentences). The scoring path is `bench.py`'s verified fixed-shape Neuron
scorer; it reproduces `eval/bpb.py`'s `score_texts` to ~1e-8.

**The LR-state confound.** `base_main.yaml` is WSD (warmup 1B, stable 23B,
decay 6B): decay starts at **24B**. So every `*-8b`/`*-12b`/`*-15b`/`*-23b`
checkpoint is a mid-**stable** snapshot at **peak LR 3.0e-3**, while an
unsuffixed model is the **cooled** 30B final at 3.0e-4. Pairing a mono
intermediate against a cooled bilingual final hands the bilingual the entire
decay phase for free. That is exactly what the old `matched_lang` did, and
what a naive "mono-15b vs en-X-fair" pairing does:

| budget | LR-matched? | BTS range observed |
|---|---|---|
| 7.5B (`*-8b` vs `*-15b`) | yes, both @3e-3 | −0.006 .. +0.003 |
| 11.4B (`*-12b` vs `*-23b`) | yes, both @3e-3 | −0.013 .. +0.001 |
| 15B (`*-15b` vs cooled final) | **no** (3e-3 vs 3e-4) | **+0.027 .. +0.047** |

The large positive BTS values — old and new — are **substantially a cooldown
artifact**. At LR-matched budgets transfer to the partner language is
approximately nil, slightly negative.

**Checkpoint noise dominates what is left.** Between adjacent mono
checkpoints at peak LR, BPB moves erratically (`ar-fair`: −0.0009 for +52%
tokens, then −0.0081 for +26%; `ar-starved`: −0.0187 then −0.0044). That
±0.008–0.019 is 2–4x every LR-matched BTS effect measured, and ~5x the
bootstrap CIs — which capture only eval-sentence sampling, **not** which
mid-stable checkpoint was grabbed. So the tight-looking CIs on the LR-matched
rows understate the real uncertainty, and those BTS values are within
training noise of zero.

**Status of the headline interaction: not established.** The only
computable interaction (+0.0091 [+0.0052, +0.0128], same={fr} vs cross={ar})
comes from the LR-mismatched 15B budget and is not quotable. At LR-matched
budgets the interaction is not computable at all — `de` has **no starved
monolingual at any budget**, and the `fr`/`zh` cells at 7.5B/11.4B are not
uploaded. The one clean same-vs-cross penalty available (7.5B, destarved,
de vs ar) is **+0.0015 [−0.0017, +0.0047]** — indistinguishable from zero.

**To settle it properly** from checkpoints, the WSD design has the right
tool: branch `cooldown_run()` from the *stable* checkpoints at a matched
per-language budget for both mono and bilingual, then compare
cooled-vs-cooled. But there is a cheaper route that needs no compute at all —
see below.

### BTS from the W&B training curves (no compute, cooldown-clean)

The trainer already logs `eval/{flores,holdout}_{lang}_bpb` against `tokens_b`
at every checkpoint interval, so the full loss-vs-tokens curve exists for
every run in W&B (`jonathan-von-rad/XScript-Pretraining`, 25 runs with usable
history). `scripts/external_bench/bts_from_wandb.py` pulls them and restricts
to the **stable-LR window (1B–24B)**, which makes the comparison
cooldown-clean *by construction* — mono and bilingual are read at an
identical LR state. It is also denser than any checkpoint grid and recovers
**holdout** BPB, whose shards are not on the eval box at all.

Two estimators, because the repo and ATLAS do not define BTS the same way:

| | definition | null |
|---|---|---|
| repo (`eval/bts.py`) | `(BPB_mono − BPB_bi)/BPB_mono` at matched **per-language** tokens | 0 |
| ATLAS (2510.22037) | `D_mono(L)/D_bi(L)`, **total** tokens to reach loss L | **0.5** (pure 50/50 dilution); 1.0 = second language free |

**ATLAS BTS, stable window, both eval sets:**

| cell | script | FLORES | holdout | anchor-sensitivity (FLORES) |
|---|---|---|---|---|
| de/destarved | same | **0.639** | **0.729** | 0.51–0.72 |
| fr/starved | same | **0.969** | **0.873** | 0.80–0.97 |
| ar/destarved | cross | **0.373** | **0.380** | 0.37–0.51 |
| ar/starved | cross | **0.211** | **0.466** | 0.21–0.68 |
| zh/destarved | cross | – | **0.420** | 0.38–0.47 |
| zh/starved | cross | **0.301** | **0.434** | 0.30–0.53 |

**Same-script sits above the 0.5 dilution null, cross-script below it, on
both eval sets independently.** Adding a same-script partner costs *less*
total compute than dilution predicts; adding a cross-script partner costs
*more* — i.e. genuine interference, not merely dilution. This is the first
result in the project that supports the cross-script penalty on the intrinsic
metric with the confounds controlled.

**But it is scale-dependent, and that matters.** The repo-style BTS at
matched per-language tokens is ≈0 in every cell (−0.021..+0.016 on both
sources) — i.e. by the largest matched budget available the bilingual has
converged to dilution-parity (ATLAS BTS → 0.5). The anchor-sensitivity column
shows the same thing: each cell's ATLAS BTS drifts toward 0.5 as the anchor
moves later. So the separation is real but **shrinks with scale**, and
quoting a single BTS number (as both `bts.py` and the ATLAS framing invite)
misrepresents it. Report the curve, or at least the anchor range.

**Extraction gotcha (cost us the interaction once — do not repeat).** The
trainer logs `tokens_b` and the `eval/*_bpb` metrics in *separate*
`wandb.log()` calls, so they usually land on different steps. Any pull that
requires both on the same row silently drops most eval points — it made
`en-fr__unigram_destarved` look like a 2-point run when it has **29 points
across full training**. Always pull eval rows on their own and reconstruct
tokens as `step x tokens_per_step` (the relation is exactly linear; take the
median ratio over rows that do have both). `bts_from_wandb.py`'s puller
asserts recovered-records == eval-rows for every run.

~~The only genuinely missing monolingual is `de` starved~~ — ✅ **retrained
2026-08-03 to 16.10B (§6h); it is no longer missing.** The original run
(still in `EXCLUDE_RUNS`, and still the one this puller sees) did not
"collapse" as stated here — it **diverged at the warmup/peak-LR seam**, see
§6h for the curve. Its anchor BPB ~1.72 vs the
destarved twin's ~1.06 is the symptom, not the diagnosis. `fr`
has both conditions, so **the interaction is computable** — see below.

Also note the non-English-anchor bilinguals (`de-ar`, `de-fr`, `de-zh`,
`fr-*`, `ar-zh`) appear in W&B with ~20 eval points each up to ~11.75B but
**never actually ran** — excluded in `load()`; do not use them.

### Repo-style BTS at a fixed budget, token- vs content-matched

`bts_content_matched.py` computes the repo's own BTS at a fixed per-language
budget, in two flavours. The requested "bilingual 24B vs mono 12B" is only
reachable for **ar** (11.91B/lang); de/fr monolingual curves stop at 7.75B and
zh's bilingual at 11.75B, so those are reported at the largest budget both
tokenizer conditions support.

| cell | X/lang | BTS (FLORES) | BTS (holdout) |
|---|---|---|---|
| de/destarved (same) | 7.75B | −0.0042 | +0.0014 |
| fr/starved (same) | 7.75B | +0.0088 | +0.0155 |
| **ar/destarved (cross)** | **11.91B** | **+0.0028** | **+0.0070** |
| **ar/starved (cross)** | **11.91B** | **−0.0033** | **+0.0130** |
| zh/destarved (cross) | 5.88B | −0.0210 | −0.0147 |
| zh/starved (cross) | 5.88B | −0.0080 | −0.0149 |

Every value is |BTS| ≤ 0.026 — at matched per-language tokens the bilingual is
indistinguishable from the monolingual, corroborating both the checkpoint-based
result and the ATLAS-BTS-→0.5 drift above.

**Content matching.** Within one tokenizer condition mono and bilingual share
a tokenizer, so content-matching cannot change BTS — it only matters for
comparing the *conditions*, i.e. for the fair-vs-starved gap and hence the
interaction. The starved tokenizer needs more tokens for the same text
(fertility ratios starved/fair on FLORES: **ar 1.476, de 1.371, zh 1.304,
fr 1.301, en 1.200**), so at equal tokens a starved run has seen strictly less
content — and the distortion is largest for exactly the cross-script language
the thesis is about. `bts_content_matched.py` evaluates each condition at
`tokens = bytes x fertility(cond, lang)` for a shared byte target:

### The headline numbers (same=fr, cross={ar,zh}; de/starved absent)

| quantity | FLORES | holdout |
|---|---|---|
| penalty(starved) | +0.0212 | +0.0311 |
| penalty(destarved) | +0.0155 | +0.0255 |
| **interaction, content-matched** | **+0.0058** | **+0.0056** |
| interaction, token-matched | −0.0054 | +0.0062 |

**Two results survive every variant tried:**

1. **The cross-script penalty is real and positive** — `penalty > 0` in all
   8 measurements (both eval sets × both estimators × token/content matching).
   On the ATLAS estimator it is large (+0.40: fr ≈0.86 vs ar/zh ≈0.44); on the
   repo estimator small (+0.02..+0.03). Same sign, very different magnitude —
   the estimators are not interchangeable.
2. **The interaction is positive, i.e. de-starving the tokenizer shrinks the
   penalty** — the thesis's predicted direction. ~18–27% of the starved
   penalty is attributable to tokenizer starvation on the repo estimator
   (0.0056/0.0311 to 0.0058/0.0212).

**Content-matching is what makes (2) reproducible.** Token-matched, the
interaction flips sign between eval sets (−0.0054 FLORES vs +0.0062 holdout);
content-matched, the two agree to three decimals (+0.0058 / +0.0056). This is
the fertility correction doing real work: at equal *tokens* the starved runs
have seen ~30–48% less text, and that deficit is confounded with the tokenizer
quality being measured. **Quote the content-matched interaction, not the
token-matched one.**

Caveats, none small: the same-script group is **one language** (`fr`) because
de/starved diverged (not "collapsed" — §6h has the diagnosis), so the penalty
is really "fr vs {ar,zh}"; there are **no
confidence intervals** (one training run per cell, and unlike the downstream
deltas there is no per-example data to bootstrap); fertility is measured on
FLORES as a proxy for the training pools; and cells are compared at the
largest budget each supports (fr 7.75B, ar/zh ~11.2B), so the penalty mixes
budgets. Landing `de-starved` fixes the first and is the single highest-value
run remaining. ✅ **DONE — the retrain landed 2026-08-03 (§6h)** with
checkpoints pinned to de-fair's exact budgets, so this table can now have a
second same-script language at 7.753B, and a properly content-matched `de`
cell at 16.100B (= 11.754 x 1.371). **This table has not yet been
recomputed** — that is the single highest-value remaining analysis. Note §6h
measures de's pool fertility at **3.291 B/tok**, so the FLORES-proxy caveat
above can also be retired for `de`.

### ~~Only XNLI discriminates; MMLU & Belebele are at chance~~ (RETRACTED, §6e)
Kept for the record; every headline in this subsection is superseded.
Own-language and correctly scored, HellaSwag/Taxi-1500/XNLI/SIB-200/Belebele-cloze all discriminate, and Global-MMLU clears chance in all
five languages (21 of 23 cells). The original text:
From the `--limit 200` matrix over all 15 models (chance: MMLU/Belebele 0.25,
XNLI 0.333):
- ~~**Global-MMLU: 0/23 model×lang cells above chance.** Confirmed real (not an
  artifact): letter, cloze, and cloze+PMI scoring all stay ≈0.21–0.23 for en.
  World-knowledge MCQ is beyond a 1B/30B model.~~ **RETRACTED — see §6e.**
  This was measured at `--limit 200`, where the smallest effect detectable at
  2σ is **+6.1 accuracy points**; the real effect is **+5.9**, i.e. the
  experiment sat exactly on its own detection boundary and could not have
  found it however often it was run. On the full `CohereForAI/Global-MMLU`
  (n=14,042, not the n=400 `-Lite` build lm-eval defaults to), `en-fair` in
  **cloze** format scores **0.310 acc_norm against a 0.251 empirical null,
  +0.059 ≈ 15σ**. The **letter** (A/B/C/D) format genuinely is at chance
  (0.2499) and stays there after calibration (+0.004) — so these models do
  hold measurable world knowledge but cannot do the A/B/C/D indirection. Note
  the old cloze figure of 0.21–0.23 is *below* chance, which was itself the
  tell that length bias was dragging it under.
- **Belebele: at chance** under lm-eval's letter format. cloze+PMI on en gives a
  faint lift (0.26 → ~0.34, ~+1.8σ at n=80) — suggestive only, not confirmed.
- **XNLI: signal on en/de/fr** out of the box; ar/zh sat at exactly chance (0.335).

### AR/ZH XNLI at chance was an EVALUATION ARTIFACT, not a training failure
Diagnosed (tokenization clean, 0% `<unk>`; loading correct; en works with
identical code). lm-eval's XNLI is a cloze over the whole
`premise, {Q}? {LABEL}, hypothesis` string differing only by a connective, scored
by raw loglik → **surface-form competition** (Holtzman et al. 2021): weak models
pick the highest-prior connective and collapse to majority class. Two distinct
defects, one per language:

| lang | root cause | fix | corrected result (full val, n=2490) |
|------|-----------|-----|-----|
| Arabic | lm-eval connectives **mistranslated** (`رقم`="number" for contradiction, `لذا` for neutral) | correct to `لا` / `أيضا`, standard scoring | **0.44–0.47** |
| Chinese | surface-form competition (connectives fine) | **PMI** (prior-normalized) scoring | **0.41–0.42** |

After the fix, **all 23 XNLI model×lang cells are above chance (+7.8 to +20.2σ)**.
Mean corrected accuracy by language: **en 0.503, de 0.471, fr 0.470, ar 0.455,
zh 0.414**.

**Thesis implication:** with a correct evaluation, the cross-script languages
(AR, ZH) learn XNLI *comparably* to the same-script ones (EN/DE/FR) at 30B tokens
— i.e. the apparent cross-script downstream penalty in the raw numbers was largely
a **measurement artifact**, consistent with the repo's argument about ATLAS's
penalty. (This section previously called intrinsic **BPB→BTS** "the primary
discriminator". That no longer holds: as of the recompute at the top of §6,
BTS is confounded by LR state and swamped by checkpoint noise, and the
interaction is not established at any LR-matched budget. Downstream `acc`
can't carry the cross-script question either — MMLU/Belebele are at chance
and XNLI needs the debiasing above. Right now **no** single metric in this
repo cleanly answers it; the matched-token downstream deltas in §6's transfer
section are the best-powered evidence available.)

`bench.py`'s `XNLI_CONNECTIVES` / `XNLI_DEBIAS_METHOD` / `_xnli_debiased()`
implement the fix and are wired into `run()`: `xnli_ar` and `xnli_zh` are
*always* routed through the debiased path (corrected connectives + `standard`
scoring for ar, `pmi` scoring for zh) instead of lm-eval's task registry, for
every caller — no flag needed. `run_xnli_debiased.py` (scripts/external_bench/)
still exists standalone and reports **both** `standard` and `pmi` per language,
useful if you want to eyeball which method wins on a new checkpoint before
trusting the hardcoded choice above.

### Appendix C.5 replication (Messmer et al. 2025, arXiv:2502.10361)

`scripts/external_bench/run_appendix_c5.py` replicates the per-language
benchmark tables in that paper's Appendix C.5 across our 5 languages and all
15 checkpoints. It evaluates **every model on every language**, not just each
checkpoint's own training languages — this is also a zero-shot cross-lingual
transfer readout, not just a same-language score.

Suite (loglikelihood-scorable subset only — see script docstring for why the
F1-extractive-QA and single-language-knowledge-exam parts of Table 21 are
excluded):

| task | languages | notes |
|---|---|---|
| XNLI | en/de/fr/ar/zh | reuses the debiased ar/zh routing above |
| Belebele | en/de/fr/ar/zh | **custom cloze task** — see below |
| ARC | en/de/fr/ar/zh | native `arc_easy` (en) + `okapi` M-ARC translations |
| HellaSwag | en/de/fr/ar | zh is **not** absent from the data, only from lm-eval's task list — see §6d; the zh column is filled by `run_extra_bench.py`, not by this script |
| XStoryCloze | en/ar/zh | dataset doesn't cover de/fr |
| XWinograd | en/fr/zh | dataset doesn't cover de/ar |

**lm-eval's registered `belebele` task uses A/B/C/D letter-choice prompting,
which is NOT the paper's methodology.** Appendix D is explicit: 0-shot, cloze
multiple-choice (the answer's own text as the scored continuation, not a
letter token) — "shown to serve as a more reliable performance indicator
earlier in training" (Kydlíček et al., 2024). This also matches what
§6 already suspected from an earlier ad-hoc probe ("Belebele: at
chance under lm-eval's letter format. cloze+PMI on en gives a faint lift").
`src/xscript/eval/c5_tasks/belebele_cloze/` defines a proper cloze variant
(`belebele_cloze_{eng_Latn,deu_Latn,fra_Latn,arb_Arab,zho_Hans}`), loaded via
`lm_eval.tasks.TaskManager(include_path=...)` alongside the standard
registry. Metric preference is `acc_norm` over `acc` where both exist
("normalized accuracy" per the paper); XNLI/XStoryCloze/XWinograd only report
`acc`.

```bash
python run_appendix_c5.py --repo jvonrad/xscript-eval --device xla \
  --batch-size 8 --workdir $WORK          # full suite, all 15 models x 5 langs
# --runs / --limit / --langs subset flags mirror run_benchmarks.py
```

Results: `$WORK/results/appendix_c5/<run>_final.json`, scores nested
`{lang: {task: accuracy}}`. Deliberately writes **no shared summary.json**
(unlike the other two scripts) to sidestep the concurrent-writer clobbering
noted in NEURON.md §5 — aggregate from the per-run files.

Only runs from inside this repo (needs the local `c5_tasks/` dir); not
bundled into the portable HF export.

**Every model is scored on all 5 languages, not just its own training
languages** — this is the point: it turns the suite into a zero-shot
cross-lingual transfer readout, not just a same-language score. `run()`'s
`tasks_for_langs` restriction (used by `run_benchmarks.py`/`bench.py`'s
DEFAULT_TASKS) does not apply here.

### C.5 results (all 26 models, full test/val splits)

Mean accuracy per language, averaged across all 26 models regardless of each
model's own training languages (chance: XNLI 0.333, Belebele/ARC/HellaSwag
≈0.25, XStoryCloze/XWinograd 0.5). The 26 models span heterogeneous token
budgets (30B original mono/bilingual, plus this session's 15B/12B/23B
matched-token checkpoints — see the transfer-delta section below), so this
table pools different training regimes together; read it as "typical
accuracy across the roster," not a controlled comparison — the matched-token
table below is the controlled version.

| benchmark | en | de | fr | ar | zh |
|---|---|---|---|---|---|
| XNLI (debiased) | 0.47 (n=26) | 0.36 (n=26) | 0.38 (n=26) | 0.36 (n=26) | 0.35 (n=26) |
| Belebele (cloze) | 0.30 (n=26) | 0.28 (n=26) | 0.30 (n=26) | 0.27 (n=26) | 0.27 (n=26) |
| ARC | 0.42 (n=26) | 0.24 (n=26) | 0.25 (n=26) | 0.25 (n=26) | 0.26 (n=26) |
| HellaSwag | 0.37 (n=26) | 0.29 (n=26) | 0.32 (n=26) | 0.29 (n=26) | n/a |
| XStoryCloze | 0.58 (n=26) | n/a | n/a | 0.49 (n=26) | 0.50 (n=26) |
| XWinograd | 0.65 (n=26) | n/a | 0.55 (n=26) | n/a | 0.58 (n=26) |

Belebele's cloze fix confirms the §6 XNLI-era suspicion at full scale: a
real but modest lift over the letter-format numbers (chance 0.25 → ~0.27–0.30
vs the letter-format ~0.21–0.31 in §6's original matrix). Global-MMLU-style
world knowledge stays flat regardless of format (unchanged from §6).

> ⛔ **The ARC half of the next paragraph is RETRACTED (§6e).** English was
> scored on ARC-**Easy** (n=2376) and the other languages on okapi m_arc,
> which is 100% ARC-**Challenge** (n=1169) — different difficulty tiers.
> Like-for-like on ARC-Challenge, English is .268–.285 vs the others'
> .24–.26: a ~2-point gap, not 25. ARC-Challenge carries almost no signal
> in ANY language at this scale, English included.

**ARC and XStoryCloze show a striking English-only pattern**: clear signal in
English (0.42, 0.58) but every other language sits at or within noise of
chance (ARC: 0.24–0.26; XStoryCloze ar/zh: 0.49–0.50) — despite those same
models showing real signal on XNLI in ar/zh. XWinograd is the exception:
French (0.55) and **Chinese (0.58)** both clear chance there, unlike on
ARC/StoryCloze. Read this as "most of this benchmark suite outside XNLI is
mainly measuring English competence at this model scale," modulo XWinograd's
oddly-strong Chinese result. (English's own mean dropped a few points vs the
15-model version, e.g. XNLI 0.49→0.47 — this is the pooling effect above:
the 11 new checkpoints include lower-token-budget runs (12–15B vs the
original 30B) with weaker English, plus zh-fair-12b/zh-starved-12b which are
zh-only mono runs with a comparatively weak English zero-shot score.)

> ⚠️ The full per-model breakdown (26 checkpoints × 25 `(lang, task)` cells)
> now lives in [`results/appendix_c5/per_model_table.md`](results/appendix_c5/per_model_table.md)
> — it is raw data rather than findings, and CLAUDE.md is loaded into context
> every session. Three corrections apply to it (§6e): its **ARC** columns
> compare Easy (en) against Challenge (others), its **XNLI** column predates
> the connective calibration, and it scores every model on languages it never
> trained on — averaging those cells is what produced the retracted "only
> XNLI discriminates" verdict.

### Same-script vs. cross-script transfer (matched-token, bootstrap CIs)

Because every model is scored on every language, `bilingual_score(lang) -
monolingual_score(lang)` is a direct transfer-delta measurement. The first
pass at this (kept below in git history, not reproduced here) used the
original 30B-token monolingual baselines against 30B-token bilinguals whose
per-language exposure is only 15B (token-level 50/50 mixing, per README.md)
— a **token-dilution confound**: a negative Δ-on-English was expected from
the mono model simply seeing 2x more English, independent of any real
cross-lingual interference. That pass also had no Chinese monolingual
baseline at all, and no confidence intervals (single point estimate per
cell, `log_samples=False`).

Both gaps are now closed. New checkpoints were uploaded specifically to
match: `{lang}-{tok}-15b` monolingual runs cut to ~14.75B tokens (matching
the de/fr/ar bilinguals' ~15B-per-language share), `zh-{tok}-12b` (~11.75B
tokens, the first Chinese monolingual baseline to exist) and
`en-zh-{tok}-23b` bilinguals (~22.76B total, ~11.4B/language) to pair with
it. `run_appendix_c5.py` now runs with `log_samples=True` and
`_xnli_debiased(..., return_correct=True)`, giving per-example 0/1 hit
lists; `scripts/external_bench/bootstrap_transfer.py` bootstraps a paired
95% CI per (partner language, tokenizer, benchmark) cell from those (B=2000
replicates, resampling doc indices once and applying the same resample to
both models being compared, valid since both score the identical fixed doc
order) plus a stratified-bootstrap aggregate across all benchmarks for that
cell. One caveat remains: zh's English-anchor comparison uses `en-*-15b`
(~14.76B tokens) since no ~11.4B-token English checkpoint was uploaded to
exactly match the en-zh bilingual's English share — an approximation,
marked `~` below; every other cell is a near-exact token match.

Mean Δ across every applicable benchmark (XNLI, Belebele, ARC, HellaSwag,
plus XStoryCloze/XWinograd where the pair has coverage), with 95% CIs:

| partner | script | tok | mean Δ on partner-lang [95% CI] | mean Δ on English [95% CI] |
|---|---|---|---|---|
| de | same-script | fair | **+0.027 [+0.018, +0.036]** | **+0.037 [+0.028, +0.046]** |
| de | same-script | starved | n/a (no de-starved-15b) | **+0.013 [+0.004, +0.021]** |
| fr | same-script | fair | **+0.039 [+0.014, +0.063]** | **+0.030 [+0.023, +0.038]** |
| fr | same-script | starved | **+0.036 [+0.011, +0.060]** | **+0.017 [+0.009, +0.024]** |
| ar | cross-script | fair | **+0.024 [+0.015, +0.031]** | **+0.021 [+0.014, +0.029]** |
| ar | cross-script | starved | **+0.023 [+0.015, +0.031]** | +0.006 [−0.001, +0.013] |
| zh | cross-script | fair | +0.011 [−0.001, +0.024] | ~+0.005 [−0.004, +0.014] |
| zh | cross-script | starved | **+0.015 [+0.002, +0.028]** | ~**−0.015 [−0.024, −0.006]** |

(Bold = CI excludes 0, i.e. a statistically supported effect at this sample
size, not just a point-estimate sign.)

> ⚠️ **THE SAME LR-STATE CONFOUND AS §6's BPB→BTS APPLIES TO 5 OF THESE 7
> CELLS.** The `*-15b` monolinguals are mid-stable snapshots at peak LR
> 3.0e-3; the unsuffixed 30B bilinguals they are paired against are **cooled**
> finals at 3.0e-4 (decay starts at 24B). So de/fair, fr/{fair,starved} and
> ar/{fair,starved} hand the bilingual an entire decay phase for free, which
> inflates Δ positive. Only the **zh** cells are LR-matched
> (`zh-*-12b` vs `en-zh-*-23b`, both mid-stable @3e-3). The English anchors
> (`en-*-15b`) are mid-stable, so Δ-on-English is confounded for exactly the
> same five cells and clean only for zh.
>
> The pattern is the tell: every confounded cell lands at +0.023..+0.039,
> while the two clean cells are the smallest in the table (+0.011 n.s.,
> +0.015). Consequently **these three claims below are NOT established**:
> (a) same-script > cross-script — every same-script cell is confounded and
> every clean cell is cross-script; (b) bilingual training helps English —
> the clean English deltas are ~+0.005 (n.s.) and **−0.015**, possibly the
> opposite sign; (c) cross-script positive in 3 of 4 cells — only
> zh/starved (+0.015) is clean *and* significant.
>
> Unaffected: the raw per-model C.5 accuracy tables (no cross-checkpoint
> pairing). Also unresolved: for zh/fair the clean downstream Δ is +0.011
> (bilingual better) while the clean BPB→BTS at the same budget is −0.0129
> (bilingual worse) — they disagree in sign.
>
> Fix is the same as §6's: `cooldown_run()` branches at a matched
> per-language budget so mono and bilingual are both cooled, then re-derive.
> §6b's alignment deltas use the same checkpoint families and need the same
> audit before being quoted.

**Full per-benchmark breakdown** (every individual benchmark behind the
means above, not just the aggregate row; bold = CI excludes 0):

| partner | script | tok | benchmark | Δ on partner-lang [95% CI] | Δ on English [95% CI] |
|---|---|---|---|---|---|
| de | same-script | fair | xnli | **+0.018 [+0.003, +0.034]** | **+0.039 [+0.022, +0.057]** |
| de | same-script | fair | belebele | +0.003 [-0.021, +0.028] | **+0.032 [+0.009, +0.057]** |
| de | same-script | fair | arc | **+0.031 [+0.011, +0.051]** | **+0.030 [+0.014, +0.046]** |
| de | same-script | fair | hellaswag | **+0.056 [+0.049, +0.063]** | **+0.046 [+0.040, +0.053]** |
| de | same-script | fair | **mean (4 benchmarks)** | **+0.027 [+0.018, +0.036]** |  |
| de | same-script | fair | **mean (4 benchmarks)** |  | **+0.037 [+0.028, +0.046]** |
| de | same-script | starved | xnli | - | -0.002 [-0.021, +0.016] |
| de | same-script | starved | belebele | - | +0.020 [-0.003, +0.043] |
| de | same-script | starved | arc | - | +0.011 [-0.005, +0.028] |
| de | same-script | starved | hellaswag | - | **+0.021 [+0.015, +0.028]** |
| de | same-script | starved | **mean (4 benchmarks)** |  | **+0.013 [+0.004, +0.021]** |
| fr | same-script | fair | xnli | +0.002 [-0.012, +0.018] | +0.017 [+0.000, +0.036] |
| fr | same-script | fair | belebele | **+0.027 [+0.002, +0.050]** | +0.020 [-0.006, +0.043] |
| fr | same-script | fair | arc | **+0.024 [+0.003, +0.046]** | **+0.024 [+0.008, +0.039]** |
| fr | same-script | fair | hellaswag | **+0.068 [+0.061, +0.075]** | **+0.046 [+0.039, +0.052]** |
| fr | same-script | fair | xwinograd | +0.072 [-0.048, +0.181] | **+0.046 [+0.027, +0.065]** |
| fr | same-script | fair | **mean (5 benchmarks)** | **+0.039 [+0.014, +0.063]** |  |
| fr | same-script | fair | **mean (5 benchmarks)** |  | **+0.030 [+0.023, +0.038]** |
| fr | same-script | starved | xnli | **+0.037 [+0.020, +0.053]** | -0.007 [-0.025, +0.010] |
| fr | same-script | starved | belebele | +0.007 [-0.016, +0.030] | +0.010 [-0.013, +0.033] |
| fr | same-script | starved | arc | **+0.022 [+0.001, +0.044]** | **+0.027 [+0.011, +0.043]** |
| fr | same-script | starved | hellaswag | **+0.066 [+0.058, +0.073]** | **+0.041 [+0.035, +0.048]** |
| fr | same-script | starved | xwinograd | +0.048 [-0.060, +0.157] | +0.012 [-0.009, +0.032] |
| fr | same-script | starved | **mean (5 benchmarks)** | **+0.036 [+0.011, +0.060]** |  |
| fr | same-script | starved | **mean (5 benchmarks)** |  | **+0.017 [+0.009, +0.024]** |
| ar | cross-script | fair | xnli | -0.010 [-0.027, +0.007] | **+0.020 [+0.002, +0.037]** |
| ar | cross-script | fair | belebele | +0.016 [-0.009, +0.039] | **+0.024 [+0.002, +0.047]** |
| ar | cross-script | fair | arc | **+0.052 [+0.031, +0.072]** | **+0.029 [+0.013, +0.044]** |
| ar | cross-script | fair | hellaswag | **+0.036 [+0.030, +0.043]** | **+0.029 [+0.023, +0.035]** |
| ar | cross-script | fair | xstorycloze | **+0.024 [+0.008, +0.039]** | +0.005 [-0.009, +0.020] |
| ar | cross-script | fair | **mean (5 benchmarks)** | **+0.024 [+0.015, +0.031]** |  |
| ar | cross-script | fair | **mean (5 benchmarks)** |  | **+0.021 [+0.014, +0.029]** |
| ar | cross-script | starved | xnli | **+0.077 [+0.057, +0.097]** | -0.012 [-0.031, +0.006] |
| ar | cross-script | starved | belebele | +0.002 [-0.020, +0.026] | +0.007 [-0.014, +0.031] |
| ar | cross-script | starved | arc | -0.006 [-0.027, +0.014] | **+0.019 [+0.003, +0.037]** |
| ar | cross-script | starved | hellaswag | **+0.032 [+0.026, +0.038]** | **+0.019 [+0.012, +0.025]** |
| ar | cross-script | starved | xstorycloze | +0.009 [-0.007, +0.024] | -0.001 [-0.017, +0.013] |
| ar | cross-script | starved | **mean (5 benchmarks)** | **+0.023 [+0.015, +0.031]** |  |
| ar | cross-script | starved | **mean (5 benchmarks)** |  | **+0.006 [-0.001, +0.013]** |
| zh | cross-script | fair | xnli | +0.018 [-0.005, +0.041] | ~+0.002 [-0.017, +0.021] |
| zh | cross-script | fair | belebele | -0.014 [-0.038, +0.009] | ~+0.010 [-0.017, +0.036] |
| zh | cross-script | fair | arc | +0.009 [-0.013, +0.032] | ~+0.006 [-0.011, +0.024] |
| zh | cross-script | fair | xstorycloze | +0.005 [-0.009, +0.019] | ~-0.009 [-0.024, +0.009] |
| zh | cross-script | fair | xwinograd | +0.040 [-0.008, +0.087] | ~+0.015 [-0.005, +0.035] |
| zh | cross-script | fair | **mean (5 benchmarks)** | **+0.011 [-0.001, +0.024]** |  |
| zh | cross-script | fair | **mean (5 benchmarks)** |  | **~+0.005 [-0.004, +0.014]** |
| zh | cross-script | starved | xnli | **+0.024 [+0.002, +0.046]** | ~**-0.031 [-0.049, -0.014]** |
| zh | cross-script | starved | belebele | +0.009 [-0.016, +0.033] | ~-0.020 [-0.044, +0.007] |
| zh | cross-script | starved | arc | -0.013 [-0.032, +0.007] | ~-0.005 [-0.023, +0.011] |
| zh | cross-script | starved | xstorycloze | +0.008 [-0.005, +0.023] | ~-0.009 [-0.026, +0.007] |
| zh | cross-script | starved | xwinograd | +0.046 [-0.002, +0.095] | ~-0.008 [-0.028, +0.012] |
| zh | cross-script | starved | **mean (5 benchmarks)** | **+0.015 [+0.002, +0.028]** |  |
| zh | cross-script | starved | **mean (5 benchmarks)** |  | **~-0.015 [-0.024, -0.006]** |

Note per-benchmark rows are individually noisier than the aggregate (fewer
examples per cell, e.g. XWinograd's n is much smaller than XNLI's 2490) —
several show wide, zero-crossing CIs (fr/fair xwinograd partner-lang:
`+0.072 [-0.048, +0.181]`) even where the aggregate is tight. Treat
individual rows as directional, the aggregate mean row per (partner, tok)
as the load-bearing number.

This revises the earlier read materially:
- **Cross-script transfer is not just dilution-neutral — it's significantly
  positive in 3 of 4 cells** (ar/fair, ar/starved, zh/starved all have CIs
  clear of 0; only zh/fair falls just short, CI `[−0.001, +0.024]`). The
  original 30B-baseline analysis called Arabic transfer "flat" partly
  because it lacked the statistical power to distinguish a small positive
  effect from noise — with matched tokens and n≈2490-11000 per cell, that
  effect resolves as real.
- **Same-script (de, fr) transfer is also significantly positive and
  somewhat larger in magnitude** (+0.027 to +0.039 vs ar/zh's +0.011 to
  +0.024) — same-script and cross-script transfer look like the same
  phenomenon at different strengths, not qualitatively different regimes.
- **Δ-on-English is no longer confounded by dilution** (mono/bilingual now
  matched in per-language token count) and comes out **positive** for
  de/fr/ar (+0.006 to +0.037) — bilingual training modestly *helps* English
  too for same-script and Arabic pairs, the opposite of the old analysis's
  (confound-driven) uniformly-negative reading. zh's English deltas are the
  exception (one negative, one near-zero) but both use the `~`-flagged
  approximate baseline, so are the least trustworthy numbers in this table.
- **Fair vs. starved has NO consistent effect on Δ-on-partner-lang, but a
  clear, significant one on Δ-on-English.** Direct paired bootstrap of
  `(Δ_fair − Δ_starved)`, resampling doc indices jointly across all four
  models per cell (bi-fair, mono-fair, bi-starved, mono-starved — a
  straightforward extension of the same paired-doc-order logic
  `paired_bootstrap_delta` already relies on), rather than eyeballing two
  separate CIs:

  | partner | Δ-on-English (fair − starved) | Δ-on-partner-lang (fair − starved) |
  |---|---|---|
  | de | **+0.025 [+0.015, +0.035]** | n/a (no de-starved-15b) |
  | fr | **+0.011 [+0.000, +0.020]** | +0.003 [−0.024, +0.032] |
  | ar | **+0.020 [+0.011, +0.030]** | +0.001 [−0.011, +0.012] |
  | zh | **+0.015 [+0.005, +0.026]** | −0.003 [−0.021, +0.013] |

  On the English side, fair is significantly larger than starved for **all
  four** partners, and broadly so: 20 of 22 (partner x benchmark) cells are
  positive, 9 significantly. On the partner-language side the AGGREGATE is
  ~0 for every pair, but that mean **hides real, opposite-signed,
  individually-significant effects, not an absence of effect**:

  | partner | benchmark | Δ_fair - Δ_starved on partner-lang |
  |---|---|---|
  | ar | xnli | **-0.087 [-0.116, -0.059]** |
  | ar | arc | **+0.058 [+0.030, +0.086]** |
  | fr | xnli | **-0.034 [-0.056, -0.012]** |

  XNLI and ARC point opposite directions for Arabic, similar enough in
  magnitude to nearly cancel in the mean (+0.001) — "no consistent effect"
  described the average, not the underlying reality.

  **One confound is confirmed, and it only covers part of this.** Correction:
  this is NOT about `bench.py`'s debiasing path -- `XNLI_DEBIAS_METHOD` has no
  "fr" key, so `xnli_fr` is never routed through `_xnli_debiased()`; it always
  scores via lm-eval's own registered `xnli_fr` task. That task independently
  uses the same connective words ("Oui"/"Aussi"/"Non", confirmed against
  `lm_eval/tasks/xnli/utils.py`), and in the real
  `"{premise}, correct? {c}, {hypothesis}"` template, "Oui"/"Aussi" cost 1 MORE
  token than "Non" under `unigram_starved` (0 extra under `unigram_destarved`,
  verified in-template, not just standalone). lm-eval's XNLI scores via
  unnormalized `acc` (raw summed loglikelihood, no length normalization), so
  that token-count asymmetry is a real, tokenizer-dependent scoring bias
  toward "Non" specifically under starved -- plausibly contributing to fr's
  -0.034. Checked and **ruled out** for the other two languages: ar has 0
  marginal tokens per connective under BOTH tokenizers (no asymmetry at all,
  in-template), and zh has a real length asymmetry (2/1/2 tokens) but it's
  IDENTICAL under both tokenizers, so it can't produce a fair-vs-starved
  difference there. Neither explains ar/xnli's larger -0.087 effect.

  **The cross-partner check came back the wrong sign.** If content-dilution
  were the general mechanism, partners with worse fertility at a FIXED
  tokenizer should show smaller Δ-on-English (less content reaching English
  regardless of starvation). Across the four partners at fair: r = -0.77
  (n=4, not statistically meaningful, but the wrong direction to trust the
  mechanism as a general law rather than a within-partner story). FLORES
  fertility itself was checked against actual training-pool text for de/zh
  (1.34/1.27 measured vs FLORES's 1.37/1.30) and holds up fine — the gap is
  in extrapolating the mechanism across languages, not in the fertility
  numbers.

  **Speculative cross-link to §6b:** ar's XNLI (entailment/reasoning) favors
  STARVED, its ARC (factual/surface) favors FAIR. This loosely echoes §6b's
  finding that cross-script REPRESENTATION alignment gains exceed same-script
  ones (opposite of the downstream ordering) — consistent with, but not
  proof of, a story where forced vocabulary-sharing across ~419 languages
  under starvation pushes toward more abstract shared representations that
  help reasoning-style transfer while fertility degradation still hurts
  factual/surface tasks. Not verified mechanistically.

  Plausible (unconfirmed) high-level mechanism for the English-side effect:
  a starved tokenizer spends more of English's token budget subsidizing ~419
  other languages, leaving less capacity for a second training language to
  improve English specifically.

Full per-benchmark breakdown (not just the aggregate row) is in
`bootstrap_transfer.py`'s output — rerun it against a results directory to
regenerate; it's pure stdlib and takes about a minute for the full 26-model
set. An interactive per-model, per-benchmark matrix (the original 15 models
× 25 `(lang, task)` cells, sortable) was generated earlier in this project's
history as a claude.ai artifact, not a repo file, and does not yet reflect
the matched-token models above.

---

## 6b. Cross-lingual representation alignment (MEXA-style)

The representation-side counterpart to the downstream story: embed FLORES+
parallel sentences by mean-pooling each layer's hidden states, then measure how
well one language's sentences retrieve their translations in another.

### Every model is scored on every language pair — and here that is essential

`run_alignment.py` evaluates all 26 checkpoints on all 10 pairs, not only the
pairs a model trained on (the old code did the latter, and only for the 8
EN-anchored bilinguals). The reason is not symmetry with §6's C.5 suite — it
is that **the trained-pair numbers are meaningless without the controls**:

| model | trained on | EN-AR | EN-FR | EN-ZH | EN-DE |
|---|---|---|---|---|---|
| **lexical floor**, destarved (model-free) | — | 0.134 | 0.580 | 0.198 | 0.434 |
| **lexical floor**, starved (model-free) | — | 0.133 | 0.645 | 0.196 | 0.457 |
| `ar-fair` | **ar only** | **0.963** | **0.909** | 0.172 | 0.420 |
| `zh-fair-12b` | **zh only** | 0.015 | 0.478 | **0.965** | 0.337 |
| `zh-starved-12b` | **zh only** | 0.022 | 0.516 | **0.953** | 0.352 |

(bidirectional top-1, `centered` variant, ref layer, n=2009 dev+devtest.)

Note the floor is **tokenizer-dependent** (starved's EN-FR floor is 0.645 vs
destarved's 0.580), so a raw starved-vs-fair comparison is confounded by the
floor difference before any model effect — `analyze_alignment.py` prints one
floor row per tokenizer for this reason.

An **Arabic-only** model retrieves EN↔AR translations at 0.963, and a
**Chinese-only** model does EN↔ZH at 0.965. Neither trained on English.
Restricted to trained pairs you would read a bilingual model's ~0.96 EN-AR as
"cross-script alignment emerges from bilingual pretraining" — the monolingual
controls show that number is essentially free. Three consequences, all now
handled in-code:

1. **A model-free lexical floor is mandatory.** `alignment.lexical_baseline()`
   computes TF-IDF retrieval over shared token ids — no model at all. FLORES
   leaks hard across scripts (digits, dates, Latin-script named entities
   survive translation verbatim), so the floor is already 0.43 EN-DE / 0.58
   EN-FR. Every "same-script alignment" cell for `ar-fair`/`zh-fair-12b` above
   is **at or below** that floor (`zh-fair-12b`: EN-DE 0.337 vs floor 0.434,
   EN-FR 0.478 vs 0.580): those cells contain no representational signal
   whatsoever, only token overlap.
2. **The metric saturates.** Top-1 over one split (997) hit 0.97 for controls,
   so `--split both` (dev+devtest, n=2009) is the default pool, and `cka` /
   `cosine_margin` are reported alongside because they don't ceiling. Where a
   control is already >0.90 `analyze_alignment.py` marks the row `SAT`: a
   bilingual−monolingual delta there measures **headroom, not transfer**.
3. **"Monolingual" checkpoints are not monolingual.** The pattern of *which*
   off-language pairs light up tracks corpus contamination, not architecture:
   `ar-fair` scores 0.909 EN-FR and 0.802 FR-AR (Maghreb web text is heavily
   French-Arabic bilingual), `zh-fair-12b` scores 0.970 EN-ZH (English is
   ubiquitous in Chinese web text). Incidental exposure inside FineWeb2-HQ's
   per-language subsets is doing real work.

### Sweep result (all 26, n=2009): top-1 retrieval is SATURATED and unusable

All 26 models are done (`/mnt/scratch/xscript_align/results/alignment/`). The
headline: **every bilingual scores 0.966-0.995 on its own EN-partner pair**, and
7 of 8 delta rows are flagged `SAT`. The bilingual-minus-monolingual deltas
against the partner-language control collapse to noise at that ceiling
(-0.007 to +0.042), so **top-1 retrieval cannot answer the same-script vs
cross-script question at this scale.** Do not quote those deltas.

The `SAT` reading also makes the "same-script vs cross-script" summary row a
trap: it reports cross-script gap +0.299 > same-script +0.185, but that is
entirely an artifact of the EN-only control being *below the lexical floor* on
Arabic (0.051) while being near-ceiling on German/French. It measures how bad
the control is, not how good the bilingual is.

**Use CKA instead.** It does not saturate (observed range 0.13-0.83) and
carries the structure retrieval loses — each model's own trained pair is
visibly elevated (`en-ar-starved` EN-AR 0.805, `en-fr-fair` EN-FR 0.819,
`en-zh-fair-23b` EN-ZH 0.747) against controls (`en-fair` EN-AR 0.433).
Preliminary CKA deltas vs the matched-token partner-language monolingual:

| partner | fair | starved |
|---|---|---|
| de (same-script) | -0.034 | n/a |
| fr (same-script) | -0.001 | +0.410 ‼ |
| ar (cross-script) | +0.026 | +0.068 |
| zh (cross-script) | +0.036 | +0.102 |

Cross-script positive, same-script ~zero — the *opposite* ordering to the
downstream C.5 transfer deltas. **This is not yet a result**: CKA has no
confidence intervals here (it is a matrix statistic, not per-example, so the
paired bootstrap in `analyze_alignment.py` does not apply to it), and the fr
starved cell is contaminated by the anomaly below. Getting CIs on CKA needs a
different resampling scheme than the one implemented.

### Resolution: alignment transfer, corrected for layer-selection bias

`d' = (matched - mean_nonmatched) / std_nonmatched` per query (unbounded,
scale-free, per-example so the existing paired bootstrap applies) resolves
the saturation problem cleanly, but an earlier pass scored every model at one
fixed layer (`REF_LAYER_FRAC`, chosen up front to avoid cherry-picking a
model's own best layer) and got it wrong: it produced negative same-script
(de/fr) deltas alongside positive cross-script (ar/zh) ones — since
retracted. Cause: **bilinguals develop cross-lingual alignment DEEPER in the
network than monolinguals do** (bilingual peaks cluster at L15-16,
monolingual at L12-16), so a fixed 75%-depth probe systematically
undersamples the bilingual and manufactures a negative delta that isn't
there once each model is scored at its own peak layer instead:

| partner | tok | Δ @ peak layer |
|---|---|---|
| de | fair | **+0.66** (bi L16 vs mono L15) |
| fr | fair | **+0.43** (bi L16 vs mono L14) |
| fr | starved | **+0.53** (bi L15 vs mono L14) |
| ar | fair | +1.04 |
| ar | starved | +2.23 |
| zh | fair | +2.34 |
| zh | starved | +2.33 |

**What survives:** cross-script deltas (ar +1.04/+2.23, zh +2.34/+2.33) are
larger than same-script (de +0.66, fr +0.43/+0.53) — that ordering held under
both the (retracted) fixed-layer scoring and this peak-layer scoring, so it's
the one robust finding here, and it is the reverse of §6's downstream C.5
ordering, where same-script transfer was larger. **What does not survive:**
any claim that same-script alignment transfer is negative or absent — that
was purely the fixed-layer artifact.

**Neither layer rule is clean.** Peak-layer is selection-on-the-metric (inflates,
and inflates more for noisy profiles); fixed-layer is biased whenever peak depth
differs systematically, which it does here. Report both, or bootstrap the layer
choice, before quoting a number from a future rerun. The per-layer profile is
the honest object; `load_embeddings()` regenerates it on CPU in seconds.

**Caveat on the "same-script vs cross-script" summary row:** it averages *both*
controls, and the EN-only control is catastrophically bad at Arabic (d' 1.45),
which inflates the cross-script mean exactly as it did under top-1. Read the
per-row **partner-mono** column above, never that summary.

### Cached embeddings — do metric work from these, not from a rerun

`run_alignment.py --emb-dir` (used by the fan-out) persists the pooled per-layer
embeddings: `(n_layers+1, 2009, 2048)` fp32 per language, one `.npz` per model.
Verified to reproduce the in-run top-1, d' and per-example hit lists
**bit-for-bit** (hence fp32, not fp16). Load with
`alignment.load_embeddings(emb_dir, run_name)`.

**They now live on HF: `jvonrad/xscript-embeddings` (public dataset, 107
files, 139.4 GB).** Public rather than private because HF's private-repo
storage quota rejected the upload at ~109 GB with
`403 Forbidden: Private repository storage limit`; public repos have no such
cap and ingest far faster (570 MiB/s vs 55). The
eval box they were computed on was ephemeral and has been torn down —
`/mnt/scratch/xscript_align/embeddings/` no longer exists. Fetch with
`huggingface_hub.snapshot_download(repo_id="jvonrad/xscript-embeddings",
repo_type="dataset")`, or pull single models with `hf_hub_download`, which is
what you usually want (1.3 GiB each).

Size note: an earlier version of this section said "34 GB for all 26". That is
stale — the sweep was extended to **107 checkpoints**, so it is **~140 GB**;
34 GB is just the 26-checkpoint subset that §6b's deltas are computed on.

This matters because the forward pass is 84% of runtime and a rerun otherwise
costs ~100 GB of checkpoint re-download. Any *new* statistic — CKA CIs via
Gram-matrix resampling, anisotropy/effective-rank probes, a different pooling or
layer — is now a pure-CPU pass over local arrays needing neither Neuron nor the
network. Do not rerun the sweep to try a new metric.

### RESOLVED — the `fr-starved` "anomaly" is the same layer artifact

The low CKA on French pairs in `fr-starved`/`fr-starved-15b` is **not** a broken
checkpoint, and CKA and retrieval do **not** disagree (an earlier note here
claimed they did — that was drawn from a single layer). Per-layer EN-FR profiles
show both metrics collapsing and recovering together:

```
layer      0    2    4    6    8   10   12*  14   15   16
fr-fair   .25  .38  .40  .53  .77  .80  .82  .84  .82  .79   CKA
fr-starv  .24  .24  .03  .09  .10  .13  .23  .54  .73  .80   CKA
fr-starv  3.5  6.8  2.7  2.0  3.1  5.1  7.5  8.1  8.5  7.5   d'
                    ^^^^^^^^^^ both metrics dip together
```
(* = the fixed `ref` layer.)

**Peak CKA is 0.797 (fr-starved) vs 0.840 (fr-fair)** — a modest gap, not the
0.23-vs-0.82 the fixed layer implied. What differs is *depth*: the layer at
which CKA first reaches ~0.75 is L6 (ar-fair), L7 (fr-fair), L9 (ar-starved),
**L15 (fr-starved)**. So **starved tokenizers delay the depth at which
cross-lingual alignment emerges**, with French the extreme case — a real and
thesis-relevant effect, and the same mechanism behind the retracted deltas
above. Fertility does not explain *which* models are affected (fr's
starved/destarved ratio is 1.30, below ar's 1.48), so depth-of-emergence is the
better description than "starvation degrades French".

**Read this as a warning, not a result.** It is the same failure mode as §6's
XNLI (an evaluation artifact masquerading as a training finding) and the
Belebele letter-format probe — the third time in this project that an
uncontrolled downstream number was mostly measuring the benchmark. Alignment
deltas from `analyze_alignment.py` (paired bootstrap, matched-token
checkpoints, same estimator as `bootstrap_transfer.py`) are the numbers to
quote; raw per-model alignment scores are not.

---

## 6c. Language-specific neurons (LAPE, arXiv 2402.16438)

Port of Tang et al. 2024's LAPE to this repo: for every SwiGLU FFN neuron
(16 layers × 5632 = 90,112 per model), record `P(silu(w1·x) > 0)` per language
on FLORES+ dev+devtest (2009 parallel sentences, 60–96k counted tokens/lang,
BOS/pad excluded), then keep the bottom-1%-entropy neurons under the paper's
95th-percentile activation filters. `src/xscript/eval/neurons.py` (recording +
faithful `identify.py` port), swept over **all 109 checkpoints × 5 languages**
including the full token-budget series. Raw counts:
`/mnt/scratch/xscript_lape/results/lape/*.npz` (376 MB, one 16×5632×5 count
tensor + per-lang token totals each — re-analysis needs no forward pass);
identification + per-model tables committed in `results/lape/`.
XLA↔CPU parity verified (117 of 450k cells off by exactly 1 count, fp32 ties).

**The paper's picture inverts at this scale: language-specific neurons mark
foreignness, not competence.** Mean specific-neuron count per language over
final checkpoints: **trained** languages 27–49 (fair) / 8–28 (starved) vs
**untrained** 83–542 (fair) / 48–689 (starved) — a ~10× gap in the opposite
direction from the multilingual-LLM setting the paper studies, where each
trained language owns hundreds of neurons.

**Layer structure reproduces the paper's bottom+top concentration, but the two
ends carry different things.** Trained-language neurons sit almost entirely in
the top layers (L14–15 hold ~61% in both tokenizer conditions; layer 0 has
essentially none). Untrained-language neurons split between layer 0 (embedding
script-detectors) and the top.

**The thesis-relevant result — tokenizer starvation polarizes the script
divide at the neuron level.** For *untrained* languages at final checkpoints:

| untrained lang group | fair | starved |
|---|---|---|
| same-script (Latin) | mean 199 | **mean 27** (dissolves) |
| cross-script (ar/zh) | mean 340 | **mean 432** (grows) |

And *where* the foreign ar/zh neurons live flips with the tokenizer:

| foreign ar/zh | layer-0 share | top-2-layer share |
|---|---|---|
| fair | 0.03–0.10 | 0.44–0.70 |
| starved | **0.69–0.76** | 0.09–0.15 |

Under the fair tokenizer a foreign cross-script language is handled by
top-of-stack (prediction-side) machinery; under the starved tokenizer it is
segregated at the embedding layer by dedicated script-detector neurons, while
foreign *same*-script text becomes nearly transparent (shared vocab pieces).
This is a clean mechanistic companion to §6b's depth-of-emergence finding
(starved tokenizers delay the layer at which cross-lingual alignment appears):
starvation keeps cross-script input segregated at the bottom of the network.

**Bilingual training absorbs the partner's neurons.** Adding X as a training
language collapses X-specific counts, at every matched budget (en-mono →
en-X bilingual @30B: ar 402→47 fair / 741→15 starved; zh 225→58 / 142→12;
de 196→73 / 58→25; fr 192→54 / 68→30). Once trained, cross-script partners
need barely more dedicated neurons than same-script ones (fair finals:
de 73, fr 54, ar 47, zh 58) — dedicated-neuron count is NOT where the
cross-script penalty lives; the fair-condition model integrates ar/zh into
shared circuitry about as well as de/fr. Starvation halves own-language
neurons across the board (partner counts 73/54/47/58 → 25/30/15/12; en-mono
27→7): the starved tokenizer forces *more* sharing for trained languages.

**Dynamics: neuron sets consolidate slowly and never settle.** Jaccard overlap
between consecutive checkpoints' selected sets rises from ~0.15 (1B→2B) to
only ~0.3–0.48 late in training; trained-language sets are more stable than
foreign ones (mean J 0.44 vs 0.28 over the LR-matched 15B→23B step). The
mono 15B→30B transitions (which include the entire cooldown, so LR state is
mixed — same caveat as §6) churn hardest, down to J 0.08 for ar/starved.
Counts per language are roughly flat across training; what changes is *which*
neurons are selected.

Caveats: neuron identity is only comparable within a family (same run/init),
so cross-family Jaccards are meaningless and were not computed; LAPE
thresholds are per-model percentiles, making counts relative measures; single
corpus (FLORES+, news-ish register); no error bars (one run per cell), though
the fair-vs-starved contrasts replicate across every family. English is an
outlier throughout: it never accumulates many specific neurons in any model
(max ~200, early checkpoints only) — as the highest-resource, most-shared
language its activations are broadly distributed.

---

## 6d. Five-language benchmarks: SIB-200, Taxi-1500, zh HellaSwag

The C.5 suite cannot answer "how does this model compare *across* the five
languages" on its own, for two independent reasons: at the time only XNLI
appeared to carry signal (§6 — that verdict is retracted in §6e; the others
were being read from pooled cross-language tables), and **four of its six benchmarks do not even cover all five
languages** — HellaSwag has no zh, XStoryCloze no de/fr, XWinograd no de/ar.
`run_extra_bench.py` + `analyze_extra_bench.py` add three benchmarks that do.
Results: `$WORK/results/extra_bench/<run>_final.json`, **41 checkpoints** — the
15 finals, the `*-15b`/`*-12b` matched-token monolinguals and the `*-23b`
bilinguals, which is what makes the LR-clean transfer table below possible.

### The blank ZH-HellaSwag column was a corrupt data file, not a missing translation

§6 recorded it as "no Chinese translation exists in this lm-eval build". Half
right. `alexandrainst/m_hellaswag` — the dataset behind the registered
`hellaswag_{ar,de,fr,…}` okapi tasks — **does** ship `data/zh/val.jsonl`. It
does not load:

```
ArrowInvalid: JSON parse error: Column(/endings/[]) changed from string to object in row 153
```

4 of the split's 37,064 `endings` (all in doc index 5074) were written as
`{"zh": …, "en": …}` dicts instead of bare strings — a translation-pipeline
leak. pyarrow infers the schema from the first chunk, hits the type change and
rejects the whole file, which is why lm-eval 0.4.12 ships 31 okapi HellaSwag
languages and no `zh`. Reading the jsonl with the stdlib and taking the `"zh"`
member recovers all 9266 docs (`c5_tasks/hellaswag_zh/utils.py`), reusing
upstream's `process_docs` **verbatim** so the column stays comparable.

| checkpoint | zh tokens | acc | acc_norm |
|---|---|---|---|
| zh-fair-12b | ~11.75B | .3158 | .3664 |
| zh-starved-12b | ~11.75B | .3050 | .3501 |
| zh-fair-15b | ~14.76B | .3194 | .3728 |
| zh-starved-15b | ~14.76B | .3091 | .3532 |
| en-zh-fair-23b | ~11.4B | .3178 | .3706 |
| en-zh-starved-23b | ~11.4B | .3100 | .3554 |
| en-zh-fair (30B) | ~15B | .3383 | .4112 |
| en-zh-starved (30B) | ~15B | .3302 | .3950 |

Chinese HellaSwag is **nowhere near chance** (.250) — `en-zh-fair`'s .411 sits
between its own English (.456) and en-de-fair's German (.422) in §6's C.5
table. And **fair > starved in all four matched pairs** (+.016/.020/.015/.016
acc_norm), with no exceptions.

**The LR-state confound reproduces exactly here, on a benchmark this project
had never run.** Bilingual − monolingual on zh HellaSwag, the same comparison
computed two ways (per-metric, so the two rows are like-for-like):

| | fair | starved |
|---|---|---|
| **LR-matched** (`zh-*-12b` vs `en-zh-*-23b`, both mid-stable @3e-3), acc | **+0.002 [−0.003, +0.007]** | **+0.005 [+0.000, +0.010]** |
| LR-mismatched (`zh-*-15b` vs cooled 30B final), acc | +0.019 | +0.021 |
| **LR-matched**, acc_norm | **+0.004** | **+0.005** |
| LR-mismatched, acc_norm | +0.038 | +0.042 |

An **8–9× inflation on both metrics**, and the clean deltas are
indistinguishable from zero. Independent corroboration of the warning §6
attaches to 5 of its 7 transfer cells — on a benchmark that did not exist in
this project when that warning was written.

### SIB-200 (primary) and Taxi-1500 (backup)

| | SIB-200 | Taxi-1500 |
|---|---|---|
| source | FLORES-200 sentences | Parallel Bible Corpus verses |
| topics / chance | 7 / 0.143 | 6 / 0.167 |
| **majority class** | **0.251** | **0.261** |
| n per language | 1004 | 1077 |

SIB-200 is the one to quote. Topic classification is largely lexical, so it is
a task an undertrained model can actually do — and it is built on the **same
FLORES sentences** as §6's BPB and §6b's alignment, so downstream capability
and representation alignment are measured on identical text. Both are scored
0-shot cloze (the topic's own words as continuation, never a letter), docs
sorted by FLORES sentence id / PBC verse id so **the doc order is identical
across all five languages**.

**SIB-200 is only the second benchmark in this repo, after XNLI, to
discriminate at this scale** — `en-fair` scores .690 on English against a .251
majority baseline, where ARC, Global-MMLU and Belebele all sit at chance.

### Three controls, and all three changed the answer

1. **`acc_norm` is degenerate on this task.** Averaged over the 41 models it
   stays in .181–.372 across every (lang, task) cell, i.e. pinned near the
   majority rate, while `acc` spans .204–.529 over the same cells. It divides
   loglikelihood by the answer's length (**characters**, lm-eval's
   `completion_len` — not bytes as originally written here), which favours the
   longest option — and SIB-200's longest label ("science and technology") is
   *also* its majority class. ~~**Quote `acc`.**~~
   ⛔ **"Quote `acc`" is RETRACTED (§6e).** `acc` is degenerate in the
   *opposite* direction and just as badly: it never ranks the longest label
   first, so the 25.1%-majority class has **median recall 0.000** across all
   205 cells. Both estimators decide the argmax before reading the document.
   **Quote `acc_cal`** (prior-calibrated), and report `acc − null` and
   prediction entropy alongside. (The runner storing every metric is still the
   right call, and it now stores the raw loglikelihoods too, so this was
   fixable without re-running the accelerator. Note the `scores` field in the
   JSONs prefers `acc_norm` for shape compatibility and remains **the wrong
   field to read** — use `metrics`, or better, the raw sidecar.)

2. **Constant-prediction collapse, which accuracy alone cannot reveal.**
   `analyze_extra_bench.py` flags a cell whose 0/1 hit vector is exactly the
   indicator `gold == c`: the model ranked the same label first for all ~1000
   documents and "scored" that class's frequency having learned nothing.
   **16 of 574 cells collapsed**, every one of them in a language the model
   was not trained on, and 10 of the 16 in Arabic. **No cell used by the
   LR-matched transfer table below is among them.** The clearest case: `en-fair` on
   Arabic SIB-200 scores .110 — *below* uniform chance — purely because its
   constant choice happens to be a minority class. Without this check that
   reads as "very poor Arabic"; with it, it reads as **zero input-dependent
   Arabic discrimination**, which is a different and much stronger statement.
   Note `en-starved` on the same cell does **not** collapse (.199) — the
   starved tokenizer leaves an English-only model with *some* Arabic
   discrimination where the fair one has none, the same direction as §6's
   `ar/xnli` fair−starved delta (−0.087, favouring starved) and §6c's
   forced-vocabulary-sharing story. Single models, no CIs — suggestive only.

3. **Label language, which is worth ~14 points.** Both datasets ship English
   label words for every language, so the naive setup asks a zh-only model to
   rank English strings. `sib200_{code}` localizes prompt and labels;
   `sib200_enlab_{code}` is the same texts with the English ones. Paired
   bootstrap over `localized − English`, split by whether the model trained on
   that language:

   | | n cells | mean Δ |
   |---|---|---|
   | trained on that language | 35 | **+0.046** |
   | not trained on it | 129 | **−0.094** |

   English labels **inflate** scores on languages the model doesn't know (it
   falls back on reading the label) and **deflate** them on languages it does.
   `en-fair` on German: .281 localized vs .519 with English labels; `de-fair`
   on German: .515 vs .536. A cross-language table built on the shipped
   English labels would have been wrong in both directions simultaneously.

### The headline: no cross-script downstream penalty on SIB-200

> ⛔ The `acc` numbers in this subsection and the next are superseded by §6e.
> The *conclusion* here (no cross-script penalty in attained capability)
> survives and is strengthened; the transfer deltas below do not.

Own-language accuracy, localized labels, `acc` (chance .143, majority .251):

| partner | script | bilingual 30B final (fair) | monolingual final (fair) |
|---|---|---|---|
| de | same | .518 | .515 |
| fr | same | .571 | .544 |
| **ar** | **cross** | **.557** | **.570** |
| **zh** | **cross** | **.569** | .546 (15b) |
| en | — | .624–.690 | .690 |

**Arabic and Chinese sit inside the same-script range on both the bilingual and
the monolingual side**, and the Arabic monolingual is the strongest non-English
model in the table. On the first benchmark in this project that both covers all
five languages and clears the majority baseline, the cross-script penalty does
not appear downstream. That is consistent with §6's corrected-XNLI conclusion
(the apparent cross-script downstream penalty was largely a measurement
artifact) and inconsistent with reading §6's ATLAS-BTS separation as a
capability gap.

Fair ≥ starved on own-language SIB-200 in 6 of 8 pairs, and **by far the most
on English** (en-fair .690 vs en-starved .581, **+.109**). On partner languages
the effect is smaller and not uniform: ar-mono +.055, en-ar +.033, en-de +.024,
zh-15b +.022, en-fr +.013, but fr-mono −.004 and en-zh −.007. That is the same
asymmetry §6 found in the matched-token transfer deltas — the tokenizer's
effect is large and consistent on English, ~0 and sign-unstable on the partner
language.

~~**Taxi-1500 is much weaker and should only corroborate.** Own-language
accuracy is .188–.363 against a .261 majority baseline — barely separable.~~
⛔ **The "much weaker" half is RETRACTED (§6e): that was the scoring rule, not
the benchmark.** Calibrated, Taxi-1500 clears its own null by **+0.175 to
+0.249 at 11–17σ in all five languages**, with prediction entropy 0.98 and
**all six classes recalled** — the "Violence class is never predicted" problem
existed under `acc`/`acc_norm`/PMI and vanishes under `acc_cal`.

What does **not** go away is a defect calibration cannot fix: its register is
uneven *across* languages. The only full-coverage open Arabic edition is from
**1865** while the pinned de/fr/en editions are 20th–21st century, so the
Arabic column is disadvantaged for reasons unrelated to the model — and Arabic
is indeed its weakest column (+0.175 vs +0.204…+0.249). Therefore:

* **exclude from any cross-language aggregate** — difficulty varying by
  language for corpus reasons breaks exactly the comparison such a mean makes;
* **include for within-language work** (bilingual−monolingual, fair−starved),
  where both models read identical text and the edition confound cancels, and
  where scripture is a genuinely independent domain from the rest of the suite.

Calibrated, its transfer deltas disagree in sign with SIB-200 on 5 of 7 cells
(including reversing `ar/fair`, +0.036 vs −0.040), so treat agreement between
them as evidence and disagreement as a flag — do not average them.

### Transfer deltas, now LR-clean in 7 of 8 cells

Scoring the `*-12b` monolinguals and `*-23b` bilinguals closed the gap §6 flags
as its biggest weakness. `{lang}-{tok}-12b` (~11.75B/lang) vs
`en-{lang}-{tok}-23b` (~11.4B/lang) are **both mid-stable at peak LR 3.0e-3**
— decay starts at 24B — so this pairing is LR-matched *by construction*, needs
no new training, and the English anchor is `en-{tok}-12b` at 11.75B against the
bilingual's 11.4B English share (a ~3% match, retiring the `~` stand-in
CLAUDE.md's §6 table had to use for zh). Only `de/starved` is still missing, for
the usual reason: no `de-starved` monolingual exists at any budget.

SIB-200, localized labels, `acc`, bilingual − monolingual (paired bootstrap,
B=2000, n=1004; `*` = CI excludes 0):

| partner | script | tok | mono | bi | **Δ on partner** | Δ on English |
|---|---|---|---|---|---|---|
| de | same | fair | .488 | .541 | **+0.053** [+.032, +.073]* | +0.001 |
| fr | same | fair | .529 | .540 | +0.011 [−.011, +.031] | +0.015 |
| fr | same | starved | .520 | .566 | **+0.046** [+.019, +.074]* | **−0.093*** |
| ar | cross | fair | .531 | .527 | −0.004 [−.024, +.016] | −0.011 |
| **ar** | **cross** | **starved** | .538 | .428 | **−0.110** [−.135, −.086]* | **−0.053*** |
| zh | cross | fair | .542 | .580 | **+0.038** [+.017, +.059]* | **+0.024*** |
| zh | cross | starved | .524 | .534 | +0.010 [−.015, +.034] | **−0.073*** |

> ⛔ **EVERY NUMBER IN THE TABLE ABOVE IS SUPERSEDED — see §6e.** The `acc`
> column it is built on is a length-degenerate estimator: on SIB-200 the
> majority class ("science and technology", 25.1%, and the longest of the seven
> shared labels) has **median recall 0.000** across all 205 cells, because an
> unnormalized summed loglikelihood can never rank the longest candidate first.
> Re-scored with prior calibration (`acc_cal`), the same checkpoints and the
> same documents give: de/fair **+0.010**, fr/fair **−0.021**, fr/starved
> **+0.016**, ar/fair **−0.040\***, ar/starved **−0.037\***, zh/fair
> **+0.000**, zh/starved **+0.021**. The retractions that follow from that are
> listed in §6e; the two biggest are that **no same-script cell is significant
> any more**, and that **`ar/fair` and `ar/starved` are statistically identical
> (−0.040 vs −0.037)**, so the cross-script × starvation interaction the
> paragraph below builds on does not exist.

~~Mean Δ: **same-script +0.037** (n=3) vs **cross-script −0.016** (n=4), a gap of
+0.053 in the thesis's predicted direction — adding English helps a same-script
partner and does not help a cross-script one.~~ (Calibrated: same-script
**+0.002**, cross-script **−0.014**, gap **+0.016**.)

~~**But read that gap as one cell, not four.** It is carried almost entirely by
`ar/starved` (−0.110); drop that cell and cross-script becomes +0.015 against
same-script's +0.037, a gap of +0.022 with no cell individually significant in
the negative direction. What `ar/starved` *is* — the one condition where
cross-script and tokenizer starvation apply **together** — makes it exactly the
cell the interaction hypothesis predicts should be worst, and Taxi-1500
independently agrees on its sign (−0.095 [−.122, −.065]).~~ **All three legs of
this argument fail under calibration** (§6e): the gap is not one cell (it is one
*language* — Arabic, in both tokenizer conditions equally, so there is no
interaction); Taxi-1500 does not corroborate (it disagrees in sign with SIB-200
on 5 of 7 cells, including reversing `ar/fair` to +0.036); and XNLI, the only
other benchmark here that discriminates, puts Arabic transfer at **+0.018**,
the opposite sign. Two caveats that do survive: `en-ar-starved-23b` is still
worse than the English monolingual **on English** (−0.041 calibrated), i.e.
that bilingual underperforms on *both* its languages, which is what
interference looks like but is equally consistent with one weak training run;
and there is one run per cell, so the group means have no error bars even
though the individual deltas do.

**The mismatched rows are kept alongside, and the bias is large.** Same cells,
scored `*-15b` mono vs the cooled 30B bilingual: de/fair +0.026 (vs +0.053
matched), fr/fair +0.033 (vs +0.011), ar/fair +0.002 (vs −0.004), ar/starved
+0.004 (vs **−0.110**). The confound does not merely inflate — for `ar/starved`
it **flips the sign and hides a 0.11 effect**, which is the strongest argument
yet for §6's warning.

**Do not read the `sib200_enlab_*` rows as transfer.** They move hugely
(fr/fair +0.325, ar/starved +0.134) because an English-trained bilingual is much
better at ranking *English label words* — that is the label-language effect
above, not partner-language capability.

**This does not contradict the flat own-language table above.** The two measure
different things and both hold: cross-script models *reach* comparable absolute
accuracy (ar .557, zh .569 vs de .518, fr .571 at 30B), while adding English to
a cross-script run *buys less* than adding it to a same-script one. A penalty in
transfer, not in attained capability.

**Caveats.** One training run per cell, so the fair-vs-starved contrasts have
no error bars (the *deltas* do — paired bootstrap over docs — but not the runs).
`de` has no starved monolingual at any budget, as everywhere else in this repo.
Mid-stable checkpoints sit in the noisy regime §6 documents for BPB (adjacent
checkpoints move ±0.008–0.019), so a single-budget delta should not be
over-read even when LR-matched — the fix is the curve, as with ATLAS-BTS.

---

## 6e. Scoring recalibration: the fifth format artifact, and what it retracts

Everything in §6d, plus §6's XNLI columns and its Global-MMLU verdict, rested
on estimators that rank candidates by a score containing a **per-candidate
constant that does not depend on the document**. Removing that constant changes
several headline conclusions and dissolves three others. This is the fifth time
in this project an uncontrolled scoring choice turned out to be measuring the
benchmark rather than the training (after XNLI connectives, the Belebele letter
format, the alignment fixed-layer probe, and §6d's own `acc_norm`).

### The defect

SIB-200 gives every document the same 7 topic labels. `acc` sums an
unnormalized loglikelihood over the label, so the longest label loses on every
document alike; `acc_norm` divides by its length, so the longest label wins on
every document alike. Measured per-gold-class recall over all **205** SIB-200
model×lang cells:

| estimator | recall on `science/technology` (25.1% majority, longest label) | classes recalled >10% (of 7) | exactly-constant predictors |
|---|---|---|---|
| `acc` (what §6d quotes) | mean 0.023, **median 0.000**, max 0.433 | mean 3.69 | 9 / 205 |
| `acc_norm` | mean 0.899, **median 0.996** | mean 1.84 | 30 / 205 |

A quarter of the evaluation set is therefore decided before the document is
read, in every cell, under either estimator. What remains is "which of the
other labels this checkpoint happens to be able to reach", and that drifts
between checkpoints — which is the entire source of the non-monotonic
trajectories (`en-starved` English: .609 @12B, .537 @15B, .581 @30B) and of the
single-cell swings §6d treats as training effects.

lm-eval's XNLI for en/de/fr has the same defect at the connective
("Ja"/"Auch"/"Nein"): §6 diagnosed and fixed exactly this for ar/zh and left
the other three languages on raw loglikelihood.

Note also: **`acc_norm` normalizes by CHARACTER count, not bytes** — §6d says
bytes. lm-eval's `completion_len`, not its `byte_length`, is what divides.

### The fix

`src/xscript/eval/rawscores.py`. Two parts; the first matters more.

1. **The runners now persist raw per-candidate loglikelihoods** to
   `<results>/raw/<run>_raw.json` (conditional, unconditional where the task
   requests PMI, plus each candidate's gold index and char/byte/token lengths).
   Every estimator is then a pure-CPU re-derivation and **a scoring change
   never costs another accelerator pass** — the same lesson as §6b's cached
   embeddings. This is why the defect survived three prior audits: only 0/1 hit
   lists were kept, so the scoring rule could not be re-examined without a full
   re-run.

2. **`acc_cal`** — subtract each candidate's mean loglikelihood over the
   evaluation set: `s(d,c) = ll(c|x_d) − mean_d' ll(c|x_d')`. Contextual
   calibration (Zhao et al. 2021, arXiv:2102.09690). Removes the per-candidate
   offset exactly, whatever its cause (length, prior, tokenizer fertility), at
   zero extra compute.

Guards, because this is the failure mode the project keeps repeating:

- **Verified against lm-eval bit-for-bit.** `check_reproduces()` re-derives
  lm-eval's own `acc`/`acc_norm`/`acc_mutual_info` from the stored raw scores;
  0 mismatches, and `acc_pmi` returns 0.5817 against lm-eval's 0.5817 on real
  data. Nothing derived is trusted until that identity holds.
- **`acc_cal_loo`** (leave-one-out) answers the transductive objection; agrees
  with `acc_cal` to three decimals in every cell measured.
- **Empirical null** `Σ_c P(pred c)·P(gold c)` — the accuracy a given
  prediction *distribution* would get independent of gold. This is the honest
  baseline, not 1/k and not the majority rate: a constant predictor scores
  exactly its class frequency and `acc − null` is 0.
- **`SHARED_CHOICE_TASKS` allowlist.** `acc_cal` is only emitted where the
  choice *index* means the same thing in every document — `sib200`,
  `taxi1500`, `xnli` (index c is always connective c), `gmmlu_letter`. It is
  withheld for HellaSwag / ARC / Belebele-cloze / XStoryCloze / XWinograd,
  whose candidates are document-specific text. `_calibrate` would still *run*
  on a fixed-arity task like HellaSwag and return a number, but that number is
  a position-bias correction, a different and far smaller effect, and reading
  it as the label-prior fix would be exactly the error this section documents.

Two independent pre-registered checks that the fix is noise removal and not
tuning: trajectory monotonicity (XNLI summed backwards movement / range:
**0.55 → 0.00**, with the range preserved) and accuracy over the empirical null
(`en-ar-starved-23b` Arabic: +0.294 → +0.465).

### Re-derived results (41 checkpoints, `--own-langs`, full splits)

**SIB-200 transfer, LR-matched `{lang}-{tok}-12b` vs `en-{lang}-{tok}-23b`:**

| cell | script | `acc` (§6d) | **`acc_cal`** |
|---|---|---|---|
| de/fair | same | **+0.053\*** | +0.010 [−.019, +.038] |
| fr/fair | same | +0.011 | −0.021 [−.048, +.006] |
| fr/starved | same | **+0.046\*** | +0.016 [−.013, +.045] |
| ar/fair | cross | −0.004 | **−0.040\*** [−.068, −.012] |
| ar/starved | cross | **−0.110\*** | **−0.037\*** [−.067, −.004] |
| zh/fair | cross | **+0.038\*** | +0.000 [−.026, +.026] |
| zh/starved | cross | +0.010 | +0.021 [−.006, +.051] |

Means: same-script +0.037 → **+0.002**; cross-script −0.016 → **−0.014**; gap
+0.053 → **+0.016**. Per language: de +0.010, fr −0.003, **ar −0.038**, zh
**+0.011**.

**The "cross-script" effect is Arabic, not script.** Chinese is equally
cross-script and equally non-Latin, and its transfer is *positive* and
statistically identical to German's. Excluding Arabic, the cross-script mean
(+0.011) is **higher** than the same-script mean (+0.002) — the sign reverses.

**XNLI transfer, same pairing, does not reproduce the Arabic effect:**

| cell | `acc` | `acc_cal` |
|---|---|---|
| de/fair | **−0.053\*** | +0.006 |
| fr/fair | −0.008 | +0.019 |
| fr/starved | **+0.041\*** | **+0.050\*** |
| ar/fair | +0.004 | +0.020 |
| ar/starved | **+0.023\*** | +0.016 |
| zh/fair | +0.002 | +0.012 |
| zh/starved | +0.002 | +0.004 |

Calibrated, every XNLI cell is positive; same +0.025, cross +0.013, gap +0.012.
Arabic is **+0.018**, the opposite sign to SIB-200's −0.038. So the Arabic
transfer cost is a **SIB-200 finding that the other discriminating benchmark
reverses**, and is not established.

**Own-language SIB-200, calibrated — and the degeneracy is gone.** 30B finals:
ar-fair **.734**, fr-fair .738, en-fair .761, de-fair .694, zh (12B, no 30B zh
mono exists) .712. All 58 own-language cells have prediction entropy
**0.95–1.00** and beat their own null by **+0.46 to +0.61** — not one
degenerate cell, against 9 exactly-constant and 61 near-constant under `acc`.
Arabic is the strongest non-English model in the table. §6d's "cross-script
costs nothing in attained capability" is **strengthened**.

**Tokenizer effect on own-language, fair − starved (paired over docs):**
positive in 8 of 10 cells and significant in 7 (ar 12B +0.064\*, ar 30B
+0.055\*, fr 12B +0.080\*, fr 30B +0.030\*, en 30B +0.030\*, en-ar bi +0.061\*,
en-fr bi +0.043\*). Under `acc` the sign flips six times across the same ten
cells. **But the direction is the reverse of §6d's**: §6d says the tokenizer
effect is "large and consistent on English, ~0 and sign-unstable on the partner
language" and builds a mechanism on it (English's budget subsidizing ~419
languages). Calibrated, **English is the weakest and least consistent cell**
(−0.016, −0.001, +0.030) and the **partner languages are the strongest**
(+0.030..+0.080). The +0.109 English gap that motivated that mechanism was the
most length-inflated number in the table; the mechanism as written is not
supported.

**Taxi-1500 does not corroborate and should be dropped.** Calibrated, it
disagrees in sign with SIB-200 on 5 of 7 cells, including reversing `ar/fair`
(+0.036 vs −0.040) — the exact cell where SIB-200 finds its effect. §6d cites
Taxi as independently supporting `ar/starved`; that one agreement is a coin
landing the same way, not corroboration. Its "Violence" class has recall
0.00–0.03 in every cell under every estimator, i.e. it is effectively a 5-way
task at best.

### Global-MMLU: the models DO have world knowledge

§6's verdict was underpowered rather than wrong-headed. `en-fair`, English,
full `CohereForAI/Global-MMLU` (n=14,042; lm-eval defaults to the n=400
`-Lite` build, and §6 used `--limit 200`):

| format | estimator | acc | null | over null | pred entropy |
|---|---|---|---|---|---|
| **cloze** (answer text) | `acc` | .292 | .247 | **+0.045** | 0.98 |
| **cloze** | **`acc_norm`** | **.310** | .251 | **+0.059** | 0.99 |
| cloze | `acc_tokennorm` | .309 | — | — | — |
| letter (A/B/C/D) | `acc` | .2499 | .248 | +0.002 | 0.80 |
| letter | `acc_cal` | .256 | .252 | +0.004 | 0.97 |

SE at n=14,042 is 0.0039, so cloze `acc_norm` clears its own null by **≈15σ**,
and all three cloze estimators agree. The letter format genuinely is at chance
and **stays there after calibration** — as it must, since calibration removes a
selection bias but cannot manufacture knowledge.

So: **these checkpoints hold ~6 points of measurable world knowledge but cannot
do the A/B/C/D indirection at all.** Two reasons §6 missed it: at `--limit 200`
the 2σ detection threshold is +6.1 points against a true effect of +5.9 (the
experiment sat on its own detection boundary), and the letter format — the one
that is genuinely flat — was weighted equally with cloze.

⚠️ One model, English only. Extend to the other four languages and to
`en-starved` before quoting "+6 points" as a general property.

### Prompt language: the harness hardcodes English, and it matters less than expected

Two findings from chasing why Arabic Global-MMLU sat at chance. The first is a
property of lm-eval that anyone reading these numbers inherits; the second
stopped a plausible-sounding explanation from being written down as fact.

**lm-eval hardcodes English scaffolding in multilingual tasks.** Verified in
0.4.12:

| task | prompt actually shown for a non-English document |
|---|---|
| `global_mmlu_*` (**all 15 languages** it ships) | `{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:` — English cue, Latin option letters, regardless of language |
| `arc_{de,fr,ar,zh}` (okapi m-arc) | `Question: {translated question}\nAnswer:` — e.g. `'Question: آنا تحمل مكعب ثلج…\nAnswer:'` |
| `hellaswag_{de,fr,ar}` (okapi) | clean — uses the translated `activity_label`, no English |
| `belebele_cloze_*` (ours) | clean — passage + question only |
| `sib200_*`, `taxi1500_*` (ours) | clean — localized cue (`الموضوع:`, `Thema:`, …) |

Note the asymmetry this creates for **ARC**, which §6d reads as "a striking
English-only pattern" (en 0.42 vs 0.24–0.26 elsewhere): English `arc_easy` gets
a natively-English prompt while `arc_{de,fr,ar,zh}` get a translated question
inside English scaffolding. The comparison is not like-for-like. Whether that
*explains* the gap is a separate question — see below, which suggests it does
not.

**Cue language ≠ label language, by an order of magnitude.** §6d measured the
label-language effect on SIB-200 at **~14 points**, so the English cue looked
like a strong candidate for Arabic's flat Global-MMLU. Measured directly on
`ar-fair`, same 2000 documents, localized `الإجابة:` vs English `Answer:`:

| | acc_norm | null | over null |
|---|---|---|---|
| localized cue | .267 | .251 | +0.016 |
| English cue | .267 | .251 | +0.016 |

**Identical.** The distinction is *what* is in the wrong language:

* SIB-200's control swapped the **candidates being ranked** into English — the
  model had to score English label strings against non-English text. Large.
* This swaps only the **scaffolding**; question and all four options stay in
  the document's language. Nil.

So Arabic Global-MMLU is at chance for real, not because of the prompt — and
by the same logic the ARC scaffolding asymmetry above is probably minor rather
than the explanation for ARC's English-only pattern. Do not quote it as one
without measuring it; `gmmlu_cloze_encue_*` exists precisely so this class of
question is answerable from stored data instead of costing a run.

**Practice going forward:** any new multilingual task ships with a
localized/English control pair, the way `sib200_*` / `sib200_enlab_*` already
does. The first version of `gmmlu_probe` hardcoded `Answer:` for all five
languages — this project's sixth format artifact, introduced while fixing the
fifth.

### Dataset identity: a benchmark can be "the same benchmark" in name only

ARC exposed a defect no scoring fix could catch. §6d's C.5 table compares
English `arc_easy` (**ARC-Easy**, n=2376) against `arc_{de,fr,ar,zh}`, which
come from okapi's `alexandrainst/m_arc` and are **100% ARC-Challenge**
(n=1169; verified by question matching -- 1169/1169 of m_arc's English rows
appear in ARC-Challenge, 1/1169 in ARC-Easy). Easy and Challenge differ by
~20 points at this scale. Scored like-for-like on ARC-Challenge, English is
**.268-.285** against the others' .24-.26 -- a ~2-point gap, not the
25-point one §6d reports as "a striking English-only pattern". **That claim
is retracted**, and ARC-Challenge turns out to carry almost no signal in any
language, English included (1.4-2.7σ), so ARC leaves the aggregate because it
is too hard for a 1B/30B model rather than because non-English is weak.

**Item-count audit of the whole C.5 suite** (equal counts across languages is
the cheapest possible identity check, and it is what caught ARC):

| family | counts | verdict |
|---|---|---|
| XNLI | 2490 x 5 | same pool |
| Belebele-cloze | 900 x 5 | same pool |
| XStoryCloze | 1511 x 3 | same pool (3 languages only) |
| HellaSwag | en 10042, de 9368, fr 9338, ar 9176, zh 9266 | **mismatch** (probably okapi filtering, not a different pool -- unverified) |
| XWinograd | en 2325, fr **83**, zh 504 | **severe mismatch** -- at n=83 French SE is ~5.5 points, so §6d's `fr/fair xwinograd` delta `+0.072 [-0.048, +0.181]` was never measurable |
| ARC | en 2376 vs others 1169 | **Easy vs Challenge** |

**Standing rule:** before comparing a benchmark across languages, verify the
non-English version renders the same pool as the English one. Check item
counts first (free), then match questions against the English original.
`scripts/external_bench/verify_mubench.py` does this for the MuBench tasks and
is the template for any new multilingual benchmark.

### MuBench: same items in every language, by construction

`aialt/MuBench` ships 12 benchmarks x 61 languages aligned by `_id`, which
removes this failure mode structurally. Verified before use: ARC-Easy 100% in
real ARC-Easy, HellaSwag 100% in real HellaSwag, MNLI 100% in real MNLI
validation, SNLI 100% in real SNLI -- with `_id` sequences identical across
en/de/fr/ar/zh and every gold index in range.

What it is and is not good for. It **fixes** ARC (Easy in all five languages),
gives StoryCloze and WinoGrande the de/fr coverage they never had (XWinograd's
French n=83 becomes 1213 x 5), and adds MNLI/SNLI/BMLAMA. It does **not**
improve translation quality -- the dataset ships no README, so provenance is
undocumented, which is weaker than MMMLU's professional translation and no
better than okapi. XNLI and Belebele gain nothing from it and stay as they are.

Two traps found while wiring it up, both silent:

* **Use `local_template`, not `en_template`.** MuBench ships both; the former
  localizes the instructions, the latter leaves English scaffolding around
  translated content.
* **The option markers are localized too** -- Arabic writes `الخيار A:`,
  Chinese `选项 A:` or `选项A:`, while ARC-Easy and BMLAMA use a bare `A:`. A
  parser matching the English `Option `/`Choice ` prefixes silently yields
  **zero rows** for ar/zh on StoryCloze, WinoGrande and MMLU. `mubench/utils.py`
  instead requires a single Latin capital immediately before a colon (ASCII or
  full-width) with any prefix, and requires the recovered letters to run
  A, B, C, ... consecutively -- the ordering constraint is what makes the loose
  prefix safe. On `en_template` the English-only regex worked, so this bug
  would have produced plausible numbers and never raised an error.

**NLI triple-counting.** XNLI, MNLI and SNLI are the same task (3-way
entailment) on disjoint item sets -- verified: 0% premise overlap pairwise,
and XNLI's dev/test were newly collected under MNLI's protocol rather than
taken from MNLI's splits (XNLI ∩ MNLI-val = 1/2499). Domains differ (SNLI is
Flickr30k captions with strong hypothesis-only artifacts; MNLI spans 10
genres) and only XNLI is professionally translated. A plain mean over
benchmarks would therefore weight NLI 3x, and NLI is already the
strongest-signal family. Weight by task family for the aggregate, and treat
XNLI as the representative with MNLI/SNLI as robustness checks.

### What to quote now

- `acc_cal` for SIB-200 / Taxi-1500 / XNLI; `acc_norm` for cloze Global-MMLU;
  unchanged estimators for HellaSwag / ARC / Belebele-cloze / XStoryCloze /
  XWinograd, which have per-document choices and whose trajectories were
  already monotone (HellaSwag backwards-movement ratio 0.00 vs XNLI's 0.55).
- Always report `acc − null` and prediction entropy alongside, never accuracy
  alone.
- Untouched by all of this: §6 BPB/BTS (per-token NLL), §6b alignment
  (embeddings), §6c LAPE (activations). None use multiple-choice scoring.

---

## 6f. State of play — read this before continuing the evaluation

Written 2026-07-31 at the end of the recalibration session (§6e). Everything
below is what a fresh agent needs and cannot infer from the code.

> ⚠️ **Superseded in part by §6g** (2026-08-02). Specifically: the estimator
> column below says `acc_norm` for the MuBench families — that is **wrong for
> ARC-Easy and BMLAMA**, use §6g's per-family table. Open items 1 and 3 are
> **done**. The Belebele/HellaSwag "no raw" rows are still accurate for the
> *okapi* HellaSwag; MuBench's HellaSwag now has raw for 100 checkpoints.

### What is measured, and with what

| benchmark | status | estimator to quote | in the cross-language aggregate? |
|---|---|---|---|
| SIB-200 | ✅ 41 ckpts, raw stored | `acc_cal` | **yes** |
| XNLI | ✅ 41 ckpts, raw stored | `acc_cal` | **yes** |
| Taxi-1500 | ✅ 41 ckpts, raw stored | `acc_cal` | **no — within-language only** (1865 Arabic edition vs 20-21C de/fr/en) |
| native MMLU (ArabicMMLU/CMMLU) | ✅ ar/zh finals, raw | `acc_norm` | **no — knowledge panel, per-language instrument** |
| Belebele-cloze | ⚠️ hit lists only, **no raw** | `acc_norm` | yes, but on nominal chance until raw is captured |
| HellaSwag | ⚠️ hit lists only, **no raw**; pool identity unverified (en 10042 vs okapi 9176-9368) | `acc_norm` | yes, same caveat |
| ARC-Easy / StoryCloze / WinoGrande / BMLAMA (MuBench) | 🔄 sweep running | `acc_norm` | candidates |
| ARC (okapi) | ❌ retired | — | no — Easy-vs-Challenge mismatch (§6e) |
| translated MMLU (Global-MMLU / MMMLU) | ✅ finals | `acc_norm` | **no** — ar has no signal, so a delta on it is undefined |
| MNLI / SNLI | dropped | — | no — same task as XNLI, would weight NLI 3x |

### Design decisions and why

* **`--own-langs` everywhere.** Only trained-language cells are scored. Every
  transfer cell pairs trained-language scores, so nothing needed is lost; the
  zero-shot cross-lingual readout is given up deliberately.
* **A benchmark enters the cross-language aggregate only if EVERY language
  clears its own empirical null** at the largest budget. This drops ARC,
  XStoryCloze and XWinograd (the last two also on coverage: 3 of 5 languages).
* **Aggregate in headroom units** `(acc − null)/(1 − null)`, not percentage
  points above nominal chance — benchmarks have different chance levels
  (SIB-200 .143, XNLI .333, ArabicMMLU .272) and different ceilings.
* **Always use the per-cell empirical null**, `Σ_c P(pred c)·P(gold c)`, not
  nominal chance. They differ exactly where it matters.
* **Knowledge is a separate panel, never pooled.** Translated MMLU asks about
  Anglocentric facts and cannot detect ar/zh knowledge; native exams can
  (ar-fair: +0.011 on MMMLU vs **+0.102 on ArabicMMLU**). Per-language best
  instrument, used for within-language contrasts only.
* **Only ONE NLI benchmark in the mean.** XNLI, MNLI and SNLI are the same
  task on disjoint item sets (verified 0% pairwise premise overlap); XNLI is
  the only professionally-translated one.

### Where the data lives — and what dies with the box

    /home/ubuntu/xscript_bench/results/{extra_bench,appendix_c5}/
        <run>_final.json        scores + metrics + per-example hit lists
        raw/<run>_raw.json      per-candidate loglikelihoods  <-- NOT in git

⚠️ **As of §6g the raw sidecars cover 100 checkpoints / 792 cells** and are
what made that session's entire estimator investigation (acc vs acc_norm vs
tokennorm vs acc_cal, the fertility proof, every re-derivation) cost **zero**
accelerator time. They are still ONLY on the eval box.

**The raw sidecars are the artifact that makes every future scoring change a
pure-CPU re-derivation.** They exist only on the eval box. Copy them off or
push them next to `jvonrad/xscript-embeddings` before the instance is torn
down; regenerating costs a full re-run (~14h on a trn2.3xlarge's two
core-pairs). `results/recalibrated/` in git holds the per-model JSONs only.
`archive_n14042/` holds the full-n Global-MMLU record for `en-fair` before it
was re-scored at n=2000.

### Open work, in priority order

*(updated 2026-08-02 — items 1 and 3 are DONE, see §6g)*

1. ~~Backfill raw for Belebele + HellaSwag~~ **DONE for MuBench HellaSwag**
   (100 checkpoints, raw stored) and the pool-identity question is **closed**:
   `mub_hellaswag` is 100% inside real HellaSwag and `_id`-aligned (§6g).
   *Belebele* still has no raw and was deliberately not re-run.
2. **Score the 9 `*-15b` monolinguals on the four MuBench families**
   (9 cells, ~2h). This is now the single highest-value missing number: the
   mono/bilingual budget rosters intersect only at 2B/5B/30B, so **there is no
   large mid-stable matched-total tier**, and 15B is the only one that is
   trainable-matched. It is the direct test of whether §6g's −.044 English
   dilution cost at 30B survives without the cooldown.
3. ~~Extend to the low-budget series~~ **DONE** — 68 checkpoints, §6g.
4. **`ar/starved` still needs a second training run.** Calibrated it is −0.037
   with `ar/fair` at −0.040, so it no longer carries an interaction — but the
   Arabic transfer effect itself is one run per cell and XNLI reverses its
   sign. §6g adds that the *same-script* side has the mirror problem: the
   advantage is carried by French alone, and ~~`de-starved` still does not
   exist~~ — ✅ **`de-starved` was retrained 2026-08-03 (§6h)**, 16.10B with
   checkpoints at de-fair's exact 7.753B/11.754B/14.755B budgets. The
   *training* half of the same-script problem is solved (French no longer has
   to carry it alone), but **the analyses have not been re-run** — §6's
   content-matched interaction and §6d/§6e's transfer table still print the
   old `fr`-only / `n/a` values. The Arabic half of this item stands unchanged.
5. **Delete `c5_tasks/arc_pmi/` and `c5_tasks/mubench_arc/`** and their
   `FAMILIES` entries — dead, superseded by `mubench/`.
6. **Rewrite §6d's tables** with calibrated numbers (was item 2; still open).

### Operational notes that cost real time

* **Anchor every `pgrep -f`/`pkill -f` pattern** (`^bash /path/...`). An
  unanchored pattern matches the shell that invokes it: it killed three shells
  in this session, and two waiters each matching the other's pattern
  deadlocked a chained sweep for 3.7h.
* `run_extra_bench.py --require-raw` (default on) treats a task scored before
  the raw sidecar existed as **not done**, so a resumed sweep backfills it.
* Long-prompt tasks (Global-MMLU, MMMLU) need `--batch-size 8` or lower;
  batch 16 fails with `NCC_EOOM002` because `_score_active_xla`'s one-hot
  materializes a `[batch, width, 65536]` float tensor.

---

## 6g. The 100-checkpoint trajectory sweep — and two estimator rules it overturns

Written 2026-08-02. Closes §6f's open items 1 and 3. **Where §6f and §6g
disagree, §6g wins**: it is measured on 792 cells against §6f's 23.

### What exists now

| | checkpoints | cells |
|---|---|---|
| low-budget series 1b/2b/5b/8b/10b/15b | 68 | 100 |
| `*-12b` mono + `en-*-23b` bilingual | 17 | 25 |
| cooled 30B finals | 15 | 23 |

Families: `sib200`, `xnli` (with `--xnli-raw-all-langs`), `mub_arceasy`,
`mub_storycloze`, `mub_hellaswag`, `mub_bmlama`. `--own-langs` throughout;
**raw sidecars stored for every cell**. Belebele and Taxi-1500 deliberately
not re-run. Readable output: `results/mubench_sweep/accuracy_table.md`
(one accuracy per model x language x benchmark); `per_cell_table.md` adds
null / pp / headroom / entropy.

### ⛔ Estimator, per family — §6f's blanket "`acc_norm` for MuBench" is wrong

Judged on §6e's own criteria (discrimination over the **empirical null**,
prediction entropy, trajectory monotonicity), never on gold accuracy:

| benchmark | estimator | over-null | entropy | backwards |
|---|---|---|---|---|
| **arceasy** | **`acc`** | **.182** | .846 | **.000** |
| | `acc_norm` | .152 | .854 | .005 |
| **bmlama** | **`acc`** | **.389** | .999 | **.033** |
| | `acc_norm` | .269 | .999 | .083 |
| **hellaswag** | **`acc_norm`** | **.068** | 1.000 | .000 |
| **storycloze** | **`acc_norm`** | **.071** | 1.000 | .092 |
| **sib200** | `acc_norm` | .066 | **.143** ⚠ | .673 |
| | **`acc_cal`** | **.535** | .987 | .593 |
| **xnli** | **`acc_cal`** | **.108** | .995 | .063 |

The split is predictable from candidate structure, so it is not
selection-on-the-metric: `acc_norm` wins where candidates are **long
free-form continuations** (HellaSwag ~139 chars, StoryCloze ~39) and loses
where they are **short fixed phrases** (ARC-Easy ~23 chars, BMLAMA bare
entity names). `acc` is `argmax P(c|x)` — Bayes-correct when candidates are a
priori exchangeable; `acc_norm` is a heuristic proxy for the candidate prior
with no probabilistic derivation (the principled correction is PMI). Note
`acc_norm` on SIB-200 reproduces §6e's collapse exactly (entropy .143).

### ⛔ NEVER use `acc_tokennorm` in this project

Dividing by token count is **tokenizer-dependent**, and this project's whole
contrast is a tokenizer. Whenever `acc_norm` and `acc_tokennorm` disagree,
tokennorm's pick provably has more tokens per character:

    ll_i/nc_i > ll_j/nc_j  and  ll_j/nt_j > ll_i/nt_i
      =>  nt_i/nt_j < nc_i/nc_j  =>  nt_i/nc_i < nt_j/nc_j

Confirmed empirically at **100.0% of disagreements in all five languages** —
it is a theorem, not a tendency. Measured fertility (tokens/char) on the eval
text, starved/fair: **en 1.14, de 1.24, fr 1.22, ar 1.32, zh 1.18**. So
tokennorm inflates starved models most in Arabic, i.e. exactly the cell the
thesis predicts an effect in. Character- and byte-normalisation are the
tokenizer-invariant choices — the same reason §6 measures **BPB**.

### Units: `pp` vs `headroom` — report pp as primary

`headroom = (acc - null)/(1 - null) = pp / (1 - null)`, i.e. a per-benchmark
**amplifier**: storycloze **2.00x**, xnli 1.50x, arceasy/hellaswag 1.33x,
sib200 1.18x, bmlama 1.11x. Within one benchmark the two give identical
orderings; across benchmarks headroom reweights. Use **pp for headline
numbers** and headroom only for the pooled aggregate — headroom assumes a
ceiling of 1.0, which is least true on StoryCloze, exactly where it amplifies
most. On a 2-way task headroom literally doubles the gap, which is how
Arabic StoryCloze's ".112 vs .262" turned out to be **.556 vs .631 accuracy**.

### SIB-200 is saturated; HellaSwag is the trajectory instrument

Bilinguals only (constant population), fraction of the 15B headroom already
reached at 2B, and trajectory non-monotonicity over 150 series:

| benchmark | solved @2B | backwards ratio |
|---|---|---|
| hellaswag | **33%** | **.000** |
| arceasy | 60% | .010 |
| bmlama | 74% | .051 |
| xnli | 55% | .063 |
| storycloze | 55% | .162 |
| **sib200** | **98%** | **.601** |

**SIB-200 is unusable for curves** — 98% solved at 2B, worst monotonicity by
4x. It is still the right benchmark for fixed-budget capability and for
continuity with §6d/§6e, but do not fit a trend through it. Its saturation is
accuracy-only: mean `P(gold)` keeps rising ~3.5x faster than accuracy, though
**all** estimators turn over at 15b, so a proper scoring rule makes the signal
bigger without making it monotone.

### Transfer: same-vs-cross is +0.014, stable — but it is French, not script

Every 1b-15b checkpoint is mid-stable at peak LR 3.0e-3 (decay starts at 24B),
so **these pairings are LR-matched by construction** — the confound §6/§6d
call their biggest weakness does not apply. Paired bootstrap over documents,
mono X B/lang vs bilingual 2X B total:

| tier | same-script | cross-script | gap |
|---|---|---|---|
| mono 1B / bi 2B | +.033 | +.019 | +.014 |
| mono 5B / bi 10B | +.018 | +.005 | +.013 |
| mono 8B / bi 15B | +.019 | +.000 | +.019 |
| mono 12B / bi 23B | +.020 | +.010 | +.010 |
| **mean (pp)** | | | **+.0138** |
| mean (headroom) | | | +.0177 |

Stable across all four tiers and robust to the estimator choice. **But by
language it is fr +.027 > de +.013 > ar +.009 > zh +.008** — German sits with
the cross-script pair, not with French. Under an all-`acc_norm` sensitivity
run de (+.015) is indistinguishable from zh (+.012) and ar (+.011). So
"same-script advantage" is one language, in both tokenizer conditions, at
every budget — structurally the same finding as §6e's "the cross-script
effect is Arabic". **Cross-script transfer also decays with budget**
(+.019 → +.005 → +.000) while same-script holds flat.

### BTS on the benchmarks, split by which language is measured

Repo-style BTS on capability instead of loss. ⚠️ **Use absolute deltas, not
the ratio**: `(cap_bi - cap_mono)/cap_mono` explodes at low budgets because a
1B monolingual is *at chance* on XNLI (denominator 1e-13) — the same "silent
degeneration" defect §6 documents in `results/bts/`, reproduced independently.

**MATCHED-LANG** (mono X B/lang vs bilingual 2X B total — equal exposure):

| tier | same partner | same En | cross partner | cross En |
|---|---|---|---|---|
| mono 1B / bi 2B | +.042 | +.034 | +.025 | +.026 |
| mono 5B / bi 10B | +.023 | −.009 | +.008 | −.015 |
| mono 8B / bi 15B | +.026 | −.004 | +.000 | −.010 |
| mono 12B / bi 23B | +.028 | −.017 | +.013 | −.016 |
| **overall** | **+.0298** | **+.0026** | **+.0115** | **−.0027** |

**MATCHED-TOTAL** (equal total tokens, bilingual sees half the language):

| tier | same partner | same En | cross partner | cross En |
|---|---|---|---|---|
| 2B | −.020 | −.047 | −.033 | −.054 |
| 5B | +.003 | −.037 | −.010 | −.039 |
| 30B (cooled) | +.017 | −.045 | +.007 | −.044 |
| **overall** | **−.0024** | **−.0424** | **−.0177** | **−.0461** |

**English's cost is DILUTION, not interference.** It is ~0 when English
tokens are held fixed (matched-lang, −.000 pooled) and ≈−.044 only when they
are halved — at *every* budget including 30B, while the partner language
recovers to positive. The partner gets most of its capability back from half
the tokens; English does not. This does **not** support §6d's proposed
mechanism (a starved tokenizer making English subsidise ~419 languages): the
effect tracks token count, not tokenizer. Note the script gap lives almost
entirely on the partner language — `gap(English)` is +.005/+.004 and decays
to ~0 at the largest budget.

Matched-total has only **three** tiers because the mono and bilingual budget
rosters intersect at 2B/5B/30B only. **15B is trainable-matched and missing
solely because the 9 15b monolinguals were never scored on MuBench** — that
is the one remaining gap, and it is the only way to get a *large mid-stable*
matched-total tier (the 30B row is cooled).

### The cooldown reproduces on four benchmarks that had never been run

Bilinguals only, headroom per B token: **15→23B gains +.0018/B; 23→30B gains
+.0088/B** — 4-5x more per token in the cooldown (arceasy +.062, bmlama +.055,
hellaswag +.060 in absolute headroom). So the 30B finals extend the curves but
are **not on the same curve**: never pair a cooled final against a mid-stable
monolingual, and never fit a scaling trend through the 30B point.

### No cross-script capability penalty; Arabic StoryCloze is real

Own-language accuracy at 30B: Chinese **beats both Latin partners** on
ARC-Easy (.401 vs de .333 / fr .335) and leads StoryCloze (.262 headroom).
Within-model gaps put ar worst and zh best, with de/fr between — **within-script
variation exceeds between-script variation**. §6d/§6e's "cross-script costs
nothing in attained capability" is confirmed on four new benchmarks.

Arabic StoryCloze (.556 acc vs .631 for zh) was checked hard and **is real**:
the independent, professionally-translated **XStoryCloze corroborates**
(ar .4815 vs en .5549 over 107 checkpoints), tokenization is clean (0.000%
`<unk>`, 22.3 tok/segment vs English's 22.0), translations are faithful and
`_id`-matched, and it is uniform across 24 checkpoints (sd .014). Not an
artifact — but a ~6-point accuracy gap on a task where every language sits
between .556 and .631 against a .500 floor.

### Pool identity — §6f open item 1 is CLOSED

`verify_mubench.py` (pure CPU, no accelerator): **`mub_arceasy` is 100% inside
real ARC-Easy** (n=2359 x 5) and **`mub_hellaswag` is 100% inside real
HellaSwag** (n=9044 x 5), both `_id`-aligned across all five languages. So
MuBench's HellaSwag is a strict aligned subset where okapi's is 9176-9368 and
unaligned. Two arity facts that matter for reading any degeneracy report:
**ARC-Easy is ragged (3/4/5 options)** so its prediction-entropy ceiling is
**0.863**, not 1.0; **BMLAMA is ragged 2-10** (91% ten-way), so nominal chance
is meaningless there and `acc_cal` is correctly withheld for both.

### Operational

* **Never edit a shell script while it is running** — bash reads scripts
  incrementally by byte offset, so an in-place edit can make a running worker
  jump to a wrong offset. Write a new file instead (this is why
  `run_finals_mubench.sh` duplicates `run_sweep68.sh` rather than
  parameterising it).
* `supervise_sweep.sh` counted only `run_sweep68.sh`/`chain_worker.sh`, so it
  declared the sweep over and exited while `run_finals_mubench.sh` was still
  running — the 12b/23b pass ran unmonitored for 4.5h. Fixed in
  `supervise_sweep2.sh`; add any new worker script to that pattern list.
* Throughput on a `trn2.3xlarge` (two core-pairs), all six families:
  **5.8 min fixed per model + 21.8 min per own-language cell** (27.6 min for a
  1-language model, 49.4 for a 2-language one). Predicted the full run to
  within ±30 min over 14h.
* `--keep-checkpoints` must be scoped to ONE model, not a whole sweep:
  68 x 4 GB = 272 GB against this box's 190 GB root.

---

## 6h. The `de-starved` retrain (2026-08-02/03, Isambard-AI, new allocation)

Closes the hole every other section has to apologise for: **`de__unigram_starved`
did not exist at any budget**, so §6's content-matched interaction had to average
same-script over `{fr}` alone, and §6d/§6e's transfer table prints
`n/a (no de-starved-12b)`. Run `de__unigram_starved`, job 5879754, on ~100
GH200 GPU-hours (25 node-hours) under project `brics.u6sg`.

### The original run did NOT "collapse" — it diverged at the warmup/peak-LR seam

§6's `EXCLUDE_RUNS` note calls it "collapsed mid-run", inferred from an anchor
BPB of ~1.72 against the destarved twin's ~1.06. The W&B curve is sharper than
that and identifies a specific, ordinary failure:

| tokens | flores_de BPB | |
|---|---|---|
| 0.75B | **1.2804** | its floor |
| 1.00B | 1.2902 | turns upward, exactly as warmup (1B) ends |
| 1.25B | 1.3769 | |
| 1.50B | 1.5642 | |
| 1.75B | 1.6759 | |

Training loss shows the same thing: min 2.6806 @0.73B, then 3.61-3.79 across
1.5-2.6B, recovering only to 2.71 by 7.69B — still behind the destarved twin's
2.70 despite having started *better*. So it is a **loss-spike divergence when
the LR pins at peak 3.0e-3**, not hardware, not data, not a collapse. `fr`
starved and `de` destarved survived the identical schedule; it was bad luck in
the (init, data-order) draw.

**The fix must not touch the LR schedule.** Lowering `peak_lr` or stretching
warmup would buy stability at the cost of de-matching this run from de-fair and
from every `-12b`/`-23b` pairing it exists to complete — §6's LR-state confound,
self-inflicted. `configs/base_de_starved_retrain.yaml` therefore changes
**only** `seed` 0->1 and `data_seed` 1234->5678. Verified by config diff:
model 9/9, optim 4/4, schedule 6/6 keys byte-identical to `base_main`.

**It worked.** Retrain vs the diverged original, flores_de BPB at matched
tokens: 1.00B **1.2595** (vs 1.2902, and *descending* where the original
turned), 1.25B **1.2216** (vs 1.3769), 1.50B **1.1889** (vs 1.5642), 1.75B
**1.1726** (vs 1.6759). The gap to de-fair holds flat at 0.088 -> 0.081 ->
0.082 — parallel tracking, which is the shape a sound run has. A 0.147 deficit
at the first eval (0.25B) closed to +0.011 by 0.75B: that was an init offset,
not a data problem, and the descent *rate* diagnosed it two evals before the
level did.

### RESULT: the run completed — 16.10B tokens, 9h36m, 76.8 GPU-hours

`COMPLETED 0:0`, 2 nodes, no restarts, no in-allocation stalls. All four marks
written (`stable_{7753,11754,14755,16100}M.pt`, 13.1GB each with optimizer
state, plus 4.4GB model-only `step*` checkpoints). **76.8 of the ~100 GPU-hours,
leaving 23.2 unspent** — the reserve held for a re-seed that was not needed.

Effective rate **209.7 Mtok/GPU-h** against the 218 predicted from the historical
runs (−3.8%). The shortfall is the four extra 13.1GB `stable_*` saves this run
adds and which no historical run paid for; 218 remains the right number for
budgeting an ordinary run, 210 for one with dense full-state marks.

The two token-matched marks landed on de-fair's checkpoints **step-for-step**:
`step8451_7753M` and `step12811_11754M` are byte-identical in name to
`de__unigram_destarved__step8451_7753M` / `step12811_11754M`. Same step, same
token count, same LR state — as clean a matched pair as this project can make.

**The same-script starved penalty on BPB, LR-matched, both mid-stable @3.0e-3:**

| budget | de-fair flores | de-starved flores | gap | holdout gap |
|---|---|---|---|---|
| 7.753B | 1.0068 | 1.0803 | +0.0735 | +0.0538 |
| 11.754B | 0.9989 | 1.0543 | +0.0554 | +0.0505 |
| 14.755B | 0.9907 | 1.0472 | **+0.0565** | **+0.0506** |
| 16.100B | — | 1.0466 (final) | — | — (0.9421) |

**Read this as a flat ~+0.05 BPB penalty, not a shrinking one.** The gap drops
once between 7.753B and 11.754B and then stops; holdout is flat throughout
(+0.054 / +0.051 / +0.051). 7.753B is the outlier, not the start of a trend —
an intermediate reading at 11.754B that looked like "narrowing with scale" did
not survive the third point. Note this is a *token-matched* penalty and so
still carries the content confound §6 documents: at equal tokens the starved
run has seen ~24% fewer bytes of German (3.291 vs 4.594 B/tok). The
content-matched comparison is what §6 wants, and `stable_16100M` is the
checkpoint for it (16.100 = 11.754 x 1.371).

**Still to do with these checkpoints:** recompute §6's content-matched
interaction with `de` as a second same-script language (it currently averages
over `fr` alone); fill §6d/§6e's `de/starved` transfer row by pairing
`step12811_11754M` against `en-de-starved-23b` (11.379B German, ~3% match);
and re-run §6b alignment / §6c LAPE on the new checkpoints if those tables are
to include the cell.

### Throughput: 218 M tokens per GPU-hour — use this to budget

Measured by integrating `tok_per_s` over 7 prior runs (both tokenizer
conditions, both world sizes, incl. in-loop eval + checkpointing):

| run | B tok | GPU-h | Mtok/GPU-h |
|---|---|---|---|
| ar-starved | 29.97 | 138.9 | 215.8 |
| de-destarved | 29.96 | 136.6 | 219.4 |
| en-destarved | 29.98 | 137.1 | 218.6 |
| fr-starved | 29.96 | 136.3 | 219.9 |

~61k tok/s per GPU either way: **1 node = 248k tok/s (983,040 tok/step), 2
nodes = 480k (917,504 tok/step)**. One node is ~1.5% cheaper per GPU-hour (no
inter-node collectives); two halve the wall-clock. Confirmed live on the
retrain at 473-478k. So **30B costs ~138 GPU-h** and 100 GPU-h buys ~21.8B.

Duty cycle inside an allocation is ~100% — every gap in the historical
timelines is >15min and sits at an allocation boundary, none inside one.
NEURON.md §9's "GH200 ... 61,750 tok/s per accelerator" is consistent with
this; it just never states the per-GPU-hour figure, which is what budgets need.

### `train.max_tokens` — stop early without perturbing the LR

`target_tokens` was hardwired to `total_tokens(sched)`, i.e. 30B, so a
fixed-grant run had no way to stop short except walltime. Added
`train.max_tokens`, which caps the loop counter and **deliberately does not
touch `self.sched`**: LR stays at peak and every checkpoint remains mid-stable,
directly comparable to de-fair's intermediates. Shortening `stable_tokens`
instead would start the decay early and reintroduce the LR-state confound.
`scripts/test_max_tokens.py` verifies all three properties on the real Trainer
(uncapped run reaches the schedule total; capped stops at the cap; LR at the
capped stop is still 3.000e-03).

Note `scripts/smoke.py` cannot run in the container: it trains the `pa`
parity-aware BPE tokenizer, whose learner is the optional `[tok]` git
dependency, and it writes a pool `stats.json` without the `budget_bytes` key
`pack()` requires. `pa` is analysis-only and no model is trained with it.

### Uploaded: 7 checkpoints, `models.json` 109 -> 116 entries

`scripts/external_bench/upload_de_starved.py` (+ `slurm/23_upload_de_starved.sbatch`)
put the whole series on `jvonrad/xscript-eval`, mirroring de-fair's roster so
every `de-fair-Xb` has a `de-starved-Xb` counterpart:

| friendly | from | vs de-fair |
|---|---|---|
| `de-starved-1b` | `step1092_1001M` | **same step + tokens** |
| `de-starved-2b` | `step2456_2253M` | **same step + tokens** |
| `de-starved-5b` | `step5181_4753M` | **same step + tokens** |
| `de-starved-8b` | `step8451_7753M` | **same step + tokens** |
| `de-starved-12b` | `step12811_11754M` | **same step + tokens** |
| `de-starved-15b` | `step16081_14754M` | 14754M vs 14755M (0.007%) |
| `de-starved-16b` | `final.pt` | content-match of de-fair-12b |

Five of seven are step-for-step identical to de-fair's uploaded checkpoint
names, so the same-script contrast is LR-matched by construction at every
budget rather than by interpolation. Same layout as the other 109 dirs
(`final.pt.part000..004` + `n_parts.txt`; 900MB parts). Verified after upload:
all 7 have 5 parts and the right `n_parts.txt`, sizes match de-fair's byte for
byte, and `de-starved-12b` was re-downloaded, reassembled and confirmed
**SHA256-identical** to the local checkpoint.

Two upload gotchas worth keeping:
* **`models.json` is NOT sorted** — it groups the 15 originals before the
  intermediates, indent 2, no trailing newline. Writing it back with
  `sort_keys=True` turns a 7-entry addition into a 109-entry diff. The
  uploader appends and re-serialises with `json.dumps(mj, indent=2)` only;
  verified as +49 lines / −0.
* `upload_chunked.py`'s 900MB chunking exists because "this login node kills
  any single process after a few minutes" — that is the §6h Linger=no
  teardown, not an HF limit. From a compute node the chunking is unnecessary
  (`fetch_checkpoint` checks for a whole `final.pt` *before* looking for
  parts), but the parts layout is kept for consistency with the other dirs.

### ⛔ `final.pt` here is NOT a cooled final — name the upload `de-starved-16b`

The run stops at 16.1B with **no cooldown**, so its `final.pt` is a mid-stable
checkpoint at peak LR. Every other run's `final.pt` is a **cooled 30B final at
3.0e-4**. Same filename, opposite LR state. If this lands in `models.json` as
bare `de-starved`, anything comparing "de-fair vs de-starved finals" silently
pits a cooled 30B model against an uncooled 16B one — precisely the confound
§6 spends its length warning about. Upload as **`de-starved-16b`**, alongside
`-8b`/`-12b`/`-15b`.

### Budget targets and what the run delivers

de starved/fair fertility is **1.371**, so content-matching needs
`starved = fair x 1.371`. `stable_marks` pins checkpoints to de-fair's exact
budgets (`results/models.json`: 7753M / 11754M / 14755M — note these are *not*
round numbers, because that run mixed world=4 and world=8 and its step->token
map drifted off the grid):

| mark | pairs with |
|---|---|
| 7.753B | de-fair-8b (token-matched) — gives §6 a *second* same-script language |
| 11.754B | de-fair-12b — **the LR-matched transfer cell §6d lists as n/a** |
| 14.755B | de-fair-15b |
| 16.100B | content-match of de-fair-12b (11.754 x 1.371) |

20.2B would content-match de-fair-15b but costs ~93 GPU-h, leaving no reserve
for a re-seed; 16.1B costs ~74.

### Operational findings — three of these contradict the current docs

* **⛔ Compute nodes DO have internet.** README.md says data prep runs on a
  login node "(internet)", implying otherwise. Verified (job 5878501,
  nid010882): a compute node listed all 570 FineWeb2-HQ `deu_Latn` parquet
  files. This matters because —
* **⛔ The login node CANNOT hold a long background job.**
  `loginctl show-user` reports `Linger=no` and `enable-linger` is
  "Access denied", so systemd kills every user process when the last session
  closes — `nohup`/`setsid`/`disown` do not help. Two attempts died this way
  (pool at ~24GB, container mid-build). Anything measured in hours must be a
  batch job: `slurm/22_pool_and_pack_de.sbatch`, `slurm/01_pull_container.sbatch`.
* **⛔ Authenticate to HF even for public datasets.** FineWeb2-HQ is public, so
  the first pool job ran anonymously — and HF 429'd every worker after ~8GB
  despite the documented-safe 8 workers. It had been sustaining 33.9MB/s.
  Authenticated, it ran 72.6GB start to finish with zero 429s. Put the token in
  `$HF_HOME/token` (mode 600); `huggingface_hub` picks it up, so no credential
  reaches a script, the environment, or job stdout.
* **Container build is ~4x faster as a job.** `00_pull_container.sh`'s
  `mksquashfs -processors 4 -mem 1G` throttling exists only to fit the 4GiB /
  500-task interactive cgroup. On a compute node with defaults: **6 min** vs
  25+ min and a death on the login node.
* **Anchor `pgrep -f` patterns** (§6g says this for the eval box; it bites
  here too — `pgrep -f "xscript pool"` matched the very shell that ran it).

### Data had to be rebuilt, and that is not optional

The previous allocation's scratch is unreachable: `/scratch/u6jh` and
`/scratch/u6sg` are both **root-owned, mode 750**, group `brics.<project>`.
Neither account can traverse the other's, and the parent is root-owned so no
`chmod` by either user helps. **There is no cluster-internal path between
allocations** — a transfer would mean an HF round-trip of the packed uint16
shards, which are the least compressible form of the data and larger than the
parquet source they came from.

Rebuilding is also provably equivalent: the manifest is `sorted()`, the holdout
is the first parquet file's first 30MB, and the pool is `files[1:]`, so the
holdout that `eval/holdout_de_bpb` is computed on comes out byte-identical.
`fast_pool.py` interleaves writes across threads, so pool *document order* is
not reproducible — harmless, since `pack` reshuffles within shards and the
seeds changed anyway.

**Measured fertility on real training text, not FLORES:** `unigram_starved` on
FineWeb2-HQ German is **3.291 bytes/token** (FLORES says 3.350 — 1.8% off).
Predicted from a 40k-doc sample as 3.292 before packing; the full 74-shard pack
returned 3.291. Use pool-measured fertility for sizing, FLORES for the
published fertility table.

Result: 72.60GB text / 19.8M docs -> **22.063B tokens in 74 shards**, 5.96B
above the 16.1B target, so no epoching. Do not resume an interrupted
`fast_pool` run without deleting shards past `stats.json`'s `shard_idx`: its
resume does not validate the last shard boundary, and in-flight worker files
get re-fetched, duplicating ~5% of documents.

---

## 8. Open / next steps

- ~~Run the alignment sweep~~ **DONE** — all 26 checkpoints, n=2009, with d'
  and cached embeddings (§6b). Results in
  `/mnt/scratch/xscript_align/results/alignment/`, embeddings alongside in
  `embeddings/`, full report `align_v2.txt`. The v1 (pre-d', pre-embeddings)
  results are archived at `results/alignment_v1_noemb/`.
- **Pick a defensible layer rule, then re-derive every alignment delta** (§6b).
  The fixed `ref` layer gave same-script transfer the wrong sign (retracted —
  bilinguals align deeper, L15-16, than monolinguals, L12-16, so a fixed
  75%-depth probe undersamples the bilingual); peak-layer scoring fixes the
  sign but is itself selection-on-the-metric, so isn't a clean final answer
  either. Options: bootstrap the layer jointly with the queries; integrate
  over the profile; or match on depth-of-emergence. Only the *ordering*
  (cross-script > same-script) is robust across both layer choices so far.
- **Re-derive the CKA table off the peak layer too** — every CKA number in §6b
  is at the fixed `ref` layer and inherits exactly the same bias (that is what
  made fr-starved look broken). Pure CPU on the cached embeddings.
- ~~Resolve the `fr-starved` CKA anomaly~~ **DONE** (§6b): it is the fixed-layer
  artifact above, not a broken checkpoint and not a CKA-vs-retrieval
  disagreement. Peak CKA 0.797 vs fr-fair's 0.840; starved tokenizers delay the
  depth at which alignment emerges (fr-starved L15 vs fr-fair L7).
- **CKA confidence intervals** via Gram-matrix resampling (resample a
  precomputed [n,n] Gram, ~4M ops/replicate — recomputing X^T Y per replicate is
  not tractable). CKA is currently the only quotable-looking number with no
  error bars.
- Confirm the Belebele cloze+PMI 0.34 at larger n (≥400) — is there real reading
  signal or is it noise?
- Optionally run the full standard `external_bench` suite (all examples) for the
  record — but expect MMLU/Belebele to stay at chance; XNLI is the story.
- If the debiased scoring should ship in the portable HF export, re-upload
  `src/xscript/**` (the debiasing is now folded into `bench.py` itself, so the
  export just needs to be refreshed — no separate script to bundle).
- **Re-derive §6's matched-token transfer table on the `*-12b`/`*-23b`
  pairing.** §6d shows that pairing is LR-matched by construction and needs no
  new training, yet §6's table (and §6b's alignment deltas, which use the same
  checkpoint families) still rest on the `*-15b`-vs-cooled-final comparison
  flagged as not quotable. On SIB-200 that confound flipped `ar/starved` from
  +0.004 to −0.110, so re-deriving is not cosmetic. `run_appendix_c5.py` and
  `run_alignment.py` both accept the new run names already.
- **Extend the Global-MMLU cloze probe** (§6e) to de/fr/ar/zh and to
  `en-starved` before the "+6 points of world knowledge" is quoted as a
  general property; it currently rests on `en-fair`/English alone. ~2.5h per
  model-language on a 3xlarge core-pair.
- **Re-run the low-budget trajectory points under `acc_cal`.** §6e covers the
  41 checkpoints carrying the headline tables; the 1B/2B/5B/8B/10B/15B series
  (68 more) is still on the old scoring, so the *curves* in §6d/§6 are mixed
  estimators. Cheap with `--own-langs` + `--only xnli` / `--families sib200`.
- ~~**`ar/starved` needs a second run before its −0.110 is load-bearing.**~~
  **MOOT (§6e):** calibrated it is −0.037, `ar/fair` is −0.040, and the two
  are statistically identical — so the cell was never carrying an interaction.
  What now needs a second run is the *Arabic* effect as a whole, since XNLI
  reverses its sign (+0.018).
  (Original wording, kept for the record: "It is the single cell carrying §6d's
  same-vs-cross-script gap, and its bilingual is worse than the monolingual on
  *both* languages — interference or one weak run cannot be told apart with
  n=1." The both-languages observation does survive calibration: −0.037 on
  Arabic and −0.041 on English.)
- **Fill the ZH-HellaSwag column for the rest of the C.5 roster (§6d).** Only
  the 8 Chinese-relevant checkpoints were scored on `hellaswag_zh`; the other
  18 in §6's per-model table still show `n/a` there. ~10 min/model on one
  core-pair, and `run_extra_bench.py --families hellaswag_zh` merges into the
  existing per-model JSONs rather than overwriting them.
- **SIB-200 is the benchmark to grow, not Belebele.** It is the only
  all-five-language task in the repo that clears its majority baseline. Worth
  running over the token-budget series (`*-1b`…`*-23b`) to get a *curve* rather
  than the single-budget snapshot §6d reports — the same lesson §6's ATLAS-BTS
  anchor-sensitivity taught.
- **LAPE follow-up (§6c): the deactivation experiment.** The paper's causal
  check — zero the identified neurons and measure per-language PPL — has not
  been run. `run_bpb.py` already emits per-sentence NLL, so ablated-vs-intact
  BPB deltas with sentence-bootstrap CIs are straightforward: mask the selected
  gates in `neurons._over_zero_batch`-style forwards for the ~26 headline
  checkpoints. Prediction from §6c: ablating a *trained* language's (top-layer)
  neurons should hurt that language selectively; ablating the starved models'
  layer-0 foreign-script detectors should barely matter for trained langs.
  The raw over-zero npz for all 109 checkpoints are on scratch
  (`/mnt/scratch/xscript_lape/results/lape/`) — threshold sensitivity or new
  statistics need no re-recording; but scratch is instance-local, so copy them
  off (or re-record, ~2.5h) if the box is torn down.
