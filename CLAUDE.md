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
across both files, so cross-references still resolve — **§3, §6, §6b, §8 are
here; §1, §2, §4, §5, §7, §9 are in NEURON.md**.

---

## TL;DR

- **Only XNLI carries downstream signal at this scale.** The apparent
  "Arabic/Chinese are at chance" result was an **evaluation artifact, not a
  training failure** — `xnli_ar`/`xnli_zh` are now debiased automatically inside
  `bench.py`. Global-MMLU is genuinely at chance; Belebele has only a faint
  recoverable signal (§6).
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
- Three times now, an uncontrolled evaluation number turned out to be measuring
  the benchmark rather than the training (XNLI connectives, Belebele letter
  format, the alignment fixed-layer probe). Control before quoting.

---

## 3. The models & sharded checkpoints

`jvonrad/xscript-eval` (private) has **15 models**, friendly names in `models.json`:

- mono: `en-{fair,starved}`, `fr-{fair,starved}`, `ar-{fair,starved}`, `de-fair`
- bilingual: `en-{ar,de,fr,zh}-{fair,starved}`

`fair` = `unigram_destarved` tokenizer, `starved` = `unigram_starved`. Each model
maps to its tokenizer + training languages in `models.json`.

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

The only genuinely missing monolingual is `de` starved, which collapsed
mid-run (confirmed by live monitoring; visible as anchor BPB ~1.72 vs the
destarved twin's ~1.06 — it is in `EXCLUDE_RUNS`) and is being retrained. `fr`
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
de/starved collapsed, so the penalty is really "fr vs {ar,zh}"; there are **no
confidence intervals** (one training run per cell, and unlike the downstream
deltas there is no per-example data to bootstrap); fertility is measured on
FLORES as a proxy for the training pools; and cells are compared at the
largest budget each supports (fr 7.75B, ar/zh ~11.2B), so the penalty mixes
budgets. Landing `de-starved` fixes the first and is the single highest-value
run remaining.

### Only XNLI discriminates; MMLU & Belebele are at chance
From the `--limit 200` matrix over all 15 models (chance: MMLU/Belebele 0.25,
XNLI 0.333):
- **Global-MMLU: 0/23 model×lang cells above chance.** Confirmed real (not an
  artifact): letter, cloze, and cloze+PMI scoring all stay ≈0.21–0.23 for en.
  World-knowledge MCQ is beyond a 1B/30B model.
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
| HellaSwag | en/de/fr/ar | no Chinese translation exists in this lm-eval build |
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

**Full per-model breakdown** (all 26 checkpoints × all 25 `(lang, task)`
cells; bold = model was trained on that language — a zero-shot readout
everywhere else; `Bele`=Belebele, `HS`=HellaSwag, `SC`=XStoryCloze,
`WG`=XWinograd; `-` = task not defined for that language):

| model | tok | trained | EN-XNLI | EN-Bele | EN-ARC | EN-HS | EN-SC | EN-WG | DE-XNLI | DE-Bele | DE-ARC | DE-HS | FR-XNLI | FR-Bele | FR-ARC | FR-HS | FR-WG | AR-XNLI | AR-Bele | AR-ARC | AR-HS | AR-SC | ZH-XNLI | ZH-Bele | ZH-ARC | ZH-SC | ZH-WG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ar-fair | fair | AR | .456 | .284 | .361 | .317 | .547 | .604 | .334 | .260 | .241 | .271 | .361 | .289 | .234 | .283 | .566 | **.469** | **.332** | **.251** | **.382** | **.555** | .337 | .226 | .253 | .469 | .548 |
| ar-fair-15b | fair | AR | .420 | .271 | .338 | .298 | .537 | .572 | .332 | .260 | .241 | .262 | .367 | .269 | .251 | .273 | .506 | **.452** | **.319** | **.224** | **.346** | **.531** | .339 | .222 | .266 | .475 | .484 |
| ar-starved | starved | AR | .425 | .294 | .342 | .292 | .522 | .550 | .334 | .276 | .233 | .274 | .337 | .291 | .244 | .288 | .518 | **.455** | **.301** | **.252** | **.349** | **.551** | .332 | .252 | .267 | .490 | .500 |
| ar-starved-15b | starved | AR | .328 | .268 | .310 | .281 | .520 | .532 | .342 | .252 | .223 | .273 | .335 | .276 | .240 | .287 | .554 | **.376** | **.306** | **.256** | **.318** | **.523** | .338 | .232 | .272 | .473 | .510 |
| de-fair | fair | DE | .483 | .290 | .390 | .353 | .570 | .649 | **.469** | **.333** | **.275** | **.419** | .367 | .298 | .232 | .300 | .506 | .352 | .260 | .257 | .259 | .477 | .335 | .264 | .245 | .483 | .560 |
| de-fair-15b | fair | DE | .427 | .294 | .348 | .312 | .544 | .589 | **.461** | **.327** | **.265** | **.367** | .373 | .282 | .253 | .285 | .542 | .334 | .241 | .251 | .257 | .473 | .329 | .246 | .272 | .475 | .522 |
| en-ar-fair | fair | EN+AR | **.504** | **.324** | **.502** | **.447** | **.623** | **.709** | .339 | .270 | .232 | .275 | .345 | .304 | .260 | .293 | .530 | **.442** | **.334** | **.276** | **.383** | **.555** | .333 | .264 | .245 | .494 | .552 |
| en-ar-starved | starved | EN+AR | **.495** | **.303** | **.473** | **.408** | **.606** | **.662** | .335 | .282 | .234 | .282 | .338 | .299 | .246 | .299 | .518 | **.453** | **.308** | **.250** | **.350** | **.533** | .322 | .262 | .241 | .486 | .562 |
| en-de-fair | fair | EN+DE | **.524** | **.332** | **.503** | **.464** | **.635** | **.728** | **.480** | **.330** | **.296** | **.422** | .383 | .316 | .237 | .304 | .566 | .334 | .240 | .256 | .263 | .478 | .339 | .273 | .243 | .489 | .583 |
| en-de-starved | starved | EN+DE | **.505** | **.317** | **.465** | **.411** | **.617** | **.683** | **.462** | **.314** | **.275** | **.378** | .340 | .294 | .240 | .305 | .530 | .333 | .257 | .234 | .278 | .477 | .322 | .261 | .252 | .492 | .558 |
| en-fair | fair | EN | **.525** | **.338** | **.532** | **.501** | **.657** | **.751** | .341 | .289 | .222 | .284 | .377 | .294 | .246 | .296 | .446 | .337 | .244 | .259 | .261 | .473 | .341 | .292 | .244 | .500 | .579 |
| en-fair-15b | fair | EN | **.484** | **.300** | **.473** | **.417** | **.619** | **.676** | .337 | .280 | .232 | .273 | .338 | .310 | .248 | .288 | .518 | .331 | .236 | .259 | .264 | .465 | .330 | .269 | .241 | .477 | .589 |
| en-fr-fair | fair | EN+FR | **.502** | **.320** | **.497** | **.463** | **.632** | **.722** | .361 | .277 | .230 | .283 | **.474** | **.339** | **.283** | **.447** | **.699** | .333 | .238 | .251 | .268 | .471 | .359 | .291 | .248 | .500 | .540 |
| en-fr-starved | starved | EN+FR | **.500** | **.307** | **.481** | **.431** | **.625** | **.687** | .341 | .278 | .204 | .284 | **.479** | **.322** | **.280** | **.424** | **.639** | .335 | .279 | .240 | .280 | .465 | .333 | .280 | .242 | .507 | .577 |
| en-starved | starved | EN | **.495** | **.311** | **.531** | **.472** | **.637** | **.728** | .349 | .274 | .220 | .285 | .336 | .289 | .229 | .302 | .542 | .332 | .273 | .240 | .278 | .469 | .328 | .251 | .244 | .496 | .593 |
| en-starved-15b | starved | EN | **.507** | **.297** | **.454** | **.389** | **.608** | **.675** | .337 | .280 | .224 | .278 | .331 | .304 | .231 | .298 | .518 | .344 | .266 | .235 | .278 | .469 | .333 | .259 | .259 | .482 | .536 |
| en-zh-fair | fair | EN+ZH | **.511** | **.322** | **.513** | **.456** | **.624** | **.725** | .337 | .270 | .246 | .274 | .354 | .303 | .258 | .286 | .470 | .336 | .249 | .264 | .262 | .472 | **.420** | **.290** | **.305** | **.570** | **.700** |
| en-zh-fair-23b | fair | EN+ZH | **.486** | **.310** | **.479** | **.392** | **.610** | **.691** | .341 | .272 | .240 | .267 | .339 | .301 | .239 | .287 | .566 | .334 | .233 | .263 | .256 | .465 | **.416** | **.302** | **.294** | **.559** | **.710** |
| en-zh-starved | starved | EN+ZH | **.480** | **.307** | **.477** | **.427** | **.621** | **.722** | .335 | .279 | .217 | .284 | .345 | .288 | .238 | .296 | .542 | .333 | .264 | .236 | .277 | .465 | **.407** | **.296** | **.285** | **.551** | **.708** |
| en-zh-starved-23b | starved | EN+ZH | **.476** | **.277** | **.448** | **.369** | **.598** | **.667** | .331 | .281 | .205 | .277 | .337 | .307 | .245 | .294 | .494 | .339 | .257 | .255 | .275 | .471 | **.398** | **.286** | **.270** | **.549** | **.687** |
| fr-fair | fair | FR | .456 | .293 | .378 | .337 | .570 | .610 | .347 | .274 | .238 | .269 | **.467** | **.342** | **.294** | **.435** | **.627** | .333 | .267 | .260 | .262 | .474 | .332 | .266 | .250 | .495 | .569 |
| fr-fair-15b | fair | FR | .457 | .281 | .349 | .304 | .533 | .583 | .340 | .271 | .243 | .263 | **.471** | **.312** | **.259** | **.379** | **.627** | .333 | .246 | .252 | .265 | .469 | .340 | .250 | .265 | .475 | .538 |
| fr-starved | starved | FR | .443 | .289 | .371 | .320 | .547 | .582 | .340 | .284 | .238 | .280 | **.466** | **.320** | **.281** | **.409** | **.602** | .342 | .270 | .252 | .279 | .471 | .333 | .259 | .249 | .495 | .528 |
| fr-starved-15b | starved | FR | .398 | .266 | .344 | .297 | .510 | .551 | .339 | .293 | .222 | .280 | **.443** | **.316** | **.257** | **.358** | **.590** | .333 | .259 | .253 | .272 | .467 | .339 | .253 | .258 | .474 | .520 |
| zh-fair-12b | fair | ZH | .420 | .274 | .332 | .291 | .526 | .584 | .330 | .257 | .241 | .262 | .350 | .263 | .251 | .267 | .482 | .331 | .242 | .263 | .252 | .467 | **.398** | **.317** | **.285** | **.554** | **.671** |
| zh-starved-12b | starved | ZH | .421 | .278 | .317 | .285 | .510 | .545 | .338 | .276 | .220 | .274 | .324 | .283 | .235 | .283 | .506 | .331 | .229 | .261 | .258 | .461 | **.374** | **.277** | **.283** | **.541** | **.641** |

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
