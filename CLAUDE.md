# CLAUDE.md

Operational manual for **evaluating the XScript-Pretraining checkpoints on AWS
Trainium (Neuron)**. The models live on the private HF repo `jvonrad/xscript-eval`
(15 checkpoints: 5 mono + bilingual, ~1B params, 30B tokens each). This file is
the context another Claude Code session needs to reproduce the setup, run the
benchmarks fast on a multi-core `trn2.48xlarge`, and interpret the numbers.

See [README.md](README.md) for the experiment design (cross-script pretraining
penalty × tokenizer starvation). This file is about **running the evals**.

---

## TL;DR

- The downstream harness (`src/xscript/eval/bench.py` + lm-eval) was CUDA/CPU-only.
  It has been adapted to run on **Neuron/XLA** with a fixed-shape scoring path
  (`--device xla`). CPU==XLA verified exact (`xnli_en=0.40` on both).
- **Two silent Neuron bugs** were found and worked around (see §4). If you touch
  the scoring code, read that section first — they fail *silently* (wrong numbers,
  not crashes).
- **Headline science:** of the three benchmarks, only **XNLI** carries signal at
  this scale. The apparent "Arabic/Chinese are at chance" result was an
  **evaluation artifact, not a training failure** — `xnli_ar`/`xnli_zh` are now
  debiased automatically inside `bench.py` (no separate script needed; see §6).
  Global-MMLU is genuinely at chance; Belebele has only a faint recoverable
  signal. See §6.

---

## 1. Hardware / environment

Verified on `trn2.3xlarge`, Ubuntu 26.04, kernel `7.0.0-1006-aws`.

- `trn2.3xlarge`: 1 Neuron device, 4 cores, 96 GB, `logical-neuroncore-config 2`
  → **2 logical cores** (pin with `NEURON_RT_VISIBLE_CORES=0-1` / `2-3`).
- `trn2.48xlarge`: 16 devices × 4 cores → **32 logical cores** (`0-1`,`2-3`,…,`62-63`).
  This is the box to use for fast/large-sample runs (§5).

### Setup (once per fresh instance)

```bash
bash setup_trainium.sh          # copied here from ../Lost-in-Mistranslation; idempotent
```

It installs the Neuron driver (DKMS — patches the kernel-7.0
`mm_get_unmapped_area` signature change), compat libs, and a Python-3.11
`~/neuron_venv` with `torch-neuronx`.

**Known gotcha:** the script ends by `source`-ing the new venv under `set -u`,
which trips on an unbound `LD_LIBRARY_PATH` and exits non-zero **after the driver
and venv are already built** but **before the `uv pip install`**. If that
happens, the driver/venv are fine — just finish the install manually:

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; export PATH="$HOME/.local/bin:$PATH"
source ~/neuron_venv/bin/activate      # sets PJRT_DEVICE=NEURON, adds neuron-ls to PATH
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url=https://pip.repos.neuron.amazonaws.com \
  torch-neuronx neuronx-cc transformers datasets sentence-transformers accelerate
```

### Activating in later shells

Always prefix with the `LD_LIBRARY_PATH` guard (the activate script appends to it
under `set -u` assumptions):

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; export PATH="$HOME/.local/bin:$PATH"
source ~/neuron_venv/bin/activate
```

Sanity check the device:
```bash
python -c "import torch, torch_neuronx, torch_xla.core.xla_model as xm; \
d=xm.xla_device(); print((torch.ones(2,2,device=d)+1).sum().item())"   # -> 8.0, 'Compiler status PASS'
```
The `libfabric.so.1 / libnccom-net.so` warning at startup is the multi-node
collectives plugin and is **irrelevant** for single-device inference — ignore it.

---

## 2. Dependency pinning (CRITICAL — do not skip)

Installing `torch-neuronx` pulls in **`datasets 5.x`, `huggingface_hub 1.x`,
`transformers 5.x`**, which are far newer than `lm_eval 0.4.12` and **break it**:
hub 1.x's strict HF-URI validator rejects lm-eval's legacy `dataset_path: xnli`
with `HfUriError: ... must be 'namespace/name'`. Pin the eval stack back to the
0.4.12 era (this does **not** touch `torch==2.9.1`):

```bash
uv pip install "lm_eval==0.4.12" sentencepiece \
  "huggingface_hub==0.26.5" "datasets==3.2.0" "transformers==4.47.1" numpy tqdm
```

These pins are also recorded in [scripts/external_bench/requirements.txt](scripts/external_bench/requirements.txt).
Verified working set: `torch 2.9.1`, `torch-xla 2.9.0`, `torch-neuronx 2.9.0.2`,
`datasets 3.2.0`, `huggingface_hub 0.26.5`, `transformers 4.47.1`, `lm_eval 0.4.12`.

`export HF_TOKEN=hf_...` — the repo is **private**; nothing downloads without it.

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

## 4. Neuron/XLA scoring — the adaptation and the silent traps

`bench.py` wraps our Transformer into lm-eval. lm-eval hands it variable-length
requests; the original code scored them with **dynamic per-batch tensor shapes**,
which is catastrophic on Neuron (recompiles constantly / silent corruption). The
adaptation (`XScriptLM._score_active_xla`, `_loglikelihood_tokens`) pads every
batch in a task to **one fixed `[batch_size, fixed_width]` shape**, so each task
compiles a single graph. The graph is **weight-independent**, so it compiles once
on the first model and is cached for all 15. `--device xla` selects this path;
CPU/CUDA paths are unchanged.

**Three Neuron bugs on this `torch-xla 2.9` / Neuron build** — the first two give
*wrong numbers, not errors*; the third is a hard compile failure. Guard all
three if you extend the scoring:

1. **`torch.gather` over the vocab dim silently returns ZEROS.** Do not use it to
   pick target-token logprobs. Instead select via one-hot multiply and score as
   `logit − logsumexp` (verified fp32-exact vs CPU). See `_score_active_xla`.
2. **`F.one_hot(idx, V)` trips `NRT_EXEC_OOB`** if `idx` was clamped on-device
   (the `-100` pad targets). **Clamp on the host** before `.to(device)`.
   Likewise, **build input tensors on the host and `.to(device)` once** — per-row
   in-place scatter on an XLA tensor also trips `NRT_EXEC_OOB`.
3. **An odd `fixed_width` reliably fails compilation**: `NCC-5266:
   non-trivial dst dims must have even step for non-FP32 transpose`, on a
   `Matmult` op inside the model's forward pass. Confirmed deterministic, not
   a race — reproduced solo, in isolation, twice (e.g. debiased XNLI-zh's
   `fixed_width=85`, odd, fails every time; ar's `fixed_width=88`, even,
   never does). `_loglikelihood_tokens` now rounds `fixed_width` up to the
   next even number unconditionally (the extra column is inert padding,
   scored the same as any other pad position) — this was the actual cause of
   the "compile race" originally suspected when two `run_appendix_c5.py`
   processes crashed simultaneously on the same odd-width graph; isolating
   them onto separate devices didn't fix it, only the even-width rounding did.

Other notes:
- Belebele's long passages compile fine at `--batch_size 8` (peak < the 24 GB
  per-graph HBM ceiling). Keep `--batch_size ≤ 8`.
- `run_benchmarks.py` prefers the **local repo `src/`** over the bundled HF export
  when run from inside this repo, so local patches to `bench.py` take effect. If
  you want the fixes in the portable export, re-upload `src/xscript/**` to the HF
  repo.
- Never `kill` a process mid-compile — a truncated entry in
  `/var/tmp/neuron-compile-cache` is loaded as garbage later. Recover with
  `rm -rf /var/tmp/neuron-compile-cache`.

---

## 5. Running the evals

Workdir holds downloads + results; keep it on a big disk. On `trn2.48xlarge` the
**root volume is small (~7 GB)** — mount an instance-store NVMe and point
`HF_HOME`, `TMPDIR`, `UV_CACHE_DIR`, and `NEURON_CC_FLAGS=--cache_dir=...` at it.
(On `trn2.3xlarge` root was 190 GB — check `df -h /` first.)

```bash
export HF_TOKEN=hf_...
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; source ~/neuron_venv/bin/activate
cd scripts/external_bench
WORK=/home/ubuntu/xscript_bench      # or an NVMe path on 48xlarge

# quick sanity matrix over all 15 (≈45 min single-core on 3xlarge):
python run_benchmarks.py --repo jvonrad/xscript-eval --device xla \
  --limit 200 --batch-size 8 --workdir $WORK
```

`xnli_ar`/`xnli_zh` in the output are already debiased (corrected connectives +
standard scoring for ar, PMI scoring for zh, see §6) — `bench.py`'s `run()`
routes those two tasks through `_xnli_debiased()` instead of lm-eval's task
registry automatically, for every `run_benchmarks.py` call. No separate script
or flag needed. `scripts/external_bench/run_xnli_debiased.py` still exists as a
standalone diagnostic that reports **both** `standard` and `pmi` per language
(useful for re-checking which method wins), but is no longer required for
normal runs.

Results: `run_benchmarks.py` → `$WORK/results/bench/<run>_final.json` +
`summary.json` (each per-run JSON has an `"xnli_debiased": {"ar": "standard",
"zh": "pmi"}` field recording which languages were debiased).

### Scaling to 16× TRN (`trn2.48xlarge`, 32 logical cores)

15 models ≤ 32 cores, so **run every model fully in parallel**, one per logical
core-pair. **Warm the compile cache first** so the parallel jobs all hit cache and
don't race on first-compile writes:

```bash
# 1) warm: compile every task-graph shape once, sequentially. One en+partner
#    model per language family covers all graphs (mono reuse the same shapes).
for m in en-de-fair en-ar-fair en-fr-fair en-zh-fair; do
  NEURON_RT_VISIBLE_CORES=0-1 python run_benchmarks.py --repo jvonrad/xscript-eval \
    --runs $m --limit 8 --device xla --batch-size 8 --workdir $WORK
done

# 2) fan out: one process per model, pinned to its own logical core-pair.
models=(ar-fair ar-starved de-fair fr-fair fr-starved en-fair en-starved \
        en-ar-fair en-ar-starved en-de-fair en-de-starved en-fr-fair \
        en-fr-starved en-zh-fair en-zh-starved)
core=0
for m in "${models[@]}"; do
  NEURON_RT_VISIBLE_CORES=$core-$((core+1)) setsid nohup \
    python run_benchmarks.py --repo jvonrad/xscript-eval --runs $m \
      --device xla --batch-size 8 --workdir $WORK \
      > $WORK/$m.log 2>&1 < /dev/null &
  core=$((core+2))
done
wait   # all 15 finish in ~the time of the single slowest model
```

**`wait` on `setsid nohup ... &` children is unreliable** — when the fan-out
loop itself runs inside another backgrounded/detached shell, `wait` can return
immediately while the 15 jobs are still running (observed in practice: `wait`
returned in seconds, but `ps aux | grep run_benchmarks` showed all 15 still
active minutes later). Don't trust `wait` finishing as proof the fleet is done
— poll for it instead: `until ! pgrep -f "run_benchmarks.py --repo jvonrad"; do sleep 15; done`,
or check that all 15 `results/bench/<run>_final.json` files exist.

**`summary.json` is not safe for concurrent writers.** Every parallel process
writes the *same* `$WORK/results/summary.json`, so with 15 running at once only
the last one to finish survives in it — don't trust that file after a fan-out
run. The per-run `results/bench/<run>_final.json` files are each written by
their own process and are safe; aggregate from those instead, e.g.:
```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('$WORK/results/bench/*_final.json')):
    d = json.load(open(f))
    print(f.split('/')[-1].removesuffix('_final.json'), '->', d['scores'])
"
```

Same fan-out pattern still works for the standalone `run_xnli_debiased.py`
diagnostic (drop `--limit` for the full validation set, or raise it for larger
MMLU/Belebele samples) — but for normal runs `run_benchmarks.py` alone is
enough now that debiasing is automatic. `neuron-ls` shows which PID owns which
core; `neuron-top` is the live util/mem monitor.

**Bigger sample size:** XNLI validation is 2490/lang (already the default full
run). For Global-MMLU-Lite / Belebele use the full test splits (drop `--limit`).

---

## 6. Scientific findings

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

| lang | root cause | fix | result (full val, n=2490) |
|------|-----------|-----|-----|
| Arabic | lm-eval connectives **mistranslated** (`رقم`="number" for contradiction, `لذا` for neutral) | correct to `لا` / `أيضا`, standard scoring | 0.335 → **0.44–0.47** |
| Chinese | surface-form competition (connectives fine) | **PMI** (prior-normalized) scoring | 0.335 → **0.41–0.42** |

After the fix, **all 23 XNLI model×lang cells are above chance (+7.8 to +20.2σ)**.
Mean corrected accuracy by language: **en 0.503, de 0.471, fr 0.470, ar 0.455,
zh 0.414**.

**Thesis implication:** with a correct evaluation, the cross-script languages
(AR, ZH) learn XNLI *comparably* to the same-script ones (EN/DE/FR) at 30B tokens
— i.e. the apparent cross-script downstream penalty in the raw numbers was largely
a **measurement artifact**, consistent with the repo's argument about ATLAS's
penalty. (Intrinsic **BPB→BTS** remains the primary discriminator; downstream
`acc` can't carry the cross-script question because MMLU/Belebele are at chance and
XNLI needs the debiasing above.)

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
CLAUDE.md §6 already suspected from an earlier ad-hoc probe ("Belebele: at
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
noted in §5 — aggregate from the per-run files.

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
- **Fair vs. starved still shows no clear, consistent effect on the transfer
  delta itself** — ar's two tokenizers are within noise of each other
  (+0.024 vs +0.023), fr similarly (+0.039 vs +0.036); zh is the only pair
  where fair/starved cross the significance threshold differently
  (fair not significant, starved is), but the point estimates themselves
  (+0.011 vs +0.015) are close enough that this may just reflect zh/fair
  sitting nearer the boundary, not a real tokenizer effect.

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

```bash
python run_alignment.py --repo jvonrad/xscript-eval --device xla --workdir $WORK
python analyze_alignment.py $WORK/results/alignment/

# or fan out over whatever cores are free, safe alongside a running trainer:
export HF_TOKEN=hf_...
bash run_alignment_fanout.sh /mnt/scratch/xscript_align
```

~100 s/model for all 5 languages × 10 pairs × 17 layers (dev+devtest, n=2009),
so the full 26 fan out over logical core-pairs exactly like §5. Per-run
`results/alignment/<model>.json` + `.md`; no shared summary file.

**Resource profile** (measured, `ar-fair`, one core-pair, unbounded threads):

| phase | time | where |
|---|---|---|
| tokenize | 0.1 s | host |
| `lexical_baseline()` | 0.9 s | host (scipy sparse) |
| **embedding forward** | **83.2 s** | **Neuron — 84% of total** |
| retrieval / CKA / centering | 14.4 s | host numpy |

So the sweep **is accelerator-bound** and does need cores; only
`analyze_alignment.py` is pure-CPU (stdlib, no device at all — safe to run any
time). The host phase is numpy-multithreaded and will grab all cores by
default: when fanning out alongside a training job, bound it
(`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` ≈ 8, so
jobs × threads ≲ `nproc`) or the analysis phase starves the trainer's
dataloaders even though the Neuron cores are disjoint. Bounding threads
lengthens the host phase roughly proportionally (~14 s → ~50 s at 8 threads),
which is usually the right trade.

Check occupancy before launching — training jobs pin themselves via
`NEURON_RT_VISIBLE_CORES`, whose entries are **physical** core ids 0-63 (see
the `neuron-core-pinning-torchrun-vs-xmp` note; each device = 4 consecutive
ids):
```bash
neuron-ls | grep -E "^\| [0-9]+ "            # device -> PID
tr '\0' '\n' < /proc/<training-pid>/environ | grep NEURON_RT_VISIBLE_CORES
```

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

### Resolution: d' de-saturates, and the sign flips by script

The sweep was rerun with **`d' = (matched - mean_nonmatched) / std_nonmatched`**
per query (unbounded, scale-free, per-example so the existing paired bootstrap
applies). It resolves cleanly where top-1 could not. Delta vs the **matched-token
partner-language monolingual** — the meaningful control, since it already knows
the partner language — all CIs excluding 0:

| partner | script | tok | bilingual d' | partner-mono d' | Δ [95% CI] |
|---|---|---|---|---|---|
| de | same | fair | 6.50 | 7.13 | **-0.63 [-0.69, -0.57]** |
| fr | same | fair | 6.65 | 7.67 | **-1.02 [-1.09, -0.95]** |
| fr | same | starved | 7.66 | 7.95 | **-0.30 [-0.40, -0.19]** |
| ar | cross | fair | 6.65 | 5.89 | **+0.76 [+0.70, +0.81]** |
| ar | cross | starved | 6.58 | 5.05 | **+1.52 [+1.46, +1.58]** |
| zh | cross | fair | 6.32 | 5.83 | **+0.49 [+0.43, +0.54]** |
| zh | cross | starved | 5.60 | 6.08 | **-0.48 [-0.57, -0.40]** |

⚠️ **RETRACTED: the negative same-script deltas above are a LAYER ARTIFACT.**
Scoring each model at its own peak layer instead of the fixed `ref` layer flips
every negative sign positive:

| partner | tok | Δ @ ref (L12) | Δ @ peak |
|---|---|---|---|
| de | fair | **-0.63** | **+0.66** (bi L16 vs mono L15) |
| fr | fair | **-1.02** | **+0.43** (bi L16 vs mono L14) |
| fr | starved | **-0.30** | **+0.53** (bi L15 vs mono L14) |
| ar | fair | +0.76 | +1.04 |
| ar | starved | +1.52 | +2.23 |
| zh | fair | +0.49 | +2.34 |
| zh | starved | **-0.48** | **+2.33** |

Cause: **bilinguals develop cross-lingual alignment DEEPER in the network than
monolinguals do** (bilingual peaks cluster at L15-16, monolingual at L12-16), so
a fixed 75%-depth probe systematically undersamples the bilingual and
manufactures a negative delta. `REF_LAYER_FRAC` was chosen to avoid the
selection bias of an argmax layer, which is a real concern — but it silently
assumed alignment emerges at comparable depth across models, and it does not.

**What survives:** cross-script deltas (ar +1.04/+2.23, zh +2.34/+2.33) are
larger than same-script (de +0.66, fr +0.43/+0.53) under *both* layer choices.
That ordering is the robust finding — it is still the reverse of §6's downstream
C.5 ordering, where same-script transfer was larger. **What does not survive:**
any claim that same-script alignment transfer is negative or absent.

**Neither layer rule is clean.** Peak-layer is selection-on-the-metric (inflates,
and inflates more for noisy profiles); fixed-layer is biased whenever peak depth
differs systematically, which it does here. Report both, or bootstrap the layer
choice, before quoting a number. The per-layer profile is the honest object;
`load_embeddings()` regenerates it on CPU in seconds.

**Caveat on the "same-script vs cross-script" summary row:** it averages *both*
controls, and the EN-only control is catastrophically bad at Arabic (d' 1.45),
which inflates the cross-script mean exactly as it did under top-1. Read the
per-row **partner-mono** column above, never that summary.

### Cached embeddings — do metric work from these, not from a rerun

`run_alignment.py --emb-dir` (used by the fan-out) persists the pooled per-layer
embeddings: `(n_layers+1, 2009, 2048)` fp32 per language, one `.npz` per model,
**34 GB for all 26** at `/mnt/scratch/xscript_align/embeddings/`. Verified to
reproduce the in-run top-1, d' and per-example hit lists **bit-for-bit** (hence
fp32, not fp16). Load with `alignment.load_embeddings(emb_dir, run_name)`.

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

## 7. Files (what changed vs the training-cluster export)

- `setup_trainium.sh` — copied from ../Lost-in-Mistranslation; Neuron env setup.
- `scripts/external_bench/requirements.txt` — pinned HF stack (§2).
- `scripts/external_bench/run_benchmarks.py` — `--device xla`; prefer local `src/`.
- `src/xscript/eval/bench.py` — fixed-shape XLA scoring path + the three Neuron
  workarounds (§4, including the odd-`fixed_width` NCC-5266 fix); `xnli_ar`/
  `xnli_zh` debiasing folded in as first-class task routing (`XNLI_CONNECTIVES`,
  `XNLI_DEBIAS_METHOD`, `_xnli_debiased()`, wired into `run()`) — automatic for
  every caller, including `run_benchmarks.py`. CPU/CUDA paths unchanged.
- `scripts/external_bench/run_xnli_debiased.py` — standalone diagnostic
  reporting both `standard` and `pmi` per language (§6); superseded for normal
  runs by the automatic debiasing in `bench.py` above.
- `scripts/external_bench/run_appendix_c5.py` — **new**; replicates Messmer et
  al. 2025 Appendix C.5 across en/de/fr/ar/zh and all checkpoints (§6). Now
  runs with `log_samples=True` and per-task batch sizing (`--batch-size` for
  Belebele, `--batch-size-short` for everything else) so it also produces the
  per-example correctness data `bootstrap_transfer.py` needs.
- `src/xscript/eval/c5_tasks/belebele_cloze/` — **new**; custom cloze-format
  Belebele task configs (lm-eval's registered task uses A/B/C/D letters
  instead, which isn't what that paper's methodology calls for).
- `scripts/external_bench/bootstrap_transfer.py` — **new**; paired bootstrap
  95% CIs on the same-script vs. cross-script transfer deltas from matched-
  token checkpoints (§6's "Same-script vs. cross-script transfer" section).
  Pure stdlib, ~1 min for the full model set.
- `src/xscript/eval/alignment.py` — **rewritten** (§9): Neuron/XLA fixed-shape
  embedding path, all-pairs instead of EN-anchored-only, every model on every
  language, `centered` variant, CKA, per-example hit lists, and the model-free
  `lexical_baseline()` TF-IDF floor. CPU/CUDA paths and the `xscript
  eval-align` CLI signature unchanged.
- `scripts/external_bench/run_alignment.py` — **rewritten**; `--device xla`,
  all 26 models, `--langs`, `--split both`, prefers local `src/`, per-run
  output only (no shared summary to clobber under fan-out).
- `scripts/external_bench/analyze_alignment.py` — **new**; aggregates the
  per-run alignment JSONs into baseline-relative tables with paired bootstrap
  CIs, mirroring `bootstrap_transfer.py`'s estimator. Pure stdlib.
- `scripts/external_bench/run_alignment_fanout.sh` — **new**; fans the sweep
  out over **free** core-pairs only, discovered from `neuron-ls` at runtime
  (devices with PID `NA`), so a concurrent training job is never scheduled
  over. Warms the compile cache per tokenizer first, bounds host threads,
  skips models that already have a result JSON (resumable), and polls for
  completion rather than trusting `wait` (§5).

Neuron writes stray `*PostSPMDPassesExecutionDuration.txt` files into the cwd —
gitignore them.

---

## 8. Open / next steps

- ~~Run the alignment sweep~~ **DONE** — all 26 checkpoints, n=2009, with d'
  and cached embeddings (§6b). Results in
  `/mnt/scratch/xscript_align/results/alignment/`, embeddings alongside in
  `embeddings/`, full report `align_v2.txt`. The v1 (pre-d', pre-embeddings)
  results are archived at `results/alignment_v1_noemb/`.
- **Pick a defensible layer rule, then re-derive every alignment delta** (§6b).
  This is now the blocking issue: the fixed `ref` layer and the per-model peak
  layer give *opposite signs* for same-script transfer (-1.02 vs +0.43 on
  fr/fair), because bilinguals align deeper (L15-16) than monolinguals (L12-16).
  Neither rule is clean — fixed-layer is biased when peak depth differs, peak is
  selection-on-the-metric. Options: bootstrap the layer jointly with the queries;
  integrate over the profile; or match on depth-of-emergence. Only the
  *ordering* (cross-script > same-script) is currently robust; no signed
  same-script claim is.
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
