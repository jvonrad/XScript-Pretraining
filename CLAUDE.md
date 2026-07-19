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
  **evaluation artifact, not a training failure** — fixed in
  `scripts/external_bench/run_xnli_debiased.py`. Global-MMLU is genuinely at
  chance; Belebele has only a faint recoverable signal. See §6.

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

**Two silent Neuron bugs on this `torch-xla 2.9` / Neuron build** — both give
*wrong numbers, not errors*, so guard them if you extend the scoring:

1. **`torch.gather` over the vocab dim silently returns ZEROS.** Do not use it to
   pick target-token logprobs. Instead select via one-hot multiply and score as
   `logit − logsumexp` (verified fp32-exact vs CPU). See `_score_active_xla`.
2. **`F.one_hot(idx, V)` trips `NRT_EXEC_OOB`** if `idx` was clamped on-device
   (the `-100` pad targets). **Clamp on the host** before `.to(device)`.
   Likewise, **build input tensors on the host and `.to(device)` once** — per-row
   in-place scatter on an XLA tensor also trips `NRT_EXEC_OOB`.

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

# the debiased XNLI runner (the metric that actually discriminates, see §6):
python run_xnli_debiased.py --repo jvonrad/xscript-eval --device xla \
  --batch-size 8 --workdir $WORK        # full validation set (n=2490/lang)
```

Results: `run_benchmarks.py` → `$WORK/results/bench/<run>_final.json` +
`summary.json`; `run_xnli_debiased.py` → `$WORK/results/xnli_debiased.json`.

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

Same pattern works for `run_xnli_debiased.py` (drop `--limit` for the full
validation set, or raise it for larger MMLU/Belebele samples). `neuron-ls` shows
which PID owns which core; `neuron-top` is the live util/mem monitor.

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

`run_xnli_debiased.py` implements the fix: corrected connectives + both `standard`
and `pmi` scoring, reported per language. Use `standard` for en/de/fr/ar and `pmi`
for zh.

---

## 7. Files (what changed vs the training-cluster export)

- `setup_trainium.sh` — copied from ../Lost-in-Mistranslation; Neuron env setup.
- `scripts/external_bench/requirements.txt` — pinned HF stack (§2).
- `scripts/external_bench/run_benchmarks.py` — `--device xla`; prefer local `src/`.
- `src/xscript/eval/bench.py` — fixed-shape XLA scoring path + the two Neuron
  workarounds (§4). CPU/CUDA paths unchanged.
- `scripts/external_bench/run_xnli_debiased.py` — **new**; debiased XNLI (§6).

Neuron writes stray `*PostSPMDPassesExecutionDuration.txt` files into the cwd —
gitignore them.

---

## 8. Open / next steps

- Confirm the Belebele cloze+PMI 0.34 at larger n (≥400) — is there real reading
  signal or is it noise?
- Optionally run the full standard `external_bench` suite (all examples) for the
  record — but expect MMLU/Belebele to stay at chance; XNLI is the story.
- If the debiased scoring should ship in the portable HF export, re-upload
  `src/xscript/**` (and consider folding PMI/corrected-connectives into `bench.py`
  as first-class task variants rather than a separate script).
