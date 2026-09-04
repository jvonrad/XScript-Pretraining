# NEURON.md

Everything about **running this repo on AWS Trainium (Neuron)**. Two stacks
(see the ⛔ note in §1): the public **XLA stack** used for every evaluation
(§1–§7: setup, pins, the fixed-shape scoring adaptation and its silent traps,
eval fan-out), and the **TorchNeuron Native beta** used for training (§10: a
recipe that reaches 47k tok/s/chip, 46% MFU — above GH200 — plus the ledger
of everything that does not work). §9 is the superseded XLA training port.

[CLAUDE.md](CLAUDE.md) is the companion file and holds the **scientific
results**. Section numbers are preserved across both files so existing
cross-references still resolve:

| section | file |
|---|---|
| §1 Hardware / environment | **NEURON.md** |
| §2 Dependency pinning | **NEURON.md** |
| §3 The models & sharded checkpoints | CLAUDE.md |
| §4 Neuron/XLA scoring + silent traps | **NEURON.md** |
| §5 Running the evals | **NEURON.md** |
| §6 / §6b Scientific findings | CLAUDE.md |
| §7 Files (vs the training-cluster export) | **NEURON.md** |
| §8 Open / next steps | CLAUDE.md |
| §9 Training on Neuron — XLA hand-port (SUPERSEDED by §10, kept for the XLA eval stack) | **NEURON.md** |
| §10 TorchNeuron Native: the training recipe (47k tok/s/chip, 46% MFU), setup, and what does NOT work | **NEURON.md** |

---

## 1. Hardware / environment

⛔ **Two incompatible stacks exist in this project — check which one a box
has before doing anything.** §1–§5 describe the **public XLA stack**
(`~/neuron_venv`, `torch 2.9.1 + torch-xla`, public driver) that every eval
in CLAUDE.md was run on. §10 describes the **TorchNeuron Native private
beta** (Docker container `neuron-native`, torch 2.12, beta driver) that all
training throughput work was done on. They cannot coexist on one instance:
the beta driver replaces the public one. **The current box
(`i-02a7b8a80604f7640`) runs the beta stack and has NO `~/neuron_venv`**; to
run evals, set up a fresh instance per §1 or port `bench.py` to the native
device (untested).

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

⛔ **Also pin `neuronx-cc==2.25.3371.0`** (2026-09-01). A fresh install pulls
`neuronx-cc 2.27.5334.0`, which **cannot compile this repo's 1B Transformer
forward at all** — every graph dies with the internal error `[NCC_ISMP902]
Simplifier error: is_subset(): incompatible function arguments`, including the
exact `[24, 64]` scoring graphs that ran for months on the eval box. Not our
code: a bare `model(x, y)` with no scoring logic reproduces it. Downgrading to
2.25.3371.0 (and clearing `/var/tmp/neuron-compile-cache`) fixes it with no
other change:

```bash
uv pip install "neuronx-cc==2.25.3371.0" "nki==0.5.0+28631259367.ga768afa6" \
  --extra-index-url=https://pip.repos.neuron.amazonaws.com --index-strategy unsafe-best-match
rm -rf /var/tmp/neuron-compile-cache /var/tmp/nki-intermediate-cache
```

Three traps found while landing the downgrade, all with misleading errors:
the **`nki` package must be downgraded in step** (2.27's `nki 0.6.0` writes
kernel binaries 2.25 rejects with `NCC_INLA001 ... incompatible with the
compiler's expected version: 1.0.0` — but only on graphs that hit an NKI
kernel, so small graphs work and a bigger one fails much later);
`/var/tmp/nki-intermediate-cache` must be cleared or stale binaries keep
failing after the package is fixed; and **failed compilations are cached** in
`/var/tmp/neuron-compile-cache` and replayed verbatim on retry — delete the
failing `MODULE_*` directory (or the whole cache) after fixing the cause.
Also note `NCC_EBVF030` ("Instructions ... exceeds the typical limit"): a
~100-row fwd+bwd graph over this 1B model is past the compiler's 5M
instruction limit — chunk the batch and reuse a smaller compiled shape
instead (values are data; shapes are graphs).

`export HF_TOKEN=hf_...` — the repo is **private**; nothing downloads without it.

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
standard scoring for ar, PMI scoring for zh, see CLAUDE.md §6) — `bench.py`'s `run()`
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

## 6b-ops. Running the alignment sweep

(The findings this produces live in CLAUDE.md §6b; this is just how to run it.)

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
  reporting both `standard` and `pmi` per language (CLAUDE.md §6); superseded for normal
  runs by the automatic debiasing in `bench.py` above.
- Both `run_appendix_c5.py` and `run_extra_bench.py` gained **`--own-langs`**
  (score each model only on its own training languages — ~74% less work; the
  bilingual-vs-monolingual comparisons only ever pair trained-language scores,
  so nothing they need is dropped, but the zero-shot cross-lingual readout is
  given up), **`--only` / `--families`** task-family filters that MERGE into the
  existing per-model JSON rather than replacing it, and raw-sidecar output.
  `run_appendix_c5.py` also gained `--xnli-raw-all-langs`, which routes
  xnli_{en,de,fr} through `_xnli_debiased()` so their per-candidate
  loglikelihoods are stored too. That path is **not** bit-identical to
  lm-eval's registered task — lm-eval prepends its `target_delimiter` (a space)
  to each candidate and this path does not, worth ~0.005 acc from near-ties
  flipping (measured: de-fair-12b xnli_en .4702 vs .4655). Compare against the
  previous sweep, do not assume equality.
- `scripts/external_bench/run_appendix_c5.py` — **new**; replicates Messmer et
  al. 2025 Appendix C.5 across en/de/fr/ar/zh and all checkpoints (CLAUDE.md §6). Now
  runs with `log_samples=True` and per-task batch sizing (`--batch-size` for
  Belebele, `--batch-size-short` for everything else) so it also produces the
  per-example correctness data `bootstrap_transfer.py` needs.
- `src/xscript/eval/c5_tasks/belebele_cloze/` — **new**; custom cloze-format
  Belebele task configs (lm-eval's registered task uses A/B/C/D letters
  instead, which isn't what that paper's methodology calls for).
- `scripts/external_bench/bootstrap_transfer.py` — **new**; paired bootstrap
  95% CIs on the same-script vs. cross-script transfer deltas from matched-
  token checkpoints (CLAUDE.md §6's "Same-script vs. cross-script transfer" section).
  Also emits a direct `Delta_fair - Delta_starved` paired bootstrap
  (`diff_of_diffs()`) — the tokenizer has no consistent effect on
  Δ-on-partner-lang but a significant one on Δ-on-English for every partner
  (CLAUDE.md §6). Pure stdlib, ~1 min for the full model set.
- `scripts/external_bench/run_bpb.py` — **new**; per-language BPB on FLORES+
  through `bench.py`'s fixed-shape Neuron scorer (verified to ~1e-8 against
  `eval/bpb.py`'s `score_texts` via `--verify-cpu`), emitting **per-sentence**
  NLL+bytes so BTS can be bootstrapped. Caches the repo file listing
  (`_repo_files.json`, `--refresh-listing` after new uploads) because N
  parallel `list_repo_files` calls 429 before any checkpoint transfers.
- `scripts/external_bench/bts_from_wandb.py` — **new**; BTS from the W&B
  training curves, restricted to the stable-LR window so the mono/bilingual
  comparison is cooldown-clean by construction (CLAUDE.md §6). Reports both the repo's
  BTS and ATLAS's iso-loss token-efficiency BTS (different nulls: 0 vs 0.5),
  with anchor-sensitivity, from `flores` or `holdout`. Needs `wandb login`.
  Zero accelerator compute — prefer this over evaluating checkpoints.
- `scripts/external_bench/bts_content_matched.py` — **new**; the repo's own
  BTS at a fixed per-language budget, reported both token-matched and
  **content-matched** via per-(tokenizer, language) fertility, so the
  fair-vs-starved gap is not confounded by the starved tokenizer having
  processed less text at equal token counts (CLAUDE.md §6). Auto-selects the largest
  budget both conditions' curves support.
- `scripts/external_bench/bts_matched.py` — **new**; matched-token BTS +
  penalty + interaction with paired bootstrap CIs, replacing the
  unreproducible, self-contradicting `results/bts/*` (CLAUDE.md §6). Enforces
  like-for-like partner sets across tokenizer conditions and flags budgets
  whose mono/bilingual are not LR-state matched.
- `src/xscript/eval/alignment.py` — **rewritten** (§9 below): Neuron/XLA fixed-shape
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

- `src/xscript/eval/neurons.py` — **new**; LAPE language-specific-neuron
  recording (arXiv 2402.16438, CLAUDE.md §6c): per-FFN-neuron over-zero counts
  per language, fixed-shape XLA path (same conventions as `alignment.py` —
  host-built tensors, float-cast before reduction to dodge the bool-`.sum()`
  bug, even widths). Verified XLA↔CPU parity. Identification is a faithful
  port of the paper's `identify.py` (percentile thresholds preserved).
- `scripts/external_bench/run_lape.py` — **new**; sweeps LAPE recording over
  checkpoints. Same fetch/reassemble machinery as `run_alignment.py`, but also
  deletes the HF-cache part-blobs after each model (109 checkpoints ≈ 450 GB
  would not fit scratch). Resumable; per-run npz only, no shared summary.
  Full 109-model sweep ran ~2.5h on two `trn2.3xlarge` core-pairs
  (~2.5 min/model: ~40s download + ~100s forward for 5 langs × 2009 sents).
- `scripts/external_bench/analyze_lape.py` — **new**; pure-CPU identification
  + report over the npz outputs (counts per model×lang, layer profiles,
  token-budget trajectories, Jaccard consolidation). Outputs committed to
  `results/lape/`.

- `scripts/external_bench/run_extra_bench.py` — **new**; the five-language
  benchmarks added on top of the C.5 suite (CLAUDE.md §6d): the missing
  `hellaswag_zh`, SIB-200, SIB-200 with English labels (control), and
  Taxi-1500. Same fetch/reassemble machinery as `run_appendix_c5.py`. Two
  differences that matter operationally: results are **merged** into
  `$WORK/results/extra_bench/<run>_final.json` per (lang, task) rather than
  overwritten, so two invocations with different `--families` on different
  core-pairs can build up one file for the same model without clobbering each
  other (and `--overwrite` re-scores at task granularity, not model
  granularity); and `correct` has an extra metric level,
  `{lang: {task: {metric: [0/1, …]}}}`, because these tasks report
  acc/acc_norm/acc_mutual_info and they disagree.
- `scripts/external_bench/verify_mubench.py` — **new**; the pool-identity gate.
  For each MuBench task it checks item counts, `_id` alignment across all five
  languages, gold-index validity, and — where an English original exists — what
  fraction of the English questions appear in it. **Run this before any new
  multilingual benchmark**: it is the check that would have caught
  arc_easy(en)-vs-ARC-Challenge(others) immediately, and it caught a silent
  zero-row failure on ar/zh when the option-marker regex was English-only.
- `src/xscript/eval/c5_tasks/mubench/` — **new**; MuBench (`aialt/MuBench`),
  12 benchmarks x 61 languages aligned by `_id`, converted to localized CLOZE.
  Two traps, both silent: use **`local_template`** not `en_template`, and note
  the **option markers are localized** (`الخيار A:`, `选项 A:`, `选项A:`, vs a
  bare `A:` for ARC-Easy/BMLAMA) — an English-only regex yields ZERO ar/zh rows
  without erroring. The parser requires a Latin capital immediately before a
  colon (ASCII or full-width) with any prefix, plus the letters running
  A, B, C, ... consecutively; that ordering check is what makes the loose
  prefix safe.
- `src/xscript/eval/c5_tasks/native_mmlu/` — **new**; ArabicMMLU (n=14455,
  ragged 2-5 options so its uniform-random null is **0.293**, not 0.25) and
  CMMLU (n=11582), localized cloze. These are the instruments that show ar/zh
  knowledge translated MMLU cannot (CLAUDE.md §6e).
- `src/xscript/eval/c5_tasks/mmmlu_probe/` — **new**; MMMLU (`openai/MMMLU`,
  professional translation) + English MMLU as cloze, item-aligned.
- `src/xscript/eval/c5_tasks/gmmlu_probe/` — **new**; full-size Global-MMLU
  (n=14042, vs the n=400 `-Lite` build lm-eval defaults to) in letter and cloze
  form, plus `gmmlu_cloze_encue_*`, the English-cue control that measured the
  scaffolding effect at exactly 0.000.
- ⚠️ `src/xscript/eval/c5_tasks/arc_pmi/` and `.../mubench_arc/` are **dead**
  — superseded by `mubench/`'s `mub_arceasy_*`. They are still registered in
  `run_extra_bench.py`'s `FAMILIES`; delete both plus their entries.
- `scripts/external_bench/watch_sweep.py` — **new**; one-pass health check for
  a running sweep, safe to poll (prints ONLY anomalies, so silence == healthy).
  Three tiers: liveness (workers alive via an **anchored** `--worker-pattern`,
  stall detection, FAILED lines, disk), integrity (every stored raw block must
  still reconstruct lm-eval's own hit lists, expected doc counts, no NaN/inf),
  and plausibility — prediction-entropy collapse, a cell not beating its own
  empirical null, an accuracy outside the benchmark's possible range. That last
  tier is the one that would have caught the format artifacts in CLAUDE.md
  6/6d/6e while a sweep was still running rather than months later.
  **The `--worker-pattern` must be anchored (`^bash /path/...`)**: an
  unanchored `pgrep -f`/`pkill -f` pattern matches the shell that invokes it,
  which silently breaks the liveness check and, with `pkill`, kills the caller
  mid-script. That mistake cost this session three shells and ~3.7h of wall
  clock (two waiters each matched the other's pattern and deadlocked, so a
  chained sweep never launched).
- `src/xscript/eval/c5_tasks/xcsqa/` — **new**; X-CSQA (`INK-USC/xcsr`), the
  CommonsenseQA half of XCSR, as localized 0-shot CLOZE for all five
  languages (CLAUDE.md §6i). Three traps, all silent: the upstream **`test`
  split is BLIND** (every `answerKey` is `""`, so it scores as garbage rather
  than erroring — only `validation` is labelled); **six German rows carry an
  empty option string**, and an empty continuation scores `ll = 0.0`, beating
  every real negative candidate under `acc`, so `_drop_ids` removes those ids
  from **all five** languages to keep the item set aligned (n=994 x 5); and
  the **options are permuted per language**, so `_rows` sorts by `id` or
  per-example hit lists do not line up across languages. `_rows` is the
  pure-python builder `verify_xcsqa.py` checks, kept separate from `_load` so
  the gate never constructs a `datasets.Dataset`. Ships `xcsqa_enopt_*`
  (English options — the ~14-point label-language control) and
  `xcsqa_encue_*` (English cue) alongside, per §6e's rule that any new
  multilingual task ships with a control pair.
- `scripts/external_bench/verify_xcsqa.py` — **new**; the X-CSQA pool-identity
  gate, pure CPU. Checks blind-split detection, equal item counts, identical
  `id` sets AND doc order, candidate/gold validity, the expected per-language
  permutation, control self-consistency, and (`--csqa`) that all 1000 English
  questions are real CommonsenseQA. Run before any X-CSQA sweep.
- `scripts/external_bench/analyze_xcsqa.py` — **new**; pure-stdlib estimator
  comparison, capability tables and paired-bootstrap transfer deltas off the
  raw sidecars. Notably it **refuses to name a single estimator when §6g's
  criteria disagree** (they do here) and prints the disagreement instead.
- `scripts/external_bench/run_xcsqa_sweep.sh` — **new**; per-core-pair X-CSQA
  worker. Resumable via `.xcsqa.done` markers, deletes each 4GB checkpoint
  after its model, batch 16 (X-CSQA prompts are the shortest in the suite).
  A separate file rather than a parameterisation of `run_finals_mubench.sh`,
  per §6g's warning about editing a script bash is mid-execution.
- `scripts/external_bench/make_sweep_lists.py` — **new**; splits models.json
  into balanced per-core-pair lists. ⚠️ It balances on own-language *cell*
  count, which §6i measured to be the wrong cost model for a single-family
  sweep — that is download-bound, so equal *model* counts matter more.
- `scripts/external_bench/verify_bucketing.py` — **new**; re-scores a task with
  the current code and diffs the per-candidate loglikelihoods against those
  stored by an earlier sweep. Used to certify the length-bucketed batching in
  `bench.py` (below): max delta 4.8e-06 over 200 docs x 7 choices, i.e. fp32
  noise from differing pad widths, not a score change.
- `src/xscript/eval/bench.py` — `_loglikelihood_tokens` now **length-buckets**
  on XLA: requests are sorted by prepared length and each batch is padded only
  to its own longest member, rounded up to `WIDTH_LADDER` so the number of
  compiled graphs stays bounded (~11) instead of growing with the data. The old
  behaviour padded every batch to the task-wide maximum, which is
  pathological on a skewed task -- Global-MMLU's cloze prompts have median 42
  tokens and max 1051, so one model-language cost ~2h. Only batch composition
  changes: padding is strictly right of each sequence so causal attention
  cannot reach a scored position, and results are restored to caller order.
  Note the speedup is smaller than the median/max ratio suggests (~2.4x, not
  ~10x) because sorting ascending puts all the long sequences in the final
  batches, which then dominate.
- `src/xscript/eval/rawscores.py` — **new**; the raw-loglikelihood contract
  behind CLAUDE.md §6e. `extract_raw()` pulls each document's per-candidate
  loglikelihoods out of lm-eval's `log_samples` records; `score_variants()`
  re-derives acc / acc_norm / acc_tokennorm / acc_pmi / **acc_cal** /
  acc_cal_loo / acc_cal_pmi from them on CPU. `check_reproduces()` asserts the
  stored scores reconstruct lm-eval's own `acc`/`acc_norm`/`acc_mutual_info`
  bit-for-bit — nothing derived is trusted until that holds. Two conventions
  must match lm-eval exactly or `acc_norm` silently drifts: it normalizes by
  **character** count (`completion_len`, not `byte_length`), of the raw choice
  **without** the `target_delimiter` that `arguments` carries. `acc_cal` is
  gated on the `SHARED_CHOICE_TASKS` allowlist — it is only meaningful where
  the choice *index* means the same thing in every document, and is withheld
  for HellaSwag/ARC/Belebele/XStoryCloze/XWinograd, where it would silently
  return a position-bias correction instead.
- `scripts/external_bench/analyze_raw_scores.py` — **new**; pure-stdlib reports
  off the raw sidecars: every estimator, a degeneracy report (prediction
  entropy + per-gold-class recall + the **empirical null**
  `sum_c P(pred c) P(gold c)`, which is the honest chance level rather than
  1/k), trajectory monotonicity over a token-budget series, and transfer
  deltas with the same paired bootstrap as `bootstrap_transfer.py`.
- `scripts/external_bench/backfill_calibrated.py` — **new**; injects the
  derived hit lists into the per-model result JSONs so `analyze_extra_bench.py`
  and `bootstrap_transfer.py` work unchanged apart from selecting `acc_cal`.
  Idempotent; refuses a cell whose raw scores fail `check_reproduces`.
- `scripts/external_bench/test_rawscores.py` — **new**; offline correctness
  test, no checkpoint and no accelerator. Runs lm-eval over the real task
  configs with a stub scorer and asserts (1) re-derivation matches lm-eval
  exactly, (2) `acc_cal` is invariant to an injected per-candidate offset while
  `acc`/`acc_norm` both move, (3) the degeneracy check fires on a deliberately
  collapsed predictor. Run it after touching `rawscores.py`.
- `src/xscript/eval/c5_tasks/gmmlu_probe/` — **new**; Global-MMLU at FULL size
  (`CohereForAI/Global-MMLU`, n=14042 for en — lm-eval defaults to the n=400
  `-Lite` build) in both answer formats, `gmmlu_letter_*` and `gmmlu_cloze_*`.
  This is what overturned §6's "world-knowledge MCQ is beyond a 1B/30B model"
  (CLAUDE.md §6e). **Operational note:** its prompts embed all four options, so
  `fixed_width` is the task-wide max (1088 tokens for en, against a median of
  77) and every batch pays it. `--batch-size 16` fails with
  `NCC_EOOM002` (28.45GB > 24GB) because `_score_active_xla`'s one-hot
  materializes a `[batch, width, 65536]` float tensor — use **`--batch-size 8`**
  or lower, and run the two formats as separate `--families` on separate
  core-pairs.
- `scripts/external_bench/analyze_extra_bench.py` — **new**; pure-stdlib
  aggregation of the above: the three-metric comparison with both baselines
  (chance *and* majority-class), a **constant-prediction degeneracy check**
  (a cell whose hit vector is exactly `gold == c` scored the majority rate
  while learning nothing), per-model × per-language tables, the SIB-200
  label-language control, and transfer deltas with the same paired bootstrap
  as `bootstrap_transfer.py` (LR-mismatched pairs flagged `BAD`).
- `src/xscript/eval/c5_tasks/hellaswag_zh/` — **new**; Chinese HellaSwag.
  lm-eval 0.4.12 ships 31 okapi HellaSwag languages but no `zh`, even though
  `alexandrainst/m_hellaswag` has `data/zh/val.jsonl` — because that file does
  not load: 4 of its 37,064 `endings` are `{"zh":…,"en":…}` dicts instead of
  strings, so pyarrow's schema inference rejects the whole split
  (`ArrowInvalid: Column(/endings/[]) changed from string to object in row
  153`). `utils.build_dataset` reads the jsonl with the stdlib and takes the
  `"zh"` member, recovering all 9266 docs; preprocessing is a verbatim copy of
  upstream's so the column stays comparable with de/fr/ar.
- `src/xscript/eval/c5_tasks/sib200/` — **new**; SIB-200 topic classification
  (7 topics over FLORES-200 sentences) for all five languages, plus the
  `sib200_enlab_*` English-label control. `utils.build_dataset` merges
  `Davlan/sib200`'s train+dev+test TSVs into one 1004-doc `test` split sorted
  by FLORES sentence id — nothing is finetuned on SIB-200 here, so all splits
  are unseen data, and the sort makes the doc order identical across
  languages.
- `src/xscript/eval/c5_tasks/taxi1500/` — **new**; Taxi-1500 (6 topics over
  Bible verses), all five languages. Only the *English* labels are
  distributable; the per-language text is joined onto them by PBC verse id
  from the openly-licensed **Taxi1500-c v3.0** corpus zip (786 MB, downloaded
  and pruned to five editions on first use — pre-warm with
  `python src/xscript/eval/c5_tasks/taxi1500/utils.py` before a fan-out).
  Editions are pinned in `EDITIONS`, chosen for full coverage of all 1077
  labelled verses and the most modern register available.

- `scripts/neuron_native/` — **new** (§10); the TorchNeuron-Native (beta)
  training replica `bench_train.py` (every recipe flag and every dead-end
  flag), the bitwise-identical rope rewrite `rope_fix.py`, the roofline /
  component / shape timers (`comp_bench.py`, `block_bench.py`, `block_mb.py`,
  `block_shapes.py`, `subblock_bench.py`, `chunk_bench.py`, `glue_bench.py`,
  `head_bench*.py`, `accum_bench.py`, `emb_bench.py`, `transpose_bench.py`),
  the diagnostics (`graph_diag.py`, `profile_step.py`), the custom NKI
  kernels and fused modules that were measured and rejected (`nki_rope.py`,
  `nki_rmsnorm.py`, `nki_ce.py`, `fast_modules.py`, `fused_shadow.py`,
  `flat_zero.py`, `sdpa_patch.py`), and `results/*.json` for every
  configuration run. Runs only inside the `neuron-native` container.

Neuron writes stray `*PostSPMDPassesExecutionDuration.txt` files into the cwd —
gitignore them.

---

## 9. Training on Neuron — the XLA hand-port (`train_neuron.py`) — SUPERSEDED

**Status: superseded by §10.** The TorchNeuron Native beta runs the same
trainer replica at **47k tok/s/chip vs this hand-port's 23.5k**, with
`model.py` needing only two numerically-identical edits. Use §10 for any new
training. This section is kept because (a) the XLA stack is what the public
SDK ships and the eval pipeline runs on, and (b) several of its findings are
reusable.

### What was wrong with the original reasoning (still worth knowing)

NeuronX Distributed **Training** (NxDT) was correctly rejected: it ships its
own Llama with split-half RoPE and its own init, needs NeMo, and would have
made new checkpoints architecturally different from the 15 Isambard ones.
But that reasoning was wrongly generalised to everything AWS ships.
**NxD (`neuronx-distributed`) is a model-agnostic parallelism library** (ZeRO-1
+ TP + activation checkpointing around an *unmodified* `nn.Module`, no
NeMo, no Megatron code); NxDT is a framework that replaces your model. NxD
runs our model at world=32 (`scripts/neuron_train/nxd_test.py`); the
hand-rolled ZeRO in `train_neuron.py` fails at world=16 (`NRT_INVALID ...
invalid send/recv targets`) — a defect in our code, not the platform.

```python
nxd_config = neuronx_distributed_config(
    tensor_parallel_size=1,
    optimizer_config={"zero_one_enabled": True, "grad_clipping": True, "max_grad_norm": 1.0},
    activation_checkpoint_config=Block)
model = initialize_parallel_model(nxd_config, model_fn)
optim = initialize_parallel_optimizer(nxd_config, torch.optim.AdamW, model.parameters(), lr=...)
```
Open issue with NxD: the 1B model at world=32 fails to *compile* (27.6 GB
scratchpad) because NxD fuses fwd+bwd+optimizer into one graph — memory
tuning (mb=1, `mark_step`s, `fused_ce_chunk`), not a wall.

### Facts that still apply on the XLA stack

* **Memory is the optimizer and the vocab, not activations.** At mb=1 the
  persistent bucket is ~19.5 GB (fp32 master 4.06 + fp32 grads 4.06 + AdamW
  m 4.06 + v 4.06 + bf16 autocast copies 2.03) + ~8 GB compiler scratchpad
  against a 24 GB core; checkpointed activations are ~0.08 GB. Levers in
  `cfg["train"]`: ZeRO-1 (`zero`), `bf16_params`, `fused_ce_chunk: N`
  (chunked lm_head+CE with recompute — unlocks mb>=8), per-Block
  checkpointing (unconditional). The full-vocab CE alone needs ~23 GB of
  scratchpad at mb=8 (`NCC_EOOM002`).
* **Silent numeric bugs on this XLA build (no crash, wrong numbers):** a bool
  `.sum()` returns `-1` (negates and un-normalises the loss — cast to float
  before reducing); `torch.gather` over the vocab dim returns zeros (§4).
  Verify any new reduction against CPU.
* **Launch with `xmp.spawn`, not torchrun, to pin a core subset**: torchrun's
  `Initializer` overwrites `NEURON_RT_VISIBLE_CORES` with `LOCAL_RANK`.
  `xmp.spawn` honours it. Per concurrent job set distinct `MASTER_PORT`
  (default 12355 collides -> silent hang at init) and
  `NEURON_RT_ROOT_COMM_ID` (else `CCOM WARN Timeout waiting for RX`). Core
  ids are the physical ids `neuron-ls` shows (0-63, 4 per device).
* **Concurrency hazards:** launch staggered (parallel first-compiles of one
  graph deadlock); a mid-compile kill leaves `MODULE_*/*.lock` files that
  block every later job (`find $CACHE -name '*.lock' -delete`); xmp workers
  are `python3 -c from multiprocessing.spawn...`, so kill by PID from
  `neuron-ls --show-all-procs`; cores take 5-10 s to free after SIGKILL;
  failures present as hangs — a step-timeout watchdog is mandatory.
* **Compiler flags:** `--optlevel=2/3` ballooned the scratchpad 8 -> 23 GB
  (OOM where O1 fit); `--auto-cast=none` also grew it. Keep `-O1`.
* **Measured:** world=8 (2 chips) = 47k tok/s = **23,500 tok/s/chip, MFU
  23.1% (HFU 29.8%)** at 6.54 GF/token (+1.91 GF ckpt recompute) vs 667
  TF/s; GH200 = 61,750 tok/s/GPU, MFU 40.9%.
* **`warm_start`** (`cfg["warm_start"]["from"]`) loads model weights only and
  sets the token/step counter so the WSD schedule continues; applied before
  the optimizer is built. ZeRO saves per-rank `<tag>.optim.rank<r>.pt`
  sidecars; resume needs the same world size; model-only saves stay
  byte-compatible with the CUDA trainer.
* Production launcher (outside the repo, holds a wandb key):
  `/home/ubuntu/xscript_prod/` — `prod_train.py`, `run_prod.sh`,
  `orchestrate_zh15.sh` (staggered launch + restart-on-hang), `STATUS.md`.
  Proven config: fp32 params, mb=2, ZeRO-1, full CE, `-O1`, world=8.

---

## 10. TorchNeuron Native (private beta): the training recipe, setup, and what does NOT work

Native-PyTorch backend for Trainium (`torch.device("neuron")`, eager +
`torch.compile`, standard `torch.distributed`), set up 2026-08-25 on this
`trn2.3xlarge` and tuned over ~50 measured experiments (2026-08-25/26).

**Bottom line: the recipe below trains the de-starved 1B model at
47,079 tok/s per chip, MFU 46.2% (47,225 / 46.3% with an optional runtime
flag) — 2.0x the XLA hand-port (§9) and above GH200's 40.9% MFU. It
generalises unchanged to 1.7B (44.9% vs 44.8% per Block). The search is
converged: hardware counters show the device 98% busy with the remaining
gap inside the compiler's kernels, and every framework-level lever has been
A/B'd (§10f).** Every number here is in `scripts/neuron_native/results/`.

### 10a. Quick start (3 commands, ~6 min; ~2.5 min more on a cold NEFF cache)

On a fresh instance do §10h first (bare box -> container with the repo
mounted); on this box the container already exists.

```bash
sudo docker exec -it neuron-native bash                 # repo is mounted at /repo
cd /repo/scripts/neuron_native
export TORCH_NEURONX_SEGMENT_ALLOCATOR_CONFIG='kMaxSplitSizeBytes=268435456,kMaxNonSplitRoundingBytes=67108864'
NEURON_RT_NUM_CORES=4 torchrun --nproc_per_node 4 --rdzv_backend c10d \
    --rdzv_endpoint localhost:29500 bench_train.py \
    --micro-bsz 2 --compile --ce-compile --tail-compile --bf16-shadow-hooks --zero \
    --steps 4 --warmup 1
```

Expected `[bench] RESULT {... "tok_per_s": ~47100, "mfu_vs_chip": ~0.46}`,
and — because the synthetic data is seeded — a **deterministic loss trace:
step 1 = 11.4956, step 2 = 11.3108**. A different loss means different
numerics, not noise; a different tok/s beyond +-0.5% means a contaminated
measurement (see the benchmarking trap in §10g). Add
`NEURON_RT_DISABLE_EXECUTION_BARRIER=1` for +0.3% (optional; it removes a
runtime safety barrier — validate on the real trainer first). `--accum 8`
is a ~2-min smoke run (~41k; per-step overhead amortised over 8 micros).

### 10b. The recipe — every item is load-bearing and was measured

| # | what | why / what happens without it |
|---|---|---|
| 1 | **`torch.compile(backend="neuron", dynamic=False)` on each Block, plus a compiled embedding graph and a compiled (final-norm + lm_head + CE) tail graph** (`--compile --ce-compile --tail-compile`) | eager runs at 200-500 tok/s/core. Granularity is settled in both directions (ms per 2048 tokens at mb=2): whole Block **9.14**, norm+attn \| norm+ffn 9.26, attn \| ffn with eager norms 12.93, 4-way split 12.88, group-of-4 slightly worse; and the WHOLE forward+loss as one graph is fine at mb=1 (37.1k) but pathologically slow at mb=2 (11k) even with memory to spare. Compiled tail vs eager norm/embedding: +1.1% |
| 2 | **Rewrite `model.py`'s `_apply_rope` view-based** (`rope_fix.apply_rope_viewbased`; `x.view(...,D//2,2)` + `stack`) — verified **bitwise identical** fwd and bwd | the strided even/odd slice-assign lowers to kernels ~60x slower than bandwidth-bound: 13.5 of every Block's 22 ms. This single edit was worth +76% (20.4k -> 35.9k) |
| 3 | **`.contiguous()` after each `transpose(1,2)` in `Attention.forward`** | required for `torch.compile` to lower the Block at all (`failed to legalize 'torch.constant.int'` on the non-contiguous view). Free: the compiler elides the copies (9.13-9.14 ms with or without) |
| 4 | **CE head as `logsumexp(logits) - gather(logits, target)`** (default; `--ce-fce` restores `F.cross_entropy`) | same value to 1e-6 and same gradient, but the compiler skips materialising the (N, 65536) fp32 log-softmax: 22.8 vs 30.5 ms fwd+bwd. Every other form loses (one-hot, manual max/exp, bf16-kept gather, in-graph chunking, AWS's `nkilib` CE kernel) |
| 5 | **micro-batch 2, no activation checkpointing** | Block ms per 2048 tokens: mb=1 **11.20**, mb=2 **9.11**, mb=4 8.72, mb=8 8.54 — the curve is flat past 2, and mb>=3 OOMs at 24 GiB (5.1 GiB of saved activations per unit of mb; per-Block graphs keep the backward transient at 1.2 GiB vs 4 GiB whole-graph, which is what lets mb=2 fit at all: peak 21.0 GiB, 0 OOM events) |
| 6 | **bf16 shadow copies of the Linear weights, refreshed once per optimizer step; fp32 master params in the optimizer; each parameter's bf16 grad added into the fp32 `.grad` from a post-accumulate-grad hook** (`--bf16-shadow-hooks`) | +4.8% (44.9k -> 47.1k). Removes ~6 GB/micro of autocast weight casts and halves the fresh-grad write. Numerically the same bf16 matmul outputs autocast produces (fp32 weights are cast to the identical bf16 values every forward). Costs 2.8 GiB — the first thing to drop when memory binds (§10d) |
| 7 | **world = 4 = one process per LNC2 logical core (a Trainium2 chip = 8 physical cores = 4 logical x 24 GB); ZeRO-1 via `torch.distributed.optim.ZeroRedundancyOptimizer`; one explicit `all_reduce(AVG)` per parameter per step, no DDP wrapper** | fp32 AdamW needs 17.4 GB/rank unsharded and nothing else fits. `NEURON_RT_VIRTUAL_CORE_SIZE` / `NEURON_LOGICAL_NC_CONFIG=1` do not change the 4 x 24 GB layout |
| 8 | **`TORCH_NEURONX_SEGMENT_ALLOCATOR_CONFIG='kMaxSplitSizeBytes=268435456,kMaxNonSplitRoundingBytes=67108864'`** | the caching allocator fragments and OOMs on iteration 2+ even when iteration 1 fits (largest free chunk 6-46 MB vs the 512 MB `[65536,2048]` tensors); this is the CUDA `max_split_size` analogue |
| 9 | Compiler at `-O1` (torch_neuronx's default); int32 token ids; `dist.init_process_group(backend="neuron")`; device `neuron:{torch_neuronx.current_device()}` | `-O2`/`-O3` are 4x slower whole-graph and neutral per-Block; int64 is auto-cast with a warning every run |

`scripts/neuron_native/` (mounted at `/repo/scripts/neuron_native`):
`bench_train.py` is the trainer replica with every flag above and every
dead-end flag below; `rope_fix.py` the rope rewrite + bitwise test;
`block_mb.py` / `block_shapes.py` / `comp_bench.py` / `block_bench.py` the
roofline and component timers; `profile_step.py` the profiler harness;
`results/*.json` every configuration ever run.

### 10c. Results

| stack (all: same `model.py`, AdamW 0.9/0.95 wd 0.1, bf16 autocast + fp32 master, clip 1.0, 999,424-token steps) | tok/s per accelerator | MFU |
|---|---|---|
| GH200 (CUDA `train.py`) | 61,750 | 40.9% |
| **Trn2 native beta — the recipe** | **47,079 /chip** | **46.2%** |
| … + `NEURON_RT_DISABLE_EXECUTION_BARRIER=1` | 47,225 | 46.3% |
| … without bf16 shadow weights (fp32 weights cast in-graph) | 44,905 | 44.0% |
| … mb=1 instead of 2 | 37,096 | 36.4% |
| Trn2 XLA hand-port (§9) | 23,500 | 23.1% (HFU 29.8%) |
| native, whole-graph compile, `model.py` rope as-is | 20,428 | 20.0% |
| native, first attempt (per-Block + checkpointing + eager chunked CE) | 6,958 | 6.8% |
| native, eager | ~200-500 /core | — |

MFU = tok/s x 6.54 GF/token / 667 TF/s (dense bf16 chip peak); tok/s is
wall-clock over full optimizer steps including all-reduce, clip and ZeRO
step, `torch_neuronx.synchronize()`d. Synthetic data (throughput is
data-independent; the packed shards are on Isambard).

**Why it stops at ~46% — the hardware counters.** `neuron-explorer` (the
replacement for the removed `neuron-profile`) on the compiled Block NEFF:

```bash
find /tmp/neff_cache -name '*.neff' -printf '%s %p\n' | sort -rn | head    # biggest = Block graphs
neuron-explorer capture -n <neff> -s p.ntff        # device must be otherwise idle
neuron-explorer view -s p.ntff -n <neff> --output-format summary-text
```

| counter | value | reading |
|---|---|---|
| `neuroncore_utilization` (`neuron-monitor`, live, all 4 ranks) | **~98%** | the device is saturated — there is no idle time to reclaim |
| `tensor_engine_active_time_percent` | **0.229** | …but the PE array is active only 23% of it |
| `scalar_engine_active_time_percent` | **0.872** | the scalar engine (softmax, silu, norms, casts) is the busy one |
| `transpose_flops / hardware_flops` | **17.7%** | nearly a fifth of PE work is transposes |
| `spill_reload_bytes` | **3.57 GB** | heavy SBUF -> HBM spilling per graph |
| `throttle_avg_util_limit_nc*_percent` | **0.58** | the hardware caps utilisation to 50% for 83% of the time |

Consistent with the roofline (`comp_bench.py`): pure matmuls reach 43% of
peak at 2048^3 but 71% at N=5632 and 90% for the lm_head, and the Block is
45.5% of peak at mb=2 vs only 48.5% at mb=8 (the batch asymptote). The
ceiling is the compiler's instruction mix on **these shapes** — which is
also why the 7B-shaped Block below reaches 54%. Host CPU is 99% idle,
`torch._dynamo.explain` shows one graph with zero breaks per Block, and the
`torch.profiler` trace has no inter-graph gaps: nothing framework-level is
left (the ledger in §10f is the proof).

### 10d. Other model sizes (1.7B) — compute generalises, memory is the constraint

Per compiled Block at mb=2 (`block_shapes.py`; reproduces to +-0.2%):

| shape | dim / ffn / heads (layers for ~1.7B) | TF/s | % of peak |
|---|---|---|---|
| 1B (current) | 2048 / 5632 / 16 (16) | 74.7 | 44.8% |
| **1.7B deep** | 2048 / 5632 / 16 (28) | 74.8 | **44.9%** |
| **1.7B wide** | 2560 / 6912 / 20 (18) | 74.3 | **44.6%** |
| 2.5B | 3072 / 8192 / 24 | 70.8 | 42.5% |
| 7B-ish | 4096 / 11008 / 32 | 90.1 | **54.0%** |

Memory at 1.7B, MEASURED at world=4 (`--memprobe` peak of 24 GiB):
mb=1 no ckpt with shadow — **OOM**; mb=1 + full ckpt with shadow — 20.4 GiB
but still **OOM**; mb=1 + full ckpt, no shadow — **17.5 GiB, runs**. The
persistent floor (fp32 master 6.4 + fp32 grads 6.4 + bf16 shadow 2.8 =
15.6 GiB) does not shrink with world size under ZeRO-1; only Adam does
(3.2 GiB at world=4 -> ~0.2 at world=64, so on the 16-chip box mb=1 fits
without checkpointing). Prefer the **wide** geometry (same MFU, ~20% less
activation memory).

Measured cost of each workaround (1B, isolated, 3 repeats, +-0.4%):
activation checkpointing **-20%** (exactly its +29% FLOPs — it is only
"3x slower" *inside* a whole-graph compile); dropping the shadow weights
-4.8%; mb=2 -> 1 -23%. ZeRO-2 gradient sharding would free 6.4 GiB and is
the way to keep mb=2 without checkpointing (untested on this beta; each
world=64 shard is ~100 MB, well inside what worked).

**Planning numbers for four 1.7B x 100B-token runs on a trn2.48xlarge**
(44.9% x 667 TF/s, 10.13 GF/token, linear scaling): mb=2 no ckpt (needs
ZeRO-2) 29.6k tok/s/chip -> **~9.8 days**; mb=2 + ckpt 22.9k -> ~12.6 days;
mb=1 no ckpt no shadow 21.7k -> ~13.4 days. Run them as four concurrent
4-chip jobs (world=16, grad_accum 30) rather than sequentially on 16 chips
(world=64, grad_accum 7): same chip-hours, 4x better all-reduce
amortisation, failure isolation. ⚠ Multi-chip scaling is UNMEASURED —
spend the first hour of any reservation on it (`bench_train.py` runs
unchanged; that is also where `TORCH_NEURONX_ENABLE_HOST_CC=1` becomes
worth testing).

### 10e. Porting `xscript.train` for real — the diff list

`bench_train.py` already implements all of this; `train.py` needs:

1. `device = torch.device(f"neuron:{torch_neuronx.current_device()}")`,
   `dist.init_process_group(backend="neuron")`, torchrun with
   `NEURON_RT_NUM_CORES=<ranks>`; `torch.autocast("neuron", bfloat16)` works.
2. `model.py`: `_apply_rope` -> `rope_fix.apply_rope_viewbased`; `.contiguous()`
   after the three transposes in `Attention.forward`. Both numerically
   identical, so safe to land for the CUDA trainer too (not done — the bench
   monkeypatches).
3. Compile: `layer.compile(backend="neuron", dynamic=False)` per Block, plus
   the embedding and the (norm + lm_head + lse-CE) tail as two more graphs.
   Keep the loop (backward, all-reduce, clip, step) eager. ~2.5 min TTFI per
   new shape, then the persistent NEFF cache.
4. Shadow weights: a bf16 copy of every Linear weight, refreshed after each
   `optim.step()`; post-accumulate-grad hooks add each bf16 grad into the fp32
   master `.grad` and free it. Masters go to `ZeroRedundancyOptimizer(...,
   AdamW)`; save with `consolidate_state_dict()` on rank 0 — the model
   `state_dict` is unchanged and byte-compatible.
5. `micro_batch_size: 2` for the 1B model (world=4 -> grad_accum 61). For
   1.7B use §10d's table.
6. int32 token ids; the allocator env var; `-O1`; optionally
   `NEURON_RT_DISABLE_EXECUTION_BARRIER=1`.

**Minimal reference implementation** — the whole recipe in ~50 lines, the
same code paths `bench_train.py` runs (`--compile --ce-compile
--tail-compile --bf16-shadow-hooks --zero`), stripped of its dead-end
flags. **Verified** (`ref_test.py` runs this code verbatim with the bench's
seeding): loss trace 11.4956 / 11.3108 / 12.3931 — identical to §10a — at
46,874 tok/s, MFU 46.0%. Launch with `NEURON_RT_NUM_CORES=4 torchrun
--nproc_per_node 4 ...` and the allocator env var from §10a:

```python
# ---- one-time setup (torchrun sets RANK/WORLD_SIZE; NEURON_RT_NUM_CORES=4) ----
import copy, torch, torch.distributed as dist, torch.nn.functional as F, torch_neuronx
from torch.distributed.optim import ZeroRedundancyOptimizer
import xscript.model as M
from rope_fix import apply_rope_viewbased
M._apply_rope = apply_rope_viewbased          # item 2 (item 3 = .contiguous() in Attention.forward)
dist.init_process_group(backend="neuron")
dev = torch.device(f"neuron:{torch_neuronx.current_device()}")
master = M.Transformer(cfg).to(dev)            # fp32; owns the optimizer state
optim = ZeroRedundancyOptimizer(master.parameters(), optimizer_class=torch.optim.AdamW,
                                lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
# bf16 shadow model: Linear weights bf16, everything else SHARES the fp32 storage
shadow = copy.deepcopy(master)
lin = {id(m.weight) for m in shadow.modules() if isinstance(m, torch.nn.Linear)}
pairs = []
for ps, pm in zip(shadow.parameters(), master.parameters()):
    ps.data = pm.data.to(torch.bfloat16) if id(ps) in lin else pm.data
    pairs.append((ps, pm))
def _hook(pm):                                  # bf16 grad -> fp32 master .grad, freed immediately
    def h(ps):
        g = ps.grad
        pm.grad = (g.float() if g.dtype != torch.float32 else g.clone()) if pm.grad is None else pm.grad.add_(g)
        ps.grad = None
    return h
for ps, pm in pairs: ps.register_post_accumulate_grad_hook(_hook(pm))
for blk in shadow.layers: blk.compile(backend="neuron", dynamic=False)      # item 1: per-Block graphs
emb = torch.compile(lambda w, idx: F.embedding(idx, w), backend="neuron", dynamic=False)
def _tail(x, nw, eps, w, t):                    # final RMSNorm + lm_head + lse-gather CE, one graph
    f = x.float(); f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    lg = ((f * nw.float()).to(x.dtype).reshape(-1, x.size(-1)) @ w.t()).float()
    lse = torch.logsumexp(lg, -1); tl = lg.gather(1, t.clamp(min=0).long().unsqueeze(1)).squeeze(1)
    valid = (t != -100).to(lg.dtype); return ((lse - tl) * valid).sum() / valid.sum().clamp(min=1.0)
tail = torch.compile(_tail, backend="neuron", dynamic=False)
def forward_loss(idx, tgt):                     # idx, tgt: int32 [mb, T]
    x = emb(shadow.tok_emb.weight, idx)
    cos, sin = shadow._rope_for(idx.shape[1], idx.device, x.dtype)
    for blk in shadow.layers: x = blk(x, cos, sin)
    return tail(x, shadow.norm.weight, shadow.norm.eps, shadow.lm_head.weight, tgt.reshape(-1))
# ---- one optimizer step (mb=2, grad_accum=61 at world=4 -> 999,424 tokens) ----
optim.zero_grad(set_to_none=True)
for x, y in micro_batches:
    with torch.autocast("neuron", dtype=torch.bfloat16):
        loss = forward_loss(x, y)
    (loss / grad_accum).backward()              # hooks accumulate into master .grad
for pm in master.parameters(): dist.all_reduce(pm.grad, op=dist.ReduceOp.AVG)
torch.nn.utils.clip_grad_norm_(master.parameters(), 1.0)
optim.step()
with torch.no_grad():                           # refresh the bf16 shadows ONCE per step
    for ps, pm in pairs:
        if id(ps) in lin: ps.data.copy_(pm.data)
```

### 10e′. Multi-chip on the trn2.48xlarge — MEASURED (2026-09-04), and the real trainer

`scripts/neuron_native/train_native.py` is the port of 10e wired to the
repo's `MixedStream`, WSD schedule, resumable checkpoints (same keys as the
CUDA/XLA trainers; `ZeroRedundancyOptimizer.consolidate_state_dict` on rank
0) and W&B; `run_native.sh` / `orchestrate_native.sh` launch and supervise
runs from the host via `docker exec`. Four concurrent runs, one per 4 chips
(world=16, mb=2, grad_accum 15, 983,040 tok/step — the GH200 1-node batch).

| config | tok/s per run | per chip | MFU/chip |
|---|---|---|---|
| 1 chip (world=4), box idle | 40.3k | 40.3k | 39.5% |
| 4 chips (world=16), box idle, one job | 133k | 33.3k | 32.7% |
| 4 x (4-chip jobs) concurrent, **unpinned** | 140–160k for ~100 steps, then **decays to 79–80k** | 20k | 19.6% |
| 4 x (4-chip jobs) concurrent, **NUMA-pinned + `OMP_NUM_THREADS=2`** | **155–163k**, flat through 180+ steps | 40k | **38–40%** |

So: the 16-way collective costs ~17% on an idle box, but with four jobs
running the box is *host*-bound unless each job is pinned to the NUMA node
its chips hang off (`neuron-ls` CPU AFFINITY: devices 0–3 and 12–15 →
CPUs 48–95,144–191; devices 4–11 → 0–47,96–143; `taskset -c` in
`run_native.sh`) and host threads are bounded. Unpinned, the scheduler drifts
13 of 16 ranks of every job onto one node and throughput decays over the
first ~150 steps to an identical 80k plateau for every job — a pattern that
looks exactly like thermal throttling and is not. Pinned, four concurrent
jobs each run FASTER than one 4-chip job alone did unpinned. Two traps on
the way: the native compiler rejects `NEURON_CC_FLAGS=--cache_dir`
(`NCC_EARG002`; use `NEURON_COMPILE_CACHE_URL`), and `neuron-ls` omits its
PID column entirely when no process holds a device, so any "is device N
busy" check that reads a positional column silently reports every device
busy on an idle box. The host driver/runtime/collectives (dkms 2.30.2.0,
runtime-lib 2.34.10.0) were already the same versions as the beta debs on
this instance, so the public XLA venv and the native container coexisted
without a driver swap — §1's "cannot coexist" holds only when the versions
differ.

### 10f. What does NOT work — the measured dead-end ledger (don't re-try on this SDK)

Compile / graph structure
* Whole-graph compile at mb=2: 11k tok/s (pathological) — fine only at mb=1.
  Checkpointing *inside* a compiled graph: 3x slower (`--ckpt-blocks`,
  `--full-compile --ckpt`, `--compiled-autograd` all OOM or crawl).
  Sub-Block and group-of-4 graphs: worse (§10b row 1).
* Compiler flags (19 A/B'd on the Block; list them with `strings
  .../driver/commands/CompileCommand*.so | grep -oE '^--[a-z][a-z0-9-]+'`,
  read with `neuronx-cc compile --help-hidden`): all neutral or worse.
  ⛔ `--internal-autotune=1` reads +0.2% on the Block but is **36% slower on
  the real step** — never judge a flag on the micro-bench alone.
  `--internal-tensorizer-opt-level=operator-fusion` +12% time;
  `--fast-math=fp32-cast-all`, `--scheduler=none`,
  `--enable-ccop-compute-overlap`, `--vectorize-strided-dma`,
  `--experimental-multi-level-tensorization` fail to compile.
* Backend env knobs: `NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE=1` 4.5% slower;
  `TORCH_NEURONX_ENABLE_CONCATENATION`, `TORCH_NEURONX_MLIR_ATEN_OPS`,
  `--model-type=transformer` identical; tensorizer backend crashes.
* Runtime knobs (`strings libnrt.so | grep NEURON_RT_`):
  `ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=16` neutral; `XU_COMPUTE_MAX_QUEUED_REQUESTS=64`
  crashes; `DISABLE_EXECUTION_BARRIER=1` is the only winner (+0.3% at full
  accumulation, +3% at accum 8).

Memory / batch
* mb=3 and mb=4 OOM; FFN token-chunking to cut spill (identical math):
  1 chunk 9.16, 2 -> 10.29, 4 -> 11.33 ms. Spilling is the compiler's choice.
* Flat-buffer ZeRO-1 (`flat_zero.py`): same 20.2 GiB peak as
  `ZeroRedundancyOptimizer`, then OOM (two 4 GB contiguous buffers fragment
  the pool). Manual `autograd.grad` + `_foreach_add_` accumulation: OOM
  (holds a full fresh grad set). `bf16_params` (no fp32 master): 3.3x slower
  AND deviates.

Gradient accumulation (a ~9% tax: `AccumulateGrad`'s 147 eager adds = 32 ms
per micro; one `_foreach_add_` 22.6 ms; a flat 4.4 GB `add_` 21.3 ms = the
HBM floor at ~0.58 TB/s per logical core)
* Winner: bf16 shadow + per-parameter hooks (§10b row 6). Batching the hook
  adds (`--hook-batch 8`) is slower (46.0k vs 47.2k). `foreach=True`
  clipping / AdamW: no change.

Collectives
* 4 GB `all_gather_into_tensor` and `all_reduce_coalesced` fail to allocate
  ("NRT model scheduling failed" / "Failed to load model with collectives").
  DDP-style async all-reduce overlapping the last backward (`--overlap-ar`):
  **1.6x slower** — async collectives stall the compiled graphs. The 219
  synchronous per-parameter all-reduces (~2% of a step) are right.

Kernels
* Custom NKI kernels lose to compile-friendly torch for elementwise ops: an
  interleaved-rope kernel (`nki_rope.py`, bitwise-exact, faster standalone)
  is slower in-graph (custom-op boundary blocks fusion); an RMSNorm fwd+bwd
  kernel loses 4.4 vs 0.81 ms, and the ACT engine's `rsqrt` is a ~2^-12
  table. AWS's own `nkilib` cross-entropy kernel (`nki_ce.py`) is exact but
  slower (29.6 vs 22.9 ms); it needs `chunk_size=16384` for fp32
  (`chunk*4B*2 <= 229,376`). NKI 0.6 API notes: `import nki` (not
  `neuronxcc.nki`); no `nl.arange` — slice with `nl.ds`; tile math via
  `nl.multiply/add/subtract` or `nisa.tensor_tensor(dst=...)`; `(1, D)` does
  not partition-broadcast — `nl.broadcast_to`; `nisa.activation` fuses op +
  reduce.
* Attention: SDPA already dispatches to `nkilib`'s flash kernels (fwd
  `attention_cte`, bwd `attention_bwd`, ~30% util). The math decomposition
  (`TORCH_NEURONX_ENABLE_NKI_SDPA=0`) is much worse (16.6 vs 11.2 ms/Block).
  ⛔ **Trn2 is NKI `gen3`**: `mm_out_dtype=bfloat16` (2-byte PSUM) is
  Trn3+ only; `mixed_precision=False` in the backward is a wash and changes
  numerics (`sdpa_patch.py` keeps both for a later SDK). The 8 transposes
  per attention are free inside compiled graphs.
* Fusions and reformulations, all ~zero at the component level
  (`fast_modules.py`, `fused_shadow.py`, `glue_bench.py`, `head_bench*.py`):
  fused `[3D,D]` qkv (even materialised once per step: 1% slower — `split`'s
  backward concatenates), fused `[2FF,D]` w1|w3, `F.rms_norm` and three
  other norm forms, `g*sigmoid(g)*u` / fp32 silu, the NKI
  embedding-backward path (`TORCH_NEURONX_EMBEDDING_BWD_NKI_THRESHOLD=1`,
  4.7 vs 2.8 ms).

**Beta feedback candidates** (per the guide's ask): the strided slice-assign
rope lowering (60x — the biggest single perf bug found); the non-contiguous
MLIR lowering failure; fragmentation OOM at default allocator settings;
whole-graph mb=2 pathology; `-O2`/`-O3` whole-graph regression;
`--internal-autotune` regression; tensorizer crash; 4 GB collectives failing
to allocate; async collectives stalling compiled graphs; 43% PE util on
2048^3 matmuls with 17.7% transpose FLOPs and 3.5 GB spill per graph; ACT
`rsqrt` precision; LNC1 not selectable.

### 10g. Operational lessons (each cost real time)

* ⛔ **Benchmarking trap:** chaining runs of NEW model shapes back-to-back
  contaminates timings — neuronx-cc subprocesses from run N are still
  compiling while run N+1 times its steps, and `--warmup 1` does not absorb
  it. The same 28-layer config read 2.9k, 15.5k and 3.6k tok/s in chains,
  while isolated warm-cache repeats are stable to +-0.4%. **Warm the cache
  for a shape, then re-run it isolated**, and keep the canonical control
  (§10a reproduces 47.2k with an identical loss trace every time).
* Never chain device jobs with `until [ $(ps aux | grep -c '<script>') -eq 0 ]`
  waiters: each waiter's own command line contains the next script's name,
  so queued waiters deadlock (the §6f trap, again). When the Claude Code
  process exits, `docker exec` clients die but container-side loops survive
  as orphans. Do: ONE detached sequential chain
  (`docker exec -d ... 'a; b; c; echo done > marker'`), every job's output
  to a file, poll the marker. Kill by PID from `ps -eo pid,args`, never
  `pkill -f <pattern>` (it matched its own shell).
* `neuron-explorer capture` fails silently ("exited with an error") if the
  device is busy; run it alone. A `torch.profiler` trace with
  `torch_neuronx.profiling.NeuronConfig` works but its chrome categories
  nest, so sum nothing from it — use `key_averages()` and `neuron-explorer`.
* `torch.compile` on `Attention` failed only in composition (every submodule
  compiled alone) — bisect at the composition level, not per op.
* Kill a stalled `torchrun` with `pkill -9 -f bench_train.py; pkill -9 -f
  torchrun` from the HOST (`sudo docker exec neuron-native ...`), then wait
  ~8 s for cores to free.

### 10h. Environment — how to use it, and the fresh-instance setup

**On this box:** `sudo docker exec -it neuron-native bash` (torch 2.12.1,
torch-neuronx 2.12.3, python 3.12; repo at `/repo`). The container is
`--privileged`, restart-policy `unless-stopped`. Host tools (`neuron-ls`,
`neuron-top`, `neuron-monitor`, `neuron-explorer`) live in
`/opt/aws/neuron/bin` (in `~/.bashrc` PATH). Beta artifacts (wheels, runtime
debs, `torch_neuron_eager` source + examples + the torchtitan diffs) are in
`/home/ubuntu/workspace/`.

**Fresh instance (~20 min, mostly the image pull):**

1. Attach an EC2 instance role with `AmazonEC2ContainerRegistryReadOnly`
   (no keys needed; cross-account access to the beta registry worked; the
   registry is us-east-1 regardless of the instance's region).
2. `sudo apt-get update && sudo apt-get install -y docker.io awscli dkms
   build-essential "linux-headers-$(uname -r)" libhwloc-dev && sudo usermod
   -aG docker ubuntu` — `libhwloc-dev` is an undocumented dependency of
   `aws-neuronx-collectives`.
3. `git clone <this repo> /home/ubuntu/XScript-Pretraining` (it is bind-mounted
   into the container as `/repo` in the next step), then:
   ```bash
   aws ecr get-login-password --region us-east-1 | sudo docker login --username AWS --password-stdin 421672808698.dkr.ecr.us-east-1.amazonaws.com
   sudo docker pull 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
   imageID=$(sudo docker images -q --filter reference=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b)
   cd ~ && sudo docker create --name tmp $imageID && sudo docker cp tmp:/workspace . && sudo docker rm tmp && sudo chown -R ubuntu:ubuntu ~/workspace
   sudo dpkg -i ~/workspace/runtime_artifacts/*.deb && sudo modprobe neuron && /opt/aws/neuron/bin/neuron-ls
   sudo docker run -d --privileged --restart unless-stopped --name neuron-native \
     -v /home/ubuntu/XScript-Pretraining:/repo $imageID sleep infinity
   ```
4. Smoke test: `sudo docker exec neuron-native bash -c "pip install
   transformers && cd /workspace/torch_neuron_eager/examples/gpt2-train-loop
   && python3 train.py"` (2 iterations, loss ~11.05, ~1 min).

What the guide doesn't say: the beta DKMS driver builds clean on kernel 7.0
(no `mm_get_unmapped_area` patch needed); the host-venv "Option B" is
impossible on Ubuntu 26.04 (cp312 wheels, only Python 3.14 available) — use
Docker; `:latest` is a moving target (the 5/15 guide says torch 2.11, the
Aug-5 image ships 2.12.1 / torch-neuronx 2.12.3.0 / neuronx-cc 2.27 / nki
0.6.0 — record `pip list`); int64 and float64 are auto-downcast with a
warning; `NKI_ENABLE_TRACE_CACHE=1` persists the NKI kernel cache across
processes (set 0 if kernels behave inconsistently).
