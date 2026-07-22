# NEURON.md

Everything about **running this repo on AWS Trainium (Neuron)**: environment
setup, dependency pins, the XLA scoring adaptation and its silent traps, how to
fan the eval sweeps out across cores, how the **training** port works, and the
Neuron-specific bugs that cost real time.

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
| §9 Training on Neuron | **NEURON.md** |

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

Neuron writes stray `*PostSPMDPassesExecutionDuration.txt` files into the cwd —
gitignore them.

---

## 9. Training on Neuron (`train_neuron.py`)

The eval sections above are about *scoring* existing checkpoints. This section is
about **training new ones** on Trainium — used to finish the models that were
missing from the Isambard (GH200/CUDA) matrix.

### Why not AWS's own training framework

NeuronX Distributed Training (NxDT) is the sanctioned, optimized path, and it was
**deliberately not used**: it ships its own Llama implementation with
**split-half RoPE** (HuggingFace convention), whereas `model.py` uses
**interleaved RoPE** (GPT-NeoX convention), plus a different init scheme and its
own data/optimizer plumbing. Training with it would have made the new
checkpoints architecturally different from the 15 already trained on Isambard —
exactly the comparability the cross-script comparison cannot afford. It also
depends on NVIDIA **NeMo**, which is not installed here.

So `src/xscript/train_neuron.py` is a hand-port that keeps `model.py`, the data
pipeline, tokenizers, schedule and checkpoint format **byte-identical** to the
CUDA trainer (`xscript.train`, which is untouched and still runs on GH200). Only
the execution mechanics differ. Everything below is the cost of that choice.

### The memory picture — it is the OPTIMIZER and the VOCAB, not activations

Measured from Neuron's own allocation ledger (`nrt_mem_log_*.csv`), at
micro_batch=1 the persistent "tensor" bucket is **~19.5GB** (fp32 master weights
4.06 + fp32 grads 4.06 + AdamW m 4.06 + v 4.06 + bf16 autocast copies 2.03) plus
~8.25GB compiler scratchpad → ~28GB against a **24GB** per-core budget. The
`act` (activations) bucket is only **~0.08GB** once checkpointed.

That is why shrinking `micro_batch_size` from 8→1 barely moved the number: it
only touches the 0.08GB bucket while ~27GB is **batch-independent**. Levers that
actually work, all opt-in via `cfg["train"]`:

| knob | what it does | when to use |
|---|---|---|
| ZeRO-1 (`zero`, auto when world>1) | shards AdamW m/v + fp32 master across replicas | always for world>1 |
| `bf16_params: true` | bf16 forward weights/grads, **fp32 master kept by ZeRO** (`optimizer_dtype` defaults fp32) | to halve fwd/bwd weight I/O |
| `fused_ce_chunk: N` | `_ChunkedLMHeadCE`: chunks lm_head+cross_entropy over the token dim, recomputes logits in backward | to raise micro_batch past ~2 |
| activation checkpointing (`_checkpointed_forward`) | recomputes each Block in backward | always (it is unconditional) |

**The real micro_batch ceiling is the vocabulary projection.** With
`vocab_size=65536` at seq 2048, the logits tensor is `mb*seq*vocab`; at mb=8 the
cross-entropy over it alone needs **~23GB** of compiler scratchpad
(`NCC_EOOM002`, 29GB peak) regardless of ZeRO or bf16. `fused_ce_chunk` is what
unlocks mb≥8. Note the chunk size trades memory against compile time: chunk=2048
(8 chunks) compiles very slowly because the loop unrolls and the backward nests
autograd per chunk; prefer a larger chunk (fewer unrolls).

### Silent numerical bugs found here (CLAUDE.md §4 class — these do NOT crash)

- **A bool `.sum()` returns `-1`** on this build instead of the count.
  `n_valid = (targets != -100).sum()` gave −1, which silently **negated and
  un-normalized the loss** (reported −188243 instead of ~11.5, i.e. `−sum`).
  Fix: cast to float *before* reducing —
  `(targets != -100).to(torch.float32).sum().clamp(min=1.0)`. CPU never
  reproduces it, so the CPU unit test passed while the device was wrong.
- `torch.gather` over the vocab dim returns zeros (see §4).

**Verify any new loss/reduction against CPU before trusting a run.**

### Launch mechanics — `xmp.spawn`, NOT `torchrun`

`torchrun` **cannot** pin a job to a subset of cores. `torch_neuronx`'s
`Initializer` overwrites `NEURON_RT_VISIBLE_CORES` with `LOCAL_RANK` via
`__set_envvar_defaulted_and_save("NEURON_RT_VISIBLE_CORES", key_from="LOCAL_RANK",
default=<your cores>)` — and since `os.environ.get(key_from, default)` finds
`LOCAL_RANK`, your value is ignored. Every torchrun job therefore lands on cores
`0..nproc-1` and collides with anything already there (`NRT_FAILURE ... lnc0
Available:0, cores busy`).

`xmp.spawn` **does** honor it (`Initializer.reset()` early-returns when
`is_torchelastic_launched()` is False, and the xmp path remaps via
`cores_list[local_rank]`). Recipe:

```bash
export NEURON_RT_VISIBLE_CORES="8,9,10,11,12,13,14,15"  # one core id per replica
export NEURONCORE_NUM_DEVICES=8        # xmp.spawn accepts nprocs=1 or None only
export NEURON_RT_ROOT_COMM_ID=127.0.0.1:48711   # DISTINCT per concurrent job
export MASTER_ADDR=localhost MASTER_PORT=48811  # DISTINCT per concurrent job
python3 your_entry.py                  # entry calls xmp.spawn(fn, args=())
```

Core ids are the **same physical ids** `neuron-ls` / `free_cores.py` report
(0-63; each device = 4 consecutive ids) — not a separate logical space.

### Concurrency hazards (each one cost hours)

1. **Rendezvous port collision → silent hang.** Every job defaults to the same
   port for `dist.init_process_group("xla", init_method="xla://")` (12355), so
   the **2nd job hangs forever at init with no error**. Set a distinct
   `MASTER_PORT` per job. This was the single worst time-sink; with it fixed,
   three world=8 jobs coexist with almost no contention.
2. **Distinct `NEURON_RT_ROOT_COMM_ID` per job**, or collectives cross-talk →
   `CCOM WARN Timeout waiting for RX (waited 240 sec)`.
3. **Parallel compile-cache race.** Launching N jobs at once that compile the
   same graph can deadlock them. Launch **staggered** — let one compile and
   reach a step (warming the cache) before starting the next.
4. **Stale compile-cache locks.** Killing a job mid-compile leaves
   `MODULE_*/model.hlo_module.pb.lock`; other jobs then wait forever on
   "Another process must be compiling…". Recover: kill all workers, then
   `find $CACHE -name '*.lock' -delete` and `rm -rf` the locked MODULE_* dirs.
   A clean *runtime* OOM does NOT corrupt the cache — only mid-compile kills do.
5. **xmp workers are `neuron_venv/bin/python3 -c from multiprocessing.spawn…`**,
   not your script name, so `pkill -f your_script.py` kills only the parent and
   orphans workers that keep holding cores. Kill by PID from
   `neuron-ls --show-all-procs`, or `pkill -9 -f "neuron_venv/bin/python3 -c from"`.
6. **Core-release lag** (~5-10s) after SIGKILL — relaunching immediately gives
   "cores busy". Verify with `neuron-ls` first.
7. **Failures present as HANGS, not errors.** A watchdog that restarts when no
   step has been logged for N minutes is mandatory for unattended runs.

### The world>8 collective limit (the main throughput cap)

A single job at **world=16 fails** at the first ZeRO collective:
`NRT_INVALID ... invalid execution input, such as incorrect number of inputs or
invalid send/recv targets`. **world=8 works.** This caps one model to 8 of 64
cores (2 of 16 devices) at **~47k tok/s**, i.e. ~7 days for a 30B-token run.
EFA/libfabric is *not* the fix (that is inter-node; single-node collectives use
on-chip NeuronLink).

**RESOLVED — the limit was ours, not the platform's.** `scripts/neuron_train/
nxd_test.py` runs **world=32 to completion** using `neuronx-distributed`'s
ZeRO-1, with **no** `invalid send/recv targets`. So the wall is a defect in the
hand-rolled `torch_xla` ZeRO path in `train_neuron.py`, not a Trainium or
collective-library limit.

The key correction to §9's opening: **NxD (`neuronx-distributed`) is not NxDT**.
NxDT ships its own Llama (split-half RoPE) and needs NeMo — that is what was
correctly rejected. NxD is a bring-your-own-model library that wraps an
*arbitrary* `nn.Module`, needs no NeMo, and preserves our interleaved RoPE:

```python
nxd_config = neuronx_distributed_config(
    tensor_parallel_size=1,          # our model uses plain nn.Linear; TP would need edits
    optimizer_config={"zero_one_enabled": True, "grad_clipping": True, "max_grad_norm": 1.0},
    activation_checkpoint_config=Block,   # a module CLASS is the supported path
)
model = initialize_parallel_model(nxd_config, model_fn)          # model_fn() -> our Transformer
optim = initialize_parallel_optimizer(nxd_config, torch.optim.AdamW, model.parameters(), lr=...)
```
Launch with **torchrun** here: its rank-i→core-i pinning is *correct* when you
take the whole box; the pinning problem above only bites for a core subset.

Still open: the **1B** model at world=32 fails to *compile* — 36.75GB peak,
**27.63GB scratchpad**, because NxD fuses fwd+bwd+optimizer into one graph
where our own trainer's `mark_step` boundaries kept the full-vocab CE at ~8GB.
That is memory tuning (mb=1, added `mark_step`s, or the existing
`fused_ce_chunk`), not a wall. Note NxD's ZeRO also sets `use_fp32_grad_acc`,
adding a full fp32 grad buffer our implementation did not have.

### Measured throughput vs GH200 (for MFU comparisons)

Model FLOPs/token = **6.54 GF** (fwd+bwd over body+head+attention); activation
checkpointing adds **1.91 GF** of recompute on Neuron that the CUDA trainer does
not pay (it has no checkpointing), hence MFU vs HFU below.

| | tok/s per accelerator | model TF/s | MFU |
|---|---|---|---|
| GH200 (Isambard, 4/node, W&B `tok_per_s` 247k) | 61,750 | 404 | **40.9%** |
| Trn2 (world=8 = 2 chips, 47k) | 23,500 | 154 | **23.1%** (HFU 29.8%) |

Peak bf16 dense assumed: GH200 989 TF/s, Trn2 667 TF/s. The per-accelerator gap
is **2.63x**, of which **1.48x is raw peak** and **1.77x is efficiency** — so
per-chip efficiency was never the main problem. At 23.5k tok/s/chip all 16 chips
would give **376k tok/s, ~1.5x a 4xGH200 node**; the world=8 cap is what made it
0.19x instead.

### Resuming from a foreign checkpoint (`warm_start`)

`cfg["warm_start"] = {"from": <path>}` loads *model weights only* from a
checkpoint that has no optimizer state (e.g. the Isambard 12B partials) and sets
the token/step counter so the WSD schedule continues from there. It is applied
in `__init__` **before** the optimizer is built, so ZeRO's fp32 master is sharded
from the loaded weights rather than the random init, and only on the first
launch (later restarts resume from `last.pt` with real optimizer state).
Verified exact: resumed at step 12820 / 11.76B with lr and loss continuing.

Checkpoint layout note: under ZeRO the optimizer state is **sharded per rank**,
so `save()` writes `<tag>.optim.rank<r>.pt` sidecars (xm.save writes only the
master ordinal). Resume requires the **same world size**. Model-only saves
(`resumable=False`) are unaffected and stay byte-compatible with the CUDA
trainer's checkpoints.

### Production launcher

`/home/ubuntu/xscript_prod/` (outside the repo, holds a wandb key):
`prod_train.py` (xmp entry, forces the proven config), `run_prod.sh` (per-model
env + core pinning), `orchestrate_zh15.sh` (staggered launch + keep-alive that
restarts on death **or** hang, resuming from `last.pt`), `STATUS.md`.
Proven-stable config for unattended runs: **fp32 params, micro_batch=2, ZeRO-1,
full cross-entropy, `--optlevel=1`, world=8**.

Compiler-flag caveat: `--optlevel=2/3` **ballooned the scratchpad 8GB → 23GB**
for this model and caused OOM where O1 fit. `--auto-cast=none` was also
implicated in scratchpad growth. Keep `--optlevel=1` for memory-bound models.
