# Neuron/Trainium pretraining

Trainium (Neuron/XLA) adaptation of `src/xscript/train.py` for finishing the
**3 unfinished models** from the intended 18-cell EN-anchored 30B matrix
(`src/xscript/runmatrix.py`) that never made it into the 15 checkpoints
uploaded to `jvonrad/xscript-eval`:

| run name (`xscript runs --only-30b`) | friendly name | langs |
|---|---|---|
| `de__unigram_starved`   | de-starved | `[de]` |
| `zh__unigram_destarved` | zh-fair    | `[zh]` |
| `zh__unigram_starved`   | zh-starved | `[zh]` |

This directory is **new code only** — nothing under `src/xscript/{train,model,
schedule,runmatrix,data/loader}.py` or `slurm/` is modified, so the CUDA/Slurm
path on Isambard-AI keeps working exactly as before.

## What's different from the CUDA path

See the module docstring in [`../../src/xscript/train_neuron.py`](../../src/xscript/train_neuron.py)
for the full mechanical rundown (XLA device instead of CUDA, `xm.reduce_gradients`
instead of `DistributedDataParallel`, one `xm.mark_step()` per optimizer step,
`xm.save`/`xm.add_step_closure`, in-loop BPB eval off by default). Everything
about the *experiment itself* — model architecture, WSD schedule, deterministic
loader, checkpoint format, run configs — is identical; only execution mechanics
differ, because Trainium has no CUDA/NCCL.

Checkpoints written by `NeuronTrainer.save()` use the **same dict keys** as the
CUDA trainer's, so they load with plain `torch.load` and work unmodified with
the existing eval path (`src/xscript/eval/bench.py --device xla`, already
Neuron-native per the top-level `CLAUDE.md`) and with `run_benchmarks.py`'s
`fetch_checkpoint()`/upload tooling.

## Prerequisites

1. **Neuron venv** set up per the top-level `CLAUDE.md` (§1) — `~/neuron_venv`
   with `torch-neuronx`, plus the pinned eval-stack packages are *not* needed
   for training itself, but `zstandard` and `sentencepiece` (pool/pack/tokenizer
   code) must be present:
   ```bash
   export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; export PATH="$HOME/.local/bin:$PATH"
   source ~/neuron_venv/bin/activate
   uv pip install zstandard sentencepiece
   ```
2. **Tokenizers** (`unigram_starved`, `unigram_destarved`) — already trained;
   reuse the artifacts fetched for eval rather than retraining:
   ```bash
   mkdir -p "$XSCRIPT_SCRATCH/tokenizers"
   cp -r /home/ubuntu/xscript_bench/xscript/tokenizers/unigram_starved   "$XSCRIPT_SCRATCH/tokenizers/"
   cp -r /home/ubuntu/xscript_bench/xscript/tokenizers/unigram_destarved "$XSCRIPT_SCRATCH/tokenizers/"
   ```
   (Confirmed present with `sp.model` + `meta.json` at that path on this box.)
3. **`HF_TOKEN`** (private repo terms + FLORES+ access) — **not currently set
   in this shell**, and required for `xscript pool` to download FineWeb2-HQ
   `deu_Latn`/`cmn_Hani`.
4. **Text pools + packed shards for `de` and `zh`** — **do not exist on this
   box yet** (confirmed: no `pools/` or `shards/` dirs under any candidate
   `XSCRIPT_SCRATCH`). These use the existing, unmodified CLI (data prep is
   device-agnostic, no Neuron-specific code needed):
   ```bash
   export HF_TOKEN=hf_...
   export XSCRIPT_SCRATCH=/home/ubuntu/xscript_train/scratch   # big disk; see launch.sh
   PYTHONPATH=src python -m xscript.cli pool --lang de
   PYTHONPATH=src python -m xscript.cli pool --lang zh
   PYTHONPATH=src python -m xscript.cli pack --lang de --tok unigram_starved   --workers 32
   PYTHONPATH=src python -m xscript.cli pack --lang de --tok unigram_destarved --workers 32
   PYTHONPATH=src python -m xscript.cli pack --lang zh --tok unigram_starved   --workers 32
   PYTHONPATH=src python -m xscript.cli pack --lang zh --tok unigram_destarved --workers 32
   ```
   **Sizing** (`runmatrix.plan_budgets()`, worst-case destarved tokenizer, 30B
   tokens): ~155GB of raw text *per language* before the 1.15x safety margin
   and real bytes/token measurement kick in — plan for real downloads in the
   tens-to-~150GB range per language, i.e. **hours of download time and
   meaningful disk**, not a quick step. `de__unigram_destarved` pool is shared
   with the already-uploaded `de-fair` model's pack — only `de`'s
   `unigram_starved` pack and both of `zh`'s packs are strictly new work, but
   `xscript pool --lang de` still needs to be (re)run if its pool isn't cached
   from that earlier run.
   README.md's design doc also flags: FineWeb2-HQ's `arb_Arab` and `fra_Latn`
   splits were previously found short of budget (needed a fallback source,
   `FALLBACK_SOURCES` in `data/fineweb.py`); worth watching for the same on
   `deu_Latn`/`cmn_Hani` — `xscript pool` prints `WARNING ... corpus exhausted`
   if so (training then just epochs over the smaller pool; not fatal).

## Validating the code path (cheap, done before any real run)

`smoke_neuron.py` is the Neuron twin of the repo's `scripts/smoke.py`: fully
synthetic data (no network, no `HF_TOKEN`), tiny model
(`configs/base_smoke.yaml`), but drives the **real** `NeuronTrainer` loop
(forward/backward/`xm.reduce_gradients`/`xm.mark_step`/`xm.save`/resume) on an
**actual XLA device** — not CPU emulation. Confirms the port compiles and runs
on real hardware before committing to a 30B-token job.

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; export PATH="$HOME/.local/bin:$PATH"
source ~/neuron_venv/bin/activate
export NEURON_RT_VISIBLE_CORES=32-33   # pick a free core pair; check with neuron-ls
python scripts/neuron_train/smoke_neuron.py
```

Prints `ALL STAGES PASSED` on success. Single-process (`world=1`) is a valid
degenerate case of the XLA path — no `torchrun` needed for this check.

## Launching a real run

```bash
RUN=zh__unigram_destarved bash scripts/neuron_train/launch.sh
```

See [`launch.sh`](launch.sh) for env vars (`NPROC`, `WORK`, `BASE`, `FLAVOR`).
Defaults to all 32 logical cores on `trn2.48xlarge` via
`torchrun --nproc_per_node=32`. Resumable: re-running the same `RUN` continues
from `checkpoints/last.pt` (the loader is deterministic and world-size-
independent, matching the CUDA trainer's resume semantics exactly).

Rough sizing: at 1M tokens/optimizer step (`base_main.yaml`'s
`global_batch_tokens`) and 30B tokens, that's ~30k optimizer steps per model.
Actual wall-clock depends on measured Neuron throughput for this model
size/shape, which hasn't been benchmarked on this box yet — check
`[train-neuron] step N | ... | Xk tok/s` in the first few minutes of a real
run and extrapolate before committing all 3 models to a long unattended run.

## After training: eval + upload

Checkpoints land in `$XSCRIPT_SCRATCH/runs/<run_name>/checkpoints/{last,final,
stepN_*}.pt` — the same layout `eval/bpb.py`, `eval/bench.py`, and
`eval/alignment.py` already expect. Run the existing (unmodified) eval CLI
commands from the top-level `README.md` §Commands, or the Neuron-native
`bench.py --device xla` path documented in the top-level `CLAUDE.md` after
uploading to `jvonrad/xscript-eval` and adding an entry to `models.json`
following the existing 15 models' pattern (`orig_run`/`tok`/`langs`).
