# XScript-Pretraining

Does ATLAS's reported cross-script pretraining penalty (EN-AR-ZH transferring
worse than EN-DE-FR) survive a **fair tokenizer**? This repo tests whether that
penalty is largely a **tokenizer vocabulary-allocation / byte-tax artifact**
rather than an intrinsic script-transfer effect. The untested contribution is
the **interaction term**: mixture (same- vs cross-script) × tokenizer
(starved vs destarved).

See [thesis-plan.txt](thesis-plan.txt) for the full design and the audit of
ATLAS (arXiv 2510.22037). This README is the operational manual.

## The experiment in one paragraph

Five languages: **EN** (anchor), **DE**, **FR** (same script as EN),
**AR**, **ZH** (cross-script). We pretrain a ~1B Llama-style model on 15
mixtures (5 monolingual + all 10 pairwise bilingual mixtures) under **2 tokenizer
conditions** at 30B tokens each = **30 runs**, plus a cheap **100B cooldown
extension** of 4 runs (EN-DE and EN-AR, both conditions). The two tokenizer
conditions are trained on the same corpus family (raw FineWeb/FineWeb2 — see
below) so **allocation is the only manipulated variable**:

| condition   | vocab competition                                    |
|-------------|------------------------------------------------------|
| `starved`   | 64k BPE, raw FineWeb/FineWeb2, T=100 over ~419 langs (ATLAS-scale) |
| `destarved` | 64k BPE, raw FineWeb/FineWeb2, our 5 languages only, byte-premium-adjusted |

Headline readout is **BPB** (bits-per-byte, tokenizer-independent) → **BTS**
(bilingual transfer score) → the **interaction**. If de-starving the tokenizer
collapses the cross-script penalty, the penalty was largely an artifact.

## Three tokenizer flavors (a decision gate before model training)

We train **five** tokenizers and inspect fertility/allocation before committing
model compute:

- **`unigram`** — SentencePiece **Unigram**, `character_coverage=0.999995`, byte
  fallback. This is **ATLAS's actual algorithm** (their 64k Unigram) — the
  faithful replication point. Both conditions.
- **`bpe`** — classical **byte-level** BPE via
  [swiss-ai/parity-aware-bpe](https://github.com/swiss-ai/parity-aware-bpe)'s
  `learn_bpe.py` (`Whitespace()+ByteLevel` pre-tokenization). Both conditions.
- **`pa`** — **parity-aware** byte-level BPE from the same repo
  (`parity_aware_learn_bpe.py`, window variant for ZH), fertility-equalized over
  the 5-way-parallel FLORES+ dev set. **Destarved only** — it balances a fixed
  dev-language set, so a 420-language "starved" version is undefined. `bpe` and
  `pa` share the exact same pipeline; the merge criterion is the *only*
  difference, giving a clean fairness contrast. (optional)

So: `unigram_{starved,destarved}`, `bpe_{starved,destarved}`, `pa_destarved`.

`xscript tok-analyze` writes `results/tok_analysis/{report.md, samples.md}`.
**Pick one MODEL flavor — `unigram` or `bpe`** (inspect segmentation fidelity
and the starved/destarved fertility gap for AR/ZH), then pack and train models
with it. `pa` is an **analysis-only** fidelity reference: it has no starved
condition, so it can't carry the starved-vs-destarved contrast the interaction
term needs. All model/eval code is flavor-agnostic (ids 0-3 are always
`<unk>/<bos>/<eos>/<pad>`; vocab is exactly 65536 so tokens fit `uint16`).

## Pipeline

```
flores-download ─┐
                 ├─ byte-premium ─┐
tok-corpus ──────┴─ tok-train ────┴─ tok-analyze   ← GATE: choose flavor
                                        │
        pool ── pack (per lang × chosen tok)
                                        │
                                     train         ← one run of the matrix
                                        │
              eval-bpb / eval-align ── bts          ← headline analysis
                       eval-bench                    ← downstream accuracy
```

Every step caches and is resumable. The `xscript` CLI is the single interface;
the Slurm jobs in [slurm/](slurm/) just call it, so behaviour is identical on a
laptop and on the compute nodes.

### Commands

```bash
# 0. install ([tok] adds the byte-level BPE + parity-aware learners)
pip install -e ".[tok]"     # base + tok; add [train] inside the GPU container
# One-time Python-3.12 runtime for the NGC container (Tok, W&B, lm-eval):
#   sbatch --account=brics.u6jh --partition=workq slurm/02_setup_runtime.sbatch

# 1. data prep (needs internet + HF_TOKEN with FLORES+ terms accepted) — login node
export HF_TOKEN=...
xscript flores-download
xscript byte-premium                     # -> results/byte_premium/
xscript tok-corpus both --gb 4           # raw FineWeb/FineWeb2 corpora (starved + destarved)

# 1b. tokenizer training + gate — CPU/memory-heavy, no internet needed, so
# submit as a real job (slurm/11_tok_train.sbatch) rather than running on the
# login node (its interactive-session cgroup caps out at 4GB RAM):
#   sbatch --account=brics.u6jh --partition=workq slurm/11_tok_train.sbatch
xscript tok-train --flavor all           # unigram/bpe x conds + pa_destarved
xscript tok-analyze                      # -> results/tok_analysis/  (THE GATE)

# 2. model data (per language), then pack with the CHOSEN model tokenizer(s)
for L in en de fr ar zh; do xscript pool --lang $L; done
xscript pack --lang en --tok bpe_destarved --workers 64     # repeat per lang × tok

# 3. train one run (see `xscript runs` for names)
xscript train en-ar__bpe_destarved --base configs/base_main.yaml --flavor bpe

# 4. eval + headline
xscript eval-bpb   en-ar__bpe_destarved --tok bpe_destarved
xscript eval-align en-ar__bpe_destarved --tok bpe_destarved
xscript bts --flavor bpe --source flores  # -> results/bts/ (BTS + interaction)
xscript eval-bench en-ar__bpe_destarved --tok bpe_destarved
# The full 5-language Global-MMLU + Belebele + XNLI suite is normally submitted
# per final checkpoint with slurm/41_bench.sbatch, not run inside training.
# Each run evaluates only its own one or two languages unless --tasks overrides it.
```

## Run matrix

`xscript runs --flavor <unigram|bpe>` lists all 38 runs. Names are
`<mix>__<flavor>_<condition>[__trunk|__100b]`:

- **30 deliverables @ 30B**: all 5 monolingual and 10 pairwise bilingual
  mixtures ×
  {starved,destarved}`.
- **4 deliverables @ 100B**: EN-DE and EN-AR (both conditions), as `__100b`.
- **4 trunks**: for the extended cells, a long constant-LR (WSD) stable trunk;
  the 30B and 100B deliverables are cheap **cooldown branches** off it
  (`branch.from` a `stable_<N>M` checkpoint), so the 0-24B compute is shared.

`bash slurm/submit_matrix.sh` submits everything with the trunk→cooldown
dependencies wired (`--dependency=afterok`).

To run only the 30 independent 30B cells (no 80B trunks or 100B extensions),
use `NODES=2 bash slurm/submit_30b.sh`. Each allocation restarts a failed
training step in-place from the latest checkpoint while allocation time remains.

## Design choices worth knowing (and defending in the thesis)

- **BPB, not per-token loss.** Byte denominator makes losses comparable across
  tokenizers of different fertility — the whole point.
- **Bilingual mixing is token-level 50/50** (`probs: null` → uniform). Each
  language contributes half the *tokens*; under a starved tokenizer the
  cross-script partner therefore sees less *content* — that is exactly the
  effect under audit. BTS is reported **both** matched-total and matched-per-
  language-tokens (`eval.bts`) so the token-count confound is visible. A
  content-matched mode is available by setting `probs` from the byte premiums.
- **Tokenizer corpus is FineWeb-family, not MADLAD-400** (unlike ATLAS's actual
  tokenizer). Both conditions are trained on raw (unfiltered) FineWeb/FineWeb2 —
  the same corpus family as the model-training pools — so there's no tokenizer-
  corpus-vs-model-corpus domain mismatch that could hit AR/ZH harder than
  DE/FR. State this as a deliberate, documented deviation from ATLAS's literal
  MADLAD-trained tokenizer (see `data/tokcorpus.py`). The `unigram` flavor
  still replicates ATLAS's *algorithm* (SentencePiece Unigram + byte fallback);
  only the training corpus differs.
- **WSD schedule** enables the 100B extension without retraining the trunk.
- **Deterministic, world-size-independent loader.** The token stream is fixed
  by `(data_seed, global slot)`; a run resumes bit-identically even on a
  different node count.
- **Log-spaced checkpoints** (dense early: 250M→500M→1B→2B) because dynamics
  are front-loaded. Downstream benchmarks run at major/final checkpoints only;
  BPB remains the frequent in-loop signal.

## Two things to verify before committing compute (plan's explicit asks)

1. **Classifier identity.** The whole EN-vs-rest comparison rests on
   FineWeb-HQ (EN) and FineWeb2-HQ (DE/FR/AR/ZH) using the **same** XLM-R
   quality classifier and top-10% threshold. The READMEs say so; confirm the
   paper appendix has no English-specific tuning. Note this differences out of
   the headline **interaction** term (tokenizer × mixture) regardless, but it
   affects absolute BPB comparisons across languages.
2. **AR token volume at 30B.** `arb_Arab` is byte-heavy; after byte-premium
   scaling the monolingual AR run may need to epoch its pool. `xscript pool
   --lang ar` prints `WARNING ... corpus exhausted` if the pool can't cover the
   budget without repetition, and `xscript pack` reports exact token counts.
   Check AR specifically before launching; top up with a larger `--gb` if
   needed. (`byte-premium` also compares our FLORES+-recomputed premiums
   against `catherinearnett/byte-premium-tool` as an external sanity check.)

## Layout

```
src/xscript/
  langs.py          5 study languages + tokenizer grid (single source of truth)
  paths.py          scratch/results layout (XSCRIPT_SCRATCH, XSCRIPT_RESULTS)
  flores.py         FLORES+ download + parallel loading
  byte_premium.py   byte-premium calibration (FLORES+; vs Arnett tool)
  data/tokcorpus.py raw FineWeb/FineWeb2 tokenizer corpora (starved/destarved)
  data/fineweb.py   FineWeb(-2)-HQ model-training pools (column-pruned)
  data/pack.py      tokenize pools -> uint16 shards
  data/loader.py    deterministic memmap windows + bilingual mixer
  tok/{train,wrapper,analyze}.py   5 tokenizers; uniform API; the fertility gate
  model.py          Llama-style decoder (RMSNorm/RoPE/SwiGLU, untied)
  schedule.py       WSD LR + log-spaced checkpoint schedules
  train.py          DDP/bf16 loop, cooldown branch, resume, in-loop BPB
  eval/{bpb,bts,alignment,bench}.py   BPB, BTS, MEXA, lm-eval benchmarks
  runmatrix.py      the 26-run matrix from a base config
  cli.py            `xscript` subcommands
configs/            base_main (1B) / base_pilot (300M) / base_smoke (CPU test)
slurm/              container/runtime setup, prep, pack, train, eval
scripts/smoke.py    end-to-end CPU smoke test on synthetic data
```

## Smoke test

```bash
python scripts/smoke.py     # ~1 min on CPU; needs numpy/torch/tokenizers/…
```

Fabricates synthetic FLORES/corpora/pools on a throwaway scratch (no network,
no HF_TOKEN) and drives the real code paths end-to-end: tokenizer train →
analyze → pack → loader determinism → model train (mono, bilingual, and a WSD
cooldown branch) → BPB → MEXA alignment → BTS. Prints `ALL STAGES PASSED`.

## Target hardware

Isambard-AI (HPE Cray EX, GH200 ×4/node, Slingshot). Data prep runs on a login
node (internet); training runs multi-node inside the NGC PyTorch container via
`module load brics/apptainer-multi-node` + `torchrun`. That module bind-mounts
the admin-built aws-ofi-nccl (Slingshot) plugin stack into the container and
its `/host/adapt.sh` wrapper exports the tuned NCCL/CXI env — don't set
`NCCL_*`/`FI_*` yourself (see [slurm/env.sh](slurm/env.sh)). Storage on 5TB
Lustre scratch (`XSCRIPT_SCRATCH`). Slurm account/partition default to
`brics.u6jh` / `workq` ([Isambard-AI docs](https://docs.isambard.ac.uk/)).

Login-node interactive shells run inside a systemd session cgroup capped at
`MemoryMax=4GiB` / `TasksMax=500` (`systemctl show user-$(id -u).slice`),
independent of the node's real specs. Fine for network-bound, disk-streaming
steps (`flores-download`, `tok-corpus`, `pool`); CPU/memory-heavy steps that
don't need internet (`tok-train`) should run as a real Slurm job instead —
see [slurm/11_tok_train.sbatch](slurm/11_tok_train.sbatch).
