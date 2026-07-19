# External benchmark evaluation

Isambard-AI is blocked by a CPU-minutes quota, so we evaluate the trained
checkpoints on a GPU elsewhere. The models are a custom LLaMA-style Transformer
(`src/xscript/model.py`) + SentencePiece tokenizer — pure PyTorch, using
`F.scaled_dot_product_attention`, no flash-attn / triton / custom kernels — so
they run on any stock GPU (or CPU, slowly). Each model is ~1B params (fits any
16GB GPU).

The benchmark harness (`src/xscript/eval/bench.py`) wraps our model into
lm-evaluation-harness and scores Global-MMLU, Belebele, and XNLI on each run's
training languages. It is the *same* harness we would have run on-cluster, so
numbers are directly comparable.

## 1. Export from Isambard (already done by `upload_to_hf.py`)

The private HF repo mirrors the on-cluster layout:

```
src/xscript/**                       # bundled model + harness code
tokenizers/unigram_{starved,destarved}/{sp.model,meta.json}
runs/<name>/checkpoints/final.pt     # 15 checkpoints, fp32, ~4GB each
models.json                          # friendly name -> tokenizer + langs + orig run
run_benchmarks.py  requirements.txt  README.md
```

Models use friendly names `<mixture>-<starved|fair>` (e.g. `en-fair`,
`en-ar-starved`). `models.json` maps each to its real tokenizer.

## 2. Run on your GPU

```bash
# clone just the runner (or download run_benchmarks.py + requirements.txt from the repo)
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
export HF_TOKEN=hf_...        # while the repo is private

# quick validation pass over all 15 runs (~200 examples/task) -- do this FIRST
python run_benchmarks.py --repo jvonrad/xscript-eval --limit 200

# full suite once the quick pass looks sane
python run_benchmarks.py --repo jvonrad/xscript-eval
```

The runner downloads one checkpoint at a time and deletes it after eval
(`--keep-checkpoints` to retain), so peak disk is ~5GB. Results:

```
xscript_bench/results/bench/<run>_final.json    # per-run task accuracies
xscript_bench/results/summary.json              # everything combined
```

Send those JSONs back for analysis.

## 3. Cross-lingual representation alignment (optional, same GPU)

MEXA-style embedding retrieval between EN and each partner language, at every
layer (`src/xscript/eval/alignment.py`). Only the 8 EN-anchored bilingual
models have a pair to align (mono runs are skipped automatically):

```bash
python run_alignment.py --repo jvonrad/xscript-eval
```

Results in `xscript_bench/results/alignment/<model>.{json,md}`.

## Notes
- `--runs en-starved en-fair` limits to a subset (friendly names).
- `--tasks xnli_en xnli_de` overrides the task list (default = the run's langs).
- Mono runs get 3 tasks (their one language), bilingual runs get 6 (both langs).
- Scores are ordinary accuracy (`acc,none`); raw harness output is preserved in
  each per-run JSON for length-normalized variants.
