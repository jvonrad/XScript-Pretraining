#!/usr/bin/env python
"""Trainium entry point for one run of the matrix — the Neuron twin of
`xscript.cli train`.

Kept out of `xscript/cli.py` on purpose: the CUDA CLI stays untouched so the
Nvidia workflow is exactly as before. This script resolves a run name through
the *same* `runmatrix` (identical configs, schedule, mixture, tokenizer) and
dispatches to `xscript.train_neuron` instead of `xscript.train`.

Run under torchrun for data parallelism across Neuron cores; see
`scripts/neuron_train/README.md` and `scripts/neuron_train/launch.sh`.

    torchrun --nproc_per_node=32 scripts/neuron_train/run_train.py \
        zh__unigram_destarved --base configs/base_main.yaml --flavor unigram --only-30b

The positional/flag surface intentionally matches `xscript cli train`.
"""
import argparse
import sys
from pathlib import Path

# Prefer the local repo `src/` (mirrors run_benchmarks.py) so local patches win.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from xscript import runmatrix  # noqa: E402
from xscript import train_neuron  # noqa: E402
from xscript.langs import MODEL_FLAVORS  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="run_train (neuron)",
        description="Train one matrix run on AWS Trainium (Neuron/XLA).")
    ap.add_argument("name", help="run name (see `xscript runs`)")
    ap.add_argument("--base", default="configs/base_main.yaml")
    ap.add_argument("--flavor", default="unigram", choices=MODEL_FLAVORS)
    ap.add_argument("--only-30b", action="store_true",
                    help="self-contained 30B WSD config, never a trunk branch")
    ap.add_argument("--output-name", default=None,
                    help="store an independent diagnostic replicate under this name")
    ap.add_argument("--seed", type=int, default=None,
                    help="override model/optimizer RNG seed")
    ap.add_argument("--data-seed", type=int, default=None,
                    help="override packed-stream order seed")
    ap.add_argument("--wandb-id", default=None,
                    help="override the stable W&B run ID")
    ap.add_argument("--eval-in-loop", action="store_true",
                    help="force in-loop BPB eval on (off by default on Neuron: "
                         "variable-length eval docs cause XLA recompiles)")
    args = ap.parse_args(argv)

    cfg = runmatrix.get_run(args.base, args.flavor, args.name, args.only_30b)
    if args.output_name is not None:
        cfg["name"] = args.output_name
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.data_seed is not None:
        cfg["data_seed"] = args.data_seed
    if args.wandb_id is not None:
        cfg["wandb_id"] = args.wandb_id
    if args.eval_in_loop:
        cfg.setdefault("train", {})["eval_in_loop"] = True

    train_neuron.run_from_config(cfg)


if __name__ == "__main__":
    main()
