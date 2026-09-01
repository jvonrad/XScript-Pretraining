#!/usr/bin/env python
"""Extend the W&B BPB curves with points scored from uploaded checkpoints.

WHY THIS EXISTS
===============
`pull_wandb_curves.py` can only report what the trainer logged. Two Chinese
runs were resumed under new W&B ids (`zh__*__neuron`, 11.76B -> 14.99B) with
the in-loop eval switched off, so **the last ~3B of both zh curves does not
exist in W&B under any key** and no puller can recover it. What does exist is
the checkpoints: `zh-{fair,starved}-15b` are `step15865_14756M`, taken from
inside exactly that unlogged stretch.

`run_bpb.py` scores them, and the result is the SAME quantity the trainer
logged, not a proxy -- both are FLORES+ **dev** (n=997, `bpb.py`'s
`flores.load_parallel(langs, "dev")`), same NLL/bytes definition. This script
checks that claim rather than asserting it: `zh-{fair,starved}-12b` ARE the
checkpoints behind the logged step-12811 points, so scoring them reproduces a
number W&B already holds. Agreement there is what licenses splicing the 15b
points onto the curve with no calibration offset.

    # score (see the module docstring of run_bpb.py for the workdir layout)
    XSCRIPT_FLORES=$WORK/xscript/flores_plus python run_bpb.py \
        --repo jvonrad/xscript-eval --langs zh --split dev --device cpu \
        --runs zh-fair-12b zh-starved-12b zh-fair-15b zh-starved-15b \
        --workdir $WORK
    # then
    python bpb_fill_from_checkpoints.py --bpb-dir $WORK/results/bpb

Writes `results/wandb_curves/bpb_curves_ckpt.csv`, same long schema as
`bpb_curves.csv` plus provenance columns. It is a SEPARATE file on purpose:
`bpb_curves.csv` is "whatever W&B holds, pulled reproducibly", and quietly
mixing a locally-computed point into it would destroy exactly the provenance
this directory exists to have. Join on `run` to plot them together.

⛔ **FLORES only.** The trainer also logs `eval/holdout_*_bpb` from 500 docs of
the reserved FineWeb2-HQ holdout shard (`bpb.load_holdout`). Those shards are
not on the eval box, so the holdout half of the gap CANNOT be filled this way
-- it needs the language's pool rebuilt (CLAUDE.md 6h did that for German at
72.6GB). Any analysis that reads `--source holdout` still stops where W&B does.
"""
import argparse
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "wandb_curves"

# friendly name -> the W&B curve it belongs on, derived from `orig_run`.
_STEP_RE = re.compile(r"^(?P<run>.+?)__step(?P<step>\d+)_(?P<tokens_m>\d+)M$")


def parse_orig_run(orig: str):
    """`zh__unigram_destarved__step15865_14756M` -> (curve, step, tokens).

    The `M` figure is the TRAINER's own token count at that step, which is the
    authoritative position on the token axis -- `step x tokens_per_step` is a
    reconstruction, and for these resumed runs it needs a fitted offset before
    it even agrees (see `pull_wandb_curves.py`'s RUN_MERGE).
    """
    m = _STEP_RE.match(orig)
    if not m:
        raise ValueError(f"cannot parse orig_run {orig!r}")
    return m["run"], int(m["step"]), int(m["tokens_m"]) * 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bpb-dir", required=True, type=Path,
                    help="run_bpb.py's results/bpb directory")
    ap.add_argument("--models-json", type=Path, default=REPO / "results" / "models.json")
    ap.add_argument("--curves", type=Path, default=OUT / "bpb_curves.csv",
                    help="W&B curves, used for the control comparison")
    ap.add_argument("--out", type=Path, default=OUT / "bpb_curves_ckpt.csv")
    args = ap.parse_args()

    models = json.loads(args.models_json.read_text())

    # the W&B points, for the control check: {(curve, step, metric): value}
    logged = {}
    with open(args.curves) as fh:
        for r in csv.DictReader(fh):
            logged[(r["run"], int(r["step"]), r["metric"])] = float(r["value"])

    rows, controls = [], []
    for path in sorted(args.bpb_dir.glob("*_bpb.json")):
        d = json.loads(path.read_text())
        model = d["run"]
        if model not in models:
            raise SystemExit(f"{model} not in {args.models_json}")
        orig = models[model]["orig_run"]
        curve, step, tokens = parse_orig_run(orig)
        for lang, res in d["per_lang"].items():
            metric = f"eval/{d.get('source', 'flores')}_{lang}_bpb"
            rows.append((curve, model, orig, step, tokens, metric, res["bpb"],
                         d["split"], d["n_sentences"]))
            ref = logged.get((curve, step, metric))
            if ref is not None:
                controls.append((model, curve, step, metric, res["bpb"], ref))

    # --- the control: does a checkpoint-scored point reproduce the logged one?
    if not controls:
        print("[fill] WARNING: no control point -- nothing here is validated "
              "against W&B, so do not splice these onto the curve yet.")
    else:
        print("[fill] control -- checkpoint-scored vs the value W&B logged for "
              "the SAME checkpoint:")
        worst = 0.0
        for model, curve, step, metric, got, ref in controls:
            worst = max(worst, abs(got - ref))
            print(f"    {model:16s} step {step:6d} {metric:24s} "
                  f"{got:.6f} vs {ref:.6f}   {got - ref:+.2e}")
        print(f"    worst |difference| = {worst:.2e}")
        assert worst < 1e-4, (
            f"control disagrees by {worst:.2e}; the checkpoint-scored points are "
            f"NOT on the same axis as the logged curve -- do not splice them")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "model", "orig_run", "step", "tokens", "metric",
                    "value", "split", "n_sentences"])
        for row in sorted(rows):
            w.writerow(row)

    fills = [r for r in rows if (r[0], r[3], r[5]) not in logged]
    print(f"\n[fill] wrote {len(rows)} points ({len(controls)} control, "
          f"{len(fills)} NEW) to {args.out}")
    for r in sorted(fills):
        print(f"    NEW  {r[0]:24s} {r[4] / 1e9:7.3f}B  {r[5]:24s} {r[6]:.6f}"
              f"   ({r[1]})")


if __name__ == "__main__":
    raise SystemExit(main())
