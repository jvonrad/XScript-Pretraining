#!/usr/bin/env python
"""Extend the W&B BPB curves with points scored from uploaded checkpoints.

WHY THIS EXISTS
===============
`pull_wandb_curves.py` can only report what the trainer logged. Two kinds of
W&B logging gap have shown up so far, and this script fills both the same way
-- score the retrain's OWN checkpoint at the missing budget and verify it
against a point W&B DOES hold before trusting anything new:

1. **zh, 2026-09-01** (FLORES only -- its holdout was never rebuilt).
   Two Chinese runs were resumed under new W&B ids
   (`zh__*__neuron`, 11.76B -> 14.99B) with the in-loop eval switched off, so
   the last ~3B of both zh curves does not exist in W&B under any key.
   `zh-{fair,starved}-15b` (`step15865_14756M`) sit inside exactly that
   unlogged stretch. Control: `zh-{fair,starved}-12b` ARE the checkpoints
   behind the logged step-12811 points -- scoring them reproduces a number
   W&B already holds, to <1e-5 in both tokenizer conditions.

2. **de/starved, 2026-09-02 (FLORES) + 2026-09-03 (holdout).**
   The 2026-08-03 retrain resumed the SAME W&B id
   as the diverged original (CLAUDE.md 6h), so its own early points collided
   with the original's existing steps and W&B dropped them -- **below 7.75B,
   `de__unigram_starved` has no history under any key**, the same failure
   mode as (1) even though the mechanism (id collision, not eval-off) differs.
   `de-starved-{1,2,5}b` (`step{1092,2456,5181}_*M`) sit inside that unlogged
   stretch. Control: `de-starved-8b` IS the checkpoint behind the logged
   step-8451 point (the retrain's first surviving point) -- scoring it
   reproduces 1.0803, W&B's logged value, to <1e-5.

`run_bpb.py` scores them, and the result is the SAME quantity the trainer
logged, not a proxy -- both are FLORES+ **dev** (n=997, `bpb.py`'s
`flores.load_parallel(langs, "dev")`), same NLL/bytes definition. The control
point in each case is what licenses splicing the *other* points onto the
curve with no calibration offset -- it is not assumed, it is checked below
before anything is written.

    # score (see the module docstring of run_bpb.py for the workdir layout)
    XSCRIPT_FLORES=$WORK/xscript/flores_plus python run_bpb.py \
        --repo jvonrad/xscript-eval --langs zh --split dev --device cpu \
        --runs zh-fair-12b zh-starved-12b zh-fair-15b zh-starved-15b \
        --workdir $WORK
    XSCRIPT_FLORES=$WORK/xscript/flores_plus python run_bpb.py \
        --repo jvonrad/xscript-eval --langs de --split dev --device cpu \
        --runs de-starved-1b de-starved-2b de-starved-5b de-starved-8b \
        --workdir $WORK
    # then, ONE call covers everything currently scored under --bpb-dir --
    # the script regenerates the whole output file from every *_bpb.json
    # present, so zh's rows and de's rows coexist without re-running zh:
    python bpb_fill_from_checkpoints.py --bpb-dir $WORK/results/bpb

Writes `results/wandb_curves/bpb_curves_ckpt.csv`, same long schema as
`bpb_curves.csv` plus provenance columns. It is a SEPARATE file on purpose:
`bpb_curves.csv` is "whatever W&B holds, pulled reproducibly", and quietly
mixing a locally-computed point into it would destroy exactly the provenance
this directory exists to have. Join on `run` to plot them together;
`bts_from_wandb.py`'s `load()` does this automatically (its `--ckpt-csv`
defaults to this file's path).

HOLDOUT (2026-09-03): NO LONGER FLORES-ONLY, BUT ONLY WHERE A CONTROL PASSES
============================================================================
The trainer also logs `eval/holdout_*_bpb` from 500 docs of the reserved
FineWeb2-HQ holdout shard (`bpb.load_holdout`). Those shards died with the
Isambard-AI allocation, which is why this script was FLORES-only. They are,
however, *reconstructible*: the holdout is a deterministic function of the
public corpus (`files[0]` of the sorted manifest, first 30 MiB of text, never
entering the pool), so `rebuild_holdout.py` re-derives it by calling the pool
builder's own functions. `run_bpb.py --source holdout` then scores it with
`bpb.score_texts` -- the TRAINER's function, which windows long documents;
the FLORES fixed-shape path would truncate them instead, and the two are not
the same number.

⚠️ Reconstruction is only trustworthy where its own control passes, and the
control is per LANGUAGE (each language has its own manifest, and fr/ar draw
on FALLBACK_SOURCES once the primary is exhausted). Status:

  de  ✅ control PASSED at -5.84e-06 (de-starved-8b vs W&B's logged 0.9699255
         at step 8451). The de holdout fill below is licensed by that.
  zh, fr, ar, en  ❌ NOT ATTEMPTED. Do not assume the recipe transfers --
         rebuild and re-control per language before filling any of them.
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
    skipped = []
    for path in sorted(args.bpb_dir.glob("*_bpb.json")):
        d = json.loads(path.read_text())
        if d.get("limit"):
            # truncated eval (run_bpb.py --limit): not a curve point. These
            # appear when a checkpoint-STAGING run shares this directory.
            skipped.append(f"{path.name} (limit={d['limit']})")
            continue
        if "error" in d:
            skipped.append(f"{path.name} ({d['error'][:40]})")
            continue
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

    if skipped:
        print(f"[fill] ignored {len(skipped)} non-curve record(s): "
              + ", ".join(skipped[:6]) + (" ..." if len(skipped) > 6 else ""))

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
