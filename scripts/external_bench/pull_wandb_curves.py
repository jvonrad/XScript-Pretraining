#!/usr/bin/env python
"""Pull dense BPB-vs-tokens curves from W&B into results/wandb_curves/.

This is the puller `results/wandb_curves/README.md` says to use; it was never
actually committed (`bts_from_wandb.py` only *reads* the `histories.json` this
produces), so the cached CSV had no reproducible provenance. It does now.

    export WANDB_API_KEY=...        # or `wandb login`
    python pull_wandb_curves.py [--project jonathan-von-rad/XScript-Pretraining]

Writes, into `results/wandb_curves/`:
    histories.json   {run_id: {name, tokens_per_step, history: [...]}}  <- bts_from_wandb.py
    bpb_curves.csv   long format: run, step, tokens, metric, value
    runs_meta.json   per-run state, tokens_per_step, final tokens_b

TWO THINGS THIS HANDLES THAT A NAIVE PULL GETS WRONG
====================================================

1. **tokens_b and eval/*_bpb are logged in SEPARATE `wandb.log()` calls**, so
   they land on different steps. Requiring both on one row silently drops most
   eval points -- CLAUDE.md 6 records this costing the interaction once
   (`en-fr__unigram_destarved` looked like a 2-point run when it has 29).
   Eval rows are therefore pulled on their own and tokens reconstructed as
   `step x tokens_per_step`, with the ratio taken as the median over rows that
   *do* carry both. The relation is exactly linear.

2. **SPLICED RUNS.** `de__unigram_starved` is one W&B id containing TWO
   different training runs. The original diverged at the warmup/peak-LR seam
   (CLAUDE.md 6h) and ran to step 7361; the 2026-08-03 retrain resumed the
   same W&B id, so its early points collided with existing steps and were
   dropped, and only its history past the original's last step survived. The
   result is a single page whose BPB falls 1.3012 -> 1.0803 in one eval
   interval -- a 0.221 drop that is not learning, it is the model changing
   identity. Read as one curve it mixes a diverged run with a healthy one.

   `RUN_MIN_STEP` cuts each spliced run at its seam. Verified for this one:
   step x 917,504 reproduces 6h's documented retrain budgets on all three
   checkpoints (7.754/11.754/14.754 B against 7.753/11.754/14.755), so the
   retrain's step counter is its own and NO token offset is needed.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "wandb_curves"

# run name -> first step that belongs to the run we actually want.
# See the module docstring, point 2. Keep the reason with the number.
RUN_MIN_STEP = {
    # Steps <=7361 are the DIVERGED original (BPB 1.55->1.28->1.74->1.30).
    # Steps >=8451 are the 2026-08-03 retrain (1.0803 -> 1.0472, monotone).
    "de__unigram_starved": 8451,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="jonathan-von-rad/XScript-Pretraining")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    import wandb
    api = wandb.Api()
    runs = list(api.runs(args.project))
    print(f"[wandb] {len(runs)} runs in {args.project}")

    histories, meta, csv_rows = {}, {}, []
    for r in runs:
        # --- tokens_per_step, from rows that carry BOTH fields --------------
        ratios = []
        for h in r.scan_history(keys=["_step", "tokens_b"], page_size=10000):
            s, t = h.get("_step"), h.get("tokens_b")
            if s and t:
                ratios.append(t * 1e9 / s)
        tps = statistics.median(ratios) if ratios else None

        # --- eval rows, pulled ON THEIR OWN ---------------------------------
        keys = ["_step"] + [k for k in r.summary.keys()
                            if k.startswith("eval/") and k.endswith("_bpb")]
        hist, n_eval = [], 0
        if len(keys) > 1 and tps:
            cut = RUN_MIN_STEP.get(r.name, 0)
            for h in r.scan_history(keys=keys, page_size=10000):
                step = h.get("_step")
                vals = {k: v for k, v in h.items()
                        if k.startswith("eval/") and k.endswith("_bpb") and v is not None}
                if step is None or not vals:
                    continue
                n_eval += 1
                if step < cut:
                    continue          # spliced-run seam; see RUN_MIN_STEP
                tokens_b = step * tps / 1e9
                hist.append({"step": step, "tokens_b": tokens_b, **vals})
                for k, v in vals.items():
                    csv_rows.append((r.name, step, step * tps, k, v))

        # Assert what CLAUDE.md 6's gotcha demands: every eval row we saw is
        # either kept or deliberately cut, never silently lost to a join.
        kept = len(hist)
        dropped = n_eval - kept
        histories[r.id] = {"name": r.name, "tokens_per_step": tps, "history": hist}
        meta[r.id] = {"name": r.name, "state": r.state, "tokens_per_step": tps,
                      "final_tokens_b": hist[-1]["tokens_b"] if hist else None,
                      "n_eval_points": kept, "n_cut_at_seam": dropped}
        flag = f"  CUT {dropped} pre-seam rows" if dropped else ""
        print(f"  {r.name:32s} {r.state:9s} eval={kept:4d}"
              f" tps={(f'{tps:,.0f}' if tps else '-'):>9s}{flag}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "histories.json").write_text(json.dumps(histories, indent=1))
    (args.out / "runs_meta.json").write_text(json.dumps(meta, indent=1))
    with open(args.out / "bpb_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "step", "tokens", "metric", "value"])
        for row in sorted(csv_rows):
            w.writerow(row)
    print(f"\nwrote {len(csv_rows)} curve points to {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
