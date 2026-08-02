#!/usr/bin/env python
"""Unattended plausibility check over a running 68-checkpoint sweep.

Prints ONE line per (model, lang, task) cell plus an ANOMALY line for anything
that looks wrong, so a human returning to a finished sweep can grep for
"ANOMALY" instead of reading 100 cells.  Pure CPU, no accelerator, safe to run
against a directory that is still being written.

What counts as an anomaly, and why each one is here rather than "accuracy looks
low" (CLAUDE.md 6/6d/6e -- five format artifacts, every one of which produced
plausible-looking accuracy):

  CONSTANT     the hit vector is exactly `gold == c`: the cell ranked one label
               first for every document and "scored" that class's frequency
               having learned nothing.  16 of 574 cells did this in 6d.
  BELOW-NULL   accuracy under the cell's OWN empirical null
               `sum_c P(pred c) P(gold c)`.  Nominal chance is the wrong
               baseline: a constant predictor scores its class frequency, which
               on StoryCloze is .533, not .500.
  LOW-ENTROPY  prediction entropy far under what this task's arity permits.
               Thresholds are per-task because ragged sets cannot reach 1.0 --
               ARC-Easy's ceiling is 0.863 (3/4/5 options) and flagging it at
               0.9 would fire on every healthy cell.
  RAW-MISMATCH the stored raw loglikelihoods no longer reconstruct lm-eval's
               own hit lists.  Nothing derived is trustworthy if this fires.
  NO-RAW       scored before the raw sidecar existed, so no estimator other
               than the one lm-eval reported can ever be derived for it.

Usage:
    python health_check.py <workdir> [--quiet]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from xscript.eval.rawscores import (  # noqa: E402
    check_reproduces, degeneracy, has_shared_choices, prediction_profile,
    score_variants,
)

# Per-task entropy floor.  These come from a FULL arity scan of each pool, not
# from nominal chance: a perfectly uniform predictor cannot exceed these.
#   arceasy  {3:7, 4:2348, 5:4}          -> ceiling 0.863
#   bmlama   {2:22,3:45,5:10,6:233,8:141,9:69,10:5496} -> ceiling 0.999
# Flag at roughly two thirds of the ceiling; below that a cell is collapsing
# onto a handful of labels regardless of arity.
ENTROPY_FLOOR = {
    "sib200": 0.60, "xnli": 0.60, "taxi1500": 0.60,
    "mub_arceasy": 0.55,      # ceiling 0.863
    "mub_storycloze": 0.60,
    "mub_hellaswag": 0.60,
    "mub_bmlama": 0.55,       # ragged 2-10, most docs 10-way
}

# Which estimator each family is quoted on (CLAUDE.md 6e "what to quote now").
PREFERRED = {
    "sib200": "acc_cal", "xnli": "acc_cal", "taxi1500": "acc_cal",
    "mub_arceasy": "acc_norm", "mub_storycloze": "acc_norm",
    "mub_hellaswag": "acc_norm", "mub_bmlama": "acc_norm",
}


def family_of(task: str) -> str:
    for fam in PREFERRED:
        if task.startswith(fam):
            return fam
    return task


def empirical_null(raw: dict, variant: str) -> float:
    """sum_c P(pred c) * P(gold c) -- the accuracy this prediction DISTRIBUTION
    would get if its predictions were independent of gold."""
    prof = prediction_profile(raw, variant)
    frac = prof["pred_frac"]
    gold = raw["gold"]
    n = len(gold)
    k = len(frac)
    gfrac = [0.0] * k
    for g in gold:
        if 0 <= g < k:
            gfrac[g] += 1.0 / n
    return sum(p * q for p, q in zip(frac, gfrac))


def main() -> None:
    work = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/xscript_bench")
    quiet = "--quiet" in sys.argv
    anomalies = 0
    cells = 0

    for sub in ("extra_bench", "appendix_c5"):
        res = work / "results" / sub
        if not res.is_dir():
            continue
        for f in sorted(res.glob("*_final.json")):
            run = f.name.removesuffix("_final.json")
            try:
                d = json.loads(f.read_text())
            except json.JSONDecodeError:
                print(f"ANOMALY {run:22s} {sub}: result JSON unreadable (mid-write?)")
                anomalies += 1
                continue
            if "error" in d:
                print(f"ANOMALY {run:22s} {sub}: error stub -- {d['error'][:80]}")
                anomalies += 1
                continue
            rawf = res / "raw" / f"{run}_raw.json"
            rawall = {}
            if rawf.exists():
                try:
                    rawall = json.loads(rawf.read_text()).get("raw", {})
                except json.JSONDecodeError:
                    pass

            for lang, tasks in (d.get("correct") or {}).items():
                for task, hits_by_metric in tasks.items():
                    cells += 1
                    fam = family_of(task)
                    raw = (rawall.get(lang) or {}).get(task)
                    if raw is None:
                        print(f"ANOMALY {run:22s} {lang} {task:24s} NO-RAW "
                              "(cannot be recalibrated)")
                        anomalies += 1
                        continue

                    mism = {k: v for k, v in
                            check_reproduces(raw, hits_by_metric).items() if v}
                    if mism:
                        print(f"ANOMALY {run:22s} {lang} {task:24s} "
                              f"RAW-MISMATCH {mism}")
                        anomalies += 1

                    variants = score_variants(
                        raw, shared_choices=has_shared_choices(task))
                    want = PREFERRED.get(fam, "acc_norm")
                    if want not in variants:
                        want = "acc_norm" if "acc_norm" in variants else "acc"
                    hits = variants[want]
                    acc = sum(hits) / len(hits)
                    null = empirical_null(raw, want)
                    prof = prediction_profile(raw, want)
                    deg = degeneracy(raw, hits)
                    ent = prof["pred_entropy"]

                    flags = []
                    if deg["constant"]:
                        flags.append("CONSTANT")
                    if acc < null:
                        flags.append("BELOW-NULL")
                    if ent < ENTROPY_FLOOR.get(fam, 0.5):
                        flags.append("LOW-ENTROPY")

                    line = (f"{run:22s} {lang} {task:24s} {want:9s} "
                            f"acc={acc:.4f} null={null:.4f} "
                            f"over={acc - null:+.4f} ent={ent:.3f} "
                            f"nrec={deg['n_recalled']} n={len(hits)}")
                    if flags:
                        print(f"ANOMALY {line}  <-- {' '.join(flags)}")
                        anomalies += 1
                    elif not quiet:
                        print(f"        {line}")

    print(f"\n[health] {cells} cells checked, {anomalies} anomalies")


if __name__ == "__main__":
    main()
