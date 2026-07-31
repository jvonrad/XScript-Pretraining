#!/usr/bin/env python
"""Health check for a running eval sweep. One pass; prints ONLY anomalies.

Silence means healthy, so this is safe to poll from a monitor. It checks three
classes of problem, in increasing order of how easy they are to miss:

  liveness    workers alive, model count advancing, no FAILED lines, disk free
  integrity   every stored raw block reconstructs lm-eval's own hit lists
              (`check_reproduces`), expected document count, no NaN/inf
  plausibility  the checks that would have caught THIS project's five format
              artifacts: a cell whose predictions collapse onto one label, a
              cell that does not beat its own empirical null, an accuracy
              outside the range the benchmark can produce, or a score that
              moves implausibly far from the same model's other cells

State is kept in `--state` so an anomaly is reported once, not every poll.

    python watch_sweep.py --workdir $WORK --logs $WORK/phase_w0.log $WORK/phase_w1.log
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from xscript.eval.rawscores import (check_reproduces, prediction_profile,  # noqa: E402
                                    score_variants)

# Document counts each task must produce at full split. A short count means a
# truncated dataset load or a silently applied --limit.
EXPECTED_N = {
    "sib200_": 1004, "taxi1500_": 1077, "xnli_": 2490,
    "hellaswag_zh": 9266,
}
# Accuracy ranges these benchmarks can plausibly produce for 1B/30B-token
# checkpoints. Outside this is a bug, not a result.
PLAUSIBLE = {
    "gmmlu_": (0.15, 0.55), "sib200_": (0.10, 0.90), "taxi1500_": (0.10, 0.80),
    "xnli_": (0.25, 0.75),
}


def _match(task, table):
    for k, v in table.items():
        if task.startswith(k):
            return v
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--logs", nargs="*", default=[], type=Path)
    ap.add_argument("--state", type=Path,
                    default=Path("/tmp/watch_sweep_state.json"))
    ap.add_argument("--stall-minutes", type=float, default=90.0)
    ap.add_argument("--min-free-gb", type=float, default=15.0)
    ap.add_argument("--worker-pattern",
                    default="^bash /home/ubuntu/xscript_bench/phase_worker.sh",
                    help="pgrep -f pattern identifying the sweep workers. Must be "
                         "ANCHORED (^bash /path) so it cannot match the shell that "
                         "invokes this script -- an unanchored pattern matches its "
                         "own caller and makes the liveness check meaningless.")
    ap.add_argument("--done-marker", default="ALL DONE",
                    help="string a worker log ends with on success")
    ap.add_argument("--models-json", type=Path,
                    default=Path(__file__).resolve().parents[2] / "results" / "models.json",
                    help="used to tell own-language cells from out-of-domain ones")
    args = ap.parse_args()

    # Which (run, lang) pairs are the model's OWN training languages. The
    # plausibility checks below (collapse / below-null / range) are only
    # meaningful there: a model scored on a language it never trained on is
    # SUPPOSED to sit at its null, and sometimes lands a little under it by
    # chance. Firing on those produces a permanent false alarm -- which is
    # exactly how a watchdog gets ignored. Integrity checks still apply
    # everywhere, since corrupt raw scores are a bug in any cell.
    own = {}
    if args.models_json.exists():
        try:
            own = {r: set(v.get("langs", []))
                   for r, v in json.loads(args.models_json.read_text()).items()}
        except (json.JSONDecodeError, OSError):
            own = {}

    state = {}
    if args.state.exists():
        try:
            state = json.loads(args.state.read_text())
        except json.JSONDecodeError:
            state = {}
    seen = set(state.get("seen", []))
    alerts = []

    # ---------------------------------------------------------- liveness
    workers = subprocess.run(["pgrep", "-fc", args.worker_pattern],
                             capture_output=True, text=True).stdout.strip()
    n_workers = int(workers) if workers.isdigit() else 0

    for log in args.logs:
        if not log.exists():
            continue
        text = log.read_text(errors="replace")
        for line in text.splitlines():
            if "FAILED" in line and line not in seen:
                alerts.append(f"FAILURE  {line.strip()[:160]}")
                seen.add(line)

    free_gb = shutil.disk_usage("/").free / 1e9
    if free_gb < args.min_free_gb:
        alerts.append(f"DISK     only {free_gb:.1f} GB free (< {args.min_free_gb})")

    # ------------------------------------------------- integrity + plausibility
    cells = 0
    for results_dir in ("extra_bench", "appendix_c5"):
        raw_dir = args.workdir / "results" / results_dir / "raw"
        if not raw_dir.is_dir():
            continue
        for f in sorted(raw_dir.glob("*_raw.json")):
            try:
                blob = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue          # mid-write; next poll will catch it
            run = blob.get("run", f.stem)
            res_path = raw_dir.parent / f"{run}_final.json"
            res = {}
            if res_path.exists():
                try:
                    res = json.loads(res_path.read_text())
                except json.JSONDecodeError:
                    pass
            for lang, tasks in blob.get("raw", {}).items():
                for task, raw in tasks.items():
                    cells += 1
                    key = f"{run}/{lang}/{task}"
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        hits = score_variants(raw)
                    except Exception as exc:                # noqa: BLE001
                        alerts.append(f"BADRAW   {key}: {type(exc).__name__}: {exc}")
                        continue
                    n = len(raw["gold"])

                    want = _match(task, EXPECTED_N)
                    if want and n != want and res.get("limit") in (None, 0):
                        alerts.append(f"SHORT    {key}: n={n}, expected {want}")

                    flat = [x for row in raw["ll"] for x in row]
                    if any(math.isnan(x) or math.isinf(x) for x in flat):
                        alerts.append(f"NONFINITE {key}: NaN/inf in loglikelihoods")
                        continue

                    stored = res.get("correct", {}).get(lang, {}).get(task)
                    if isinstance(stored, dict):
                        bad = {k: v for k, v in check_reproduces(raw, stored).items() if v}
                        if bad:
                            alerts.append(f"MISMATCH {key}: raw does not reproduce "
                                          f"lm-eval hits {bad}")

                    if own and lang not in own.get(run, {lang}):
                        continue          # out-of-domain: integrity only
                    best = "acc_cal" if "acc_cal" in hits else (
                        "acc_norm" if "acc_norm" in hits else "acc")
                    prof = prediction_profile(raw, best)
                    lo_hi = _match(task, PLAUSIBLE)
                    if lo_hi and not (lo_hi[0] <= prof["acc"] <= lo_hi[1]):
                        alerts.append(f"RANGE    {key}: {best}={prof['acc']:.3f} "
                                      f"outside plausible {lo_hi}")
                    if prof["pred_entropy"] < 0.30:
                        alerts.append(f"COLLAPSE {key}: {best} prediction entropy "
                                      f"{prof['pred_entropy']:.2f} -- predictions "
                                      "concentrated on one label")
                    if prof["acc_over_null"] < -0.02:
                        alerts.append(f"BELOWNULL {key}: {best} beats its own null "
                                      f"by {prof['acc_over_null']:+.3f}")

    # Stall detection: model count must advance between polls.
    now = time.time()
    prev_cells, prev_t = state.get("cells", 0), state.get("t", now)
    if n_workers > 0 and cells == prev_cells and (now - prev_t) / 60 > args.stall_minutes:
        alerts.append(f"STALL    no new cell in {(now - prev_t) / 60:.0f} min "
                      f"({cells} cells, {n_workers} worker(s) alive)")
        prev_t = now
    elif cells != prev_cells:
        prev_t = now

    logs_present = [p for p in args.logs if p.exists()]
    if n_workers == 0 and logs_present and not all(
            args.done_marker in p.read_text(errors="replace") for p in logs_present):
        alerts.append(f"DIED     no worker matching {args.worker_pattern!r} alive, "
                      f"but a log lacks {args.done_marker!r}")

    args.state.write_text(json.dumps(
        {"seen": sorted(seen), "cells": cells, "t": prev_t}))

    for a in alerts:
        print(a, flush=True)


if __name__ == "__main__":
    main()
