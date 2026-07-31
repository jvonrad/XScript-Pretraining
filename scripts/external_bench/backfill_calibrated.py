#!/usr/bin/env python
"""Inject calibrated hit lists into the per-model result JSONs, from the raw
sidecars. Pure CPU, idempotent, no accelerator.

`analyze_extra_bench.py` and `bootstrap_transfer.py` already implement the
whole of CLAUDE.md §6d -- both baselines, the constant-prediction check, the
LR-mismatch flagging, the paired bootstrap. None of that needs rewriting to
use the corrected estimator; they just need the hit list to exist under a
metric name they can select. This walks `<results>/raw/*_raw.json`, re-derives
every estimator in `rawscores.VARIANTS`, and writes the ones that are missing
into each model's `correct` block:

    extra_bench:  correct[lang][task][metric] = [0/1, ...]   <- metric-keyed
                  already, so `acc_cal` slots in beside `acc` / `acc_norm` /
                  `acc_mutual_info` and nothing downstream breaks.

    appendix_c5:  correct[lang][task] is a FLAT list (the published
                  estimator). Changing its shape would break
                  bootstrap_transfer.py, so the calibrated lists go to a
                  sibling key `correct_calibrated[lang][task][metric]`
                  instead, and the flat list is left exactly as it was.

Run after a sweep:

    python backfill_calibrated.py $WORK/results/extra_bench
    python backfill_calibrated.py $WORK/results/appendix_c5

Re-running is safe: existing keys are overwritten with identical values, and
a model with no raw sidecar is skipped rather than emptied.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from xscript.eval.rawscores import (VARIANTS, check_reproduces,  # noqa: E402
                                    score_variants)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw_dir = args.results_dir / "raw"
    if not raw_dir.is_dir():
        sys.exit(f"no raw sidecars under {raw_dir}")

    n_models = n_cells = n_bad = 0
    for raw_file in sorted(raw_dir.glob("*_raw.json")):
        run = json.loads(raw_file.read_text()).get("run") or \
            raw_file.stem.removesuffix("_raw")
        blob = json.loads(raw_file.read_text())["raw"]
        res_path = args.results_dir / f"{run}_final.json"
        if not res_path.exists():
            print(f"[backfill] {run}: no result JSON, skipping")
            continue
        res = json.loads(res_path.read_text())
        if "error" in res:
            continue
        flat_c5 = any(isinstance(v, list)
                      for ts in res.get("correct", {}).values()
                      for v in ts.values())
        target_key = "correct_calibrated" if flat_c5 else "correct"
        res.setdefault(target_key, {})
        changed = False
        for lang, tasks in blob.items():
            for task, raw in tasks.items():
                hits = score_variants(raw)
                # Guard: the stored raw must still reproduce whatever lm-eval
                # recorded, or the derived hits are meaningless.
                existing = res.get("correct", {}).get(lang, {}).get(task)
                if isinstance(existing, dict):
                    bad = {k: v for k, v in check_reproduces(raw, existing).items() if v}
                    if bad:
                        print(f"[backfill] MISMATCH {run}/{lang}/{task}: {bad} "
                              "-- leaving this cell alone")
                        n_bad += 1
                        continue
                dest = res[target_key].setdefault(lang, {})
                slot = dest.setdefault(task, {}) if flat_c5 else dest.setdefault(task, {})
                for v in VARIANTS:
                    if v in hits:
                        slot[v] = hits[v]
                changed = True
                n_cells += 1
        if changed and not args.dry_run:
            tmp = res_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(res, indent=2))
            tmp.replace(res_path)
        n_models += 1

    print(f"[backfill] {n_models} model(s), {n_cells} cell(s) updated"
          + (f", {n_bad} cell(s) SKIPPED on reproduction mismatch" if n_bad else "")
          + (" (dry run)" if args.dry_run else ""))
    if n_bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
