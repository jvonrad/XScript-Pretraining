#!/usr/bin/env python
"""Re-derive every scoring rule from the stored raw loglikelihoods, on CPU.

The runners (`run_extra_bench.py`, `run_appendix_c5.py`) now persist each
document's per-candidate loglikelihoods to `<results>/raw/<run>_raw.json`.
This script turns those into:

  * every estimator in `xscript.eval.rawscores.VARIANTS` -- acc, acc_norm,
    acc_tokennorm, acc_pmi, acc_cal, acc_cal_loo, acc_cal_pmi -- so the scoring
    rule can be
    changed without another accelerator pass;
  * a **degeneracy report**: the predicted-label distribution and its
    normalized entropy, plus per-gold-class recall. This is what shows that
    lm-eval's shipped `acc` and `acc_norm` are not measuring classification on
    SIB-200 at all (see `rawscores` module docstring);
  * **trajectory monotonicity** over a token-budget series, the cheapest
    available check on whether an estimator is measuring the model or the
    label prior;
  * transfer deltas with the paired bootstrap `bootstrap_transfer.py` uses,
    computed under whichever estimator is selected.

Pure stdlib. Usage:

    python analyze_raw_scores.py $WORK/results/extra_bench          # all reports
    python analyze_raw_scores.py $WORK/results/extra_bench --report degeneracy
    python analyze_raw_scores.py $WORK/results/appendix_c5 --tasks xnli_de
"""
import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from xscript.eval.rawscores import (VARIANTS, degeneracy,  # noqa: E402
                                    prediction_profile, score_variants)


def load(results_dir: Path) -> dict:
    """{run: {lang: {task: raw_block}}} from the raw sidecars."""
    raw_dir = results_dir / "raw"
    if not raw_dir.is_dir():
        sys.exit(f"no raw sidecars in {raw_dir} -- re-run the sweep with the "
                 "current runner (it stores per-choice loglikelihoods).")
    out = {}
    for f in sorted(raw_dir.glob("*_raw.json")):
        d = json.loads(f.read_text())
        out[d.get("run", f.stem.removesuffix("_raw"))] = d["raw"]
    if not out:
        sys.exit(f"no *_raw.json under {raw_dir}")
    return out


def budget(run: str) -> int:
    """Token budget in B from the run-name suffix; unsuffixed finals are 30B."""
    m = re.search(r"-(\d+)b$", run)
    return int(m.group(1)) if m else 30


def family(run: str) -> str:
    return re.sub(r"-\d+b$", "", run)


# ------------------------------------------------------------------ reports

def report_variants(data: dict, tasks: list[str] | None) -> None:
    print("=" * 100)
    print("ACCURACY UNDER EVERY ESTIMATOR")
    print("  acc            raw summed loglikelihood         (lm-eval `acc`)")
    print("  acc_norm       / character length               (lm-eval `acc_norm`)")
    print("  acc_tokennorm  / token count")
    print("  acc_pmi        - log P(candidate)               (lm-eval `acc_mutual_info`)")
    print("  acc_cal        - mean_docs log P(candidate|doc) <-- quote this for a")
    print("                                                     shared choice set")
    print("  acc_cal_loo    calibrated, leave-one-out (non-transductive check)")
    print("  acc_cal_pmi    calibrated PMI (== acc_cal for a shared choice set)")
    print("=" * 100)
    for run in sorted(data):
        for lang in sorted(data[run]):
            for task, raw in sorted(data[run][lang].items()):
                if tasks and task not in tasks:
                    continue
                hits = score_variants(raw)
                n = len(raw["gold"])
                cells = "  ".join(
                    f"{v}={sum(hits[v]) / n:.3f}" for v in VARIANTS if v in hits)
                print(f"{run:24s} {lang:3s} {task:28s} n={n:6d}  {cells}")


def report_degeneracy(data: dict, tasks: list[str] | None,
                      variants: tuple[str, ...] = ("acc", "acc_norm", "acc_cal")) -> None:
    print("=" * 100)
    print("DEGENERACY: is the argmax reading the document, or the label prior?")
    print("  null     accuracy this prediction DISTRIBUTION would get if it were")
    print("           independent of gold: sum_c P(pred c) P(gold c). The honest")
    print("           chance level -- not 1/k, and not the majority rate.")
    print("  over     acc - null: what the document actually bought")
    print("  pred_H   normalized entropy of the PREDICTED-label distribution")
    print("           (1.00 = uniform over choices, 0.00 = one label for every doc)")
    print("  n_rec    how many gold classes are recalled above 10%")
    print("  CONST    hit vector is exactly `gold == c` -- scored a class")
    print("           frequency having learned nothing")
    print("=" * 100)
    for run in sorted(data):
        for lang in sorted(data[run]):
            for task, raw in sorted(data[run][lang].items()):
                if tasks and task not in tasks:
                    continue
                if raw.get("n_choices", -1) <= 0:
                    continue          # ragged choice set: no shared label index
                hits = score_variants(raw)
                row = [f"{run:24s} {lang:3s} {task:28s}"]
                for v in variants:
                    if v not in hits:
                        continue
                    prof = prediction_profile(raw, v)
                    deg = degeneracy(raw, hits[v])
                    flag = " CONST" if deg["constant"] else ""
                    row.append(f"{v}: acc={prof['acc']:.3f} null={prof['null']:.3f} "
                               f"over={prof['acc_over_null']:+.3f} "
                               f"pred_H={prof['pred_entropy']:.2f} "
                               f"n_rec={deg['n_recalled']}{flag}")
                print("  ".join(row))


def report_recall(data: dict, task: str, variant: str) -> None:
    print("=" * 100)
    print(f"PER-GOLD-CLASS RECALL  task={task}  estimator={variant}")
    print("=" * 100)
    for run in sorted(data):
        for lang in sorted(data[run]):
            raw = data[run][lang].get(task)
            if raw is None:
                continue
            hits = score_variants(raw)
            if variant not in hits:
                continue
            deg = degeneracy(raw, hits[variant])
            k = raw["n_choices"]
            rec = " ".join(f"{deg['recall'][c]:5.2f}" if deg["recall"][c] is not None
                           else "    -" for c in range(k))
            acc = sum(hits[variant]) / len(hits[variant])
            print(f"{run:24s} {lang:3s} acc={acc:.3f}  {rec}")


def report_trajectory(data: dict, tasks: list[str] | None) -> None:
    """Backwards movement along each family's token-budget series.

    A learning curve should not go down as tokens increase. Summed downward
    steps, divided by the curve's total range, is a scale-free "how much of
    this estimator's signal is noise" number -- and it is the check that
    caught the SIB-200 problem in the first place (en-starved English: .609
    @12B, .537 @15B, .581 @30B under `acc`).
    """
    print("=" * 100)
    print("TRAJECTORY MONOTONICITY over each family's token-budget series")
    print("  drops   summed downward movement (0 = perfectly monotone)")
    print("  range   max - min over the series")
    print("  ratio   drops / range -- lower is better")
    print("=" * 100)
    series = defaultdict(lambda: defaultdict(dict))   # (fam,lang,task)[budget][variant]
    for run in data:
        for lang in data[run]:
            for task, raw in data[run][lang].items():
                if tasks and task not in tasks:
                    continue
                hits = score_variants(raw)
                for v, h in hits.items():
                    series[(family(run), lang, task)].setdefault(v, {})[budget(run)] = \
                        sum(h) / len(h)
    agg = defaultdict(lambda: [0.0, 0.0, 0])
    for key, by_variant in sorted(series.items()):
        for v, by_budget in by_variant.items():
            if len(by_budget) < 3:
                continue
            xs = [by_budget[b] for b in sorted(by_budget)]
            drops = sum(max(0.0, xs[i] - xs[i + 1]) for i in range(len(xs) - 1))
            rng = max(xs) - min(xs)
            agg[v][0] += drops
            agg[v][1] += rng
            agg[v][2] += 1
    print(f"{'estimator':16s} {'series':>7s} {'mean drops':>11s} {'mean range':>11s} {'ratio':>8s}")
    for v in VARIANTS:
        if v not in agg:
            continue
        d, r, n = agg[v]
        print(f"{v:16s} {n:7d} {d / n:11.4f} {r / n:11.4f} "
              f"{(d / r if r else float('nan')):8.2f}")


def paired_bootstrap(hits_a: list[int], hits_b: list[int], b: int = 2000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Mean(a) - mean(b) with a 95% CI, resampling doc indices jointly.

    Same estimator as bootstrap_transfer.py: both models scored the identical
    fixed doc order, so one resample applies to both.
    """
    n = len(hits_a)
    rnd = random.Random(seed)
    point = sum(hits_a) / n - sum(hits_b) / n
    reps = []
    for _ in range(b):
        idx = [rnd.randrange(n) for _ in range(n)]
        reps.append(sum(hits_a[i] for i in idx) / n - sum(hits_b[i] for i in idx) / n)
    reps.sort()
    return point, reps[int(0.025 * b)], reps[int(0.975 * b) - 1]


def report_transfer(data: dict, pairs: list[tuple[str, str, str, str]],
                    variant: str) -> None:
    """bilingual - monolingual on a named (partner-lang, task) cell."""
    print("=" * 100)
    print(f"TRANSFER DELTAS (bilingual - monolingual), estimator={variant}")
    print("=" * 100)
    print(f"{'label':28s} {'lang':4s} {'task':24s} {'mono':>6s} {'bi':>6s} "
          f"{'delta':>7s}  95% CI")
    for label, mono, bi, spec in pairs:
        lang, task = spec.split(":", 1)
        try:
            ra = data[bi][lang][task]
            rb = data[mono][lang][task]
        except KeyError:
            print(f"{label:28s} {lang:4s} {task:24s}  (missing raw for "
                  f"{bi!r} or {mono!r})")
            continue
        ha = score_variants(ra).get(variant)
        hb = score_variants(rb).get(variant)
        if ha is None or hb is None or len(ha) != len(hb):
            print(f"{label:28s} {lang:4s} {task:24s}  (estimator unavailable "
                  "or doc counts differ)")
            continue
        d, lo, hi = paired_bootstrap(ha, hb)
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"{label:28s} {lang:4s} {task:24s} {sum(hb)/len(hb):6.3f} "
              f"{sum(ha)/len(ha):6.3f} {d:+7.3f}{star} [{lo:+.3f}, {hi:+.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path,
                    help="$WORK/results/extra_bench or .../appendix_c5")
    ap.add_argument("--report", nargs="*",
                    default=["variants", "degeneracy", "trajectory"],
                    choices=["variants", "degeneracy", "trajectory", "recall",
                             "transfer"])
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="restrict to these task names")
    ap.add_argument("--variant", default="acc_cal",
                    help="estimator for the recall/transfer reports")
    ap.add_argument("--pairs", type=Path, default=None,
                    help="JSON list of [label, mono_run, bi_run, 'lang:task'] "
                         "for --report transfer")
    args = ap.parse_args()

    data = load(args.results_dir)
    print(f"[raw] {len(data)} run(s) from {args.results_dir / 'raw'}\n")
    if "variants" in args.report:
        report_variants(data, args.tasks)
        print()
    if "degeneracy" in args.report:
        report_degeneracy(data, args.tasks)
        print()
    if "trajectory" in args.report:
        report_trajectory(data, args.tasks)
        print()
    if "recall" in args.report:
        for t in (args.tasks or []):
            report_recall(data, t, args.variant)
            print()
    if "transfer" in args.report:
        if not args.pairs:
            sys.exit("--report transfer needs --pairs")
        report_transfer(data, [tuple(p) for p in json.loads(args.pairs.read_text())],
                        args.variant)


if __name__ == "__main__":
    main()
