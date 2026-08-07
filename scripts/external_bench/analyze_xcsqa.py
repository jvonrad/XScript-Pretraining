#!/usr/bin/env python
"""X-CSQA reports: estimator choice, capability tables, transfer deltas.

Pure stdlib, reads only the raw sidecars, so every number here is a CPU
re-derivation and no scoring question costs another accelerator pass
(CLAUDE.md 6e).

Three reports:

  estimator  Judged on CLAUDE.md 6g's criteria and never on gold accuracy:
             discrimination over the EMPIRICAL null, prediction entropy, and
             trajectory monotonicity over the token-budget series. 6g's rule
             predicts `acc` for X-CSQA -- its candidates are short fixed
             phrases (median 9 chars, the ARC-Easy/BMLAMA regime) rather than
             long free-form continuations (HellaSwag/StoryCloze). Note
             `acc_tokennorm` is reported ONLY to show it is never selected;
             6g proves it is tokenizer-dependent and so must never be used in
             a project whose contrast IS a tokenizer.

  capability Own-language accuracy by language and budget: does cross-script
             cost attained capability?

  transfer   bilingual - monolingual, paired bootstrap over documents, at the
             LR-matched tiers. Every 1b-15b checkpoint is mid-stable at peak
             LR 3.0e-3 (decay starts at 24B), so mono X B/lang vs bilingual
             2X B total is LR-matched BY CONSTRUCTION -- the confound
             CLAUDE.md 6/6d call their biggest weakness does not apply.
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
from xscript.eval.rawscores import score_variants, prediction_profile  # noqa: E402

LANGS = ("en", "de", "fr", "ar", "zh")
SCRIPT = {"de": "same", "fr": "same", "ar": "cross", "zh": "cross"}
ESTIMATORS = ("acc", "acc_norm", "acc_pmi", "acc_tokennorm")


def load(results_dir: Path) -> dict:
    out = {}
    for p in sorted((results_dir / "raw").glob("*_raw.json")):
        d = json.loads(p.read_text())
        for lang, tasks in d.get("raw", {}).items():
            for task, block in tasks.items():
                if task.startswith("xcsqa"):
                    out.setdefault(d["run"], {}).setdefault(lang, {})[task] = block
    return out


def hits(block: dict, est: str) -> list[int]:
    """Per-document 0/1 correctness under one estimator.

    `shared_choices=False` is passed explicitly: X-CSQA candidates are
    per-document text, so the acc_cal family must stay withheld (CLAUDE.md 6e).
    The sidecars already record the flag; this makes it non-negotiable here.
    """
    return score_variants(block, shared_choices=False)[est]


def profile(block: dict, est: str) -> dict:
    """acc, empirical null (sum_c P(pred c) P(gold c)) and entropy.

    `rawscores.prediction_profile` covers every estimator except
    `acc_tokennorm`, which it deliberately does not implement -- CLAUDE.md 6g
    proves that estimator is tokenizer-dependent and must never be selected in
    this project. It is computed here ONLY so the comparison table can show it
    losing rather than silently omitting it. X-CSQA is uniformly 5-way, so the
    non-ragged branch of the upstream formula applies exactly.
    """
    if est != "acc_tokennorm":
        return prediction_profile(block, est)
    gold, ll, ntok = block["gold"], block["ll"], block["ntok"]
    scores = [[x / n for x, n in zip(r, nt)] for r, nt in zip(ll, ntok)]
    n, k = len(scores), max(len(r) for r in scores)
    preds = [max(range(len(r)), key=r.__getitem__) for r in scores]
    frac = [preds.count(c) / n for c in range(k)]
    gold_frac = [sum(1 for g in gold if g == c) / n for c in range(k)]
    null = sum(frac[c] * gold_frac[c] for c in range(k))
    acc = sum(int(p == g) for p, g in zip(preds, gold)) / n
    ent = abs(-sum(p * math.log(p) for p in frac if p > 0) / math.log(k))
    return {"acc": acc, "null": null, "acc_over_null": acc - null,
            "pred_entropy": ent, "pred_frac": frac, "ragged": False}


# ---------------------------------------------------------------- estimator
_BUDGET = re.compile(r"^(?P<base>.+?)-(?P<b>\d+)b$")


def series(runs) -> dict:
    """{family: [(budget, run), ...]} over the token-budget checkpoints."""
    fam = defaultdict(list)
    for r in runs:
        m = _BUDGET.match(r)
        if m:
            fam[m.group("base")].append((int(m.group("b")), r))
    return {k: sorted(v) for k, v in fam.items() if len(v) >= 3}


def report_estimator(data: dict) -> None:
    print("=" * 78)
    print("ESTIMATOR CHOICE  (CLAUDE.md 6g criteria; never gold accuracy)")
    print("=" * 78)
    rows = []
    for est in ESTIMATORS:
        overs, ents, back, rng = [], [], 0.0, 0.0
        for run, langs in data.items():
            for lang, tasks in langs.items():
                b = tasks.get(f"xcsqa_{lang}")
                if b is None:
                    continue
                pr = profile(b, est)
                overs.append(pr["acc_over_null"])
                ents.append(pr["pred_entropy"])
        # trajectory monotonicity, pooled over every budget series x language
        for base, pts in series(data).items():
            for lang in LANGS:
                seq = []
                for _, run in pts:
                    b = data[run].get(lang, {}).get(f"xcsqa_{lang}")
                    if b is not None:
                        seq.append(profile(b, est)["acc"])
                if len(seq) < 3:
                    continue
                back += sum(max(0.0, seq[i] - seq[i + 1]) for i in range(len(seq) - 1))
                rng += max(seq) - min(seq)
        rows.append((est, sum(overs) / len(overs), sum(ents) / len(ents),
                     back / rng if rng else float("nan")))
    print(f"{'estimator':16s} {'over-null':>10s} {'entropy':>9s} {'backwards':>10s}")
    usable = [r for r in rows if r[0] != "acc_tokennorm"]
    best_disc = max(usable, key=lambda r: r[1])[0]
    best_mono = min(usable, key=lambda r: r[3])[0]
    for est, o, e, bk in rows:
        note = ""
        if est == "acc_tokennorm":
            note = "  (NEVER USE: tokenizer-dependent, 6g)"
        else:
            tags = ([f"best discrimination"] if est == best_disc else []) + \
                   ([f"most monotone"] if est == best_mono else [])
            if tags:
                note = "  <-- " + ", ".join(tags)
        print(f"{est:16s} {o:10.3f} {e:9.3f} {bk:10.3f}{note}")
    print("\n'backwards' = summed downward movement / total range over the budget")
    print("series, pooled across families and languages. Lower is better.")
    if best_disc != best_mono:
        print(f"\n*** CRITERIA DISAGREE: {best_disc} discriminates best, {best_mono} is")
        print("    most monotone. No single estimator is selected -- report the")
        print("    contrast under both, and treat any conclusion that flips between")
        print("    them as NOT ESTABLISHED. Picking the one that gives the wanted")
        print("    answer is exactly the failure mode CLAUDE.md 6e/6g document.")


# --------------------------------------------------------------- capability
def report_capability(data: dict, est: str) -> None:
    print("\n" + "=" * 78)
    print(f"OWN-LANGUAGE CAPABILITY  ({est}, nominal chance 0.200)")
    print("=" * 78)
    finals = [r for r in data if not _BUDGET.match(r)]
    print("\n30B cooled finals (and 12b/15b where no 30B monolingual exists):")
    print(f"{'model':22s} {'lang':5s} {'acc':>7s} {'null':>7s} {'over':>8s}")
    for run in sorted(finals):
        for lang in LANGS:
            b = data[run].get(lang, {}).get(f"xcsqa_{lang}")
            if b is None:
                continue
            pr = profile(b, est)
            print(f"{run:22s} {lang:5s} {pr['acc']:7.3f} {pr['null']:7.3f} "
                  f"{pr['acc_over_null']:+8.3f}")

    print("\nBy language, averaged over the 30B finals that trained on it:")
    agg = defaultdict(list)
    for run in finals:
        for lang in LANGS:
            b = data[run].get(lang, {}).get(f"xcsqa_{lang}")
            if b is not None:
                agg[lang].append(profile(b, est)["acc_over_null"])
    print(f"{'lang':6s} {'script':7s} {'n':>3s} {'mean over-null':>15s}")
    for lang in LANGS:
        if agg[lang]:
            print(f"{lang:6s} {SCRIPT.get(lang,'-'):7s} {len(agg[lang]):3d} "
                  f"{sum(agg[lang])/len(agg[lang]):+15.3f}")


# ----------------------------------------------------------------- transfer
def paired_bootstrap(a: list[int], b: list[int], n_rep=2000, seed=0):
    """CI on mean(b)-mean(a), resampling DOC INDICES once and applying the
    same resample to both models -- valid because both scored the identical
    fixed doc order (the estimator bootstrap_transfer.py uses)."""
    assert len(a) == len(b)
    n = len(a)
    point = sum(b) / n - sum(a) / n
    rng = random.Random(seed)
    reps = []
    for _ in range(n_rep):
        idx = [rng.randrange(n) for _ in range(n)]
        reps.append(sum(b[i] for i in idx) / n - sum(a[i] for i in idx) / n)
    reps.sort()
    return point, reps[int(0.025 * n_rep)], reps[int(0.975 * n_rep)]


# mono X B/lang  vs  bilingual 2X B total -- equal per-language exposure, and
# both mid-stable at peak LR (decay starts at 24B), so LR-matched by design.
TIERS = [("1B", "1b", "2b"), ("5B", "5b", "10b"), ("8B", "8b", "15b"),
         ("12B", "12b", "23b")]


def report_transfer(data: dict, est: str) -> None:
    print("\n" + "=" * 78)
    print(f"TRANSFER: bilingual - monolingual  ({est}, paired bootstrap B=2000)")
    print("=" * 78)
    print("All tiers LR-matched by construction (mid-stable @3.0e-3).\n")
    per_lang = defaultdict(list)
    per_tier = defaultdict(lambda: defaultdict(list))
    print(f"{'tier':6s} {'partner':8s} {'script':7s} {'tok':8s} "
          f"{'d partner':>22s} {'d English':>22s}")
    for tier, mb, bb in TIERS:
        for partner in ("de", "fr", "ar", "zh"):
            for tok in ("fair", "starved"):
                mono = f"{partner}-{tok}-{mb}"
                bi = f"en-{partner}-{tok}-{bb}"
                en_mono = f"en-{tok}-{mb}"
                if mono not in data or bi not in data:
                    continue
                out = []
                for lang, m in ((partner, mono), ("en", en_mono)):
                    if m not in data:
                        out.append(None)
                        continue
                    ba = data[m].get(lang, {}).get(f"xcsqa_{lang}")
                    bb_ = data[bi].get(lang, {}).get(f"xcsqa_{lang}")
                    if ba is None or bb_ is None:
                        out.append(None)
                        continue
                    out.append(paired_bootstrap(hits(ba, est), hits(bb_, est)))
                def fmt(r):
                    if r is None:
                        return f"{'n/a':>22s}"
                    p, lo, hi = r
                    star = "*" if lo > 0 or hi < 0 else " "
                    return f"{p:+7.3f}{star}[{lo:+6.3f},{hi:+6.3f}]"
                print(f"{tier:6s} {partner:8s} {SCRIPT[partner]:7s} {tok:8s} "
                      f"{fmt(out[0])} {fmt(out[1])}")
                if out[0]:
                    per_lang[partner].append(out[0][0])
                    per_tier[tier][SCRIPT[partner]].append(out[0][0])

    print("\nSame-script vs cross-script, per tier (partner-language delta):")
    print(f"{'tier':6s} {'same':>8s} {'cross':>8s} {'gap':>8s}")
    gaps = []
    for tier, _, _ in TIERS:
        s, c = per_tier[tier]["same"], per_tier[tier]["cross"]
        if s and c:
            gap = sum(s)/len(s) - sum(c)/len(c)
            gaps.append(gap)
            print(f"{tier:6s} {sum(s)/len(s):+8.3f} {sum(c)/len(c):+8.3f} {gap:+8.3f}")
    if gaps:
        print(f"{'mean':6s} {'':8s} {'':8s} {sum(gaps)/len(gaps):+8.3f}")

    print("\nBy partner language (this is where 6g found the gap was FRENCH,")
    print("not script -- German sat with ar/zh):")
    for lang in ("de", "fr", "ar", "zh"):
        if per_lang[lang]:
            v = per_lang[lang]
            print(f"  {lang} ({SCRIPT[lang]:5s}): {sum(v)/len(v):+.3f}  (n={len(v)})")


def report_percell(data: dict, est: str) -> None:
    """Emit a `## xcsqa` section in results/mubench_sweep/per_cell_table.md's
    exact column format, so X-CSQA can be appended to that file.

    That table has no committed generator -- its raw sidecars died with the box
    that produced the 100-checkpoint sweep, which is why build_tables.py treats
    the committed copy as the only surviving record. This section is therefore
    APPENDED, never regenerated in place: the other families cannot be rebuilt.
    """
    rows = []
    for run, langs in data.items():
        for lang in LANGS:
            b = langs.get(lang, {}).get(f"xcsqa_{lang}")
            if b is None:
                continue
            pr = profile(b, est)
            head = pr["acc_over_null"] / (1 - pr["null"]) if pr["null"] < 1 else float("nan")
            rows.append((budget_of(run), run, lang, len(b["gold"]), pr["acc"],
                         pr["null"], pr["acc_over_null"], head, pr["pred_entropy"]))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    print(f"## xcsqa  ({est}, n={len(rows)} cells)")
    print()
    print("| model | B | lang | n | acc | null | pp | head | ent |")
    print("|---|---|---|---|---|---|---|---|---|")
    for bud, run, lang, n, acc, nul, pp, hd, ent in rows:
        print(f"| {run} | {bud} | {lang} | {n} | {acc:.4f} | {nul:.4f} | "
              f"{pp:+.4f} | {hd:.4f} | {ent:.3f} |")
    print()


def budget_of(run: str) -> int:
    m = _BUDGET.match(run)
    return int(m.group("b")) if m else 30


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--report", nargs="*",
                    default=["estimator", "capability", "transfer"],
                    choices=["estimator", "capability", "transfer", "percell"])
    ap.add_argument("--estimator", default="acc",
                    help="estimator for the capability/transfer reports")
    args = ap.parse_args()

    data = load(args.results_dir)
    cells = sum(len(t) for l in data.values() for t in l.values())
    if args.report != ["percell"]:      # keep the appendable section clean
        print(f"[xcsqa] {len(data)} runs, {cells} cells\n")
    if "estimator" in args.report:
        report_estimator(data)
    if "capability" in args.report:
        report_capability(data, args.estimator)
    if "transfer" in args.report:
        report_transfer(data, args.estimator)
    if "percell" in args.report:
        report_percell(data, args.estimator)


if __name__ == "__main__":
    raise SystemExit(main())
