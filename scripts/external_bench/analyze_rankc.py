#!/usr/bin/env python
"""RankC (Qi et al. 2023) from the raw BMLAMA scores run_rankc.py stored.

Per row the reference computes  sum_k w_k * |top_k(L1) & top_k(L2)| / k  over
the N (position-aligned) candidates with w = softmax([N, ..., 1]); RankC is
the mean over rows. Three candidate rankings are derived from the same raw
scores, because the reference's causal-LM rule is a length-normalised
whole-sentence score and this project has learned (CLAUDE.md 6e/6g) to check
an estimator before quoting it:

  meanCE   whole sentence with the candidate substituted, ranked by mean
           per-token cross-entropy  -- the reference implementation, verbatim
  sumWhole same, ranked by SUMMED loglikelihood (no length normalisation)
  cand     candidate tokens only, given the prefix up to <mask>, summed

Also reports P@1 accuracy per language, the fraction of rows correct in both
languages, RankC restricted to that subset, a shuffled-row null (RankC of L1
against L2's rankings of *different* rows), paired-over-rows bootstrap CIs,
and fair - starved paired deltas per partner (rows intersected, since the
width filter can drop different rows under the two tokenizers).

    python analyze_rankc.py [--workdir /mnt/scratch/xscript_rankc] [--B 2000]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARTNERS = ["de", "fr", "ar", "zh"]
VARIANTS = ["meanCE", "sumWhole", "cand"]


def rankings(rec, lang, variant):
    w = np.array(rec[lang]["whole"])   # [N, 2] = (sum_ll, n_tok)
    c = np.array(rec[lang]["cand"])
    if variant == "meanCE":
        s = w[:, 0] / w[:, 1]          # mean loglik per token; higher = lower CE
    elif variant == "sumWhole":
        s = w[:, 0]
    else:
        s = c[:, 0]
    return np.argsort(-s, kind="stable")


def row_rankc(r1, r2):
    n = len(r1)
    order = np.arange(n, 0, -1, dtype=float)
    w = np.exp(order - order.max()); w /= w.sum()
    tot = 0.0
    s1, s2 = set(), set()
    for k in range(n):
        s1.add(int(r1[k])); s2.add(int(r2[k]))
        tot += w[k] * len(s1 & s2) / (k + 1)
    return tot


def load(path):
    d = json.load(open(path))
    l1, l2 = d["langs"]
    out = {"run": d["run"], "set": d["set"], "langs": d["langs"], "n_dropped": d["n_dropped"],
           "n_total": d["n_rows_total"], "row_ids": [], "rc": {v: [] for v in VARIANTS},
           "acc1": {v: [] for v in VARIANTS}, "acc2": {v: [] for v in VARIANTS},
           "r1": {v: [] for v in VARIANTS}, "r2": {v: [] for v in VARIANTS}}
    for rec in d["rows"]:
        if rec.get("gold_mismatch"):
            continue
        g = rec["gold"][0]
        out["row_ids"].append(rec["row"])
        for v in VARIANTS:
            r1, r2 = rankings(rec, l1, v), rankings(rec, l2, v)
            out["rc"][v].append(row_rankc(r1, r2))
            out["acc1"][v].append(int(r1[0] == g)); out["acc2"][v].append(int(r2[0] == g))
            out["r1"][v].append(r1); out["r2"][v].append(r2)
    for k in ("rc", "acc1", "acc2"):
        out[k] = {v: np.array(x) for v, x in out[k].items()}
    return out


def null_rankc(r1, r2, rng, reps=5):
    vals = []
    for _ in range(reps):
        perm = rng.permutation(len(r1))
        vals.append(np.nanmean([row_rankc(a, r2[j]) if len(a) == len(r2[j]) else np.nan
                                for a, j in zip(r1, perm)]))
    return float(np.nanmean(vals))


def boot_mean(x, rng, B):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/mnt/scratch/xscript_rankc")
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    res_dir = Path(args.workdir) / "results" / "rankc"
    files = sorted(res_dir.glob("*_bmlama*.json"))
    data = {}
    for f in files:
        d = load(f)
        data[(d["run"], d["set"])] = d
    lines = []
    summary = {}
    for s in ("mub", "17", "53"):
        lines.append(f"\n## BMLAMA-{s}" + (" (MuBench re-extraction, 6016 items, all 5 langs)" if s == "mub" else " (original Qi et al.)") + "\n")
        lines.append("| model | n | variant | acc L1 | acc L2 | both | RankC [95% CI] | null | RankC both-correct |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for p in PARTNERS:
            for cond in ("fair", "starved"):
                run = f"en-{p}-{cond}"
                d = data.get((run, s))
                if d is None:
                    continue
                for v in VARIANTS:
                    rc, lo, hi = boot_mean(d["rc"][v], rng, args.B)
                    both = (d["acc1"][v] == 1) & (d["acc2"][v] == 1)
                    rc_both = float(d["rc"][v][both].mean()) if both.any() else float("nan")
                    null = null_rankc(d["r1"][v], d["r2"][v], rng)
                    summary[(run, s, v)] = {"rankc": rc, "lo": lo, "hi": hi, "null": null,
                                            "acc1": float(d["acc1"][v].mean()),
                                            "acc2": float(d["acc2"][v].mean()),
                                            "both": float(both.mean()), "rankc_both": rc_both,
                                            "n": int(len(d["rc"][v])), "n_dropped": d["n_dropped"]}
                    lines.append(f"| {run} | {len(d['rc'][v])} | {v} | {d['acc1'][v].mean():.3f} | "
                                 f"{d['acc2'][v].mean():.3f} | {both.mean():.3f} | "
                                 f"**{rc:.3f}** [{lo:.3f}, {hi:.3f}] | {null:.3f} | {rc_both:.3f} |")
        # fair - starved, paired over the intersection of rows
        lines.append(f"\n### fair − starved, paired over rows (BMLAMA-{s})\n")
        lines.append("| partner | variant | Δ RankC [95% CI] | Δ acc L1 | Δ acc L2 |")
        lines.append("|---|---|---|---|---|")
        for p in PARTNERS:
            f_, s_ = data.get((f"en-{p}-fair", s)), data.get((f"en-{p}-starved", s))
            if f_ is None or s_ is None:
                continue
            common = sorted(set(f_["row_ids"]) & set(s_["row_ids"]))
            fi = {r: i for i, r in enumerate(f_["row_ids"])}; si = {r: i for i, r in enumerate(s_["row_ids"])}
            a = np.array([fi[r] for r in common]); b = np.array([si[r] for r in common])
            for v in VARIANTS:
                diff = f_["rc"][v][a] - s_["rc"][v][b]
                m, lo, hi = boot_mean(diff, rng, args.B)
                d1 = f_["acc1"][v][a].mean() - s_["acc1"][v][b].mean()
                d2 = f_["acc2"][v][a].mean() - s_["acc2"][v][b].mean()
                star = "*" if lo > 0 or hi < 0 else ""
                summary[(f"delta-{p}", s, v)] = {"delta": m, "lo": lo, "hi": hi, "n": len(common)}
                lines.append(f"| {p} | {v} | **{m:+.4f}** [{lo:+.4f}, {hi:+.4f}]{star} | {d1:+.3f} | {d2:+.3f} |")
    md = "\n".join(lines)
    print(md)
    out = ROOT / "results" / "rankc"
    out.mkdir(exist_ok=True)
    (out / "bmlama_rankc.md").write_text(md + "\n")
    json.dump({"|".join(k): v for k, v in summary.items()}, open(out / "bmlama_rankc.json", "w"), indent=1)


if __name__ == "__main__":
    main()
