#!/usr/bin/env python
"""Mediation analysis: does per-fact cross-lingual representation alignment
carry the fair tokenizer's consistency gain? (CLAUDE.md 6k)

Inputs: run_fact_align.py's per-language question embeddings (all layers)
and run_polyfact_traj.py's candidate scores for the same checkpoints.

Per checkpoint and layer, the per-fact alignment score is the bidirectional
retrieval d' of the question's en embedding against its partner-language
embedding, over the other 2038 facts (alignment._discriminability, centred
per language, exactly 6b's statistic). The reported layer is the model's
own peak-mean-d' layer AND a fixed layer (L12 = 6b's `ref`), per 6b's rule
of reporting both. Per-fact consistency = same top-1 QID in both languages;
correctness = gold top-1 in both.

Tests, per partner:
  (1) alignment level: mean d' fair vs starved, paired over facts;
  (2) alignment -> consistency within model: AUC of d' for predicting a
      consistent answer, separately on all facts and on wrong-in-both facts;
  (3) mediation: logistic regression consistent ~ [fair] with and without
      d' as a covariate (pooled over the two models); if the tokenizer
      coefficient shrinks toward 0 once d' is controlled, the effect runs
      through alignment. Also on the wrong-in-both subset (error consistency).

    python analyze_fact_align.py [--workdir /mnt/scratch/xscript_rankc]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from xscript.eval.alignment import _center, _discriminability  # noqa: E402

PARTNERS = ["de", "fr", "ar", "zh"]
REF_LAYER = 12


def auc(score, label):
    """Rank AUC (Mann-Whitney)."""
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logit_fit(X, y, iters=200, l2=1e-3):
    """Plain Newton logistic regression; returns coefficients (X includes a 1 column)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def consistency_labels(traj_json):
    d = json.load(open(traj_json))
    l1, l2 = d["langs"]
    ll1, ll2 = np.array(d["ll"][l1]), np.array(d["ll"][l2])
    g1, g2 = np.array(d["gold"][l1]), np.array(d["gold"][l2])
    ids1, ids2 = d["option_ids"][l1], d["option_ids"][l2]
    perm = np.array([[ids2[i].index(q) for q in ids1[i]] for i in range(len(ids1))])
    ll2 = np.take_along_axis(ll2, perm, axis=1)
    g2 = np.array([ids1[i].index(ids2[i][g2[i]]) for i in range(len(ids1))])
    t1, t2 = ll1.argmax(1), ll2.argmax(1)
    return {"consistent": (t1 == t2).astype(int), "correct_both": ((t1 == g1) & (t2 == g2)).astype(int),
            "wrong_both": ((t1 != g1) & (t2 != g2)).astype(int), "fact_id": d["fact_id"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/mnt/scratch/xscript_rankc")
    ap.add_argument("--runs", nargs="*", default=None, help="default: the 8 cooled finals")
    args = ap.parse_args()
    work = Path(args.workdir)
    runs = args.runs or [f"en-{p}-{c}" for p in PARTNERS for c in ("fair", "starved")]
    rng = np.random.default_rng(0)
    per = {}
    for run in runs:
        f = work / "results" / "fact_align" / f"{run}.npz"
        t = work / "results" / "polyfact_traj" / f"{run}.json"
        if not (f.exists() and t.exists()):
            print(f"[falign] {run}: missing embeddings or scores, skip"); continue
        z = np.load(f, allow_pickle=True); langs = list(z["langs"]); E1, E2 = z[langs[0]], z[langs[1]]
        lab = consistency_labels(t)
        assert list(z["fact_id"]) == lab["fact_id"]
        L = E1.shape[0]
        dprime = np.zeros((L, E1.shape[1]))
        for l in range(L):
            A, B = _center(E1[l]), _center(E2[l])
            dp = _discriminability(A @ B.T)["dprime_sym"]
            dprime[l] = dp
        peak = int(dprime.mean(1).argmax())
        per[run] = {"partner": langs[1], "cond": "fair" if "fair" in run else "starved",
                    "dprime": dprime, "peak": peak, **lab}
    lines = ["## Per-fact question alignment (d') vs answer consistency, PolyFact, 30B finals", "",
             "| partner | cond | peak layer | mean d' @peak | mean d' @L12 | AUC(d'→consistent) all | AUC on wrong-in-both | P(consistent) | P(consistent \\| wrong-in-both) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for run, r in per.items():
        dp, dr = r["dprime"][r["peak"]], r["dprime"][REF_LAYER]
        wb = r["wrong_both"] == 1
        lines.append(f"| {r['partner']} | {r['cond']} | L{r['peak']} | {dp.mean():.3f} | {dr.mean():.3f} | "
                     f"{auc(dp, r['consistent']):.3f} | {auc(dp[wb], r['consistent'][wb]):.3f} | "
                     f"{r['consistent'].mean():.3f} | {r['consistent'][wb].mean():.3f} |")
    lines += ["", "## fair − starved: alignment (paired over facts) and mediation (logistic: consistent ~ fair [+ d'])", "",
              "| partner | Δ mean d' @peak [95% CI] | Δ d' @L12 | β_fair (no d') | β_fair (with d') | shrink | β_d' | same on wrong-in-both: β_fair no/with d' |",
              "|---|---|---|---|---|---|---|---|"]
    summary = {}
    for p in PARTNERS:
        F, S = per.get(f"en-{p}-fair"), per.get(f"en-{p}-starved")
        if not (F and S):
            continue
        dF, dS = F["dprime"][F["peak"]], S["dprime"][S["peak"]]
        d = dF - dS; idx = rng.integers(0, len(d), size=(2000, len(d))); m = d[idx].mean(1)
        lo, hi = np.percentile(m, 2.5), np.percentile(m, 97.5)
        d12 = (F["dprime"][REF_LAYER] - S["dprime"][REF_LAYER]).mean()
        # mediation on pooled facts (both models): standardise d' within the pooled sample
        def med(mask_f, mask_s):
            y = np.concatenate([F["consistent"][mask_f], S["consistent"][mask_s]])
            fair = np.concatenate([np.ones(mask_f.sum()), np.zeros(mask_s.sum())])
            dp = np.concatenate([dF[mask_f], dS[mask_s]]); dp = (dp - dp.mean()) / dp.std()
            w0 = logit_fit(np.stack([np.ones_like(fair), fair], 1), y)
            w1 = logit_fit(np.stack([np.ones_like(fair), fair, dp], 1), y)
            return w0[1], w1[1], w1[2]
        allf, alls = np.ones(len(dF), bool), np.ones(len(dS), bool)
        b0, b1, bd = med(allf, alls)
        wb0, wb1, wbd = med(F["wrong_both"] == 1, S["wrong_both"] == 1)
        shrink = 1 - b1 / b0 if abs(b0) > 1e-9 else float("nan")
        summary[p] = {"d_dprime_peak": float(d.mean()), "lo": float(lo), "hi": float(hi), "d_dprime_L12": float(d12),
                      "beta_fair": float(b0), "beta_fair_given_dprime": float(b1), "beta_dprime": float(bd),
                      "wrong_both_beta_fair": float(wb0), "wrong_both_beta_fair_given_dprime": float(wb1)}
        lines.append(f"| {p} | **{d.mean():+.3f}** [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''} | {d12:+.3f} | "
                     f"{b0:+.3f} | {b1:+.3f} | {shrink:+.0%} | {bd:+.3f} | {wb0:+.3f} / {wb1:+.3f} (β_d' {wbd:+.3f}) |")
    md = "\n".join(lines)
    print(md)
    out = ROOT / "results" / "rankc"; out.mkdir(exist_ok=True)
    (out / "fact_align_mediation.md").write_text(md + "\n")
    json.dump(summary, open(out / "fact_align_mediation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
