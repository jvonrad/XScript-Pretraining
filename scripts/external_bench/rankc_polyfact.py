#!/usr/bin/env python
"""RankC (Qi et al. 2023, arXiv:2310.10378) between a bilingual model's two
languages on PolyFact, re-derived from the per-candidate loglikelihoods that
run_knowneurons.py phase A persisted (results/knowneurons/<run>_selection.json).

RankC(L1, L2) = mean over facts of  sum_k w_k * |top_k(L1) & top_k(L2)| / k,
with w = softmax([N, N-1, ..., 1]) over the N candidates (the reference
implementation's default 'softmax' weighting; candidates are index-aligned
across languages). Candidates are ranked by summed loglikelihood (the
selection rule phase A used); the reference ranks causal LMs by mean
per-token CE of the whole sentence -- see rankc_bmlama.py for that variant.
Pure CPU, no model.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ["en-de-fair", "en-de-starved", "en-fr-fair", "en-fr-starved",
        "en-ar-fair", "en-ar-starved", "en-zh-fair", "en-zh-starved"]


def rankc_rows(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Per-fact RankC. r1, r2: [n_facts, N] candidate indices, best first."""
    N = r1.shape[1]
    order = np.arange(N, 0, -1, dtype=float)
    w = np.exp(order) / np.exp(order).sum()
    out = np.zeros(len(r1))
    for i, (a, b) in enumerate(zip(r1, r2)):
        for k in range(N):
            out[i] += w[k] * len(set(a[:k + 1]) & set(b[:k + 1])) / (k + 1)
    return out


def rankc(r1, r2) -> float:
    return float(rankc_rows(r1, r2).mean())


def boot(x, rng, B=2000):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _option_ids(lang):
    """Per-fact list of the 4 option QIDs in the dataset's own (per-language
    shuffled) order -- needed because PolyFact shuffles option POSITIONS
    independently per language while the option ID sets align."""
    import datasets
    ds = datasets.load_dataset("jvonrad/PolyFact", lang, split="test")
    return [list(r) for r in ds["option_ids"]], list(ds["fact_id"])


def main():
    rows = []
    ids_cache = {}
    for run in RUNS:
        d = json.load(open(ROOT / "results" / "knowneurons" / f"{run}_selection.json"))
        l1, l2 = d["langs"]
        for l in (l1, l2):
            if l not in ids_cache:
                ids_cache[l] = _option_ids(l)
        ids1, f1 = ids_cache[l1]; ids2, f2 = ids_cache[l2]
        assert f1 == f2 and len(f1) == len(d["kept_facts"]) and d["kept_facts"] == list(range(len(f1)))
        ll1, ll2 = np.array(d["ll"][l1]), np.array(d["ll"][l2])
        g1, g2 = np.array(d["gold"][l1]), np.array(d["gold"][l2])
        # re-express L2's scores in L1's option order (same QID sets, verified)
        perm = np.array([[ids2[i].index(q) for q in ids1[i]] for i in range(len(ids1))])
        assert all(sorted(a) == sorted(b) for a, b in zip(ids1, ids2))
        ll2 = np.take_along_axis(ll2, perm, axis=1)
        g2 = np.array([list(ids1[i]).index(ids2[i][g2[i]]) for i in range(len(ids1))])
        assert (g1 == g2).all(), "gold QID differs after alignment"
        r1 = np.argsort(-ll1, axis=1); r2 = np.argsort(-ll2, axis=1)
        acc1 = float((r1[:, 0] == g1).mean()); acc2 = float((r2[:, 0] == g2).mean())
        both = (r1[:, 0] == g1) & (r2[:, 0] == g2)
        rng = np.random.default_rng(0)
        rcr = rankc_rows(r1, r2)
        rc = float(rcr.mean()); lo, hi = boot(rcr, rng)
        rc_both = float(rcr[both].mean())
        # null: RankC between L1 and an independent shuffle of L2's facts
        null = np.mean([rankc(r1, r2[rng.permutation(len(r2))]) for _ in range(5)])
        rows.append((run, len(ll1), acc1, acc2, float(both.mean()), rc, lo, hi, null, rc_both, rcr))
    print(f"{'model':14s} {'n':>5s} {'acc_en':>7s} {'acc_X':>7s} {'both':>6s} {'RankC':>7s} {'[95% CI]':>16s} {'null':>7s} {'RankC|both':>10s}")
    for run, n, a1, a2, b, rc, lo, hi, null, rcb, _ in rows:
        print(f"{run:14s} {n:5d} {a1:7.3f} {a2:7.3f} {b:6.3f} {rc:7.3f} [{lo:.3f}, {hi:.3f}] {null:7.3f} {rcb:10.3f}")
    print("\nfair - starved, RankC (paired over facts, 95% CI; * = excludes 0):")
    rng = np.random.default_rng(1)
    deltas = {}
    for p in ["de", "fr", "ar", "zh"]:
        f = next(r for r in rows if r[0] == f"en-{p}-fair"); s = next(r for r in rows if r[0] == f"en-{p}-starved")
        d = f[-1] - s[-1]; lo, hi = boot(d, rng)
        deltas[p] = (float(d.mean()), lo, hi)
        print(f"  {p}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}   (fair {f[5]:.3f}, starved {s[5]:.3f})")
    out = ROOT / "results" / "rankc"
    out.mkdir(exist_ok=True)
    json.dump({"models": [dict(zip(["run", "n_facts", "acc_l1", "acc_l2", "acc_both", "rankc", "ci_lo", "ci_hi", "rankc_null", "rankc_both_correct"], r[:-1])) for r in rows],
               "fair_minus_starved": deltas},
              open(out / "polyfact_rankc.json", "w"), indent=2)


if __name__ == "__main__":
    main()
