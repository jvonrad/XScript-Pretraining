#!/usr/bin/env python
"""6j knowledge-neuron statistics on the DIFFERENT-surface-form facts only
(gold entity name differs between the two languages), for dKS AND the
ablation transfer rate -- 6j reported the split for dKS only.

    python analyze_kn_surface.py [--results results/knowneurons] [--runs ...]
"""
import argparse, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src"))
from analyze_knowneurons import Run, jaccard_matrix, jaccard_mismatched, boot_ci, boot_rate_diff, boot_diff, load_gold_strings, CONDS

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default=str(ROOT / "results" / "knowneurons"))
    ap.add_argument("--runs", nargs="*", default=[f"en-{p}-{c}" for p in ("de", "fr", "ar", "zh") for c in ("fair", "starved")])
    ap.add_argument("--k", type=int, default=100); a = ap.parse_args()
    res = Path(a.results); runs = {n: Run(res, n) for n in a.runs}
    gold = load_gold_strings(sorted({l for r in runs.values() for l in r.langs}))
    rows = {}
    print(f"{'run':<16} {'n_diff':>6} {'n_same':>6} {'dKS_diff':>9} {'dKS_same':>9} {'rate_diff':>9} {'rate_same':>9} {'rate_all':>9}")
    for n, r in runs.items():
        l1, l2 = r.langs
        same = np.array([gold[l1][fi].strip().lower() == gold[l2][fi].strip().lower() for fi in r.fact_ids])
        a_, b_ = r.topk[:, 0, :], r.topk[:, 1, :]
        dks = jaccard_matrix(a_, b_, a.k) - jaccard_mismatched(a_, b_, a.k)
        pairs = {"diff": [], "same": [], "all": []}
        for ti, tl in enumerate(r.langs):
            dmg = {c: r.damage(tl, c) for c in CONDS[1:]}
            spec = dmg["own_fact"] - dmg["own_other"]; trans = dmg["cross_fact"] - dmg["cross_other"]
            pairs["all"].append((trans, spec)); pairs["diff"].append((trans[~same], spec[~same])); pairs["same"].append((trans[same], spec[same]))
        rate = lambda P: float(np.concatenate([t for t, _ in P]).mean() / np.concatenate([s for _, s in P]).mean())
        rows[n] = {"same": same, "dks": dks, "pairs": pairs}
        print(f"{n:<16} {(~same).sum():>6d} {same.sum():>6d} {dks[~same].mean():>9.4f} {dks[same].mean() if same.any() else float('nan'):>9.4f} "
              f"{rate(pairs['diff']):>9.3f} {rate(pairs['same']) if same.any() else float('nan'):>9.3f} {rate(pairs['all']):>9.3f}")
    print("\nfair − starved on DIFFERENT-surface-form facts only (95% bootstrap CI; * excludes 0):")
    print(f"{'partner':<8} {'ΔdKS (unpaired)':>28} {'Δ transfer rate':>28}")
    for p in ("de", "fr", "ar", "zh"):
        f, s = rows.get(f"en-{p}-fair"), rows.get(f"en-{p}-starved")
        if not (f and s): continue
        m, lo, hi = boot_diff(f["dks"][~f["same"]], s["dks"][~s["same"]], 2000, paired=False)
        rm, rlo, rhi = boot_rate_diff(f["pairs"]["diff"], s["pairs"]["diff"], 2000)
        st = lambda lo, hi: "*" if lo > 0 or hi < 0 else " "
        print(f"{p:<8} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]{st(lo,hi):>4} {rm:+.4f} [{rlo:+.4f}, {rhi:+.4f}]{st(rlo,rhi)}")

if __name__ == "__main__":
    main()
