#!/usr/bin/env python
"""Analysis of the follow-up ablation passes (same-relation controls +
intersection decomposition) for CLAUDE.md 6j.

    python analyze_kn_followup.py [--results DIR] [--boot 2000]

Sections:

  1. SAME-RELATION dKS -- overlap with the mismatched-fact baseline restricted
     to facts of the same Wikidata relation, so relation-level circuitry
     (capital-of machinery, answer-type priors) is subtracted too. Pure CPU
     from the stored top-K + PolyFact metadata.

  2. SAME-RELATION TRANSFER -- phase C's transfer rate re-derived with
     same-relation different-fact controls, plus the consistency check that
     the re-run own_fact/cross_fact damages reproduce phase C.

  3. INTERSECTION DECOMPOSITION -- the Story A / Story B test:
       lift          = dmg(I) - mean(dmg(rand |I| from topK_s))   [size-fair]
       specificity   = dmg(I) - dmg(I' of same-relation f')       [size-fair]
       concentration = dmg(I) / (dmg(I) + dmg(D)) - |I|/K         [size-fair]
     Damage concentrated in I beyond its size share, and beyond size-matched
     random and different-fact controls, means the intersection neurons are
     the shared store (Story A); damage carried by D means coupled-but-
     disjoint circuits (Story B). Fair-vs-starved is compared on the
     size-fair statistics AND on the size-sensitive intersection-carried
     damage (the latter's size dependence is the quantity claim itself).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from analyze_knowneurons import (Run, jaccard_matrix, boot_ci, boot_diff, fmt,  # noqa: E402
                                 CONDS, PARTNERS)

SR_CONDS = ("none", "own_fact", "cross_fact", "own_other_sr", "cross_other_sr", "random")
INT_CONDS = ("none", "inter", "disjoint", "rand_sub1", "rand_sub2", "inter_ctrl")


def relations():
    import datasets
    return datasets.load_dataset("jvonrad/PolyFact", "en", split="test")["relation"]


def mism_same_rel(r, rel, k):
    fids = r.fact_ids.tolist()
    groups = {}
    for i, fi in enumerate(fids):
        groups.setdefault(rel[fi], []).append(i)
    out = np.full(len(fids), np.nan)
    for g in groups.values():
        if len(g) < 2:
            continue
        for j, i in enumerate(g):
            nxt = g[(j + 1) % len(g)]
            sa = set(r.topk[i, 0, :k].tolist())
            sb = set(r.topk[nxt, 1, :k].tolist())
            out[i] = len(sa & sb) / len(sa | sb)
    return out


def sr_damage(doc, tlang, cond):
    gold = np.array(doc["gold"][tlang])
    lls = np.array(doc["ll"][tlang])
    rows = np.arange(len(gold))
    return (lls[rows, SR_CONDS.index("none"), gold]
            - lls[rows, SR_CONDS.index(cond), gold])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path("/home/ubuntu/xscript_kn/results/knowneurons"))
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    rel = relations()
    runs, srs = {}, {}
    for f in sorted(glob.glob(str(a.results / "*_ablation_samerel.json"))):
        name = os.path.basename(f)[:-len("_ablation_samerel.json")]
        runs[name] = Run(a.results, name)
        srs[name] = json.loads(Path(f).read_text())

    # ---------- 1. same-relation dKS ----------
    print("== 1. dKS WITH SAME-RELATION MISMATCH BASELINE (K=%d) ==" % a.k)
    print(f"{'run':<16} {'sameJ':>7} {'mismSR':>7} {'dKS_sr [95% CI]':>28}")
    dks_sr = {}
    for name, r in runs.items():
        same = jaccard_matrix(r.topk[:, 0], r.topk[:, 1], a.k)
        msr = mism_same_rel(r, rel, a.k)
        ok = ~np.isnan(msr)
        d = (same - msr)[ok]
        m, lo, hi = boot_ci(d, a.boot)
        dks_sr[name] = d
        print(f"{name:<16} {same[ok].mean():>7.4f} {msr[ok].mean():>7.4f} "
              f"{fmt(m, lo, hi):>28}  (n={ok.sum()})")

    print("\nfair - starved on dKS_sr (unpaired):")
    for p, script in PARTNERS:
        f, s = f"en-{p}-fair", f"en-{p}-starved"
        if f in dks_sr and s in dks_sr:
            m, lo, hi = boot_diff(dks_sr[f], dks_sr[s], a.boot, paired=False)
            print(f"  {p} ({script}): {fmt(m, lo, hi)}")

    # ---------- 2. same-relation transfer ----------
    print("\n== 2. TRANSFER RATE WITH SAME-RELATION CONTROLS ==")
    print(f"{'run':<16} {'rate_sr':>8} {'rate_any (phase C)':>19} {'consistency':>12}")
    pairs_sr = {}
    for name, r in runs.items():
        doc = srs[name]
        tr, sp, cons = [], [], []
        for ti, tlang in enumerate(r.langs):
            spec = sr_damage(doc, tlang, "own_fact") - sr_damage(doc, tlang, "own_other_sr")
            trans = sr_damage(doc, tlang, "cross_fact") - sr_damage(doc, tlang, "cross_other_sr")
            tr.append(trans); sp.append(spec)
            # consistency: re-run own_fact damage vs phase C's
            cons.append(abs(sr_damage(doc, tlang, "own_fact").mean()
                            - r.damage(tlang, "own_fact").mean()))
        # phase C any-relation rate for comparison
        tc, sc = [], []
        for ti, tlang in enumerate(r.langs):
            dmg = {c: r.damage(tlang, c) for c in CONDS[1:]}
            tc.append(dmg["cross_fact"] - dmg["cross_other"])
            sc.append(dmg["own_fact"] - dmg["own_other"])
        rate_sr = np.concatenate(tr).mean() / np.concatenate(sp).mean()
        rate_any = np.concatenate(tc).mean() / np.concatenate(sc).mean()
        pairs_sr[name] = list(zip(tr, sp))
        print(f"{name:<16} {rate_sr:>8.4f} {rate_any:>19.4f} {max(cons):>12.4f}")

    print("\nfair - starved on rate_sr:")
    from analyze_knowneurons import boot_rate_diff
    for p, script in PARTNERS:
        f, s = f"en-{p}-fair", f"en-{p}-starved"
        if f in pairs_sr and s in pairs_sr:
            m, lo, hi = boot_rate_diff(pairs_sr[f], pairs_sr[s], a.boot)
            print(f"  {p} ({script}): {fmt(m, lo, hi)}")

    # ---------- 3. intersection decomposition ----------
    for K in (100, 200):
        files = sorted(glob.glob(str(a.results / f"*_intersect_k{K}.json")))
        if not files:
            continue
        print(f"\n== 3. INTERSECTION DECOMPOSITION (K={K}) ==")
        print(f"{'run':<16} {'nI':>5} {'skip':>5} {'dmg(I)':>7} {'dmg(D)':>7} "
              f"{'rand':>6} {'Ictl':>6} {'lift [CI]':>26} {'conc-|I|/K [CI]':>26}")
        for f in files:
            name = os.path.basename(f)[:-len(f"_intersect_k{K}.json")]
            doc = json.loads(Path(f).read_text())
            langs = doc["langs"]
            dI, dD, dR, dC, conc, nIs, skipped = [], [], [], [], [], [], 0
            for ti, tlang in enumerate(langs):
                gold = doc["gold"][tlang]
                for j, cell in enumerate(doc["cells"][tlang]):
                    if cell["ll"] is None:
                        skipped += 1; continue
                    g = gold[j]
                    ll = np.array(cell["ll"])          # [6, 4]
                    base = ll[0, g]
                    di = base - ll[1, g]
                    dd = base - ll[2, g]
                    dr = base - (ll[3, g] + ll[4, g]) / 2
                    dc = base - ll[5, g]
                    dI.append(di); dD.append(dd); dR.append(dr); dC.append(dc)
                    nIs.append(cell["nI"])
                    tot = di + dd
                    if tot > 0.5:                      # guard near-zero totals
                        conc.append(di / tot - cell["nI"] / K)
            dI, dD, dR, dC = map(np.array, (dI, dD, dR, dC))
            lift = dI - dR
            m, lo, hi = boot_ci(lift, a.boot)
            c, clo, chi = boot_ci(np.array(conc), a.boot)
            print(f"{name:<16} {np.mean(nIs):>5.1f} {skipped:>5d} {dI.mean():>7.3f} "
                  f"{dD.mean():>7.3f} {dR.mean():>6.3f} {dC.mean():>6.3f} "
                  f"{fmt(m, lo, hi):>26} {fmt(c, clo, chi):>26}")


if __name__ == "__main__":
    main()
