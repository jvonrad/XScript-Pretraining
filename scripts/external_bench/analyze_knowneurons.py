#!/usr/bin/env python
"""Cross-lingual knowledge-neuron analysis over run_knowneurons.py outputs.

    python analyze_knowneurons.py [--results DIR] [--k 100] [--boot 2000]

Sections:

  1. FACT SELECTION -- per-model own-language PolyFact accuracy, known /
     shared fact counts, and the fair-vs-starved shared-set overlap (the two
     conditions are measured on partly different fact sets; the paired
     contrast below restricts to the intersection).
  2. COMPLETENESS -- IG sum(Attr) vs G(1)-G(0) relative error. A gate, not a
     result: large errors mean the attribution maps are untrustworthy.
  3. dKS OVERLAP -- same-fact cross-language top-K Jaccard minus
     mismatched-fact baseline, bootstrap CI over facts. Raw overlap is NOT
     comparable across tokenizers (different fact sets, different
     fragmentation); dKS is the quotable quantity. Also split by whether the
     gold surface form is IDENTICAL in both languages (Latin-script entity
     names often are, which can inflate same-script sharing trivially --
     the same trap Ifergan et al. call out for consistency).
  4. ABLATION -- per direction (ablate source-language top-K, measure target
     language): damage = ll_none - ll_cond on the gold candidate.
       specificity   = damage(own_fact)  - damage(own_other)
       transfer      = damage(cross_fact) - damage(cross_other)
       transfer rate = transfer / specificity   (the Ifergan-style number)
     plus flip rates and the random-K control.
  5. FAIR - STARVED -- the headline contrast per partner, unpaired over each
     model's own facts AND paired over the fact intersection.

Pure numpy + stdlib except for reloading PolyFact surface forms (network).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

CONDS = ("none", "own_fact", "cross_fact", "own_other", "cross_other", "random")
PARTNERS = [("de", "same"), ("fr", "same"), ("ar", "cross"), ("zh", "cross")]


def jaccard_matrix(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Per-fact Jaccard between two [n, K_store] top-id arrays at cutoff k."""
    out = np.empty(len(a), np.float64)
    for i in range(len(a)):
        sa, sb = set(a[i, :k].tolist()), set(b[i, :k].tolist())
        out[i] = len(sa & sb) / len(sa | sb)
    return out


def jaccard_mismatched(a: np.ndarray, b: np.ndarray, k: int, shifts=(1, 2, 3)) -> np.ndarray:
    """Mean mismatched-fact Jaccard per fact (deterministic derangements)."""
    n = len(a)
    acc = np.zeros(n, np.float64)
    for s in shifts:
        rolled = np.roll(np.arange(n), -s)
        for i in range(n):
            sa, sb = set(a[i, :k].tolist()), set(b[rolled[i], :k].tolist())
            acc[i] += len(sa & sb) / len(sa | sb)
    return acc / len(shifts)


def boot_ci(values: np.ndarray, b: int, seed: int = 0):
    """Mean and 95% bootstrap CI over facts."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = values[rng.integers(0, n, (b, n))].mean(1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def boot_diff(x: np.ndarray, y: np.ndarray, b: int, paired: bool, seed: int = 0):
    """Bootstrap CI on mean(x) - mean(y); paired resamples common indices."""
    rng = np.random.default_rng(seed)
    if paired:
        assert len(x) == len(y)
        idx = rng.integers(0, len(x), (b, len(x)))
        d = x[idx].mean(1) - y[idx].mean(1)
    else:
        d = (x[rng.integers(0, len(x), (b, len(x)))].mean(1)
             - y[rng.integers(0, len(y), (b, len(y)))].mean(1))
    return (float(x.mean() - y.mean()),
            float(np.quantile(d, .025)), float(np.quantile(d, .975)))


def pooled_rate(pairs):
    """Concatenated (transfer, specificity) over both directions."""
    trans = np.concatenate([t for t, _ in pairs])
    spec = np.concatenate([s for _, s in pairs])
    return trans, spec


def boot_rate_diff(pa, pb, b, seed=0):
    """Bootstrap CI on rate(A) - rate(B), rate = mean(trans)/mean(spec).

    Facts are resampled jointly for trans and spec within each model (they are
    measured on the same facts, and the two directions of one model share
    facts, so one fact-resample drives both directions); models are resampled
    independently."""
    rng = np.random.default_rng(seed)
    (ta, sa), (tb, sb) = pooled_rate(pa), pooled_rate(pb)
    n_a, n_b = len(pa[0][0]), len(pb[0][0])   # facts per direction
    diffs = np.empty(b)
    for i in range(b):
        ia = rng.integers(0, n_a, n_a)
        ib = rng.integers(0, n_b, n_b)
        ia2 = np.concatenate([ia, ia + n_a]); ib2 = np.concatenate([ib, ib + n_b])
        diffs[i] = (ta[ia2].mean() / sa[ia2].mean()
                    - tb[ib2].mean() / sb[ib2].mean())
    m = ta.mean() / sa.mean() - tb.mean() / sb.mean()
    return float(m), float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))


def fmt(m, lo, hi, star_if_excludes_zero=True):
    star = "*" if star_if_excludes_zero and (lo > 0 or hi < 0) else " "
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}"


class Run:
    def __init__(self, res: Path, name: str):
        self.name = name
        self.sel = json.loads((res / f"{name}_selection.json").read_text())
        z = np.load(res / f"{name}_kn.npz", allow_pickle=False)
        self.fact_ids = z["fact_ids"]
        self.topk = z["topk_idx"]          # [n, 2, K_store]
        self.layer_sum = z["layer_sum"]
        self.g1, self.g0, self.attr_total = z["g1"], z["g0"], z["attr_total"]
        self.langs = [str(v) for v in z["langs"]]
        self.abl = json.loads((res / f"{name}_ablation.json").read_text())

    def damage(self, tlang: str, cond: str) -> np.ndarray:
        ci_gold = np.array(self.abl["gold"][tlang])
        lls = np.array(self.abl["ll"][tlang])   # [n, n_cond, 4]
        mi, m0 = CONDS.index(cond), CONDS.index("none")
        rows = np.arange(len(ci_gold))
        return lls[rows, m0, ci_gold] - lls[rows, mi, ci_gold]

    def flips(self, tlang: str, cond: str) -> np.ndarray:
        ci_gold = np.array(self.abl["gold"][tlang])
        lls = np.array(self.abl["ll"][tlang])
        mi = CONDS.index(cond)
        return (lls[:, mi].argmax(1) != ci_gold).astype(np.float64)


def load_gold_strings(langs, limit=None):
    from xscript.eval.c5_tasks.polyfact import utils as pf
    out = {}
    for lang in langs:
        ds = pf.build_dataset(lang=lang)["test"]
        out[lang] = [d["choices"][d["label"]] for d in ds]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path("/home/ubuntu/xscript_kn/results/knowneurons"))
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--ks", type=int, nargs="*", default=[50, 100, 200, 500])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--no-gold-strings", action="store_true",
                    help="skip the identical-surface-form split (offline)")
    a = ap.parse_args()

    runs = {}
    for f in sorted(glob.glob(str(a.results / "*_ablation.json"))):
        name = os.path.basename(f)[:-len("_ablation.json")]
        try:
            runs[name] = Run(a.results, name)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
    if not runs:
        sys.exit("no completed runs found")

    all_langs = sorted({l for r in runs.values() for l in r.langs})
    gold_str = None if a.no_gold_strings else load_gold_strings(all_langs)

    # ---------- 1. fact selection ----------
    print("== 1. FACT SELECTION (PolyFact, n after length filter; acc = argmax raw ll) ==")
    print(f"{'run':<18} {'langs':<8} {'acc_l1':>7} {'acc_l2':>7} {'known1':>7} "
          f"{'known2':>7} {'shared':>7} {'same-answer%':>12}")
    for name, r in runs.items():
        accs, known = [], []
        keep = r.sel["kept_facts"]
        for lang in r.langs:
            ll = np.array(r.sel["ll"][lang]); g = np.array(r.sel["gold"][lang])
            hit = ll.argmax(1) == g
            accs.append(hit.mean()); known.append(hit.sum())
        sa = ""
        if gold_str:
            l1, l2 = r.langs
            same = [gold_str[l1][fi].strip().lower() == gold_str[l2][fi].strip().lower()
                    for fi in r.fact_ids]
            sa = f"{100*np.mean(same):.1f}"
        print(f"{name:<18} {'/'.join(r.langs):<8} {accs[0]:>7.4f} {accs[1]:>7.4f} "
              f"{known[0]:>7d} {known[1]:>7d} {len(r.fact_ids):>7d} {sa:>12}")

    # fair/starved shared-set overlap
    print("\nfair-vs-starved shared-fact-set overlap (Jaccard of fact ids):")
    for p, script in PARTNERS:
        f, s = f"en-{p}-fair", f"en-{p}-starved"
        if f in runs and s in runs:
            sf, ss = set(runs[f].fact_ids.tolist()), set(runs[s].fact_ids.tolist())
            print(f"  {p} ({script}): J={len(sf & ss)/len(sf | ss):.3f} "
                  f"(inter={len(sf & ss)})")

    # ---------- 2. completeness ----------
    print("\n== 2. IG COMPLETENESS (|sum Attr - (G1-G0)| / |G1-G0|) ==")
    for name, r in runs.items():
        rel = np.abs(r.attr_total - (r.g1 - r.g0)) / np.maximum(np.abs(r.g1 - r.g0), 1e-9)
        print(f"  {name:<18} median {np.median(rel):.4f}  p90 {np.quantile(rel, .9):.4f}")

    # ---------- 3. dKS ----------
    print(f"\n== 3. CROSS-LANGUAGE TOP-K OVERLAP (dKS = same-fact J - mismatched-fact J) ==")
    dks_store = {}
    for k in a.ks:
        print(f"\n-- K={k} --")
        print(f"{'run':<18} {'sameJ':>7} {'mismJ':>7} {'dKS [95% CI]':>28}")
        for name, r in runs.items():
            same = jaccard_matrix(r.topk[:, 0], r.topk[:, 1], k)
            mism = jaccard_mismatched(r.topk[:, 0], r.topk[:, 1], k)
            d = same - mism
            m, lo, hi = boot_ci(d, a.boot)
            print(f"{name:<18} {same.mean():>7.4f} {mism.mean():>7.4f} "
                  f"{fmt(m, lo, hi):>28}")
            if k == a.k:
                dks_store[name] = d

    if gold_str:
        print(f"\n-- dKS at K={a.k}, split by identical gold surface form --")
        print(f"{'run':<18} {'n_same':>7} {'dKS_same':>10} {'n_diff':>7} {'dKS_diff':>10}")
        for name, r in runs.items():
            l1, l2 = r.langs
            same_ans = np.array([gold_str[l1][fi].strip().lower()
                                 == gold_str[l2][fi].strip().lower()
                                 for fi in r.fact_ids])
            d = dks_store[name]
            n1, n0 = int(same_ans.sum()), int((~same_ans).sum())
            v1 = d[same_ans].mean() if n1 else float("nan")
            v0 = d[~same_ans].mean() if n0 else float("nan")
            print(f"{name:<18} {n1:>7d} {v1:>10.4f} {n0:>7d} {v0:>10.4f}")

    # layer profile of the top-K neurons
    print(f"\n-- layer distribution of top-{a.k} neurons (share in layers 0-3 / 4-7 / 8-11 / 12-15) --")
    for name, r in runs.items():
        for li, lang in enumerate(r.langs):
            layers = (r.topk[:, li, :a.k] // 5632).ravel()
            qs = [float(((layers >= q*4) & (layers < q*4+4)).mean()) for q in range(4)]
            print(f"  {name:<18} {lang}: " + " / ".join(f"{v:.2f}" for v in qs))

    # ---------- 4. ablation ----------
    print(f"\n== 4. ABLATION (K={runs[list(runs)[0]].abl['k']}; damage = drop in gold ll; "
          f"flip = argmax leaves gold) ==")
    hdr = (f"{'run':<18} {'dir':<9} {'own':>7} {'ownCtl':>7} {'cross':>7} "
           f"{'crossCtl':>8} {'rand':>7} {'spec':>7} {'transfer [CI]':>26} {'rate':>6} {'flip_own':>8} {'flip_cross':>10}")
    print(hdr)
    rate_store = {}
    for name, r in runs.items():
        pairs = []
        for ti, tlang in enumerate(r.langs):
            slang = r.langs[1 - ti]
            dmg = {c: r.damage(tlang, c) for c in CONDS[1:]}
            spec = dmg["own_fact"] - dmg["own_other"]
            trans = dmg["cross_fact"] - dmg["cross_other"]
            m, lo, hi = boot_ci(trans, a.boot)
            rate = trans.mean() / spec.mean() if spec.mean() > 0 else float("nan")
            pairs.append((trans, spec))
            fo = r.flips(tlang, "own_fact").mean()
            fc = r.flips(tlang, "cross_fact").mean()
            print(f"{name:<18} {slang+'->'+tlang:<9} {dmg['own_fact'].mean():>7.3f} "
                  f"{dmg['own_other'].mean():>7.3f} {dmg['cross_fact'].mean():>7.3f} "
                  f"{dmg['cross_other'].mean():>8.3f} {dmg['random'].mean():>7.3f} "
                  f"{spec.mean():>7.3f} {fmt(m, lo, hi):>26} {rate:>6.3f} "
                  f"{fo:>8.3f} {fc:>10.3f}")
        rate_store[name] = pairs

    # ---------- 5. fair - starved ----------
    print("\n== 5. FAIR - STARVED (the headline contrast) ==")
    print("\n-- dKS (unpaired over each model's facts; paired over the intersection) --")
    print(f"{'partner':<8} {'script':<6} {'unpaired diff [CI]':>28} {'paired diff [CI]':>28} {'n_pair':>7}")
    for p, script in PARTNERS:
        f, s = f"en-{p}-fair", f"en-{p}-starved"
        if f not in runs or s not in runs:
            print(f"{p:<8} {script:<6} (missing run)"); continue
        rf, rs = runs[f], runs[s]
        m, lo, hi = boot_diff(dks_store[f], dks_store[s], a.boot, paired=False)
        common = sorted(set(rf.fact_ids.tolist()) & set(rs.fact_ids.tolist()))
        pf = {fi: i for i, fi in enumerate(rf.fact_ids.tolist())}
        ps = {fi: i for i, fi in enumerate(rs.fact_ids.tolist())}
        xf = dks_store[f][[pf[c] for c in common]]
        xs = dks_store[s][[ps[c] for c in common]]
        pm, plo, phi = boot_diff(xf, xs, a.boot, paired=True)
        print(f"{p:<8} {script:<6} {fmt(m, lo, hi):>28} {fmt(pm, plo, phi):>28} "
              f"{len(common):>7d}")

    print("\n-- ablation TRANSFER RATE (pooled both directions; rate = "
          "mean transfer / mean specificity, the tokenizer-comparable unit) --")
    print(f"{'partner':<8} {'script':<6} {'fair':>8} {'starved':>8} {'diff [CI]':>28}")
    for p, script in PARTNERS:
        f, s = f"en-{p}-fair", f"en-{p}-starved"
        if f not in runs or s not in runs:
            continue
        tf, sf_ = pooled_rate(rate_store[f]); ts, ss = pooled_rate(rate_store[s])
        m, lo, hi = boot_rate_diff(rate_store[f], rate_store[s], a.boot)
        print(f"{p:<8} {script:<6} {tf.mean()/sf_.mean():>8.4f} "
              f"{ts.mean()/ss.mean():>8.4f} {fmt(m, lo, hi):>28}")

    if gold_str:
        print("\n-- dKS restricted to DIFFERENT gold surface forms "
              "(the honest basis for same- vs cross-script comparisons) --")
        print(f"{'partner':<8} {'script':<6} {'fair':>8} {'starved':>8} "
              f"{'diff [CI] (unpaired)':>28}")
        for p, script in PARTNERS:
            f, s = f"en-{p}-fair", f"en-{p}-starved"
            if f not in runs or s not in runs:
                continue
            sub = {}
            for name in (f, s):
                r = runs[name]
                l1, l2 = r.langs
                diffa = np.array([gold_str[l1][fi].strip().lower()
                                  != gold_str[l2][fi].strip().lower()
                                  for fi in r.fact_ids])
                sub[name] = dks_store[name][diffa]
            m, lo, hi = boot_diff(sub[f], sub[s], a.boot, paired=False)
            print(f"{p:<8} {script:<6} {sub[f].mean():>8.4f} {sub[s].mean():>8.4f} "
                  f"{fmt(m, lo, hi):>28}")


if __name__ == "__main__":
    main()
