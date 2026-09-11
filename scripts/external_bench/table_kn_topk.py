#!/usr/bin/env python
"""Top-K robustness table for parametric sharing (thesis Table A.2), on the
DIFFERENT-surface-form facts -- the same subset as Fig 4.15 / Table A.3.

    python table_kn_topk.py [--results results/knowneurons]

PS_K = same-fact top-K Jaccard - mismatched-fact Jaccard, K in {50,100,200,500},
for the 8 cooled 30B bilingual finals, plus the fair - starved delta per
partner paired over the intersection of the two runs' different-surface-form
fact ids.

Every K column is exactly what `plot_kn_ps.py --k K` computes: same `diff`
mask, same Jaccard helpers, and the same single np.random.default_rng(0)
stream consumed in the same order (boot(PS), boot(same), rate_boot per run),
so the K=100 CIs are the figure's CIs rather than equal up to Monte-Carlo
noise. Note this is NOT the "split by identical gold surface form" block of
analyze_knowneurons.py section 3: that one computes the mismatched-fact
baseline over ALL facts and then subsets (en-de-fair .1215), whereas
plot_kn_ps.py builds the derangement inside the diff subset (.1175).

The paired delta uses each run's own per-fact PS (as in the per-run rows)
aligned by fact_id, and analyze_knowneurons.boot_diff(paired=True).

Writes <results>/table_topk_diff.tex (LaTeX rows only, for \\input into the
tabular) after checking the K=100 PS against a live run of plot_kn_ps.py.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src"))
from analyze_knowneurons import Run, jaccard_matrix, jaccard_mismatched, load_gold_strings, boot_diff  # noqa: E402
from plot_kn_ps import boot, rate_boot, PARTNERS  # noqa: E402

KS = [50, 100, 200, 500]
B = 2000
TEX_MINUS = "−"


def diff_mask(r, gold):
    """plot_kn_ps.py's construction: gold entity spelled differently in the two languages."""
    l1, l2 = r.langs
    return np.array([gold[l1][fi].strip().lower() != gold[l2][fi].strip().lower() for fi in r.fact_ids])


def ps_column(runs, gold, k):
    """plot_kn_ps.py's main loop at cutoff k, RNG stream included."""
    rng = np.random.default_rng(0)
    col = {}
    for name, r in runs.items():
        diff = diff_mask(r, gold)
        A, B_ = r.topk[diff, 0, :], r.topk[diff, 1, :]
        same_j = jaccard_matrix(A, B_, k); mism_j = jaccard_mismatched(A, B_, k)
        ps = same_j - mism_j
        ps_ci = boot(ps, rng)
        boot(same_j, rng)                       # consumed by plot_kn_ps.py; keeps the stream aligned
        trans, spec = [], []
        for tl in r.langs:
            d = {c: r.damage(tl, c)[diff] for c in ("own_fact", "own_other", "cross_fact", "cross_other")}
            spec.append(d["own_fact"] - d["own_other"]); trans.append(d["cross_fact"] - d["cross_other"])
        rate_boot(trans, spec, rng)             # likewise
        ids = r.fact_ids[diff].tolist()
        assert len(set(ids)) == len(ids), f"{name}: duplicate fact ids"
        col[name] = {"n": int(diff.sum()), "ps": ps_ci, "per_fact": dict(zip(ids, ps))}
    return col


def paired_delta(col, p):
    f, s = col[f"en-{p}-fair"]["per_fact"], col[f"en-{p}-starved"]["per_fact"]
    common = sorted(set(f) & set(s))
    xf = np.array([f[i] for i in common]); xs = np.array([s[i] for i in common])
    return len(common), boot_diff(xf, xs, B, paired=True)


def f3(x, minus="-", plus=""):
    """.118 not 0.118; a negative value that rounds to zero keeps its sign (-.000)."""
    s = f"{abs(x):.3f}"
    s = s[1:] if s.startswith("0.") else s
    return (minus if np.signbit(x) else plus) + s


def cell(m, lo, hi, minus="-", plus=""):
    return f"{f3(m, minus, plus)} [{f3(lo, minus, plus)},{f3(hi, minus, plus)}]"


def check_against_plot_kn_ps(res, col100):
    """Run the real plot_kn_ps.py (figures go to a temp dir, not results/) and
    compare its printed K=100 PS and n_diff to ours."""
    with tempfile.TemporaryDirectory() as td:
        for pat in ("*_kn.npz", "*_selection.json", "*_ablation.json"):
            for f in res.glob(pat):
                os.symlink(f.resolve(), Path(td) / f.name)
        out = subprocess.run([sys.executable, str(HERE / "plot_kn_ps.py"), "--results", td, "--k", "100"],
                             capture_output=True, text=True, check=True).stdout
    ref = {m[1]: (int(m[2]), float(m[3]))
           for m in re.finditer(r"^(\S+)\s+n_diff=\s*(\d+) PS=(-?[\d.]+)", out, re.M)}
    bad = []
    for name, c in col100.items():
        if name not in ref:
            bad.append(f"{name}: not in plot_kn_ps.py output"); continue
        n_ref, ps_ref = ref[name]
        if n_ref != c["n"] or abs(ps_ref - c["ps"][0]) >= 5e-4:
            bad.append(f"{name}: ours n={c['n']} PS={c['ps'][0]:.4f}, plot_kn_ps.py n={n_ref} PS={ps_ref:.4f}")
    return ref, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "knowneurons")
    a = ap.parse_args()
    res = a.results
    runs = {f"en-{p}-{c}": Run(res, f"en-{p}-{c}") for p in PARTNERS for c in ("fair", "starved")}
    gold = load_gold_strings(["en", *PARTNERS])
    cols = {k: ps_column(runs, gold, k) for k in KS}
    deltas = {p: {k: paired_delta(cols[k], p) for k in KS} for p in PARTNERS}

    # ---- plain text -----------------------------------------------------------
    w = 22
    print("PS_K on different-surface-form facts (95% bootstrap CI over facts, B=2000)")
    print(f"{'run':<16}{'n':>5}  " + "".join(f"{'K=' + str(k):<{w}}" for k in KS))
    for name in runs:
        print(f"{name:<16}{cols[100][name]['n']:>5}  " + "".join(f"{cell(*cols[k][name]['ps']):<{w}}" for k in KS))
    print("-" * (23 + w * len(KS)))
    print("fair - starved, paired over the intersection of different-surface-form fact ids")
    for p in PARTNERS:
        n = deltas[p][100][0]
        assert all(deltas[p][k][0] == n for k in KS)
        print(f"{'d en-' + p:<16}{n:>5}  " + "".join(f"{cell(*deltas[p][k][1], plus='+'):<{w}}" for k in KS))

    # ---- requirement: K=100 reproduces plot_kn_ps.py (Fig 4.15 / Table A.3) --
    ref, bad = check_against_plot_kn_ps(res, cols[100])
    if bad:
        sys.exit("K=100 does NOT reproduce plot_kn_ps.py -- table not written:\n  " + "\n  ".join(bad))
    print(f"\ncheck: K=100 PS matches plot_kn_ps.py for all {len(ref)} runs "
          f"(max |diff| {max(abs(ref[n][1] - cols[100][n]['ps'][0]) for n in runs):.1e}, n identical)")

    # ---- LaTeX rows -----------------------------------------------------------
    L = ["% generated by scripts/external_bench/table_kn_topk.py -- do not edit by hand",
         "% PS_K = same-fact top-K Jaccard - mismatched-fact Jaccard, different-surface-form facts,",
         "% 95% bootstrap CI over facts (B=2000); delta rows paired over the fact-id intersection.",
         "% n per run: " + ", ".join(f"{n} {cols[100][n]['n']}" for n in runs),
         "% n per intersection: " + ", ".join(f"{p} {deltas[p][100][0]}" for p in PARTNERS)]
    for name in runs:
        L.append(f"{name} & " + " & ".join(cell(*cols[k][name]["ps"], minus=TEX_MINUS) for k in KS) + r" \\")
    L.append(r"\midrule")
    for p in PARTNERS:
        L.append(rf"$\Delta$ en-{p} & " + " & ".join(cell(*deltas[p][k][1], minus=TEX_MINUS, plus="+") for k in KS) + r" \\")
    out = res / "table_topk_diff.tex"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
