#!/usr/bin/env python
"""6b alignment (centred d', CKA) for explicit fair/starved checkpoint pairs --
the matched-loss and 30B-final comparisons of CLAUDE.md 6k. Reads
run_alignment.py's per-run JSONs; deltas are paired over the 2009 FLORES+
sentences at each model's own peak layer and at the fixed ref layer (6b's
rule: report both).

    python analyze_align_pairs.py [--results /mnt/scratch/xscript_rankc/results/alignment]
"""
import argparse, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
PAIRS = {"de": [("en-de-fair", "en-de-starved"), ("en-de-fair-5b", "en-de-starved-23b"), ("en-de-fair-5b", "en-de-starved-15b")],
         "fr": [("en-fr-fair", "en-fr-starved"), ("en-fr-fair-10b", "en-fr-starved-23b"), ("en-fr-fair-5b", "en-fr-starved-10b")],
         "ar": [("en-ar-fair", "en-ar-starved"), ("en-ar-fair-5b", "en-ar-starved-23b"), ("en-ar-fair-10b", "en-ar-starved-23b")],
         "zh": [("en-zh-fair", "en-zh-starved"), ("en-zh-fair-10b", "en-zh-starved-23b"), ("en-zh-fair-5b", "en-zh-starved-10b")]}

def boot(x, rng, B=2000):
    idx = rng.integers(0, len(x), size=(B, len(x))); m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="/mnt/scratch/xscript_rankc/results/alignment"); a = ap.parse_args()
    rng = np.random.default_rng(0); R = Path(a.results)
    lines = ["| partner | fair | starved | d' fair @peak (L) | d' starved @peak (L) | Δ d' @peak [95% CI] | Δ d' @L12 [95% CI] | CKA fair/starved @peak |", "|---|---|---|---|---|---|---|---|"]
    out = {}
    for p, prs in PAIRS.items():
        for f, s in prs:
            fp, sp = R / f"{f}.json", R / f"{s}.json"
            if not (fp.exists() and sp.exists()):
                lines.append(f"| {p} | {f} | {s} | (missing) | | | | |"); continue
            F = json.load(open(fp))["pairs"][f"en-{p}"]["centered"]; S = json.load(open(sp))["pairs"][f"en-{p}"]["centered"]
            df = np.array(F["best"]["dprime_sym_q"]) - np.array(S["best"]["dprime_sym_q"]); lo, hi = boot(df, rng)
            dr = np.array(F["ref"]["dprime_sym_q"]) - np.array(S["ref"]["dprime_sym_q"]); lo2, hi2 = boot(dr, rng)
            out[f"{f}|{s}"] = {"d_peak": float(df.mean()), "lo": lo, "hi": hi, "d_ref": float(dr.mean()), "lo_ref": lo2, "hi_ref": hi2,
                               "fair_peak": F["best"]["dprime_sym"], "starved_peak": S["best"]["dprime_sym"], "fair_layer": F["best_layer"], "starved_layer": S["best_layer"]}
            lines.append(f"| {p} | {f} | {s} | {F['best']['dprime_sym']:.2f} (L{F['best_layer']}) | {S['best']['dprime_sym']:.2f} (L{S['best_layer']}) | "
                         f"**{df.mean():+.3f}** [{lo:+.3f}, {hi:+.3f}]{'*' if lo>0 or hi<0 else ''} | {dr.mean():+.3f} [{lo2:+.3f}, {hi2:+.3f}]{'*' if lo2>0 or hi2<0 else ''} | {F['best'].get('cka', float('nan')):.3f} / {S['best'].get('cka', float('nan')):.3f} |")
    md = "\n".join(lines); print(md)
    (ROOT / "results" / "rankc" / "alignment_pairs.md").write_text(md + "\n"); json.dump(out, open(ROOT / "results" / "rankc" / "alignment_pairs.json", "w"), indent=1)

if __name__ == "__main__":
    main()
