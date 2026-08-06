#!/usr/bin/env python
"""Per-layer x per-checkpoint x per-pooling alignment for the bilingual runs.

Consumes the 48-checkpoint sweep (`run_pooling_trajectory.sh`) and emits the
trajectory view the single-budget tables in `results/alignment_pooling/` cannot
give: how cross-lingual alignment develops with depth AND with token budget,
under each of the four sentence poolings.

    python analyze_pooling_trajectory.py RESULTS_DIR [--out-dir DIR]

Emits, per pooling:
  - a per-layer x per-budget grid of `mutual_nn` for every (pair, tokenizer)
  - the fair-minus-starved gap on the same grid
  - depth-to-threshold and peak-layer summaries vs budget
  - a SATURATION flag per cell, because a peak-layer gap between two ceilinged
    numbers is uninterpretable (see results/alignment_pooling/README.md T6) and
    is the single easiest way to misread these tables.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

POOLINGS = ["mean", "mean_nobos", "weighted", "last"]
PARTNERS = ["de", "fr", "ar", "zh"]
TOKS = ["fair", "starved"]
# Suffix -> per-language token budget label. The bilinguals mix 50/50, so a
# "-23b" run has seen ~11.4B of each language; the label is the TOTAL, matching
# how models.json and every other table in this repo name them.
BUDGETS = [("-2b", "2B"), ("-5b", "5B"), ("-10b", "10B"),
           ("-15b", "15B"), ("-23b", "23B"), ("", "30B")]
VARIANT = "centered"
METRIC = "mutual_nn"
SAT = 0.95            # both arms above this -> peak/late gaps are ceilinged
THRESH = 0.90


def load(results: Path):
    """{(partner, tok, budget_label): doc}, skipping anything not yet run."""
    out, missing = {}, []
    for p in PARTNERS:
        for t in TOKS:
            for suf, label in BUDGETS:
                f = results / f"en-{p}-{t}{suf}.json"
                if f.exists():
                    out[(p, t, label)] = json.loads(f.read_text())
                else:
                    missing.append(f.name)
    return out, missing


def series(doc, pooling, pair):
    pairs = doc["poolings"][pooling]["pairs"] if "poolings" in doc else doc["pairs"]
    return np.array([L[METRIC] for L in pairs[pair][VARIANT]["per_layer"]])


def depth_to(v, thresh=THRESH):
    hit = np.nonzero(v >= thresh)[0]
    return int(hit[0]) if len(hit) else None


def grid_table(docs, pooling, p, tok, n_layers):
    """Per-layer rows x per-budget columns of mutual_nn."""
    labels = [l for _, l in BUDGETS if (p, tok, l) in docs]
    if not labels:
        return []
    cols = {l: series(docs[(p, tok, l)], pooling, f"en-{p}") for l in labels}
    md = [f"#### en-{p} / {tok}", "",
          "| layer | " + " | ".join(labels) + " |",
          "|---" * (len(labels) + 1) + "|"]
    for ly in range(n_layers):
        md.append(f"| L{ly} | " + " | ".join(f"{cols[l][ly]:.3f}" for l in labels) + " |")
    md.append("| **peak** | " + " | ".join(f"**{cols[l].max():.3f}**" for l in labels) + " |")
    md.append("| **depth to .90** | "
              + " | ".join(str(depth_to(cols[l])) if depth_to(cols[l]) is not None
                           else "-" for l in labels) + " |")
    return md + [""]


def gap_table(docs, pooling, n_layers):
    """fair - starved per layer x budget, one block per pair, with SAT flags."""
    md = []
    for p in PARTNERS:
        labels = [l for _, l in BUDGETS
                  if (p, "fair", l) in docs and (p, "starved", l) in docs]
        if not labels:
            continue
        f = {l: series(docs[(p, "fair", l)], pooling, f"en-{p}") for l in labels}
        s = {l: series(docs[(p, "starved", l)], pooling, f"en-{p}") for l in labels}
        md += [f"#### en-{p}: fair - starved", "",
               "| layer | " + " | ".join(labels) + " |",
               "|---" * (len(labels) + 1) + "|"]
        for ly in range(n_layers):
            md.append(f"| L{ly} | "
                      + " | ".join(f"{f[l][ly] - s[l][ly]:+.3f}" for l in labels) + " |")
        md.append("| **peak gap** | "
                  + " | ".join(f"{f[l].max() - s[l].max():+.3f}" for l in labels) + " |")
        # The flag that stops the peak-gap row being misread.
        md.append("| **ceilinged?** | "
                  + " | ".join("**SAT**" if min(f[l].max(), s[l].max()) > SAT else "no"
                               for l in labels) + " |")
        md += [""]
    return md


def dip(v):
    """The mid-stack trough: (layer, depth below the best pre-trough layer).

    Per-layer alignment is NOT monotone in depth. Every model past ~10B tokens
    develops a trough somewhere in L2-L12 -- the same feature CLAUDE.md section
    6b diagnosed for `fr-starved` (where CKA and d\' collapse and recover
    together). Its LAYER moves with token budget and differs between tokenizer
    conditions, which is why a fixed layer band is not a safe trajectory
    summary: the band ends up measuring where each model\'s trough happens to
    sit rather than how well it aligns.
    """
    i = int(np.argmin(v[2:13])) + 2
    return i, float(v[1:i + 1].max() - v[i])


def dip_table(docs, pooling):
    md = [f"#### trough position and depth (`{pooling}`)", "",
          "`Lx(d)` = trough at layer x, d below the best pre-trough layer. "
          "A depth near 0 means no trough exists yet.", "",
          "| pair | tok | " + " | ".join(l for _, l in BUDGETS) + " |",
          "|---" * (len(BUDGETS) + 2) + "|"]
    for p in PARTNERS:
        for tok in TOKS:
            cells = []
            for _, l in BUDGETS:
                if (p, tok, l) not in docs:
                    cells.append("-")
                    continue
                i, d = dip(series(docs[(p, tok, l)], pooling, f"en-{p}"))
                cells.append(f"L{i}({d:.2f})")
            md.append(f"| en-{p} | {tok} | " + " | ".join(cells) + " |")
    return md + [""]


def summary(docs, n_layers):
    """One row per (pooling, pair, budget): peak, gap, depth, saturation."""
    rows = []
    for pooling in POOLINGS:
        for p in PARTNERS:
            for _, l in BUDGETS:
                if (p, "fair", l) not in docs or (p, "starved", l) not in docs:
                    continue
                f = series(docs[(p, "fair", l)], pooling, f"en-{p}")
                s = series(docs[(p, "starved", l)], pooling, f"en-{p}")
                df, ds = depth_to(f), depth_to(s)
                rows.append({
                    "pooling": pooling, "pair": f"en-{p}", "budget": l,
                    "fair_peak": float(f.max()), "starved_peak": float(s.max()),
                    "peak_gap": float(f.max() - s.max()),
                    "mid_gap_L5_8": float((f[5:9] - s[5:9]).mean()),
                    "depth_fair": df, "depth_starved": ds,
                    "delay": None if df is None or ds is None else ds - df,
                    "saturated": bool(min(f.max(), s.max()) > SAT),
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    docs, missing = load(args.results)
    if not docs:
        sys.exit(f"no model JSONs in {args.results}")
    n_layers = next(iter(docs.values()))["n_layers"]
    print(f"loaded {len(docs)}/48 checkpoints, {n_layers} layers"
          + (f"; MISSING {len(missing)}: {missing}" if missing else "; complete"))

    rows = summary(docs, n_layers)
    out = args.out_dir or args.results
    out.mkdir(parents=True, exist_ok=True)

    head = [f"# Bilingual alignment trajectory: layer x budget x pooling", "",
            f"`{METRIC}`, `{VARIANT}` variant, each bilingual on its OWN trained "
            f"pair, FLORES+ dev+devtest (n=2009). {len(docs)}/48 checkpoints.", "",
            "**Budgets are TOTAL tokens**; the bilinguals mix 50/50, so a 23B run "
            "has seen ~11.4B of each language. Every checkpoint at 23B and below "
            "is mid-stable at peak LR 3.0e-3 (decay starts at 24B), so those "
            "columns are LR-matched by construction; the **30B column is cooled** "
            "and is NOT on the same curve -- see CLAUDE.md section 6.", "",
            f"`SAT` = both arms above {SAT}: the peak gap there is a ceiling "
            "effect, not a measurement. Do not read a decay-with-budget trend "
            "across a row that changes from `no` to `SAT`.", ""]

    (out / "trajectory_summary.json").write_text(json.dumps(rows, indent=2))

    for pooling in POOLINGS:
        md = list(head) + [f"## pooling: `{pooling}`", "", "### Absolute mutual_nn", ""]
        for p in PARTNERS:
            for tok in TOKS:
                md += grid_table(docs, pooling, p, tok, n_layers)
        md += ["### fair - starved", ""] + gap_table(docs, pooling, n_layers)
        md += ["### mid-stack trough (why a fixed layer band is unsafe)", ""] \
            + dip_table(docs, pooling)
        (out / f"trajectory_{pooling}.md").write_text("\n".join(md) + "\n")
        print(f"wrote {out}/trajectory_{pooling}.md")

    # Compact cross-pooling view: the mid-stack gap and the peak gap vs budget.
    md = list(head) + ["## Cross-pooling summary", "",
                       "`mid` = mean fair-starved gap over L5-8; `peak` = gap "
                       "between each arm's own best layer (⚠ = ceilinged).", ""]
    for p in PARTNERS:
        labels = [l for _, l in BUDGETS if any(r["pair"] == f"en-{p}"
                                               and r["budget"] == l for r in rows)]
        md += [f"### en-{p}", "",
               "| pooling | " + " | ".join(f"{l} mid / peak" for l in labels) + " |",
               "|---" * (len(labels) + 1) + "|"]
        for pooling in POOLINGS:
            cells = []
            for l in labels:
                r = next((r for r in rows if r["pooling"] == pooling
                          and r["pair"] == f"en-{p}" and r["budget"] == l), None)
                cells.append("-" if r is None else
                             f"{r['mid_gap_L5_8']:+.3f} / {r['peak_gap']:+.3f}"
                             + (" ⚠" if r["saturated"] else ""))
            md.append(f"| {pooling} | " + " | ".join(cells) + " |")
        md += [""]
    (out / "trajectory_summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}/trajectory_summary.md")


if __name__ == "__main__":
    main()
