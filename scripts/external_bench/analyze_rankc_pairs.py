#!/usr/bin/env python
"""RankC on MuBench-BMLAMA for explicit (fair, starved) checkpoint PAIRS --
the matched-validation-loss comparison (CLAUDE.md 6k). Pairs are chosen on
FLORES BPB (results/rankc/polyfact_traj_rankc.json), mid-stable only, and
each pair's residual Δbpb (partner, English) is printed next to its Δ RankC
so the reader sees how well the match holds.

    python analyze_rankc_pairs.py [--workdir /mnt/scratch/xscript_rankc]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_rankc import load, boot_mean, VARIANTS  # noqa: E402  (same dir)

ROOT = Path(__file__).resolve().parents[2]
PAIRS = {
    "de": [("en-de-fair-5b", "en-de-starved-23b"), ("en-de-fair-5b", "en-de-starved-15b")],
    "fr": [("en-fr-fair-10b", "en-fr-starved-23b"), ("en-fr-fair-5b", "en-fr-starved-10b")],
    "ar": [("en-ar-fair-5b", "en-ar-starved-23b"), ("en-ar-fair-10b", "en-ar-starved-23b")],
    "zh": [("en-zh-fair-10b", "en-zh-starved-23b"), ("en-zh-fair-5b", "en-zh-starved-10b")],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/mnt/scratch/xscript_rankc")
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    res = Path(args.workdir) / "results" / "rankc"
    bpb = {r["run"]: r["bpb"] for r in json.load(open(ROOT / "results" / "rankc" / "polyfact_traj_rankc.json"))["per_checkpoint"]}
    lines = ["## MuBench-BMLAMA RankC at matched validation loss (fair − starved, paired over items)", "",
             "| partner | fair ckpt | starved ckpt | Δbpb partner | Δbpb en | variant | RankC fair | RankC starved | Δ RankC [95% CI] |",
             "|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for p, prs in PAIRS.items():
        for f, s in prs:
            fp, sp = res / f"{f}_bmlamamub.json", res / f"{s}_bmlamamub.json"
            if not (fp.exists() and sp.exists()):
                lines.append(f"| {p} | {f} | {s} | | | (missing) | | | |")
                continue
            F, S = load(fp), load(sp)
            common = sorted(set(F["row_ids"]) & set(S["row_ids"]))
            fi = {r: i for i, r in enumerate(F["row_ids"])}; si = {r: i for i, r in enumerate(S["row_ids"])}
            a = np.array([fi[r] for r in common]); b = np.array([si[r] for r in common])
            db = bpb[f][p] - bpb[s][p]; de = bpb[f]["en"] - bpb[s]["en"]
            for v in VARIANTS:
                d = F["rc"][v][a] - S["rc"][v][b]
                m, lo, hi = boot_mean(d, rng, args.B)
                star = "*" if lo > 0 or hi < 0 else ""
                out[f"{f}|{s}|{v}"] = {"delta": m, "lo": lo, "hi": hi, "dbpb_partner": db, "dbpb_en": de,
                                       "rc_fair": float(F["rc"][v][a].mean()), "rc_starved": float(S["rc"][v][b].mean())}
                lines.append(f"| {p} | {f} | {s} | {db:+.4f} | {de:+.4f} | {v} | {F['rc'][v][a].mean():.3f} | "
                             f"{S['rc'][v][b].mean():.3f} | **{m:+.4f}** [{lo:+.4f}, {hi:+.4f}]{star} |")
    md = "\n".join(lines)
    print(md)
    (ROOT / "results" / "rankc" / "bmlama_rankc_matched.md").write_text(md + "\n")
    json.dump(out, open(ROOT / "results" / "rankc" / "bmlama_rankc_matched.json", "w"), indent=1)


if __name__ == "__main__":
    main()
