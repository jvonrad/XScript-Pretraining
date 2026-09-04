#!/usr/bin/env python
"""Cross-lingual consistency (RankC) vs validation loss across the bilingual
checkpoint trajectories, both tokenizer conditions -- the matched-loss
control for the PolyFact RankC result.

Reads run_polyfact_traj.py's per-checkpoint JSONs. For each checkpoint:
RankC(en, X) over QID-aligned candidates (summed-loglik ranking), P@1 per
language, FLORES BPB per language. Then, per partner:

  * the trajectory table (tokens, bpb_en, bpb_X, acc, RankC [CI]) per condition;
  * MATCHED-LOSS pairs: for each fair checkpoint, the starved checkpoint
    whose partner-language BPB is closest (and the reverse), reporting the
    residual BPB gap and the paired-over-facts RankC delta with a bootstrap
    CI. Only mid-stable (1b-23b) checkpoints are paired with each other and
    cooled finals with cooled finals, so LR state is matched too (CLAUDE.md 6);
  * a per-condition linear fit of RankC on partner BPB, and the vertical
    offset between the two fits at the overlapping BPB range -- the single
    number that says whether fair is more consistent AT EQUAL LOSS.

    python analyze_polyfact_traj.py [--workdir /mnt/scratch/xscript_rankc]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARTNERS = ["de", "fr", "ar", "zh"]


def rankc_rows(r1, r2):
    N = r1.shape[1]
    order = np.arange(N, 0, -1, dtype=float)
    w = np.exp(order) / np.exp(order).sum()
    out = np.zeros(len(r1))
    for i, (a, b) in enumerate(zip(r1, r2)):
        s1, s2 = set(), set()
        for k in range(N):
            s1.add(int(a[k])); s2.add(int(b[k]))
            out[i] += w[k] * len(s1 & s2) / (k + 1)
    return out


def boot(x, rng, B=2000):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def budget_of(run):
    m = re.search(r"-(\d+)b$", run)
    return float(m.group(1)) if m else 30.0


def load(path):
    d = json.load(open(path))
    l1, l2 = d["langs"]
    ll1, ll2 = np.array(d["ll"][l1]), np.array(d["ll"][l2])
    g1, g2 = np.array(d["gold"][l1]), np.array(d["gold"][l2])
    ids1, ids2 = d["option_ids"][l1], d["option_ids"][l2]
    perm = np.array([[ids2[i].index(q) for q in ids1[i]] for i in range(len(ids1))])
    ll2 = np.take_along_axis(ll2, perm, axis=1)
    g2 = np.array([ids1[i].index(ids2[i][g2[i]]) for i in range(len(ids1))])
    assert (g1 == g2).all()
    r1, r2 = np.argsort(-ll1, axis=1, kind="stable"), np.argsort(-ll2, axis=1, kind="stable")
    rc = rankc_rows(r1, r2)
    return {"run": d["run"], "partner": l2, "cond": "fair" if "fair" in d["run"] else "starved",
            "budget": budget_of(d["run"]), "cooled": not re.search(r"-\d+b$", d["run"]),
            "tokens": d.get("tokens"), "bpb": d["bpb"], "rc": rc,
            "acc1": float((r1[:, 0] == g1).mean()), "acc2": float((r2[:, 0] == g2).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/mnt/scratch/xscript_rankc")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    recs = [load(p) for p in sorted((Path(args.workdir) / "results" / "polyfact_traj").glob("*.json"))]
    lines, summary = [], {}
    for p in PARTNERS:
        R = [r for r in recs if r["partner"] == p]
        if not R:
            continue
        lines.append(f"\n## en-{p}\n")
        lines.append("| checkpoint | B tok | LR state | bpb en | bpb X | acc en | acc X | RankC [95% CI] |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(R, key=lambda r: (r["cond"], r["budget"])):
            lo, hi = boot(r["rc"], rng)
            lines.append(f"| {r['run']} | {r['budget']:.0f} | {'cooled' if r['cooled'] else 'stable'} | "
                         f"{r['bpb'].get('en', float('nan')):.4f} | {r['bpb'].get(p, float('nan')):.4f} | "
                         f"{r['acc1']:.3f} | {r['acc2']:.3f} | **{r['rc'].mean():.3f}** [{lo:.3f}, {hi:.3f}] |")
        F = [r for r in R if r["cond"] == "fair"]; S = [r for r in R if r["cond"] == "starved"]
        # matched-loss pairs (same LR state)
        lines.append(f"\n### matched partner-BPB pairs (en-{p}); Δ = fair − starved, paired over facts\n")
        lines.append("| fair ckpt | starved ckpt | bpb X fair | bpb X starved | Δ bpb | Δ RankC [95% CI] |")
        lines.append("|---|---|---|---|---|---|")
        pairs = {}
        for f in F:
            cands = [s for s in S if s["cooled"] == f["cooled"]]
            if cands:
                s = min(cands, key=lambda s: abs(s["bpb"][p] - f["bpb"][p]))
                pairs[(f["run"], s["run"])] = (f, s)
        for s in S:
            cands = [f for f in F if f["cooled"] == s["cooled"]]
            if cands:
                f = min(cands, key=lambda f: abs(f["bpb"][p] - s["bpb"][p]))
                pairs.setdefault((f["run"], s["run"]), (f, s))
        tight = []
        for f, s in sorted(pairs.values(), key=lambda fs: fs[0]["budget"]):
            d = f["rc"] - s["rc"]; lo, hi = boot(d, rng)
            star = "*" if lo > 0 or hi < 0 else ""
            dbpb = f["bpb"][p] - s["bpb"][p]
            if abs(dbpb) <= 0.01 and not f["cooled"]:
                tight.append((f["run"], s["run"], dbpb, float(d.mean()), lo, hi))
            lines.append(f"| {f['run']} | {s['run']} | {f['bpb'][p]:.4f} | {s['bpb'][p]:.4f} | "
                         f"{dbpb:+.4f} | **{d.mean():+.4f}** [{lo:+.4f}, {hi:+.4f}]{star} |")
        # all mid-stable cross pairs within |dbpb| <= 0.01 (not just nearest), pooled
        allt = []
        for f in F:
            for s in S:
                if not f["cooled"] and not s["cooled"] and abs(f["bpb"][p] - s["bpb"][p]) <= 0.01:
                    allt.append((f, s))
        if allt:
            ds = np.array([float((f["rc"] - s["rc"]).mean()) for f, s in allt])
            lines.append(f"\n**Tight matched-loss pairs (mid-stable, |Δbpb_X| ≤ 0.01), n={len(allt)}: "
                         f"mean Δ RankC (fair − starved) = {ds.mean():+.4f}, range {ds.min():+.4f}..{ds.max():+.4f}; "
                         f"pairs: " + ", ".join(f"{f['run'].split('-')[-1]}/{s['run'].split('-')[-1]} {float((f['rc']-s['rc']).mean()):+.3f}" for f, s in allt) + "**")
            summary.setdefault(p, {})["tight_pairs_mean_delta"] = float(ds.mean())
            summary[p]["tight_pairs"] = [(f["run"], s["run"], float((f["rc"] - s["rc"]).mean())) for f, s in allt]
        # linear fit RankC ~ bpb_X per condition on mid-stable checkpoints, offset at common bpb
        fit = {}
        for cond, G in (("fair", F), ("starved", S)):
            G = [r for r in G if not r["cooled"]]
            if len(G) >= 3:
                x = np.array([r["bpb"][p] for r in G]); y = np.array([r["rc"].mean() for r in G])
                fit[cond] = np.polyfit(x, y, 1)
        if len(fit) == 2:
            xs = [r["bpb"][p] for r in F + S if not r["cooled"]]
            lo_x, hi_x = max(min(r["bpb"][p] for r in F if not r["cooled"]), min(r["bpb"][p] for r in S if not r["cooled"])), \
                         min(max(r["bpb"][p] for r in F if not r["cooled"]), max(r["bpb"][p] for r in S if not r["cooled"]))
            if hi_x > lo_x:
                grid = np.linspace(lo_x, hi_x, 5)
                off = np.polyval(fit["fair"], grid) - np.polyval(fit["starved"], grid)
                lines.append(f"\nLinear fit RankC~bpb_X (mid-stable only): fair slope {fit['fair'][0]:+.3f}/bpb, "
                             f"starved slope {fit['starved'][0]:+.3f}/bpb; **fair − starved at equal BPB "
                             f"over the overlap [{lo_x:.3f}, {hi_x:.3f}]: {off.mean():+.4f}** "
                             f"(range {off.min():+.4f}..{off.max():+.4f})")
                summary.setdefault(p, {}).update({"offset_at_equal_bpb": float(off.mean()), "overlap": [lo_x, hi_x],
                              "fair_slope": float(fit["fair"][0]), "starved_slope": float(fit["starved"][0])})
            else:
                lines.append("\n(no overlapping BPB range between conditions among mid-stable checkpoints)")
    md = "\n".join(lines)
    print(md)
    out = ROOT / "results" / "rankc"; out.mkdir(exist_ok=True)
    (out / "polyfact_traj_rankc.md").write_text(md + "\n")
    json.dump({"per_checkpoint": [{k: (v if k != "rc" else float(np.mean(v))) for k, v in r.items()} for r in recs],
               "equal_bpb_offset": summary}, open(out / "polyfact_traj_rankc.json", "w"), indent=1)


if __name__ == "__main__":
    main()
