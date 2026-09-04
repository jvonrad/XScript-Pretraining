#!/usr/bin/env python
"""Figures for results/trajectories/trajectories.json (CLAUDE.md 6k Result 5):
consistency and parametric sharing over training, fair vs starved, per partner.

  fig_traj_sharing.{pdf,png}      dKS K=100 and ablation transfer rate, 4 partners
  fig_traj_consistency.{pdf,png}  RankC (PolyFact), error consistency, partner P@1
  fig_traj_deltas.{pdf,png}       fair - starved by budget, one panel per metric

Hollow markers = the cooled 30B final (different LR state from the 2b-23b
mid-stable checkpoints). Colors: fair blue / starved orange; partners in the
fixed order de, fr, ar, zh (blue, orange, aqua, yellow).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "trajectories" / "figs"
PARTNERS = ["de", "fr", "ar", "zh"]
PNAME = {"de": "en–de (same script)", "fr": "en–fr (same script)", "ar": "en–ar (cross-script)", "zh": "en–zh (cross-script)"}
COND = {"fair": ("#2a78d6", "fair (5-lang) tokenizer"), "starved": ("#eb6834", "starved (419-lang) tokenizer")}
PCOL = {"de": "#2a78d6", "fr": "#eb6834", "ar": "#1baf7a", "zh": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
METRICS = {
    "dks_k100": ("Knowledge-neuron overlap\ndKS (K=100)", ".3f"),
    "transfer_rate": ("Cross-lingual ablation\ntransfer rate", ".2f"),
    "pf_rankc": ("Answer consistency\nRankC (PolyFact)", ".3f"),
    "pf_same_wrong": ("Error consistency: same\nwrong answer in both langs", ".3f"),
    "pf_acc_partner": ("Partner-language recall\nP@1 (PolyFact)", ".3f"),
}
BUD = {"2b": 2, "5b": 5, "10b": 10, "15b": 15, "23b": 23, "30b-cooled": 30}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def series(recs, p, cond, key):
    pts = sorted(((BUD[r["budget"]], r.get(key), r["budget"] == "30b-cooled") for r in recs
                  if r["partner"] == p and r["cond"] == cond and r.get(key) is not None), key=lambda t: t[0])
    return [t[0] for t in pts], [t[1] for t in pts], [t[2] for t in pts]


def style_x(ax):
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 15, 23, 30]); ax.set_xticklabels(["2", "5", "10", "15", "23", "30*"], fontsize=8)
    ax.minorticks_off(); ax.set_xlabel("training tokens (B)")


def plot_lines(recs, keys, fname, title):
    fig, axes = plt.subplots(len(keys), 4, figsize=(13, 2.9 * len(keys)), sharex=True)
    for i, key in enumerate(keys):
        label, fmt = METRICS[key]
        for j, p in enumerate(PARTNERS):
            ax = axes[i][j]
            ends = {}
            for cond, (col, cl) in COND.items():
                x, y, cooled = series(recs, p, cond, key)
                ax.plot(x, y, color=col, lw=2, label=cl if (i == 0 and j == 0) else None, zorder=3)
                for xx, yy, c in zip(x, y, cooled):
                    ax.plot(xx, yy, marker="o", ms=7, mfc="white" if c else col, mec=col, mew=1.8, ls="none", zorder=4)
                if y:
                    ends[cond] = (x[-1], y[-1])
            # end labels: nudge apart vertically when the two lines finish close together
            if len(ends) == 2:
                (xf, yf), (xs, ys) = ends["fair"], ends["starved"]
                allv = [v for c in COND for v in series(recs, p, c, key)[1]]
                close = abs(yf - ys) < 0.11 * ((max(allv) - min(allv)) or 1)
                dy = {"fair": 5 if (close and yf >= ys) else (-5 if close else 0),
                      "starved": -5 if (close and yf >= ys) else (5 if close else 0)}
            for cond, (xe, ye) in ends.items():
                ax.annotate(format(ye, fmt), (xe, ye), xytext=(6, dy[cond] if len(ends) == 2 else 0),
                            textcoords="offset points", va="center", fontsize=8, color=INK2)
            if i == 0:
                ax.set_title(PNAME[p], fontsize=10)
            if j == 0:
                ax.set_ylabel(label, fontsize=8.5, wrap=True)
            if i == len(keys) - 1:
                style_x(ax)
            ax.set_xlim(1.7, 42)
        ys = [v for p in PARTNERS for c in COND for v in series(recs, p, c, key)[1]]
        lo, hi = min(ys), max(ys); pad = 0.08 * (hi - lo or 1)
        for ax in axes[i]:
            ax.set_ylim(lo - pad, hi + pad)
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.suptitle(title + "   (* 30B = cooled final; 2–23B mid-stable at peak LR; hollow marker = cooled)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=170)
    plt.close(fig)


def plot_deltas(recs, keys, fname):
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 3.6))
    for ax, key in zip(axes, keys):
        label, fmt = METRICS[key]
        ends = []
        for p in PARTNERS:
            xf, yf, cf = series(recs, p, "fair", key); xs, ys, _ = series(recs, p, "starved", key)
            common = sorted(set(xf) & set(xs)); df = {x: y for x, y in zip(xf, yf)}; ds = {x: y for x, y in zip(xs, ys)}
            d = [df[x] - ds[x] for x in common]
            ax.plot(common, d, color=PCOL[p], lw=2, zorder=3)
            for x, v in zip(common, d):
                ax.plot(x, v, marker="o", ms=6, mfc="white" if x == 30 else PCOL[p], mec=PCOL[p], mew=1.6, ls="none", zorder=4)
            ends.append([p, common[-1], d[-1]])
        # spread end labels that would overprint: push apart in data units until >= 7% of the y-range apart
        lo, hi = ax.get_ylim(); gap = 0.11 * (hi - lo)
        pos = sorted(ends, key=lambda e: e[2]); ys_ = [e[2] for e in pos]
        for k in range(1, len(ys_)):          # push labels apart bottom-up
            if ys_[k] - ys_[k - 1] < gap:
                ys_[k] = ys_[k - 1] + gap
        for (p, xe, ye), yl in zip(pos, ys_):
            ax.annotate(p, xy=(xe, yl), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9,
                        color=PCOL[p], fontweight="bold", annotation_clip=False)
        ax.axhline(0, color=INK2, lw=0.8, zorder=2)
        ax.set_title(label, fontsize=9); style_x(ax); ax.set_xlim(1.7, 42)
        ax.set_ylabel("fair − starved", fontsize=8.5)
    fig.suptitle("Tokenizer effect over training (fair − starved), per partner language", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = json.load(open(ROOT / "results" / "trajectories" / "trajectories.json"))
    plot_lines(recs, ["dks_k100", "transfer_rate"], "fig_traj_sharing", "Parametric knowledge sharing over training")
    plot_lines(recs, ["pf_rankc", "pf_same_wrong", "pf_acc_partner"], "fig_traj_consistency", "Cross-lingual consistency over training")
    plot_deltas(recs, ["dks_k100", "transfer_rate", "pf_rankc", "pf_same_wrong"], "fig_traj_deltas")
    print("wrote", sorted(f.name for f in OUT.iterdir()))


if __name__ == "__main__":
    main()
