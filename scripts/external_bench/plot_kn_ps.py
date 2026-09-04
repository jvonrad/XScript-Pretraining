#!/usr/bin/env python
"""6j figures in the thesis' PS formulation, on the DIFFERENT-surface-form
facts only (gold entity spelled differently in the two languages), for the 8
cooled 30B bilingual finals:

  fig_kn_ps_bars       (a) PS_{s,t} = same-fact top-K Jaccard minus the
                       mismatched-fact baseline, K=100; (b) ablation transfer
                       rate. fair vs starved, 95% bootstrap CIs over facts.
  fig_kn_ablation      ablation damage by condition per model (log scale):
                       own top-100, own different-fact control, other-language
                       top-100, other different-fact control, 100 random.
  fig_kn_layers        cumulative layer distribution of the partner language's
                       top-100 knowledge neurons, fair vs starved panels.

    python plot_kn_ps.py [--results results/knowneurons] [--k 100]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src"))
from analyze_knowneurons import Run, jaccard_matrix, jaccard_mismatched, load_gold_strings, CONDS  # noqa: E402

PARTNERS = ["de", "fr", "ar", "zh"]
FAIR, STARVED, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
PCOL = {"de": "#2a78d6", "fr": "#e87ba4", "ar": "#eda100", "zh": "#1baf7a"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})


def boot(vals, rng, B=2000):
    idx = rng.integers(0, len(vals), (B, len(vals)))
    m = vals[idx].mean(1)
    return float(vals.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def rate_boot(trans_list, spec_list, rng, B=2000):
    t = np.concatenate(trans_list); s = np.concatenate(spec_list); n = len(trans_list[0])
    out = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n); i2 = np.concatenate([i, i + n])
        out[b] = t[i2].mean() / s[i2].mean()
    return float(t.mean() / s.mean()), float(np.quantile(out, .025)), float(np.quantile(out, .975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "knowneurons"))
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()
    res = Path(a.results); out = res / "figs"; out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    runs = {f"en-{p}-{c}": Run(res, f"en-{p}-{c}") for p in PARTNERS for c in ("fair", "starved")}
    gold = load_gold_strings(["en", *PARTNERS])
    stats = {}
    for name, r in runs.items():
        l1, l2 = r.langs
        diff = np.array([gold[l1][fi].strip().lower() != gold[l2][fi].strip().lower() for fi in r.fact_ids])
        A, B_ = r.topk[diff, 0, :], r.topk[diff, 1, :]
        same_j = jaccard_matrix(A, B_, a.k); mism_j = jaccard_mismatched(A, B_, a.k)
        ps = same_j - mism_j
        dmg = {}; trans, spec = [], []
        for ti, tl in enumerate(r.langs):
            d = {c: r.damage(tl, c)[diff] for c in CONDS[1:]}
            for c, v in d.items():
                dmg.setdefault(c, []).append(v)
            spec.append(d["own_fact"] - d["own_other"]); trans.append(d["cross_fact"] - d["cross_other"])
        layers = (r.topk[diff, 1, :a.k] // 5632).ravel()
        stats[name] = {"n": int(diff.sum()), "ps": boot(ps, rng), "same": boot(same_j, rng), "mism": float(mism_j.mean()),
                       "rate": rate_boot(trans, spec, rng),
                       "dmg": {c: float(np.concatenate(v).mean()) for c, v in dmg.items()},
                       "layer_cdf": np.cumsum(np.bincount(layers, minlength=16) / len(layers))}
        print(f"{name:14s} n_diff={diff.sum():4d} PS={stats[name]['ps'][0]:.4f} rate={stats[name]['rate'][0]:.3f}")

    # ---- fig 1: bars (thesis design: blue fair, orange hatched starved) -------
    BLUE, ORANGE = "#1f77b4", "#e8a33d"
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    x = np.arange(4); w = 0.36; bw = w * 0.95
    for ax, key, title in ((axes[0], "ps", "a   Parametric sharing (PS)"), (axes[1], "rate", "b   Ablation transfer rate")):
        for cond, off in (("fair", -w / 2), ("starved", w / 2)):
            m = [stats[f"en-{p}-{cond}"][key][0] for p in PARTNERS]
            lo = [m[i] - stats[f"en-{p}-{cond}"][key][1] for i, p in enumerate(PARTNERS)]
            hi = [stats[f"en-{p}-{cond}"][key][2] - m[i] for i, p in enumerate(PARTNERS)]
            if cond == "fair":
                ax.bar(x + off, m, bw, color=BLUE, label="fair", zorder=3)
            else:
                ax.bar(x + off, m, bw, color=ORANGE, edgecolor="white", hatch="//", lw=0, label="starved", zorder=3)
            ax.errorbar(x + off, m, yerr=[lo, hi], fmt="none", ecolor="#222222", elinewidth=1.1, capsize=3, zorder=5)
            for xi, (mi, hii) in enumerate(zip(m, hi)):
                ax.text(x[xi] + off, mi + hii + (0.002 if key == "ps" else 0.012),
                        ("{:.3f}" if key == "ps" else "{:.2f}").format(mi).lstrip("0"), ha="center", va="bottom", fontsize=8.5, color="#333333")
        top = max(stats[f"en-{p}-{c}"][key][2] for p in PARTNERS for c in ("fair", "starved"))
        ax.set_ylim(0, top * 1.18)
        ax.set_xticks(x); ax.set_xticklabels(PARTNERS, fontsize=10)
        ax.annotate("same script", xy=(0.25, -0.15), xycoords="axes fraction", ha="center", fontsize=9.5, color="#666666")
        ax.annotate("cross-script", xy=(0.75, -0.15), xycoords="axes fraction", ha="center", fontsize=9.5, color="#666666")
        ax.set_title(title, loc="left", fontsize=12, color="#222222", pad=8)
        ax.set_ylabel("PS  (same-fact − mismatched-fact Jaccard)" if key == "ps" else "transfer rate", fontsize=10.5, color="#222222")
        ax.tick_params(axis="y", labelsize=10, colors="#444444"); ax.tick_params(axis="x", colors="#222222")
        ax.grid(axis="x", visible=False); ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#cccccc")
    axes[0].legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_kn_ps_bars.{ext}", dpi=170)
    plt.close(fig)

    # ---- fig 2: ablation damage by condition --------------------------------
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    names = [f"en-{p}-{c}" for p in PARTNERS for c in ("fair", "starved")]
    spec = [("own_fact", FAIR, "o", True, "own-language top-100"), ("own_other", FAIR, "o", False, "own, different fact (control)"),
            ("cross_fact", STARVED, "s", True, "other-language top-100"), ("cross_other", STARVED, "s", False, "other, different fact (control)"),
            ("random", "#6f6e6a", "D", True, "100 random neurons")]
    for yi, n in enumerate(names):
        for c, col, mk, filled, lab in spec:
            v = max(stats[n]["dmg"][c], 1e-3)
            ax.plot(v, len(names) - 1 - yi, marker=mk, ms=8, mfc=col if filled else "white", mec=col, mew=1.6, ls="none",
                    label=lab if yi == 0 else None, zorder=3)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names[::-1]); ax.set_xscale("log")
    ax.set_xlabel("drop in gold-answer log-likelihood (nats, log scale)"); ax.grid(axis="y", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_kn_ablation.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 3: layer CDF ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    L = np.arange(16)
    for ax, cond in zip(axes, ("fair", "starved")):
        ax.plot(L, (L + 1) / 16, ls="--", color="#9a998f", lw=1.2, zorder=2)
        ax.annotate("uniform", xy=(9.6, 9.0 / 16), xytext=(0, -13), textcoords="offset points", fontsize=8, color="#9a998f", rotation=36)
        for p in PARTNERS:
            cdf = stats[f"en-{p}-{cond}"]["layer_cdf"]
            ax.plot(L, cdf, color=PCOL[p], lw=2.2, zorder=3)
            k = {"zh": 6, "de": 11, "fr": 12, "ar": 4}[p]
            dx, dy = {"zh": (-4, 7), "de": (-4, 7), "fr": (0, -13), "ar": (6, -13)}[p]
            ax.annotate(p, xy=(k, cdf[k]), xytext=(dx, dy), textcoords="offset points",
                        color=PCOL[p], fontsize=10, fontweight="bold")
        ax.set_title(f"{cond} tokenizer", loc="left", fontsize=10); ax.set_xlabel("layer")
        ax.set_xticks([0, 4, 8, 12, 15]); ax.set_xlim(0, 15); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("cumulative share of top-100\nknowledge neurons (partner lang.)", fontsize=8.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_kn_layers.{ext}", dpi=170)
    plt.close(fig)
    print("wrote", sorted(f.name for f in out.iterdir() if f.name.startswith("fig_kn_")))


if __name__ == "__main__":
    main()
