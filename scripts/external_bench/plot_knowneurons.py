#!/usr/bin/env python
"""Thesis figures + LaTeX numbers for the knowledge-sharing analysis (6j).

    python plot_knowneurons.py [--results DIR] [--out DIR]

Reuses analyze_knowneurons.py's estimators (imported, not duplicated) so the
figures cannot drift from the report. Outputs PDF+PNG figures and a
latex_numbers.txt with the conductivity table rows.

Palette: Okabe-Ito, validated with the dataviz six-checks
(fair #0072B2 / starved #E69F00, CVD dE 29.2; partners
#0072B2/#CC79A7/#E69F00/#009E73, worst dE 8.5 + direct labels). Starved bars
additionally carry a hatch so the contrast is never color-alone.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_knowneurons import (Run, jaccard_matrix, jaccard_mismatched,  # noqa: E402
                                 pooled_rate, CONDS)

FAIR, STARVED = "#0072B2", "#E69F00"
PARTNER_COLORS = {"de": "#0072B2", "fr": "#CC79A7", "ar": "#E69F00", "zh": "#009E73"}
PARTNERS = ["de", "fr", "ar", "zh"]
GRAY = "#6b6b6b"


def style(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(colors="#444444", labelsize=8)
    if ygrid:
        ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def boot_mean_ci(v, b=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (b, len(v)))].mean(1)
    return v.mean(), np.quantile(m, .025), np.quantile(m, .975)


def rate_ci(pairs, b=2000, seed=0):
    """Transfer rate = mean(trans)/mean(spec), CI by joint fact bootstrap."""
    rng = np.random.default_rng(seed)
    t, s = pooled_rate(pairs)
    n = len(pairs[0][0])
    reps = np.empty(b)
    for i in range(b):
        ix = rng.integers(0, n, n)
        ix2 = np.concatenate([ix, ix + n])
        reps[i] = t[ix2].mean() / s[ix2].mean()
    return t.mean() / s.mean(), np.quantile(reps, .025), np.quantile(reps, .975)


def grouped_bars(ax, fair_vals, starved_vals, ylabel, title, decimals=3):
    x = np.arange(len(PARTNERS))
    w = 0.36
    for vals, off, color, hatch, label in (
            (fair_vals, -w / 2, FAIR, None, "fair"),
            (starved_vals, w / 2, STARVED, "///", "starved")):
        b = ax.bar(x + off, [v[0] for v in vals], w, color=color, hatch=hatch,
                   edgecolor="white", linewidth=1.0, zorder=3, label=label)
        ax.errorbar(x + off, [v[0] for v in vals],
                    yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                    fmt="none", ecolor="#333333", elinewidth=1.0, capsize=2.5, zorder=4)
        for rect, v in zip(b, vals):
            ax.annotate(f"{v[0]:.{decimals}f}".lstrip("0"),
                        (rect.get_x() + rect.get_width() / 2, v[2]),
                        xytext=(0, 2.5), textcoords="offset points",
                        ha="center", fontsize=6.5, color="#444444")
    ax.set_xticks(x)
    ax.set_xticklabels(PARTNERS, fontsize=9)
    # script grouping under the partner codes
    for x0, x1, lab in ((0, 1, "same script"), (2, 3, "cross-script")):
        ax.annotate(lab, ((x0 + x1) / 2, 0), xytext=(0, -26),
                    textcoords="offset points", xycoords=("data", "axes fraction"),
                    ha="center", fontsize=7.5, color="#777777")
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9.5, loc="left", color="#222222")
    ax.margins(y=0.14)
    style(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path(__file__).resolve().parents[2] / "results" / "knowneurons")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()
    out = a.out or (a.results / "figs")
    out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for p in PARTNERS:
        for tokn in ("fair", "starved"):
            runs[f"en-{p}-{tokn}"] = Run(a.results, f"en-{p}-{tokn}")

    # ---------- stats ----------
    dks, rates, dmg = {}, {}, {}
    for name, r in runs.items():
        same = jaccard_matrix(r.topk[:, 0], r.topk[:, 1], a.k)
        mism = jaccard_mismatched(r.topk[:, 0], r.topk[:, 1], a.k)
        dks[name] = boot_mean_ci(same - mism)
        pairs = []
        for ti, tlang in enumerate(r.langs):
            d = {c: r.damage(tlang, c) for c in CONDS[1:]}
            pairs.append((d["cross_fact"] - d["cross_other"],
                          d["own_fact"] - d["own_other"]))
            for c in CONDS[1:]:
                dmg.setdefault(name, {}).setdefault(c, []).append(d[c].mean())
        rates[name] = rate_ci(pairs)

    # ---------- figure 1: headline ----------
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
    grouped_bars(axes[0],
                 [dks[f"en-{p}-fair"] for p in PARTNERS],
                 [dks[f"en-{p}-starved"] for p in PARTNERS],
                 r"$\Delta$KS  (same-fact $-$ mismatched-fact Jaccard)",
                 r"a   Parametric sharing ($\Delta$KS)", decimals=3)
    grouped_bars(axes[1],
                 [rates[f"en-{p}-fair"] for p in PARTNERS],
                 [rates[f"en-{p}-starved"] for p in PARTNERS],
                 "transfer rate",
                 "b   Ablation transfer rate", decimals=2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(w_pad=2.6)
    fig.subplots_adjust(bottom=0.18)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_kn_sharing.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- figure 2: ablation magnitudes / controls ----------
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    order = [f"en-{p}-{t}" for p in PARTNERS for t in ("fair", "starved")]
    y = np.arange(len(order))[::-1]
    specs = [("own_fact", FAIR, "o", "full", "own-language top-100"),
             ("own_other", FAIR, "o", "none", "own, different fact (control)"),
             ("cross_fact", "#D55E00", "s", "full", "other-language top-100"),
             ("cross_other", "#D55E00", "s", "none", "other, different fact (control)"),
             ("random", GRAY, "D", "full", "100 random neurons")]
    for cond, color, marker, fill, label in specs:
        vals = [np.mean(dmg[n][cond]) for n in order]
        ax.scatter(vals, y, s=34, marker=marker, zorder=3, label=label,
                   facecolors=color if fill == "full" else "none",
                   edgecolors=color, linewidths=1.4)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 40)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("drop in gold-answer log-likelihood (nats, log scale)", fontsize=9)
    ax.set_title("Ablation damage by condition — 100 random neurons are inert; "
                 "attributed neurons are load-bearing", fontsize=9.5, loc="left")
    style(ax, ygrid=False)
    ax.grid(axis="x", color="#e6e6e6", lw=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_kn_ablation.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- figure 3: layer profile (partner language) ----------
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7), sharey=True)
    label_x = {"zh": 2.6, "ar": 5.6, "de": 9.5, "fr": 12.3}
    label_dy = {"zh": 0.055, "ar": -0.075, "de": 0.055, "fr": -0.075}
    for ax, tokn in zip(axes, ("fair", "starved")):
        for p in PARTNERS:
            r = runs[f"en-{p}-{tokn}"]
            li = 1  # partner language index (langs = [en, partner])
            layers = (r.topk[:, li, :a.k] // 5632).ravel()
            share = np.array([(layers == L).mean() for L in range(16)])
            cum = np.cumsum(share)
            ax.plot(range(16), cum, color=PARTNER_COLORS[p], lw=2.0, zorder=3)
            lx = label_x[p]
            ly = np.interp(lx, range(16), cum) + label_dy[p]
            ax.annotate(p, (lx, ly), color=PARTNER_COLORS[p], fontsize=8.5,
                        ha="center", va="center", fontweight="bold", zorder=5)
        ax.plot(range(16), np.arange(1, 17) / 16, ls="--", color="#aaaaaa",
                lw=1.0, zorder=2)
        ax.annotate("uniform", (13.3, 0.76), color="#999999", fontsize=7,
                    rotation=33, ha="center")
        ax.set_xlim(0, 15.4)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 4, 8, 12, 15])
        ax.set_xlabel("layer", fontsize=9)
        ax.set_title(f"{tokn} tokenizer", fontsize=9.5, loc="left")
        style(ax)
    axes[0].set_ylabel("cumulative share of top-100\nknowledge neurons (partner lang.)",
                       fontsize=8.5)
    fig.tight_layout(w_pad=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_kn_layers.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- conductivity numbers ----------
    lines = ["% conductivity table rows: model & acc_en & acc_p & J & J_indep & C(p|en) & C(en|p)"]
    for name, r in runs.items():
        sel = r.sel
        l1, l2 = sel["langs"]
        hit = {}
        for lang in (l1, l2):
            ll = np.array(sel["ll"][lang]); g = np.array(sel["gold"][lang])
            hit[lang] = ll.argmax(1) == g
        a1, a2 = hit[l1].mean(), hit[l2].mean()
        inter = (hit[l1] & hit[l2]).sum()
        union = (hit[l1] | hit[l2]).sum()
        j = inter / union
        j_ind = a1 * a2 / (a1 + a2 - a1 * a2)
        lines.append(f"{name} & {a1:.3f} & {a2:.3f} & {j:.3f} & {j_ind:.3f} & "
                     f"{inter/hit[l1].sum():.3f} & {inter/hit[l2].sum():.3f} \\\\")
    lines.append("")
    lines.append("% headline: partner  dKS_fair  dKS_starved  rate_fair  rate_starved")
    for p in PARTNERS:
        f, s = dks[f"en-{p}-fair"], dks[f"en-{p}-starved"]
        rf, rs = rates[f"en-{p}-fair"], rates[f"en-{p}-starved"]
        lines.append(f"% {p}: dKS {f[0]:.4f} [{f[1]:.4f},{f[2]:.4f}] vs "
                     f"{s[0]:.4f} [{s[1]:.4f},{s[2]:.4f}]; rate {rf[0]:.3f} "
                     f"[{rf[1]:.3f},{rf[2]:.3f}] vs {rs[0]:.3f} [{rs[1]:.3f},{rs[2]:.3f}]")
    (out / "latex_numbers.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nfigures in {out}")


if __name__ == "__main__":
    main()
