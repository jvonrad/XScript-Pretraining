#!/usr/bin/env python
"""Verify and characterize the bidirectional d' added to
`src/xscript/eval/alignment.py::_discriminability`.

Reuses the actual production functions (`_center`, `_retrieval_sim`,
`_discriminability`) via import -- no reimplemented formulas -- so this both
validates the change and produces the 8-cell robustness check in one pass:

  1. sanity checks: transpose-based dprime_b2a vs an independently-written
     column formula; dprime_a2b vs the original _discriminability_1d.
  2. recomputes dprime_a2b/b2a/sym for the 8 EN-anchored bilingual models
     (final checkpoint, every layer, raw + centered) from the cached FLORES+
     embeddings (HF dataset `jvonrad/xscript-embeddings`), and cross-checks
     dprime_a2b against the committed results/alignment/per_layer.csv `dprime`
     column -- confirms nothing silently changed for existing consumers.

    python scripts/dprime_bidirectional_check.py [--cache-dir ~/.cache/xscript_embeddings]

Writes results/alignment/{dprime_bidirectional.csv, fig_dprime_bidirectional.png}.
Does NOT re-run the full 26/107-model pipeline -- 8 cells is enough to decide
whether the asymmetry matters; see the written summary for the verdict.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from xscript.eval import alignment as A  # noqa: E402

OUT_DIR = REPO / "results" / "alignment"
PAIRS = ["ar", "de", "fr", "zh"]
CONDS = ["fair", "starved"]
N_LAYERS = 17
VARIANTS = ("raw", "centered")


def fetch_slim(run: str, partner: str, cache_dir: Path) -> Path:
    slim_path = cache_dir / f"{run}.npz"
    if slim_path.exists():
        return slim_path
    from huggingface_hub import hf_hub_download
    raw_dir = cache_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {run}")
    raw_path = Path(hf_hub_download("jvonrad/xscript-embeddings", f"{run}.npz",
                                     repo_type="dataset", local_dir=str(raw_dir)))
    d = np.load(raw_path)
    np.savez(slim_path, en=d["en"], partner=d[partner],
             meta=str(d["__meta__"]), partner_name=partner)
    d.close()
    raw_path.unlink()
    return slim_path


def independent_col_dprime(sim):
    """Column-wise d', written independently of _discriminability_1d(sim.T),
    to cross-check the transpose trick without trusting it blindly."""
    n = sim.shape[0]
    matched = np.diag(sim).astype(np.float64)
    col_sum = sim.sum(0).astype(np.float64) - matched
    col_sq = (sim.astype(np.float64) ** 2).sum(0) - matched ** 2
    m = n - 1
    mean_off = col_sum / m
    var_off = np.maximum(col_sq / m - mean_off ** 2, 0.0)
    margin = matched - mean_off
    return margin / np.maximum(np.sqrt(var_off), 1e-9)


def sanity_checks():
    rng = np.random.RandomState(0)
    sim = rng.randn(50, 50)
    d = A._discriminability(sim)
    indep = independent_col_dprime(sim)
    assert np.allclose(d["dprime_b2a"], indep, atol=1e-9), "b2a transpose trick mismatch!"
    _, dprime_old = A._discriminability_1d(sim)
    assert np.allclose(d["dprime_a2b"], dprime_old, atol=1e-9), "a2b changed!"
    print("[sanity] dprime_b2a (transpose) == independent column formula (atol 1e-9): OK")
    print("[sanity] dprime_a2b == _discriminability_1d(sim) (atol 1e-9): OK")


def load_committed_reference() -> pd.DataFrame:
    models = {f"en-{p}-{c}" for p in PAIRS for c in CONDS}
    rows = []
    with open(OUT_DIR / "per_layer.csv") as f:
        for row in csv.DictReader(f):
            if row["model"] in models and row["trained_pair"] == "1":
                rows.append(row)
    df = pd.DataFrame(rows)
    df["layer"] = df.layer.astype(int)
    df["dprime"] = df.dprime.astype(float)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "xscript_embeddings"))
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sanity_checks()
    ref = load_committed_reference()

    rows, mismatches = [], 0
    for cond in CONDS:
        for pair in PAIRS:
            run = f"en-{pair}-{cond}"
            path = fetch_slim(run, pair, cache_dir)
            d = np.load(path)
            en_all, partner_all = d["en"], d["partner"]
            for variant in VARIANTS:
                for layer in range(N_LAYERS):
                    E, F = en_all[layer], partner_all[layer]
                    if variant == "centered":
                        E, F = A._center(E), A._center(F)
                    m = A._retrieval_sim(E @ F.T)

                    r = ref[(ref.model == run) & (ref.pair == f"en-{pair}")
                            & (ref.variant == variant) & (ref.layer == layer)]
                    if len(r) and not np.isclose(m["dprime"], float(r.dprime.iloc[0]),
                                                  atol=1e-3, rtol=1e-3):
                        mismatches += 1
                        print(f"[MISMATCH] {run} {variant} L{layer}: "
                              f"recomputed={m['dprime']:.5f} committed={float(r.dprime.iloc[0]):.5f}")

                    for metric in ("dprime_a2b", "dprime_b2a", "dprime_sym"):
                        rows.append(dict(model=run, cond=cond, pair=f"en-{pair}", layer=layer,
                                          variant=variant, metric=metric, value=m[metric]))
                    rows.append(dict(model=run, cond=cond, pair=f"en-{pair}", layer=layer,
                                      variant=variant, metric="asymmetry",
                                      value=m["dprime_a2b"] - m["dprime_b2a"]))
            d.close()
            print(f"[done] {run}")

    print("[check] all recomputed dprime_a2b match committed per_layer.csv (atol=1e-3): OK"
          if mismatches == 0 else f"[check] {mismatches} MISMATCHES vs committed per_layer.csv!")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "dprime_bidirectional.csv", index=False)
    print(f"wrote {len(df)} rows -> {OUT_DIR / 'dprime_bidirectional.csv'}")

    piv = df.pivot_table(index=["model", "cond", "pair", "layer", "variant"],
                          columns="metric", values="value").reset_index()
    piv["rel_asym"] = piv.asymmetry.abs() / piv.dprime_sym.abs().clip(lower=1e-6)
    print(piv.groupby("variant").rel_asym.describe())

    make_figure(df, OUT_DIR / "fig_dprime_bidirectional.png")


def make_figure(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9"
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    for row, variant in enumerate(VARIANTS):
        for col, pair in enumerate(PAIRS):
            ax = axes[row, col]
            ax.set_facecolor("#fcfcfb")
            sub = df[(df.pair == f"en-{pair}") & (df.variant == variant)]
            for cond, color in [("fair", BLUE), ("starved", ORANGE)]:
                for metric, ls in [("dprime_a2b", "-"), ("dprime_b2a", "--")]:
                    s = sub[(sub.cond == cond) & (sub.metric == metric)].sort_values("layer")
                    ax.plot(s.layer, s.value, color=color, linestyle=ls, linewidth=2)
            if row == 0:
                ax.set_title(f"en-{pair}", color=INK, fontsize=13)
            if col == 0:
                ax.set_ylabel(f"d' ({variant})", color=INK, fontsize=11)
            if row == 1:
                ax.set_xlabel("layer", color=MUTED, fontsize=10)
            ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
            ax.axhline(0, color="#c3c2b7", linewidth=0.8)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color("#c3c2b7")
            ax.tick_params(colors=MUTED, labelsize=9)
    cond_handles = [Line2D([0], [0], color=BLUE, lw=2.5, label="fair"),
                    Line2D([0], [0], color=ORANGE, lw=2.5, label="starved")]
    dir_handles = [Line2D([0], [0], color=INK, lw=2, linestyle="-", label="d' A→B (en→partner)"),
                  Line2D([0], [0], color=INK, lw=2, linestyle="--", label="d' B→A (partner→en)")]
    leg1 = fig.legend(handles=cond_handles, loc="upper center", bbox_to_anchor=(0.28, 1.04),
                      ncol=2, frameon=False, fontsize=10)
    fig.legend(handles=dir_handles, loc="upper center", bbox_to_anchor=(0.7, 1.04),
               ncol=2, frameon=False, fontsize=10)
    fig.add_artist(leg1)
    fig.suptitle("Bidirectional d': raw (top, asymmetric) vs centered (bottom, symmetric)",
                 y=1.1, fontsize=14, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 1.0])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[fig] wrote {out_path}")


if __name__ == "__main__":
    main()
