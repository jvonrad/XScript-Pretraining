#!/usr/bin/env python
"""Task 1 -- replicate Limisiewicz et al. (2023) sentence retrieval on our models
and compare it against MEXA mutual-NN and plain top-1, using the cached FLORES+
embeddings from the HF dataset `jvonrad/xscript-embeddings`.

Their protocol: cosine similarity matrix between two languages' sentence reps, then
optimal one-to-one assignment via the Hungarian algorithm (scipy linear_sum_assignment
on the negated similarity matrix). We run it alongside MEXA's mutual-nearest-neighbour
rule and one-directional top-1, at their scale (n=1000, 5 subsamples) and ours
(n=2009, full FLORES+ dev+devtest pool), for the 4 EN-anchored bilingual pairs under
both tokenizer conditions, final checkpoint, every layer, raw (uncentered) embeddings.

    python scripts/task1_hungarian_retrieval.py [--cache-dir ~/.cache/xscript_embeddings]

Writes results/alignment/{hungarian_retrieval.csv, hungarian_retrieval_seedmean.csv,
hungarian_decomposition_2x2.csv, fig_hungarian_vs_mexa_vs_top1.png}.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results" / "alignment"

PAIRS = ["ar", "de", "fr", "zh"]
CONDS = ["fair", "starved"]
COND_TOK = {"fair": "unigram_destarved", "starved": "unigram_starved"}
N_LAYERS = 17
N_FULL = 2009
N_SUB = 1000
SEEDS = [0, 1, 2, 3, 4]

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9"


def fetch_slim(run: str, partner: str, cache_dir: Path) -> Path:
    """Download run's npz (all 5 langs x 17 layers x 2009 sentences), keep only
    en + partner, delete the ~1.3GB raw file. Idempotent."""
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


def retrieval_metrics(E: np.ndarray, F: np.ndarray) -> dict:
    """E, F: (n, d) L2-normalised, index-aligned. Hungarian / MEXA-mutual / top-1."""
    n = E.shape[0]
    sim = E @ F.T
    a2b, b2a, diag = sim.argmax(1), sim.argmax(0), np.arange(n)
    top1_a2b = float((a2b == diag).mean())
    top1_b2a = float((b2a == diag).mean())
    mexa_mutual = float(((a2b == diag) & (b2a[a2b] == diag)).mean())
    row_ind, col_ind = linear_sum_assignment(-sim)
    hungarian = float((col_ind[np.argsort(row_ind)] == diag).mean())
    return {"top1_a2b": top1_a2b, "top1_b2a": top1_b2a,
            "mexa_mutual": mexa_mutual, "hungarian": hungarian}


def run_analysis(cache_dir: Path) -> pd.DataFrame:
    rows = []
    for cond in CONDS:
        for pair in PAIRS:
            run = f"en-{pair}-{cond}"
            path = fetch_slim(run, pair, cache_dir)
            d = np.load(path)
            en_all, partner_all = d["en"], d["partner"]
            print(f"[analyze] {run}")
            for layer in range(N_LAYERS):
                E_full, F_full = en_all[layer], partner_all[layer]
                for metric, value in retrieval_metrics(E_full, F_full).items():
                    rows.append(dict(model=run, cond=cond, pair=f"en-{pair}", layer=layer,
                                      n=N_FULL, seed=None, metric=metric, value=value))
                for seed in SEEDS:
                    idx = np.random.RandomState(seed).choice(N_FULL, N_SUB, replace=False)
                    for metric, value in retrieval_metrics(E_full[idx], F_full[idx]).items():
                        rows.append(dict(model=run, cond=cond, pair=f"en-{pair}", layer=layer,
                                          n=N_SUB, seed=seed, metric=metric, value=value))
            d.close()
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    top1 = (df[df.metric.isin(["top1_a2b", "top1_b2a"])]
              .groupby(["model", "cond", "pair", "layer", "n", "seed"], dropna=False)["value"]
              .mean().reset_index())
    top1["metric"] = "top1_avg"
    full = pd.concat([df[~df.metric.isin(["top1_a2b", "top1_b2a"])], top1])
    full = full[full.n == N_FULL]

    styles = {"hungarian": "-", "mexa_mutual": "--", "top1_avg": ":"}
    labels = {"hungarian": "Hungarian", "mexa_mutual": "MEXA mutual", "top1_avg": "top-1 (avg dir.)"}
    colors = {"fair": BLUE, "starved": ORANGE}

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, pair in zip(axes, PAIRS):
        ax.set_facecolor("#fcfcfb")
        sub = full[full.pair == f"en-{pair}"]
        for cond in CONDS:
            for metric in ["hungarian", "mexa_mutual", "top1_avg"]:
                s = sub[(sub.cond == cond) & (sub.metric == metric)].sort_values("layer")
                if s.empty:
                    continue
                ax.plot(s.layer, s.value, color=colors[cond], linestyle=styles[metric],
                        linewidth=2, marker="o" if metric == "hungarian" else None, markersize=3)
        ax.set_title(f"en-{pair}", color=INK, fontsize=13)
        ax.set_xlabel("layer", color=MUTED, fontsize=10)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[0].set_ylabel("retrieval accuracy (n=2009)", color=INK, fontsize=11)

    cond_handles = [Line2D([0], [0], color=colors[c], lw=2.5, label=c) for c in CONDS]
    metric_handles = [Line2D([0], [0], color=INK, lw=2, linestyle=styles[m], label=labels[m])
                      for m in ["hungarian", "mexa_mutual", "top1_avg"]]
    leg1 = fig.legend(handles=cond_handles, loc="upper center", bbox_to_anchor=(0.28, 1.06),
                      ncol=2, frameon=False, fontsize=10)
    fig.legend(handles=metric_handles, loc="upper center", bbox_to_anchor=(0.68, 1.06),
               ncol=3, frameon=False, fontsize=10)
    fig.add_artist(leg1)
    fig.suptitle("Cross-script retrieval by matching rule: Hungarian vs MEXA-mutual vs top-1",
                 y=1.16, fontsize=14, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 1.0])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[fig] wrote {out_path}")


def decomposition_table(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """2x2 (rule x n) fair-minus-starved gap, mean over layers>=1 (excludes the
    layer-0 byte-fragmentation artifact), per pair."""
    rows = []
    for pair in PAIRS:
        for metric in ["hungarian", "mexa_mutual"]:
            for n in [N_SUB, N_FULL]:
                sub = df[(df.pair == f"en-{pair}") & (df.metric == metric)
                         & (df.n == n) & (df.layer >= 1)]
                fair = sub[sub.cond == "fair"].groupby("layer").value.mean()
                starved = sub[sub.cond == "starved"].groupby("layer").value.mean()
                rows.append(dict(pair=f"en-{pair}", rule=metric, n=n,
                                  mean_gap_fair_minus_starved=(fair - starved).mean()))
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(out.pivot_table(index=["pair", "rule"], columns="n", values="mean_gap_fair_minus_starved"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "xscript_embeddings"))
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = run_analysis(cache_dir)
    df.to_csv(OUT_DIR / "hungarian_retrieval.csv", index=False)
    print(f"wrote {len(df)} rows -> {OUT_DIR / 'hungarian_retrieval.csv'}")

    seedmean = (df.groupby(["model", "cond", "pair", "layer", "n", "metric"])["value"]
                  .mean().reset_index())
    seedmean.to_csv(OUT_DIR / "hungarian_retrieval_seedmean.csv", index=False)

    make_figure(df, OUT_DIR / "fig_hungarian_vs_mexa_vs_top1.png")
    decomposition_table(df, OUT_DIR / "hungarian_decomposition_2x2.csv")


if __name__ == "__main__":
    main()
