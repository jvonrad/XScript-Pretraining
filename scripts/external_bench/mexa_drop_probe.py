#!/usr/bin/env python
"""Reproduce the centered-MEXA late-training drop, then ask what drives it.

CONTEXT
=======
The thesis reports (fig_N) that with the language centroid removed, MEXA
alignment at EARLY layers holds close to 1.0 until ~15B tokens and then falls
sharply by 23B -- for en-de and en-fr under the fair tokenizer, and deeper in
the network (layers 6-9) under the starved one. Deep layers stay at ceiling
throughout. The proposed reading is that the model is progressively learning
LANGUAGE-SPECIFIC geometry: better tokenisation pulls the subspaces together
but also sharpens within-language meaning, and part of meaning is language
specific (`Schuld` = debt AND guilt), which must separate the spaces at
whatever depth that meaning lives.

This script does two things, in order:

  1. REPRODUCE the drop from the cached FLORES+ embeddings
     (`jvonrad/xscript-embeddings`, (n_layers+1, 2009, 2048) per language),
     using MEXA as the thesis defines it -- the true translation is the argmax
     in BOTH its row and its column, i.e. `alignment._retrieval_sim`'s
     `mutual_nn`. Centering is the language centroid over the 2009 FLORES
     sentences, exactly `alignment._center`.

  2. DECOMPOSE it: identify the sentences that flip from retrieved to
     not-retrieved across the drop, and test whether they are lexically
     distinguishable from the ones that survive. If the language-specific-
     semantics reading is right, the flippers should be the culturally or
     idiomatically loaded ones. If they look like a random draw of FLORES,
     the drop is real but the proposed explanation is not supported by it.

⚠️ POOLING. The cached npz stores ONE pooled embedding per language; the
thesis figure uses position-weighted pooling. If the cache was written with
mean pooling the absolute MEXA values here will not match fig_N exactly. The
SHAPE over layers and budgets is what this script is for -- if the drop does
not appear at all, that is the finding worth having.

    python mexa_drop_probe.py --emb-dir /home/ubuntu/xscript_emb \\
        --pair en-de --tok fair --budgets 2b 5b 10b 15b 23b
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def center(E):
    """alignment._center: remove the language centroid, then re-normalise."""
    C = E - E.mean(0, keepdims=True)
    return C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)


def l2(E):
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)


def mexa_sim(sim):
    """MEXA from a similarity matrix: the true translation is the argmax in
    BOTH directions (thesis Eq. mexa == `alignment._retrieval_sim`'s
    `mutual_nn`)."""
    n = sim.shape[0]
    d = np.arange(n)
    return (sim.argmax(1) == d) & (sim.argmax(0) == d)


def csls(sim, k=10):
    """Cross-domain Similarity Local Scaling (Conneau et al. 2018) -- the
    standard hubness correction for cross-lingual retrieval. Subtracts each
    item's mean similarity to its k nearest neighbours in the other language,
    so a "hub" that is close to everything stops winning every argmax.

    This exists here because MEXA is argmax-based and high-dimensional
    retrieval is hubness-prone: a metric can collapse while the underlying
    geometry improves. Comparing MEXA against MEXA-CSLS separates "the pairs
    moved apart" from "a few sentences became hubs"."""
    rt = np.sort(sim, axis=0)[-k:, :].mean(0)      # per-target hubness
    rs = np.sort(sim, axis=1)[:, -k:].mean(1)      # per-query
    return 2 * sim - rt[None, :] - rs[:, None]


def diagnose(E, F, k=10):
    """MEXA plus the statistics that say WHY it moved.

    `matched`/`margin` describe the geometry: if translation pairs are
    genuinely separating, matched cosine FALLS. `max_votes` describes the
    metric's failure mode: how many queries a single target wins. A drop in
    `mexa` with `matched` rising and `max_votes` exploding is hubness, not
    separation -- see the module docstring."""
    sim = E @ F.T
    n = sim.shape[0]
    hit = mexa_sim(sim)
    matched = float(np.diag(sim).mean())
    nonmatched = float((sim.sum() - np.trace(sim)) / (n * (n - 1)))
    votes = np.bincount(sim.argmax(1), minlength=n)
    return {
        "mexa": float(hit.mean()),
        "mexa_csls": float(mexa_sim(csls(sim, k)).mean()),
        "cosine_matched": matched,
        "cosine_nonmatched": nonmatched,
        "margin": matched - nonmatched,
        "max_votes": int(votes.max()),
        "vote_std": float(votes.std()),
    }, hit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-dir", required=True, type=Path)
    ap.add_argument("--pair", default="en-de")
    ap.add_argument("--tok", default="fair")
    ap.add_argument("--budgets", nargs="+",
                    default=["2b", "5b", "10b", "15b", "23b"])
    ap.add_argument("--variant", default="centered", choices=["centered", "raw"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = (REPO / "results" / "cultural_drift" /
                    f"mexa_{args.pair}_{args.tok}_{args.variant}.json")

    a, b = args.pair.split("-")
    grid, hits = {}, {}
    for bud in args.budgets:
        f = args.emb_dir / f"{args.pair}-{args.tok}-{bud}.npz"
        if not f.exists():
            print(f"[mexa] missing {f.name}, skipping")
            continue
        z = np.load(f)
        meta = json.loads(str(z["__meta__"]))
        EA, EB = z[a], z[b]
        n_layers = EA.shape[0]
        row, hrow = [], []
        for L in range(n_layers):
            X = center(EA[L]) if args.variant == "centered" else l2(EA[L])
            Y = center(EB[L]) if args.variant == "centered" else l2(EB[L])
            stats, h = diagnose(X, Y)
            stats["layer"] = L
            row.append(stats)
            hrow.append(h)
        grid[bud] = row
        hits[bud] = np.stack(hrow)          # (n_layers, 2009)
        print(f"[mexa] {args.pair}-{args.tok}-{bud}: n={meta['n_sentences']}, "
              f"layers={n_layers}", flush=True)

    if not grid:
        raise SystemExit("no embeddings found")
    buds = list(grid)
    n_layers = len(grid[buds[0]])
    get = lambda b, L, k: grid[b][L][k]
    print(f"\n## MEXA ({args.variant}) -- {args.pair} / {args.tok}\n")
    print(f"{'layer':>5} " + "".join(f"{b:>9}" for b in buds) + f"{'peak-last':>10}")
    for L in range(n_layers):
        vals = [get(b, L, "mexa") for b in buds]
        print(f"{L:5d} " + "".join(f"{v:9.4f}" for v in vals)
              + f"{max(vals) - vals[-1]:10.4f}")

    drops = [(max(get(b, L, "mexa") for b in buds[:-1]) - get(buds[-1], L, "mexa"), L)
             for L in range(n_layers)]
    drops.sort(reverse=True)
    print(f"\nlargest peak->final falls: " +
          ", ".join(f"L{L} {d:+.3f}" for d, L in drops[:5]))

    # --- is a fall separation, or hubness? -------------------------------
    print(f"\n## Why (layers with the largest falls) -- if MEXA falls while "
          f"`matched` RISES and `max_votes` explodes, the pairs did not "
          f"separate; a few targets became hubs.\n")
    print(f"{'layer':>5} {'bud':>6} {'MEXA':>7} {'+CSLS':>7} {'matched':>8} "
          f"{'margin':>7} {'maxvote':>8}")
    for _, L in drops[:4]:
        for b in buds:
            g = grid[b][L]
            print(f"{L:5d} {b:>6} {g['mexa']:7.4f} {g['mexa_csls']:7.4f} "
                  f"{g['cosine_matched']:8.4f} {g['margin']:7.4f} "
                  f"{g['max_votes']:8d}")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"pair": args.pair, "tok": args.tok, "variant": args.variant,
         "budgets": buds, "by_layer": grid}, indent=1))
    np.savez_compressed(args.out.with_suffix(".hits.npz"),
                        **{b: hits[b] for b in buds})
    print(f"\n[mexa] wrote {args.out} and per-sentence hits alongside")


if __name__ == "__main__":
    raise SystemExit(main())
