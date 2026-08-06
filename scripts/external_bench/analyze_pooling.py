#!/usr/bin/env python
"""T1-T5: is the fair-vs-starved alignment gap a model property or a pooling artifact?

Our alignment pipeline pools sentences with an UNWEIGHTED MEAN over all
non-pad tokens, BOS included. MEXA (arXiv 2410.05873) instead uses a
POSITION-WEIGHTED average, w_t = t / sum_k k, and only ever compares that
against LAST-TOKEN pooling -- plain unweighted mean is not evaluated anywhere in
that paper, so our estimator is unvalidated on exactly the axis that matters
here.

Why the axis matters: the starved tokenizer emits 1.14-1.32x more tokens for the
same text. Under mean pooling the weight on BOS is 1/T and half the mass sits on
the first half of the sentence, where representations are least contextualised;
under MEXA's weighting BOS carries ~2/T^2 and the early half ~25%; under
last-token both are 0. So a fair-vs-starved difference can be manufactured by
the pooling rule interacting with fertility, without any difference in the
trained models.

    python analyze_pooling.py RESULTS_DIR [--emb-dir DIR] [--out README.md]

RESULTS_DIR holds <model>.json from `run_alignment.py --poolings mean
mean_nobos weighted last`. --emb-dir enables T4 (the length control), which
needs the cached per-layer embeddings to re-run retrieval on a sentence subset.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from xscript.eval.alignment import _center, _retrieval_sim  # noqa: E402

POOLINGS = ["mean", "mean_nobos", "weighted", "last"]
PARTNERS = ["de", "fr", "ar", "zh"]
BANDS = [("L0-4", range(0, 5)), ("L5-8", range(5, 9)),
         ("L9-12", range(9, 13)), ("L13-16", range(13, 17))]
VARIANT = "centered"
METRIC = "mutual_nn"
THRESH = 0.90


def load(results: Path, suffix: str) -> dict:
    """{(partner, tok): doc} for the 8 EN-anchored bilinguals."""
    out = {}
    for p in PARTNERS:
        for tok in ("fair", "starved"):
            f = results / f"en-{p}-{tok}{suffix}.json"
            if f.exists():
                out[(p, tok)] = json.loads(f.read_text())
    return out


def series(doc, pooling, pair, metric=METRIC):
    """Per-layer metric for one pooling/pair, the `centered` variant."""
    pairs = doc["poolings"][pooling]["pairs"] if "poolings" in doc else doc["pairs"]
    return np.array([L[metric] for L in pairs[pair][VARIANT]["per_layer"]])


def depth_to(vals, thresh=THRESH):
    """First layer index reaching `thresh` (None if it never does)."""
    hit = np.nonzero(vals >= thresh)[0]
    return int(hit[0]) if len(hit) else None


# --------------------------------------------------------------------------

def t1(docs, metric=METRIC):
    """fair - starved, per pooling x pair x layer band."""
    rows = []
    for pooling in POOLINGS:
        for p in PARTNERS:
            if (p, "fair") not in docs or (p, "starved") not in docs:
                continue
            pair = f"en-{p}"
            d = series(docs[(p, "fair")], pooling, pair, metric) \
                - series(docs[(p, "starved")], pooling, pair, metric)
            cell = {"pooling": pooling, "pair": pair}
            for name, rng in BANDS:
                cell[name] = float(d[list(rng)].mean())
            # peak-layer gap: each condition scored at its OWN argmax layer,
            # which is what CLAUDE.md section 6b's surviving claim is about.
            f_, s_ = (series(docs[(p, t)], pooling, pair, metric)
                      for t in ("fair", "starved"))
            cell["peak"] = float(f_.max() - s_.max())
            rows.append(cell)
    return rows


def t2(docs, pooling_list=POOLINGS):
    """Layer at which each condition first reaches mutual_nn >= 0.90."""
    rows = []
    for pooling in pooling_list:
        for p in PARTNERS:
            if (p, "fair") not in docs or (p, "starved") not in docs:
                continue
            pair = f"en-{p}"
            r = {"pooling": pooling, "pair": pair}
            for tok in ("fair", "starved"):
                v = series(docs[(p, tok)], pooling, pair)
                r[tok] = depth_to(v)
                r[f"{tok}_max"] = float(v.max())
            r["delay"] = (None if r["fair"] is None or r["starved"] is None
                          else r["starved"] - r["fair"])
            rows.append(r)
    return rows


def t4(docs, emb_dir: Path, suffix: str):
    """Length control: recompute the L5-8 gap on the tertile of sentences whose
    starved-minus-fair token-count difference is SMALLEST.

    If the gap is driven by the pooling rule interacting with fertility it must
    shrink on sentences where the two tokenizers barely disagree on length. If
    it is a property of the trained models it should survive.

    Retrieval is recomputed from scratch on the subset (a smaller candidate pool
    is an easier task, so absolute values rise) -- only the fair-vs-starved GAP
    is comparable, and it is compared against the same-subset full-pool control
    reported alongside.
    """
    rows = []
    for p in PARTNERS:
        if (p, "fair") not in docs or (p, "starved") not in docs:
            continue
        pair, langs = f"en-{p}", ["en", p]
        tc = {}
        for tok in ("fair", "starved"):
            counts = docs[(p, tok)].get("token_counts")
            if not counts:
                print(f"[t4] {pair}: no token_counts stored, skipping")
                break
            tc[tok] = np.array([counts[l] for l in langs]).sum(0)
        if len(tc) != 2:
            continue
        diff = tc["starved"] - tc["fair"]          # per sentence, over the pair
        order = np.argsort(diff, kind="stable")
        k = len(order) // 3
        terts = [np.sort(order[i * k:(i + 1) * k]) for i in range(3)]
        sub = terts[0]                             # smallest-difference tertile

        for pooling in POOLINGS:
            gaps = {}
            for scope, idx in (("all", None), ("tertile", sub),
                               ("t_mid", terts[1]), ("t_big", terts[2])):
                per_tok = {}
                for tok in ("fair", "starved"):
                    E = load_emb(emb_dir, f"en-{p}-{tok}{suffix}", pooling, langs)
                    if E is None:
                        break
                    A, B = E[langs[0]], E[langs[1]]
                    if idx is not None:
                        A, B = A[:, idx], B[:, idx]
                    vals = []
                    for ly in range(A.shape[0]):
                        X, Y = _center(A[ly]), _center(B[ly])
                        vals.append(_retrieval_sim(X @ Y.T)[METRIC])
                    per_tok[tok] = np.array(vals)
                if len(per_tok) != 2:
                    gaps = {}
                    break
                d = per_tok["fair"] - per_tok["starved"]
                gaps[scope] = float(d[list(BANDS[1][1])].mean())   # L5-8
            if gaps:
                # Even the smallest tertile still has a median +9..+12 token
                # difference, so it is not a zero-fertility-difference control.
                # Regress the gap on each tertile's median difference and
                # extrapolate to diff = 0: `gap_at_0` estimates what would
                # remain if the two tokenizers produced identical lengths.
                x = np.array([np.median(diff[t]) for t in terts], dtype=float)
                y = np.array([gaps["tertile"], gaps["t_mid"], gaps["t_big"]])
                slope, intercept = np.polyfit(x, y, 1)
                rows.append({"pair": pair, "pooling": pooling,
                             "n_tertile": int(len(sub)),
                             "median_diff_all": float(np.median(diff)),
                             "median_diff_tertile": float(np.median(diff[sub])),
                             "gap_at_0": float(intercept),
                             "frac_retained": (float(intercept / gaps["all"])
                                               if gaps["all"] else float("nan")),
                             **gaps})
    return rows


def load_emb(emb_dir: Path, run: str, pooling: str, langs):
    f = emb_dir / f"{run}__{pooling}.npz"
    if not f.exists():
        print(f"[t4] missing {f}")
        return None
    z = np.load(f)
    return {l: z[l] for l in langs}


# --------------------------------------------------------------------------

def md_table(header, rows, fmt):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r, h) for h in header) + " |")
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--emb-dir", type=Path, default=None)
    ap.add_argument("--suffix", default="", help="e.g. '-2b' for the 2B tier")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    docs = load(args.results, args.suffix)
    if not docs:
        sys.exit(f"no model JSONs found in {args.results} (suffix {args.suffix!r})")
    print(f"loaded {len(docs)} models: {sorted(docs)}")

    md = [f"# Pooling sensitivity of the alignment gap ({args.suffix or '30B finals'})",
          "",
          f"Metric: `{METRIC}`, variant `{VARIANT}`, each bilingual on its OWN "
          f"trained pair. Values are **fair minus starved**.", ""]

    md += ["## T1. fair - starved by layer band, all four poolings", ""]
    r1 = t1(docs)
    md += md_table(["pooling", "pair"] + [b for b, _ in BANDS] + ["peak"], r1,
                   lambda r, h: (r[h] if isinstance(r[h], str) else f"{r[h]:+.3f}"))

    md += ["## T2. depth to mutual_nn >= 0.90", "",
           "`-` = never reaches 0.90 at any layer; `max` is that condition's "
           "best layer value.", ""]
    r2 = t2(docs)
    md += md_table(["pooling", "pair", "fair", "starved", "delay",
                    "fair_max", "starved_max"], r2,
                   lambda r, h: (r[h] if isinstance(r[h], str)
                                 else "-" if r[h] is None
                                 else f"{r[h]:.3f}" if h.endswith("_max")
                                 else f"{r[h]:+d}" if h == "delay" else str(r[h])))

    if args.emb_dir:
        md += ["## T4. length control (L5-8 gap, smallest-fertility-difference tertile)", ""]
        r4 = t4(docs, args.emb_dir, args.suffix)
        if r4:
            md += ["Gap measured on the smallest / middle / largest tertile of "
                   "per-sentence `starved - fair` token-count difference. "
                   "`gap_at_0` linearly extrapolates the three tertiles to a "
                   "zero length difference; `frac_retained` = gap_at_0 / all.", ""]
            md += md_table(["pair", "pooling", "all", "tertile", "t_mid", "t_big",
                            "gap_at_0", "frac_retained", "median_diff_tertile",
                            "median_diff_all"],
                           r4, lambda r, h: (r[h] if isinstance(r[h], str)
                                             else f"{r[h]:+.3f}" if isinstance(r[h], float)
                                             else str(r[h])))

    text = "\n".join(md)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"\nwrote {args.out}")
    (args.results / "tables.json").write_text(json.dumps(
        {"t1": r1, "t2": r2, "t4": r4 if args.emb_dir else None}, indent=2))


if __name__ == "__main__":
    main()
