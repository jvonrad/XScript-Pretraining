#!/usr/bin/env python
"""Export run_alignment.py's per-run JSONs as tidy CSVs for plotting.

The raw per-run JSONs are ~31 MB for 26 models, most of which is per-query
arrays (`hits`, `dprime_q`) that exist for the paired bootstrap and are useless
for visualisation. This writes the plot-shaped subset instead -- long/tidy
format, one row per (model, pair, variant, layer) -- small enough to commit.

    python export_alignment.py $WORK/results/alignment/ results/alignment/

Outputs:
  per_layer.csv     the depth profiles: every metric at every layer. This is the
                    one to plot -- alignment emerges at DIFFERENT DEPTHS across
                    models (bilinguals peak ~L15-16, monolinguals ~L12-16), which
                    is invisible in any single-layer summary and was the cause of
                    a retracted result (CLAUDE.md 6b).
  summary.csv       the two reported layers (`ref`, `best`) incl. CKA, which is
                    only computed there.
  lexical_floor.csv the model-free TF-IDF token-overlap floor, per tokenizer.
                    Any model cell at or below its floor carries no
                    representational signal -- plot it as a reference line.

Pure stdlib.
"""
import argparse
import csv
import json
from pathlib import Path

PER_LAYER_METRICS = ["top1_a2b", "top1_b2a", "mutual_nn", "cosine_matched",
                     "cosine_nonmatched", "cosine_margin", "dprime"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for f in sorted(args.results_dir.glob("*.json")):
        try:
            runs.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            print(f"[warn] skipping unreadable {f}")
    if not runs:
        raise SystemExit(f"no *.json under {args.results_dir}")

    pl = args.out_dir / "per_layer.csv"
    with open(pl, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "tok", "train_langs", "n_sentences", "pair", "a", "b",
                    "same_script", "trained_pair", "n_trained_in_pair",
                    "variant", "layer", "is_ref_layer", *PER_LAYER_METRICS])
        for d in runs:
            tl = set(d.get("train_langs") or [])
            ref = d.get("ref_layer")
            for pair, v in (d.get("pairs") or {}).items():
                shared = len({v["a"], v["b"]} & tl)
                for variant in ("raw", "centered"):
                    if variant not in v:
                        continue
                    for ly, m in enumerate(v[variant]["per_layer"]):
                        w.writerow([
                            d["run"], d.get("tok"), "+".join(sorted(tl)),
                            d.get("n_sentences"), pair, v["a"], v["b"],
                            int(v["same_script"]), int(bool(v.get("trained_pair"))),
                            shared, variant, ly, int(ly == ref),
                            *[f"{m[k]:.6g}" for k in PER_LAYER_METRICS]])

    sm = args.out_dir / "summary.csv"
    with open(sm, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "tok", "train_langs", "pair", "a", "b", "same_script",
                    "trained_pair", "n_trained_in_pair", "variant", "which_layer",
                    "layer", *PER_LAYER_METRICS, "cka"])
        for d in runs:
            tl = set(d.get("train_langs") or [])
            for pair, v in (d.get("pairs") or {}).items():
                shared = len({v["a"], v["b"]} & tl)
                for variant in ("raw", "centered"):
                    if variant not in v:
                        continue
                    for which in ("ref", "best"):
                        m = v[variant][which]
                        w.writerow([
                            d["run"], d.get("tok"), "+".join(sorted(tl)), pair,
                            v["a"], v["b"], int(v["same_script"]),
                            int(bool(v.get("trained_pair"))), shared, variant,
                            which, v[variant][f"{which}_layer"],
                            *[f"{m[k]:.6g}" for k in PER_LAYER_METRICS],
                            f"{m.get('cka', float('nan')):.6g}"])

    lf = args.out_dir / "lexical_floor.csv"
    seen = {}
    for d in runs:
        if d.get("lexical_baseline") and d.get("tok") not in seen:
            seen[d["tok"]] = d["lexical_baseline"]
    with open(lf, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tok", "pair", "top1_a2b", "top1_b2a", "mutual_nn", "cosine_margin"])
        for tok, base in sorted(seen.items()):
            for pair, m in base.items():
                w.writerow([tok, pair, f"{m['top1_a2b']:.6g}", f"{m['top1_b2a']:.6g}",
                            f"{m['mutual_nn']:.6g}", f"{m['cosine_margin']:.6g}"])

    for p in (pl, sm, lf):
        rows = sum(1 for _ in open(p)) - 1
        print(f"[export] {p}  {rows} rows  {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
