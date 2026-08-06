#!/usr/bin/env python
"""Guard: the patched multi-pooling harness must reproduce the committed sweep.

`results/alignment_v2_107/<run>.json` was produced by the pre-patch
`_embed`, which emitted a single unweighted-mean pooling. The patch adds three
more poolings to the SAME forward pass, so the accelerator now compiles a
different graph -- and on Neuron a different graph is entitled to a different
floating-point reduction order. Until the `mean` pooling is shown to come back
unchanged, NOTHING derived from the new poolings can be interpreted: a
fair-vs-starved shift would be indistinguishable from a harness change.

This is the same standing rule as `rawscores.check_reproduces()` (CLAUDE.md
section 6e): a re-derivation is not trusted until it reproduces the original.

    python verify_pooling_guard.py NEW.json --ref results/alignment_v2_107/en-fr-fair.json

Compares every (pair, variant, layer) cell on the two statistics the committed
files actually carry -- `mutual_nn` and `dprime`. Note `mutual_nn` is a rate
over n=2009, so its quantum is 1/2009 = 4.98e-4: any single flipped retrieval
moves it ~500x the 1e-6 tolerance. It either matches exactly or the embeddings
genuinely changed.
"""
import argparse
import json
import sys
from pathlib import Path

STATS = ("mutual_nn", "dprime")
VARIANTS = ("raw", "centered")


def _pairs(doc, pooling):
    """Pairs block from either schema (legacy top-level, or multi-pooling)."""
    if "poolings" in doc:
        if pooling not in doc["poolings"]:
            sys.exit(f"pooling {pooling!r} not in {list(doc['poolings'])}")
        return doc["poolings"][pooling]["pairs"]
    return doc["pairs"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("new", help="JSON written by the patched harness")
    ap.add_argument("--ref", required=True, help="committed reference JSON")
    ap.add_argument("--pooling", default="mean",
                    help="pooling in NEW to compare against REF (default: mean)")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    new = json.loads(Path(args.new).read_text())
    ref = json.loads(Path(args.ref).read_text())

    # Metadata identity first: a matching number means nothing if the two runs
    # scored different sentences or a different graph width.
    meta_bad = []
    for k in ("split", "eval_langs", "n_sentences", "n_layers", "ref_layer",
              "fixed_width", "tok"):
        if k in ref and new.get(k) != ref[k]:
            meta_bad.append(f"  {k}: new={new.get(k)!r} ref={ref[k]!r}")
    if meta_bad:
        print("METADATA MISMATCH -- the two runs are not comparable:")
        print("\n".join(meta_bad))
        sys.exit(2)

    np_, rp = _pairs(new, args.pooling), ref["pairs"]
    if set(np_) != set(rp):
        sys.exit(f"pair sets differ: new={sorted(np_)} ref={sorted(rp)}")

    worst = {s: (0.0, None) for s in STATS}
    n_cells = 0
    for pair in sorted(rp):
        for variant in VARIANTS:
            a, b = np_[pair][variant]["per_layer"], rp[pair][variant]["per_layer"]
            if len(a) != len(b):
                sys.exit(f"{pair}/{variant}: layer count {len(a)} vs {len(b)}")
            for ly, (x, y) in enumerate(zip(a, b)):
                n_cells += 1
                for s in STATS:
                    d = abs(x[s] - y[s])
                    if d > worst[s][0]:
                        worst[s] = (d, f"{pair}/{variant}/L{ly}")

    print(f"compared {n_cells} (pair x variant x layer) cells, "
          f"{len(rp)} pairs, n={ref['n_sentences']}")
    ok = True
    for s in STATS:
        d, where = worst[s]
        flag = "OK " if d <= args.tol else "FAIL"
        if d > args.tol:
            ok = False
        print(f"  [{flag}] max |delta {s}| = {d:.3e}"
              + (f"  at {where}" if where else ""))
    print(f"\ntolerance {args.tol:.0e} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
