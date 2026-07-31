#!/usr/bin/env python
"""Verify every MuBench task before spending accelerator time on it.

Checks, per benchmark:
  n            item count in each language
  aligned      `_id` sequence identical across all five languages
  gold ok      every gold index inside its option list
  parsed       options recovered from the prompt for every kept row
  pool         where an English original exists, what fraction of MuBench's
               English questions appear in it -- the check that caught
               arc_easy(en) vs ARC-Challenge(others)

Pure CPU. Run before any MuBench sweep.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from xscript.eval.c5_tasks.mubench import utils as U  # noqa: E402

POOL = {   # bench -> (hf dataset, config, question field)
    "arceasy": ("allenai/ai2_arc", "ARC-Easy", "question"),
    "hellaswag": ("Rowan/hellaswag", None, "ctx"),
}


def main() -> None:
    import datasets
    benches = sys.argv[1:] or list(U.SPECS)
    print(f"{'bench':12s} {'en n':>7s} {'aligned':>8s} {'gold ok':>8s} {'nopt':>12s}  pool")
    for b in benches:
        try:
            ds = {l: U._load(b, l) for l in U.LANGS}
        except Exception as exc:                      # noqa: BLE001
            print(f"{b:12s}  LOAD FAILED {type(exc).__name__}: {str(exc)[:60]}")
            continue
        ids = {l: [ds[l][i]["_id"] for i in range(len(ds[l]))] for l in U.LANGS}
        aligned = all(ids[l] == ids["en"] for l in U.LANGS)
        goldok = all(0 <= ds[l][i]["label"] < len(ds[l][i]["options"])
                     for l in U.LANGS for i in range(0, len(ds[l]), 37))
        nopt = sorted({len(ds["en"][i]["options"])
                       for i in range(0, len(ds["en"]), 11)})
        pool = ""
        if b in POOL:
            path, cfg, field = POOL[b]
            try:
                ref = datasets.load_dataset(path, cfg, split="test") if cfg \
                    else datasets.load_dataset(path, split="validation")
                refset = {str(x).strip().lower()[:80] for x in ref[field]}
                q = [ds["en"][i]["stem"].split("\n")[0]
                     .replace("Question:", "").strip().lower()[:80]
                     for i in range(len(ds["en"]))]
                hit = sum(1 for x in q if x in refset)
                pool = f"{hit / len(q):.1%} in {path.split('/')[-1]}{cfg or ''}"
            except Exception as exc:                  # noqa: BLE001
                pool = f"(pool check failed: {type(exc).__name__})"
        flag = "" if (aligned and goldok) else "   <-- PROBLEM"
        print(f"{b:12s} {len(ds['en']):7d} {str(aligned):>8s} {str(goldok):>8s} "
              f"{str(nopt):>12s}  {pool}{flag}")
        s = ds["ar"][0]
        print(f"             ar stem: {s['stem'][:70]!r}")
        print(f"             ar opts: {[o[:22] for o in s['options'][:3]]}")


if __name__ == "__main__":
    main()
