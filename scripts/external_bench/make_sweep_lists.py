#!/usr/bin/env python
"""Split models.json into balanced per-core-pair model lists for a fan-out.

    python make_sweep_lists.py --n 2 --out $WORK/lists

Balances on **own-language cell count**, not model count. Under `--own-langs` a
bilingual costs roughly twice what a monolingual does (CLAUDE.md 6g measured
27.6 min for a 1-language model against 49.4 for a 2-language one on the six
MuBench families), and the roster is 68 monolinguals to 48 bilinguals, so
splitting by model count alone leaves the last worker running long after the
others are idle.

Writes `<out>/list_00.txt` .. `<out>/list_<n-1>.txt`, one friendly model name
per line, for `run_xcsqa_sweep.sh <cores> <list>`.
"""
import argparse
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="number of core-pairs")
    ap.add_argument("--out", required=True, help="directory to write the lists into")
    ap.add_argument("--models", default=str(_REPO / "results" / "models.json"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these friendly names (default: all)")
    args = ap.parse_args()

    models = json.loads(Path(args.models).read_text())
    names = sorted(models if args.only is None else
                   [n for n in models if n in set(args.only)])
    if not names:
        raise SystemExit("no models selected")

    # Longest-processing-time-first greedy: assign the most expensive model to
    # the least-loaded worker. Optimal to within 4/3 of perfect balance, which
    # is far inside the noise of per-model download time.
    load = [0] * args.n
    parts: list[list[str]] = [[] for _ in range(args.n)]
    for name in sorted(names, key=lambda x: -len(models[x]["langs"])):
        i = load.index(min(load))
        parts[i].append(name)
        load[i] += len(models[name]["langs"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, part in enumerate(parts):
        p = out / f"list_{i:02d}.txt"
        p.write_text("".join(f"{n}\n" for n in sorted(part)))
        print(f"{p}  models={len(part):3d}  cells={load[i]:3d}")
    print(f"\ntotal: {len(names)} models, {sum(load)} own-language cells")


if __name__ == "__main__":
    raise SystemExit(main())
