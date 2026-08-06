#!/usr/bin/env python
"""Rebuild results/mubench_sweep/accuracy_table.md from every source we have.

The committed table was produced by an ad-hoc script on the eval box that ran
the 100-checkpoint sweep, and that script was never committed. It also had a
structural flaw: the 12b/23b/30b passes ran MuBench only -- SIB-200 and XNLI
were already calibrated for those models in the 6e pass and stored elsewhere
-- and rather than read that second source, the table froze the seam in as 96
en-dash placeholders. This reads all three sources so the seam disappears.

    python build_tables.py [--new-results DIR] [--check] [-o PATH]

Sources, most specific first:

  1. --new-results (default /mnt/scratch/xscript_bench/results) -- the sweep
     of 2026-08-05: the nine *-15b monolinguals and the seven de-starved-*.
     MuBench read from the per-model JSON; acc_cal re-derived from the raw
     sidecars via rawscores.score_variants.
  2. results/recalibrated/{extra_bench,appendix_c5} -- committed per-example
     acc_cal hit lists for the 41 calibrated checkpoints. This is what fills
     the 96 dashes.
  3. results/mubench_sweep/accuracy_table.md -- the existing table. For the
     100 older checkpoints this is the ONLY surviving record of their MuBench
     accuracies: the per-model JSONs died with the box that wrote them. Values
     are carried through verbatim, never recomputed.

Estimator per family is 6g's and is applied only when deriving from source 1
or 2; source 3's numbers already embed it:

    ARC-E, BMLAMA -> acc | Story, HSwag -> acc_norm | SIB-200, XNLI -> acc_cal

--check re-emits without writing and diffs against the committed file, so a
rerun that should be a no-op is verifiable as one.
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "results" / "mubench_sweep" / "accuracy_table.md"
COLS = [("SIB200", "sib200"), ("XNLI", "xnli"), ("ARC-E", "mub_arceasy"),
        ("Story", "mub_storycloze"), ("HSwag", "mub_hellaswag"),
        ("BMLAMA", "mub_bmlama")]
EST = {"mub_arceasy": "acc", "mub_bmlama": "acc", "mub_storycloze": "acc_norm",
       "mub_hellaswag": "acc_norm", "sib200": "acc_cal", "xnli": "acc_cal"}
LANGS = ["en", "de", "fr", "ar", "zh"]
CODE = {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn",
        "ar": "arb_Arab", "zh": "zho_Hans"}
DASH = "–"


def fam_of(task: str):
    if task.startswith("sib200_") and "enlab" not in task:
        return "sib200"
    if task.startswith("xnli_"):
        return "xnli"
    for f in ("mub_arceasy", "mub_storycloze", "mub_hellaswag", "mub_bmlama"):
        if task.startswith(f + "_"):
            return f
    return None


def budget_of(run: str) -> int:
    m = re.search(r"-(\d+)b$", run)
    return int(m.group(1)) if m else 30


def from_table(cells, langs, path=OUT):
    """Source 3: the committed table (verbatim; the only record for 100 runs)."""
    sec = None
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            sec = line[3:].split()[0].lower()
            continue
        if line.startswith("| ") and "|---" not in line and not line.startswith("| model"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) == 8 and c[1].isdigit():
                langs.setdefault(c[0], set()).add(sec)
                for (_, fam), v in zip(COLS, c[2:]):
                    if re.match(r"^[0-9.]+$", v):
                        cells.setdefault((c[0], sec, fam), float(v))


def from_recalibrated(cells, langs):
    """Source 2: committed acc_cal hit lists -> fills the 96 dashes."""
    base = REPO / "results" / "recalibrated"
    for sub, key in (("extra_bench", "correct"), ("appendix_c5", "correct_calibrated")):
        for f in sorted((base / sub).glob("*_final.json")):
            run = f.name[:-len("_final.json")]
            for lang, ts in (json.loads(f.read_text()).get(key) or {}).items():
                for task, per_metric in ts.items():
                    fam = fam_of(task)
                    if fam not in ("sib200", "xnli") or not isinstance(per_metric, dict):
                        continue
                    hl = per_metric.get("acc_cal")
                    if isinstance(hl, list) and hl:
                        cells.setdefault((run, lang, fam), sum(hl) / len(hl))


def from_new(cells, langs, newdir: Path):
    """Source 1: the 2026-08-05 sweep; acc_cal re-derived from raw."""
    from xscript.eval import rawscores as rs
    for sub in ("extra_bench", "appendix_c5"):
        d = newdir / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*_final.json")):
            run = f.name[:-len("_final.json")]
            own = set(json.loads(f.read_text()).get("langs") or [])
            for lang, ts in (json.loads(f.read_text()).get("metrics") or {}).items():
                for task, m in ts.items():
                    fam = fam_of(task)
                    if fam and fam.startswith("mub_"):
                        cells[(run, lang, fam)] = m[EST[fam]]
                        langs.setdefault(run, set()).add(lang)
            raw_f = d / "raw" / f"{run}_raw.json"
            if not raw_f.exists():
                continue
            for lang, ts in (json.loads(raw_f.read_text()).get("raw") or {}).items():
                for task, raw in ts.items():
                    fam = fam_of(task)
                    if fam not in ("sib200", "xnli"):
                        continue
                    hl = rs.score_variants(raw).get("acc_cal")
                    if hl:
                        cells[(run, lang, fam)] = sum(hl) / len(hl)
                        langs.setdefault(run, set()).add(lang)


def render(cells, langs) -> str:
    by_lang = {l: set() for l in LANGS}
    for (run, lang, _) in cells:
        if lang in by_lang:
            by_lang[lang].add(run)
    out = ["# Benchmark accuracy — every model x language", "",
           "One number per cell: **accuracy** (not headroom). `" + DASH + "` = the model",
           "was never scored on that (lang, benchmark) pair. Models are scored only on",
           "their own training languages (`--own-langs`), so a model has rows only for",
           "the languages it trained on.", "",
           "Estimator per family (CLAUDE.md §6g): ARC-E/BMLAMA `acc` | Story/HSwag",
           "`acc_norm` | SIB-200/XNLI `acc_cal`. Never `acc_tokennorm` — it is",
           "tokenizer-dependent and this project's whole contrast is a tokenizer.", "",
           "Chance: SIB200 .143 | XNLI .333 | ARC-E .25 | Story .50 | HSwag .25 | BMLAMA ~.10", "",
           "⚠️ 30B rows are COOLED (LR 3.0e-4); 1b-23b are mid-stable (3.0e-3).",
           "Do not pair across that boundary (CLAUDE.md §6/§6d).", "",
           "Regenerate with `scripts/external_bench/build_tables.py`.", ""]
    for lang in LANGS:
        runs = sorted(by_lang[lang], key=lambda r: (budget_of(r), r))
        if not runs:
            continue
        out += [f"## {lang.upper()}  ({len(runs)} checkpoints)", "",
                "| model | B | " + " | ".join(c for c, _ in COLS) + " |",
                "|---|---|" + "---|" * len(COLS)]
        for r in runs:
            vals = []
            for _, fam in COLS:
                v = cells.get((r, lang, fam))
                vals.append(f"{v:.3f}" if v is not None else DASH)
            out.append(f"| {r} | {budget_of(r)} | " + " | ".join(vals) + " |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-results", type=Path,
                    default=Path("/mnt/scratch/xscript_bench/results"))
    ap.add_argument("-o", "--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="report coverage, do not write")
    a = ap.parse_args()

    cells, langs = {}, {}
    from_new(cells, langs, a.new_results)
    from_recalibrated(cells, langs)
    from_table(cells, langs)

    # Own-language rows only. Some committed JSONs predate --own-langs and carry
    # cells for languages the model never trained on; keeping them would invent
    # rows the sweep never scored (and 6d's degeneracy check found 16 collapsed
    # cells, all of them out-of-domain, so they are not merely redundant).
    own = json.loads((REPO / "results" / "models.json").read_text())
    unknown = sorted({r for r, _, _ in cells} - set(own))
    if unknown:
        print(f"WARNING: not in models.json, dropped: {unknown}")
    cells = {(r, l, f): v for (r, l, f), v in cells.items()
             if r in own and l in own[r]["langs"]}

    runs = sorted({r for r, _, _ in cells})
    pairs = sorted({(r, l) for r, l, _ in cells})
    filled = len(cells)
    print(f"{len(runs)} checkpoints, {len(pairs)} (model, lang) rows, {filled} filled cells")
    holes = [(r, l, f) for (r, l) in pairs for _, f in COLS if (r, l, f) not in cells]
    print(f"remaining holes: {len(holes)}")
    for h in holes[:20]:
        print("   ", h)

    text = render(cells, langs)
    if a.check:
        old = a.out.read_text()
        print(f"\n--check: {'IDENTICAL' if old == text else 'DIFFERS'} from {a.out}")
        return
    a.out.write_text(text)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
