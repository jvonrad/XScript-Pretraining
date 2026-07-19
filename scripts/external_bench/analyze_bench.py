#!/usr/bin/env python
"""Turn returned benchmark JSONs into starved-vs-fair comparison tables.

Consumes either the combined summary.json or a directory of per-run JSONs
produced by run_benchmarks.py, and prints markdown tables:
  1. raw accuracy per model x task
  2. fair - starved delta per mixture (does the fair tokenizer help downstream?)
  3. same-script vs cross-script view for the EN-anchored bilingual models

Pure stdlib -- runs anywhere, no torch/GPU.

    python analyze_bench.py summary.json
    python analyze_bench.py results/bench/          # dir of <model>_final.json
"""
import json
import sys
from pathlib import Path

FAMILIES = ["global_mmlu", "belebele", "xnli"]
# lm-eval task stem -> our language code
LANG_OF = {
    "en": "en", "eng_Latn": "en",
    "de": "de", "deu_Latn": "de",
    "fr": "fr", "fra_Latn": "fr",
    "ar": "ar", "arb_Arab": "ar",
    "zh": "zh", "zho_Hans": "zh",
}
SAME_SCRIPT = {"de": "same", "fr": "same", "ar": "cross", "zh": "cross"}


def parse_task(task: str) -> tuple[str, str] | None:
    for fam in FAMILIES:
        if task.startswith(fam + "_"):
            stem = task[len(fam) + 1:]
            if stem in LANG_OF:
                return fam, LANG_OF[stem]
    return None


def load(path: Path) -> dict[str, dict[str, float]]:
    """model -> {task: acc}."""
    if path.is_dir():
        out = {}
        for f in sorted(path.glob("*_final.json")):
            d = json.loads(f.read_text())
            out[d["run"]] = {k: v for k, v in d["scores"].items() if v is not None}
        return out
    d = json.loads(path.read_text())
    scores = d.get("scores", d)
    return {m: {k: v for k, v in s.items() if isinstance(v, (int, float))}
            for m, s in scores.items() if isinstance(s, dict)}


def split_name(model: str) -> tuple[str, str]:
    mix, cond = model.rsplit("-", 1)        # "en-ar", "fair"
    return mix, cond


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    data = load(Path(sys.argv[1]))
    if not data:
        sys.exit("no scores found")

    # index: (mixture, cond, family, lang) -> acc
    cell: dict[tuple, float] = {}
    mixtures, langs_seen = set(), set()
    for model, scores in data.items():
        mix, cond = split_name(model)
        mixtures.add(mix)
        for task, acc in scores.items():
            fl = parse_task(task)
            if fl:
                fam, lang = fl
                cell[(mix, cond, fam, lang)] = acc
                langs_seen.add(lang)

    # ---- 1. raw accuracy -------------------------------------------------
    print("## Raw accuracy (acc)\n")
    print("| model | " + " | ".join(f"{fam}:{l}" for fam in FAMILIES
                                     for l in sorted(langs_seen)) + " |")
    print("|" + "---|" * (1 + len(FAMILIES) * len(sorted(langs_seen))))
    for model in sorted(data):
        mix, cond = split_name(model)
        row = [model]
        for fam in FAMILIES:
            for l in sorted(langs_seen):
                v = cell.get((mix, cond, fam, l))
                row.append(f"{v:.3f}" if v is not None else "-")
        print("| " + " | ".join(row) + " |")

    # ---- 2. fair - starved delta ----------------------------------------
    print("\n## Fair − starved (accuracy points; +ve => fair tokenizer helps)\n")
    print("| mixture | family | lang | starved | fair | Δ |")
    print("|---|---|---|---|---|---|")
    deltas = []
    for mix in sorted(mixtures):
        for fam in FAMILIES:
            for l in sorted(langs_seen):
                s = cell.get((mix, "starved", fam, l))
                f = cell.get((mix, "fair", fam, l))
                if s is not None and f is not None:
                    d = f - s
                    deltas.append(d)
                    print(f"| {mix} | {fam} | {l} | {s:.3f} | {f:.3f} | {d:+.3f} |")
    if deltas:
        print(f"\nmean Δ (fair − starved) over {len(deltas)} cells: "
              f"**{sum(deltas)/len(deltas):+.4f}**")

    # ---- 3. same vs cross script (EN-anchored bilingual, partner language) --
    print("\n## EN-anchored bilingual: partner-language accuracy by script\n")
    print("| mixture | partner | script | family | starved | fair | Δ |")
    print("|---|---|---|---|---|---|---|")
    for mix in sorted(m for m in mixtures if m.startswith("en-") and "-" in m):
        partner = mix.split("-", 1)[1]
        if partner not in SAME_SCRIPT:
            continue
        for fam in FAMILIES:
            s = cell.get((mix, "starved", fam, partner))
            f = cell.get((mix, "fair", fam, partner))
            if s is not None or f is not None:
                ss = f"{s:.3f}" if s is not None else "-"
                ff = f"{f:.3f}" if f is not None else "-"
                dd = f"{f - s:+.3f}" if (s is not None and f is not None) else "-"
                print(f"| {mix} | {partner} | {SAME_SCRIPT[partner]} | {fam} | {ss} | {ff} | {dd} |")


if __name__ == "__main__":
    main()
