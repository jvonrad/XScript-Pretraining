#!/usr/bin/env python
"""Identify language-specific neurons (LAPE, arXiv 2402.16438) from the
per-model over-zero counts recorded by run_lape.py, and analyze them across
layers, training-token budgets, tokenizer conditions, and model families.

Pure CPU/numpy over the ~4 MB npz files; safe to rerun any time.

    python analyze_lape.py /mnt/scratch/xscript_lape/results/lape \
        --out /mnt/scratch/xscript_lape/results/lape_analysis

Outputs:
  <out>/lape_summary.json   per model: neuron sets, counts, layer profiles
  <out>/report.md           human-readable tables
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from xscript.eval import neurons  # noqa: E402

LANG_ORDER = ["en", "de", "fr", "ar", "zh"]
NAME_RE = re.compile(r"^(?P<family>[a-z]{2}(?:-[a-z]{2})?)-(?P<tok>fair|starved)"
                     r"(?:-(?P<budget>\d+)b)?$")


def parse_name(run: str):
    m = NAME_RE.match(run)
    if not m:
        return None
    fam = m.group("family")
    budget = int(m.group("budget")) if m.group("budget") else 30
    return {"family": fam, "tokcond": m.group("tok"), "budget": budget,
            "langs": fam.split("-")}


def identify(res, top_rate, filter_rate, bar_ratio):
    out = neurons.lape(res["over_zero"], res["n"], top_rate=top_rate,
                       filter_rate=filter_rate, activation_bar_ratio=bar_ratio)
    langs = res["langs"]
    sets = {}
    for li, lang in enumerate(langs):
        per_layer = out["neurons"][li]
        sets[lang] = {ly: sorted(t.tolist()) for ly, t in enumerate(per_layer)
                      if len(t)}
    return sets, out


def neuron_key_set(sets_lang: dict) -> set:
    return {(ly, h) for ly, hs in sets_lang.items() for h in hs}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-rate", type=float, default=0.01)
    ap.add_argument("--filter-rate", type=float, default=0.95)
    ap.add_argument("--bar-ratio", type=float, default=0.95)
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    out_dir = Path(args.out or (rdir.parent / "lape_analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = sorted(p.stem for p in rdir.glob("*.npz"))
    if not runs:
        sys.exit(f"no npz in {rdir}")

    summary = {}
    for run in runs:
        res = neurons.load(rdir, run)
        meta = parse_name(run)
        sets, out = identify(res, args.top_rate, args.filter_rate, args.bar_ratio)
        n_layers = res["over_zero"].shape[0]
        layer_counts = {l: [len(sets.get(l, {}).get(ly, [])) for ly in range(n_layers)]
                        for l in res["langs"]}
        probs = res["over_zero"] / res["n"]                    # [L, ffn, lang]
        summary[run] = {
            "meta": meta, "langs": res["langs"],
            "train_langs": res.get("train_langs"),
            "n_tokens": res["n"].tolist(),
            "mean_act_prob": probs.mean(axis=(0, 1)).round(4).tolist(),
            "counts": {l: sum(layer_counts[l]) for l in res["langs"]},
            "layer_counts": layer_counts,
            "neuron_sets": {l: {str(k): v for k, v in s.items()}
                            for l, s in sets.items()},
            "top_prob_value": out["top_prob_value"],
            "activation_bar": out["activation_bar"],
        }
        print(f"{run:26s} " + " ".join(
            f"{l}:{summary[run]['counts'][l]:4d}" for l in res["langs"]))

    (out_dir / "lape_summary.json").write_text(json.dumps(summary))

    # ---------------- report ----------------
    md = ["# Language-specific neurons (LAPE) across XScript checkpoints", "",
          f"top_rate={args.top_rate}, filter_rate={args.filter_rate}, "
          f"activation_bar_ratio={args.bar_ratio}; corpus: FLORES+ dev+devtest, "
          "counts exclude BOS/pad.", ""]

    # 1. final-checkpoint table
    md += ["## Language-specific neuron counts (per model)", "",
           "| model | budget | tok | " + " | ".join(LANG_ORDER) + " | total |",
           "|---|---|---|" + "---|" * (len(LANG_ORDER) + 1)]
    for run in runs:
        s = summary[run]
        m = s["meta"] or {}
        cts = [s["counts"].get(l, 0) for l in LANG_ORDER]
        md.append(f"| {run} | {m.get('budget','?')}B | {m.get('tokcond','?')} | "
                  + " | ".join(str(c) for c in cts) + f" | {sum(cts)} |")
    md.append("")

    # 2. layer profile aggregated over final (30B/23B/12B budget-max) models
    fams = defaultdict(list)
    for run in runs:
        m = summary[run]["meta"]
        if m:
            fams[(m["family"], m["tokcond"])].append((m["budget"], run))
    finals = [max(v)[1] for v in fams.values()]
    n_layers = len(next(iter(summary.values()))["layer_counts"]
                   [next(iter(summary[finals[0]]["langs"]))])
    md += ["## Layer profile (largest-budget checkpoint per family, all langs)", "",
           "| layer | " + " | ".join(LANG_ORDER) + " |",
           "|---|" + "---|" * len(LANG_ORDER)]
    prof = {l: np.zeros(n_layers) for l in LANG_ORDER}
    for run in finals:
        for l, cts in summary[run]["layer_counts"].items():
            prof[l] += np.array(cts)
    for ly in range(n_layers):
        md.append(f"| {ly} | " + " | ".join(
            f"{prof[l][ly]:.0f}" for l in LANG_ORDER) + " |")
    md.append("")

    # 3. trajectories across training tokens per family
    md += ["## Neuron counts across training tokens", ""]
    for (fam, tokc), lst in sorted(fams.items()):
        if len(lst) < 2:
            continue
        lst = sorted(lst)
        md += [f"### {fam} / {tokc}", "",
               "| budget | " + " | ".join(LANG_ORDER) +
               " | J(en) vs final | J(partner) vs final |",
               "|---|" + "---|" * (len(LANG_ORDER) + 2)]
        final_run = lst[-1][1]
        fsets = {l: neuron_key_set(
            {int(k): v for k, v in summary[final_run]["neuron_sets"].get(l, {}).items()})
            for l in LANG_ORDER}
        partner = [l for l in summary[final_run]["meta"]["langs"] if l != "en"]
        partner = partner[0] if partner else "en"
        for budget, run in lst:
            s = summary[run]
            sets = {l: neuron_key_set(
                {int(k): v for k, v in s["neuron_sets"].get(l, {}).items()})
                for l in LANG_ORDER}
            j_en = jaccard(sets["en"], fsets["en"])
            j_pa = jaccard(sets[partner], fsets[partner])
            md.append(f"| {budget}B | " + " | ".join(
                str(s["counts"].get(l, 0)) for l in LANG_ORDER)
                + f" | {j_en:.2f} | {j_pa:.2f} |")
        md.append("")

    (out_dir / "report.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out_dir}/lape_summary.json and report.md")


if __name__ == "__main__":
    main()
