#!/usr/bin/env python
"""Aggregate run_alignment.py's per-run JSONs into the tables that actually
answer the cross-script question, with paired bootstrap CIs.

The load-bearing idea: **a raw alignment score is not interpretable on its
own.** FLORES sentences leak surface cues across scripts (digits, Latin-script
named entities, punctuation, length), so a model that never saw Arabic still
retrieves EN-AR translations well above 1/n. What carries evidence is

    delta = alignment_bilingual(en, X) - alignment_monolingual(en, X)

where the monolingual control is scored on the *same* pair it never trained
on. Both available controls are reported -- the EN-only model and the X-only
model -- because either alone is arguable and agreement between them is the
real signal. This mirrors bootstrap_transfer.py's estimator for the downstream
benchmarks, on the same matched-token checkpoints, so the representation-side
and behaviour-side numbers are directly comparable.

Metric bootstrapped: bidirectional top-1 retrieval accuracy (a2b and b2a hits
pooled) at the fixed `ref` layer, in the `centered` variant by default.

  - `ref` layer, not each model's argmax layer: the argmax is selected ON the
    metric, which inflates it and differs per model, so cross-model deltas
    would carry a selection bias. `--layer best` is available for the
    descriptive view.
  - `centered`: the per-language centroid dominates raw cross-lingual spaces
    and can hide the translation signal underneath it (the same failure mode
    that made XNLI ar/zh look like chance before debiasing). `--variant raw`
    for the uncorrected numbers.

Bootstrap caveat: resampling is over QUERIES, holding the 997-sentence
candidate pool fixed. The CI covers "which sentences did we ask about", not
"how hard was the pool" -- the pool is identical across every model compared,
so paired deltas are unaffected, but an absolute accuracy's CI is mildly
optimistic.

    python analyze_alignment.py $WORK/results/alignment/
"""
import argparse
import json
import random
import statistics
from itertools import combinations
from pathlib import Path

LANG_ORDER = ["en", "de", "fr", "ar", "zh"]
SCRIPT = {"en": "Latn", "de": "Latn", "fr": "Latn", "ar": "Arab", "zh": "Hans"}
B = 2000  # bootstrap replicates

# Matched-token checkpoints, identical to bootstrap_transfer.py's tables: mono
# runs cut to the same per-language token budget as their bilingual counterpart,
# so a delta is not just "the mono model saw 2x more of this language".
BI_MATCHED = {
    "de": {"fair": "en-de-fair", "starved": "en-de-starved"},
    "fr": {"fair": "en-fr-fair", "starved": "en-fr-starved"},
    "ar": {"fair": "en-ar-fair", "starved": "en-ar-starved"},
    "zh": {"fair": "en-zh-fair-23b", "starved": "en-zh-starved-23b"},
}
MONO_MATCHED = {
    "de": {"fair": "de-fair-15b", "starved": None},   # no de-starved-15b uploaded
    "fr": {"fair": "fr-fair-15b", "starved": "fr-starved-15b"},
    "ar": {"fair": "ar-fair-15b", "starved": "ar-starved-15b"},
    "zh": {"fair": "zh-fair-12b", "starved": "zh-starved-12b"},
}
EN_MATCHED = {"fair": "en-fair-15b", "starved": "en-starved-15b"}
EN_APPROX = {"zh"}  # en-*-15b is ~14.76B vs en-zh-*-23b's ~11.4B English share

# Only used with --allow-unmatched: the original 30B monolinguals. These saw ~2x
# the per-language tokens of the bilinguals they'd be compared against, so any
# delta is confounded by token dilution (CLAUDE.md's "token-dilution confound").
# Rows resolved this way are marked `!`.
BI_FALLBACK = {"zh": {"fair": "en-zh-fair", "starved": "en-zh-starved"}}
MONO_FALLBACK = {
    "de": {"fair": "de-fair", "starved": None},   # no de-starved exists at all
    "fr": {"fair": "fr-fair", "starved": "fr-starved"},
    "ar": {"fair": "ar-fair", "starved": "ar-starved"},
    "zh": {"fair": None, "starved": None},        # no 30B Chinese mono was trained
}
EN_FALLBACK = {"fair": "en-fair", "starved": "en-starved"}
PARTNERS = [("de", "same-script"), ("fr", "same-script"),
            ("ar", "cross-script"), ("zh", "cross-script")]


def load(path: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(path.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"[warn] unreadable: {f}")
    # Every model must have been scored on the SAME sentence pool: the paired
    # bootstrap resamples one set of doc indices and applies it to both models,
    # which is only valid if they scored an identical, identically-ordered set.
    # A short run (e.g. a stray `--limit` smoke/warm-up result left in the
    # directory) would otherwise be silently compared against full ones.
    counts = {n for d in out.values() if (n := d.get("n_sentences"))}
    if len(counts) > 1:
        keep = max(counts)
        odd = {k: d["n_sentences"] for k, d in out.items()
               if d.get("n_sentences") != keep}
        print(f"[warn] inconsistent sentence pools {sorted(counts)}; dropping "
              f"short runs (rerun them without --limit): {odd}")
        out = {k: d for k, d in out.items() if d.get("n_sentences") == keep}
    return out


def pair_key(a: str, b: str) -> str:
    i, j = LANG_ORDER.index(a), LANG_ORDER.index(b)
    return f"{a}-{b}" if i < j else f"{b}-{a}"


def cell(data: dict, a: str, b: str, variant: str, layer: str) -> dict | None:
    p = (data.get("pairs") or {}).get(pair_key(a, b))
    return p[variant][layer] if p and variant in p else None


def hits(data: dict, a: str, b: str, variant: str, layer: str) -> list[int] | None:
    """Per-query 0/1 hits, both directions concatenated per query index."""
    c = cell(data, a, b, variant, layer)
    if not c or "hits" not in c:
        return None
    h = c["hits"]
    return [h["a2b"], h["b2a"]]


def _acc(pair_hits: list[list[int]], idx: list[int]) -> float:
    a2b, b2a = pair_hits
    return sum(a2b[i] + b2a[i] for i in idx) / (2 * len(idx))


def paired_delta(bi: list[list[int]], mono: list[list[int]], rng: random.Random):
    """Point estimate and bootstrap replicates of bi - mono.

    Both models scored the identical, identically-ordered FLORES sentences, so
    one resample of query indices applied to both is a valid paired bootstrap.
    """
    n = len(bi[0])
    base = list(range(n))
    point = _acc(bi, base) - _acc(mono, base)
    reps = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        reps.append(_acc(bi, idx) - _acc(mono, idx))
    return point, reps


def ci95(reps: list[float]) -> tuple[float, float]:
    s = sorted(reps)
    return s[int(0.025 * len(s))], s[min(len(s) - 1, int(0.975 * len(s)))]


def fmt(pt: float, lo: float, hi: float, approx: bool = False) -> str:
    sig = "**" if (lo > 0) == (hi > 0) else ""
    return f"{sig}{'~' if approx else ''}{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}"


# --------------------------------------------------------------------------

def resolve(partner: str, tok: str, models: dict, allow_unmatched: bool):
    """Pick the (bilingual, partner-mono, en-mono) triple for this cell.

    Returns the matched-token names when present, else -- only with
    ``allow_unmatched`` -- the original 30B checkpoints, flagged `!` because
    their per-language token budgets do not match (token-dilution confound).
    """
    bi, mp, me, flag = (BI_MATCHED[partner][tok], MONO_MATCHED[partner][tok],
                        EN_MATCHED[tok], "")
    if allow_unmatched:
        if bi not in models and BI_FALLBACK.get(partner, {}).get(tok) in models:
            bi, flag = BI_FALLBACK[partner][tok], "!"
        if mp not in models and MONO_FALLBACK[partner][tok] in models:
            mp, flag = MONO_FALLBACK[partner][tok], "!"
        if me not in models and EN_FALLBACK[tok] in models:
            me, flag = EN_FALLBACK[tok], "!"
    return bi, mp, me, flag


def table_raw(models: dict, variant: str, layer: str) -> None:
    """Every model x every language pair -- the full matrix, no baselining."""
    pairs = [f"{a}-{b}" for a, b in combinations(LANG_ORDER, 2)]
    print(f"\n## Bidirectional top-1 retrieval, {variant} / {layer} layer "
          f"(all models x all pairs)\n")
    print("Cells a model was TRAINED on both languages of are marked `*`. "
          "Everything else is a zero-shot / control cell. The `LEXICAL FLOOR` "
          "row is model-free TF-IDF token overlap -- any cell at or below it "
          "shows no representational signal.\n")
    print("| model | " + " | ".join(pairs) + " |")
    print("|---" * (len(pairs) + 1) + "|")
    # The floor depends on the tokenizer (not the model), so one row per
    # tokenizer present in the results.
    floors = {d.get("tok"): d["lexical_baseline"] for d in models.values()
              if d.get("lexical_baseline")}
    for tname, lex in sorted(floors.items(), key=lambda kv: str(kv[0])):
        print(f"| **LEXICAL FLOOR** ({tname}) | " + " | ".join(
            f"{(lex[p]['top1_a2b'] + lex[p]['top1_b2a']) / 2:.3f}"
            if p in lex else "-" for p in pairs) + " |")
    for name, d in sorted(models.items()):
        row = [name]
        for p in pairs:
            a, b = p.split("-")
            c = cell(d, a, b, variant, layer)
            if not c:
                row.append("-")
                continue
            acc = (c["top1_a2b"] + c["top1_b2a"]) / 2
            trained = (d.get("pairs") or {})[p].get("trained_pair")
            row.append(f"{acc:.3f}{'*' if trained else ''}")
        print("| " + " | ".join(row) + " |")


def table_cka(models: dict, variant: str, layer: str) -> None:
    pairs = [f"{a}-{b}" for a, b in combinations(LANG_ORDER, 2)]
    print(f"\n## Linear CKA, {variant} / {layer} layer\n")
    print("Retrieval-free representation similarity; insensitive to the hubness "
          "artifacts nearest-neighbour retrieval can suffer from.\n")
    print("| model | " + " | ".join(pairs) + " |")
    print("|---" * (len(pairs) + 1) + "|")
    for name, d in sorted(models.items()):
        row = [name]
        for p in pairs:
            a, b = p.split("-")
            c = cell(d, a, b, variant, layer)
            row.append(f"{c['cka']:.3f}" if c else "-")
        print("| " + " | ".join(row) + " |")


def table_deltas(models: dict, variant: str, layer: str, seed: int,
                 allow_unmatched: bool = False) -> None:
    print(f"\n## Bilingual - monolingual alignment on the EN-partner pair "
          f"({variant} / {layer} layer)\n")
    print("Both monolingual controls are shown. `**bold**` = 95% CI excludes 0. "
          "`~` = the EN control's token budget only approximately matches "
          "(see EN_APPROX). `!` = token budgets NOT matched at all "
          "(--allow-unmatched fallback; delta is confounded by dilution).\n")
    print("`SAT` = a monolingual CONTROL already scores >0.90 on this pair, so "
          "the metric is saturated and the delta measures headroom, not "
          "transfer -- read those rows as 'no room left', never as 'no "
          "transfer'. `lex` is the model-free TF-IDF floor.\n")
    print("| partner | script | tok | bilingual | lex | vs EN-only mono | vs partner-only mono |")
    print("|---|---|---|---|---|---|---|")
    for partner, script in PARTNERS:
        for tok in ("fair", "starved"):
            bi_name, mono_p, mono_en, flag = resolve(partner, tok, models,
                                                     allow_unmatched)
            bi = models.get(bi_name)
            if not bi:
                continue
            bh = hits(bi, "en", partner, variant, layer)
            if not bh:
                continue
            base_acc = _acc(bh, list(range(len(bh[0]))))
            cols, ctrl_accs = [], []
            for ctrl, approx in ((mono_en, partner in EN_APPROX), (mono_p, False)):
                cd = models.get(ctrl) if ctrl else None
                ch = hits(cd, "en", partner, variant, layer) if cd else None
                if not ch:
                    cols.append("n/a")
                    continue
                ctrl_accs.append(_acc(ch, list(range(len(ch[0])))))
                pt, reps = paired_delta(bh, ch, random.Random(seed))
                cols.append(fmt(pt, *ci95(reps), approx=approx) + flag)
            sat = " `SAT`" if max(ctrl_accs, default=0) > 0.9 else ""
            lex = (bi.get("lexical_baseline") or {}).get(pair_key("en", partner))
            lex_s = (f"{(lex['top1_a2b'] + lex['top1_b2a']) / 2:.3f}"
                     if lex else "-")
            print(f"| {partner} | {script} | {tok} | {base_acc:.3f}{sat} | {lex_s} | "
                  f"{cols[0]} | {cols[1]} |")


def table_script_contrast(models: dict, variant: str, layer: str) -> None:
    """Same-script vs cross-script, averaged over the trained bilinguals."""
    print(f"\n## Same-script vs cross-script ({variant} / {layer} layer)\n")
    print("| script class | partners | mean bilingual acc | mean control acc | mean gap |")
    print("|---|---|---|---|---|")
    buckets: dict[str, list[tuple[float, float]]] = {}
    for partner, script in PARTNERS:
        for tok in ("fair", "starved"):
            bi = models.get(BI_MATCHED[partner][tok])
            if not bi:
                continue
            bh = hits(bi, "en", partner, variant, layer)
            if not bh:
                continue
            ctrls = [models.get(n) for n in (EN_MATCHED[tok], MONO_MATCHED[partner][tok]) if n]
            ch = [hits(c, "en", partner, variant, layer) for c in ctrls if c]
            ch = [h for h in ch if h]
            if not ch:
                continue
            idx = list(range(len(bh[0])))
            buckets.setdefault(script, []).append(
                (_acc(bh, idx), statistics.fmean(_acc(h, idx) for h in ch)))
    for script in ("same-script", "cross-script"):
        vals = buckets.get(script)
        if not vals:
            continue
        parts = ", ".join(p for p, s in PARTNERS if s == script)
        bi_m = statistics.fmean(v[0] for v in vals)
        ct_m = statistics.fmean(v[1] for v in vals)
        print(f"| {script} | {parts} | {bi_m:.3f} | {ct_m:.3f} | {bi_m - ct_m:+.3f} |")


def table_layers(models: dict, variant: str) -> None:
    """Where in the stack alignment peaks -- sanity check on the ref layer."""
    print(f"\n## Peak-alignment layer ({variant})\n")
    print("| model | pair | n_layers | ref | best | acc@ref | acc@best |")
    print("|---|---|---|---|---|---|---|")
    for name, d in sorted(models.items()):
        for p, v in (d.get("pairs") or {}).items():
            if not v.get("trained_pair"):
                continue
            var = v[variant]
            r, b = var["ref"], var["best"]
            print(f"| {name} | {p} | {d.get('n_layers', '?')} | "
                  f"{var['ref_layer']} | {var['best_layer']} | "
                  f"{(r['top1_a2b'] + r['top1_b2a']) / 2:.3f} | "
                  f"{(b['top1_a2b'] + b['top1_b2a']) / 2:.3f} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--variant", default="centered", choices=["centered", "raw"])
    ap.add_argument("--layer", default="ref", choices=["ref", "best"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-unmatched", action="store_true",
                    help="fall back to the original 30B checkpoints where no "
                         "matched-token one exists; those deltas are confounded "
                         "by token dilution and are flagged `!`")
    args = ap.parse_args()

    models = load(args.results_dir)
    if not models:
        raise SystemExit(f"no *.json under {args.results_dir}")
    n = next((d.get("n_sentences") for d in models.values() if d.get("n_sentences")), "?")
    print(f"# Cross-lingual representation alignment\n")
    print(f"{len(models)} models, FLORES+ n={n}, variant={args.variant}, "
          f"layer={args.layer}, B={B} bootstrap replicates.")

    table_deltas(models, args.variant, args.layer, args.seed, args.allow_unmatched)
    table_script_contrast(models, args.variant, args.layer)
    table_raw(models, args.variant, args.layer)
    table_cka(models, args.variant, args.layer)
    table_layers(models, args.variant)


if __name__ == "__main__":
    main()
