#!/usr/bin/env python
"""Monolingual control for the logit-lens commitment / interior-accuracy effects
(CLAUDE.md 6l).

Question: is "the fair tokenizer makes the model resolve its answer earlier in
depth" a CROSS-LINGUAL phenomenon (about how a bilingual routes between its two
languages), or simply what a better tokenizer does to any language model?

The decisive cell is the ENGLISH MONOLINGUAL pair (`en-fair` vs `en-starved`)
repeating English words: one language, one script, no partner anywhere, so
nothing cross-lingual can contribute. Also reported: each partner-language
monolingual on its own language, and the bilinguals for reference.

    python analyze_lens_mono_control.py > results/logitlens/mono_control.txt

Pure CPU over the stored raw sidecars; no accelerator, no re-scoring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import analyze_logitlens as A                                       # noqa: E402

# (lang, fair run, starved run, note). The de pair is NOT LR-matched: `de-starved`
# has no cooled 30B final (6h stops at 16.1B mid-stable), so it is reported but
# never pooled. zh has no 30B monolingual, so its pair is the 15B/15B one.
MONO = [("fr", "fr-fair", "fr-starved", "both 30B cooled"),
        ("ar", "ar-fair", "ar-starved", "both 30B cooled"),
        ("zh", "zh-fair-15b", "zh-starved-15b", "both 15B mid-stable"),
        ("de", "de-fair", "de-starved-16b", "NOT LR-matched: 30B cooled vs 16B mid-stable")]
CLEAN = {"fr", "ar", "zh"}          # LR-matched monolingual pairs


def persist(mask: np.ndarray) -> np.ndarray:
    out = np.full(len(mask), mask.shape[1] - 1, float)
    for i in range(len(mask)):
        l = mask.shape[1] - 1
        while l > 0 and mask[i, l - 1]:
            l -= 1
        out[i] = l
    return out


def commit(rows) -> np.ndarray:
    P = np.exp([r["out"]["full_lp"] for r in rows])
    return persist(P >= .5 * P[:, -1:])


def paired(fr, sr, X, cond):
    ra, rb = A.load_rows(fr, X, cond), A.load_rows(sr, X, cond)
    ok = np.array([x["out"]["greedy_ok"] and y["out"]["greedy_ok"] for x, y in zip(ra, rb)])
    return A.boot_ci(commit(ra)[ok] - commit(rb)[ok]) + (int(ok.sum()),)


def main():
    print("# Monolingual control for the logit-lens depth effects (CLAUDE.md 6l)\n")
    print("All numbers are fair - starved, paired over identical items, 95% bootstrap CI.\n")

    print("## 1. Commitment layer, English monolinguals repeating ENGLISH words\n")
    print("`en-fair` and `en-starved` are trained on English only. Nothing here is cross-lingual.\n")
    print("| word list | n | d commit layer [95% CI] |")
    print("|---|---|---|")
    for X in A.PARTNERS:
        m, lo, hi, n = paired("en-fair", "en-starved", X, "rep_en")
        print(f"| en-{X} word set | {n} | {A.fmt(m, lo, hi, 2)} |")

    print("\n## 2. Commitment layer, partner-language monolinguals on their OWN language\n")
    print("| lang | pair | n | d commit layer (rep X→X) | d commit layer (rep en→en) | note |")
    print("|---|---|---|---|---|---|")
    for X, fr, sr, note in MONO:
        a = paired(fr, sr, X, "rep_X")
        b = paired(fr, sr, X, "rep_en")
        print(f"| {X} | {fr} vs {sr} | {a[3]} | {A.fmt(a[0], a[1], a[2], 2)} | {A.fmt(b[0], b[1], b[2], 2)} | {note} |")

    print("\n## 3. The bilinguals, same word lists, for reference\n")
    print("| pair | n | d commit layer (rep X→X) | d commit layer (rep en→en) |")
    print("|---|---|---|---|")
    for X in A.PARTNERS:
        a = paired(f"en-{X}-fair", f"en-{X}-starved", X, "rep_X")
        b = paired(f"en-{X}-fair", f"en-{X}-starved", X, "rep_en")
        print(f"| en-{X} | {a[3]} | {A.fmt(a[0], a[1], a[2], 2)} | {A.fmt(b[0], b[1], b[2], 2)} |")

    print("\n## 4. Interior factual accuracy (PolyFact, 800 facts -- the set the monolinguals were scored on)\n")
    A.FACT_TAG = "factual"
    models, runs = A.models_table(), A.runs_available()
    fact = A.analyse_factual(runs, models, [], {})
    print("| kind | lang | pair | d acc @L14 [95% CI] | d acc @output [95% CI] | d settle layer [CI] (n) |")
    print("|---|---|---|---|---|---|")
    rows = [("mono", X, fr, sr, note) for X, fr, sr, note in MONO] + \
           [("mono", "en", "en-fair", "en-starved", "both 30B cooled")] + \
           [("bi", X, f"en-{X}-fair", f"en-{X}-starved", "both 30B cooled") for X in A.PARTNERS]
    means = {"mono": [], "bi": []}
    for kind, X, fr, sr, note in rows:
        a, b = fact.get(fr, {}).get(X), fact.get(sr, {}).get(X)
        if not (a and b):
            continue
        ca, cb = a[f"correct_{X}"], b[f"correct_{X}"]
        d14 = A.boot_ci(ca[:, 14].astype(float) - cb[:, 14].astype(float))
        d16 = A.boot_ci(ca[:, 16].astype(float) - cb[:, 16].astype(float))
        sa, sb = a["settle"], b["settle"]
        ok = ~np.isnan(sa) & ~np.isnan(sb)
        ds = A.boot_ci(sa[ok] - sb[ok])
        print(f"| {kind} | {X} | {fr} vs {sr} | {A.fmt(*d14)} | {A.fmt(*d16)} | {A.fmt(*ds, 2)} ({ok.sum()}) |")
        if kind == "bi" or X in CLEAN or X == "en":
            means[kind].append(d14[0])
    print(f"\nMean d acc @L14: monolingual **{np.mean(means['mono']):+.3f}** (LR-matched pairs plus English) "
          f"vs bilingual **{np.mean(means['bi']):+.3f}**.")
    print("\n**Reading.** The effect is present, and of comparable size, in models that have no second "
          "language at all. It is therefore a property of what a better tokenizer does to a language "
          "model's depth profile, NOT evidence about cross-lingual representation alignment. Note also "
          "that a monolingual shows the effect only in the language it was trained on (rep en→en is ~0 "
          "for the fr/zh monolinguals), so it requires competence in the language, not merely a shared "
          "vocabulary.")


if __name__ == "__main__":
    main()
