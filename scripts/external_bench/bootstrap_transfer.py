#!/usr/bin/env python
"""Bootstrap confidence intervals on the same-script vs. cross-script
transfer deltas discussed in CLAUDE.md's "Same-script vs. cross-script
transfer" section.

Consumes run_appendix_c5.py's per-run JSONs (needs the `correct` field --
per-example 0/1 hit lists -- which requires that script's log_samples=True
path). For each (partner language, tokenizer) pair, computes:

    delta_X  = bilingual_score(partner_lang) - monolingual_score(partner_lang)
    delta_en = bilingual_score(en)           - monolingual_score(en)

per benchmark, via a PAIRED percentile bootstrap (both models are scored on
the identical, identically-ordered fixed doc subset, so resampling doc
indices once and applying the same resample to both models is valid and more
efficient than an unpaired bootstrap). Also reports an aggregate delta
across every benchmark available for that language, bootstrapped per-family
and averaged per replicate (a simple stratified bootstrap over benchmarks).

Uses the MATCHED-TOKEN checkpoints (mono runs cut down to the same
per-language token budget as their bilingual counterpart -- see CLAUDE.md's
"token-dilution confound" note) instead of the original mismatched-token 30B
monolingual baselines:

    de/fr/ar partner: {lang}-{tok}-15b (~14.75B tokens) vs the existing 30B
        en-{lang}-{tok} bilingual (~15B tokens/language) -- a near-exact match.
    zh partner: zh-{tok}-12b (~11.75B tokens) vs the new en-zh-{tok}-23b
        bilingual (~11.4B tokens/language) -- also a near-exact match, and
        the first Chinese monolingual baseline to exist at all.
    English anchor: en-{tok}-15b (~14.76B tokens) for every pair. Exact match
        for the de/fr/ar bilinguals (~15B tokens/language); an approximation
        for zh (bilingual's English share is ~11.4B, ~30% less) since no
        ~11.4B English mono checkpoint was uploaded -- flagged in the output.
    de+starved has no matched mono baseline (no de-starved-15b was uploaded,
        same gap as the original analysis) -- that cell is skipped.

Pure stdlib -- no numpy/scipy needed.

    python bootstrap_transfer.py results/appendix_c5/   # dir of <model>_final.json
"""
import json
import random
import sys
from pathlib import Path

TASK_KEY = {
    "xnli": {l: f"xnli_{l}" for l in ["en", "de", "fr", "ar", "zh"]},
    "belebele": {"en": "belebele_cloze_eng_Latn", "de": "belebele_cloze_deu_Latn",
                "fr": "belebele_cloze_fra_Latn", "ar": "belebele_cloze_arb_Arab",
                "zh": "belebele_cloze_zho_Hans"},
    "arc": {"en": "arc_easy", "de": "arc_de", "fr": "arc_fr", "ar": "arc_ar", "zh": "arc_zh"},
    "hellaswag": {"en": "hellaswag", "de": "hellaswag_de", "fr": "hellaswag_fr", "ar": "hellaswag_ar"},
    "xstorycloze": {"en": "xstorycloze_en", "ar": "xstorycloze_ar", "zh": "xstorycloze_zh"},
    "xwinograd": {"en": "xwinograd_en", "fr": "xwinograd_fr", "zh": "xwinograd_zh"},
}
PAIRS = [("de", "same-script"), ("fr", "same-script"), ("ar", "cross-script"), ("zh", "cross-script")]
B = 2000  # bootstrap replicates

# Matched-token model names: {lang: {tok: model_name or None}}
MONO_MATCHED = {
    "de": {"fair": "de-fair-15b", "starved": None},   # no de-starved-15b uploaded
    "fr": {"fair": "fr-fair-15b", "starved": "fr-starved-15b"},
    "ar": {"fair": "ar-fair-15b", "starved": "ar-starved-15b"},
    "zh": {"fair": "zh-fair-12b", "starved": "zh-starved-12b"},
}
BI_MATCHED = {
    "de": {"fair": "en-de-fair", "starved": "en-de-starved"},
    "fr": {"fair": "en-fr-fair", "starved": "en-fr-starved"},
    "ar": {"fair": "en-ar-fair", "starved": "en-ar-starved"},
    "zh": {"fair": "en-zh-fair-23b", "starved": "en-zh-starved-23b"},
}
EN_MATCHED = {lang: {"fair": "en-fair-15b", "starved": "en-starved-15b"}
              for lang in ["de", "fr", "ar", "zh"]}
# zh's English anchor is an approximation: en-*-15b is ~14.76B tokens, but
# en-zh-*-23b's English share is only ~11.4B -- no exact-match checkpoint
# exists. Flagged inline in the output.
EN_APPROX = {"zh"}


def load(path: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(path.glob("*_final.json")):
        d = json.loads(f.read_text())
        if "error" not in d:
            out[d["run"]] = d
    return out


def hits(model_data: dict, lang: str, fam: str) -> list[int] | None:
    key = TASK_KEY[fam].get(lang)
    if key is None:
        return None
    return model_data.get("correct", {}).get(lang, {}).get(key)


def paired_bootstrap_delta(a: list[int], b: list[int], rng: random.Random) -> tuple[float, list[float]]:
    """delta = mean(b) - mean(a), plus B resampled deltas (same resampled
    indices applied to both -- valid since a/b are the identical fixed doc
    order)."""
    n = min(len(a), len(b))
    point = sum(b[:n]) / n - sum(a[:n]) / n
    reps = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        reps.append(sum(b[i] for i in idx) / n - sum(a[i] for i in idx) / n)
    return point, reps


def ci95(reps: list[float]) -> tuple[float, float]:
    s = sorted(reps)
    lo = s[int(0.025 * len(s))]
    hi = s[int(0.975 * len(s)) - 1]
    return lo, hi


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    data = load(Path(sys.argv[1]))
    if not data:
        sys.exit("no scores found (need run_appendix_c5.py output with a `correct` field)")

    rng = random.Random(0)
    print("## Transfer-delta confidence intervals, matched-token checkpoints "
         f"(paired bootstrap, B={B}, 95% CI)\n")
    print("Mono baselines are cut to the same per-language token budget as "
         "their bilingual counterpart (see script docstring). zh's English "
         "anchor (`en-*-15b`, ~14.76B tokens) is an approximation of the "
         "bilingual's ~11.4B-token English share -- marked with `~`.\n")
    print("| partner | script | tok | benchmark | Δ on partner-lang [95% CI] | Δ on English [95% CI] |")
    print("|---|---|---|---|---|---|")

    for lang, script in PAIRS:
        for tok in ["fair", "starved"]:
            mono_x_name = MONO_MATCHED[lang][tok]
            bi_name = BI_MATCHED[lang][tok]
            en_name = EN_MATCHED[lang][tok]
            mono_x = data.get(mono_x_name) if mono_x_name else None
            bi = data.get(bi_name)
            mono_en = data.get(en_name)
            en_flag = "~" if lang in EN_APPROX else ""
            if bi is None:
                continue
            agg_x_reps, agg_x_point = [], []
            agg_en_reps, agg_en_point = [], []
            for fam in TASK_KEY:
                if lang not in TASK_KEY[fam]:
                    continue
                x_cell = en_cell = "-"
                if mono_x is not None:
                    ax, bx = hits(mono_x, lang, fam), hits(bi, lang, fam)
                    if ax and bx:
                        pt, reps = paired_bootstrap_delta(ax, bx, rng)
                        lo, hi = ci95(reps)
                        x_cell = f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                        agg_x_point.append(pt); agg_x_reps.append(reps)
                if mono_en is not None:
                    ae, be = hits(mono_en, "en", fam), hits(bi, "en", fam)
                    if ae and be:
                        pt, reps = paired_bootstrap_delta(ae, be, rng)
                        lo, hi = ci95(reps)
                        en_cell = f"{en_flag}{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                        agg_en_point.append(pt); agg_en_reps.append(reps)
                print(f"| {lang} | {script} | {tok} | {fam} | {x_cell} | {en_cell} |")

            if agg_x_point:
                pt = sum(agg_x_point) / len(agg_x_point)
                agg_reps = [sum(r[b] for r in agg_x_reps) / len(agg_x_reps) for b in range(B)]
                lo, hi = ci95(agg_reps)
                print(f"| {lang} | {script} | {tok} | **mean ({len(agg_x_point)} benchmarks)** "
                     f"| **{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]** | |")
            if agg_en_point:
                pt = sum(agg_en_point) / len(agg_en_point)
                agg_reps = [sum(r[b] for r in agg_en_reps) / len(agg_en_reps) for b in range(B)]
                lo, hi = ci95(agg_reps)
                print(f"| {lang} | {script} | {tok} | **mean ({len(agg_en_point)} benchmarks)** "
                     f"| | **{en_flag}{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]** |")
            print("|   |   |   |   |   |   |")


if __name__ == "__main__":
    main()
