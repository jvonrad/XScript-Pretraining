#!/usr/bin/env python
"""Aggregate `run_extra_bench.py`'s per-model JSONs into the tables the
five-language comparison actually needs.

Four sections, in the order the conclusions depend on each other:

  1. SCORING CHOICE. SIB-200 reports acc / acc_norm / acc_mutual_info (PMI),
     and they disagree. Prints all three side by side, plus the two baselines
     that matter -- uniform chance (1/7 = 0.143) and the MAJORITY-CLASS rate,
     which is 0.251 because SIB-200's topics are far from balanced. A number
     between those two is not evidence of topic understanding, and acc_norm in
     particular has a known failure mode here (dividing loglik by the answer's
     byte length systematically favours the longest label, and
     "science/technology" is both the longest label AND the majority class).

  1b. DEGENERACY CHECK. Whether a cell's 0/1 hit vector is EXACTLY the
     indicator of one class -- i.e. the model ranks the same label first for
     every document and "scores" the majority rate having learned nothing.
     This is the XNLI surface-form-competition failure (CLAUDE.md 6) in a form
     that accuracy alone cannot reveal.

  2. PER-MODEL x PER-LANGUAGE accuracy for every task.

  3. LABEL-LANGUAGE CONTROL for SIB-200: localized labels vs the English ones
     the dataset ships, same texts, paired bootstrap. This isolates "can the
     model classify this text" from "can it read English label words" -- the
     confound that would otherwise be baked into every cross-language
     comparison in section 2.

  4. TRANSFER DELTAS (bilingual - monolingual) per language, same paired
     percentile bootstrap estimator as `bootstrap_transfer.py`, over the
     matched-token checkpoint pairs. Cells whose mono/bilingual are not
     LR-state matched are flagged: see CLAUDE.md 6: the `*-15b` monolinguals
     are mid-stable snapshots at peak LR 3.0e-3 while the unsuffixed 30B
     bilinguals are cooled finals at 3.0e-4, which inflates the delta.

Pure stdlib.

    python analyze_extra_bench.py $WORK/results/extra_bench/
    python analyze_extra_bench.py $WORK/results/extra_bench/ --metric acc
"""
import argparse
import json
import random
from pathlib import Path

LANGS = ["en", "de", "fr", "ar", "zh"]
CODE = {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn",
        "ar": "arb_Arab", "zh": "zho_Hans"}
METRICS = ["acc", "acc_norm", "acc_mutual_info"]
B = 2000  # bootstrap replicates

# Chance and (for SIB-200) the far more relevant majority-class rate. SIB-200's
# label distribution over the merged 1004 docs is
# science/technology 252, travel 198, politics 146, sports 122, health 110,
# entertainment 93, geography 83 -- identical in every language, since the
# languages are parallel translations of the same FLORES sentences.
BASELINES = {
    "sib200": {"chance": 1 / 7, "majority": 252 / 1004},
    "hellaswag_zh": {"chance": 0.25, "majority": 0.25},
    # Taxi-1500 is skewed the same way: Recommendation 281, Faith 260,
    # Description 184, Sin 153, Grace 140, Violence 59, over 1077 verses.
    "taxi1500": {"chance": 1 / 6, "majority": 281 / 1077},
}

# Matched-token pairs. Two families, and the distinction is the whole ballgame
# (CLAUDE.md 6): `base_main.yaml` is WSD with decay starting at 24B, so a
# `*-12b`/`*-23b` checkpoint is a mid-STABLE snapshot at peak LR 3.0e-3 while an
# unsuffixed 30B model is a COOLED final at 3.0e-4.
#
#   LR-MATCHED   `{lang}-{tok}-12b` (~11.75B/lang) vs `en-{lang}-{tok}-23b`
#                (~22.76B total, ~11.4B/lang) -- both mid-stable. Quotable.
#   LR-MISMATCHED `{lang}-{tok}-15b` (~14.76B, mid-stable) vs the unsuffixed
#                30B bilingual (~15B/lang, COOLED). Hands the bilingual an
#                entire decay phase for free; inflated positive, NOT quotable.
#                Kept only so the size of that bias stays visible.
#
# de+starved is absent from both families: no de-starved monolingual exists at
# any budget (the run collapsed mid-training, CLAUDE.md 6).
TRANSFER_PAIRS = [
    # (partner, script, tok, mono, bilingual, lr_matched)
    ("de", "same-script",  "fair",    "de-fair-12b",    "en-de-fair-23b",    True),
    ("fr", "same-script",  "fair",    "fr-fair-12b",    "en-fr-fair-23b",    True),
    ("fr", "same-script",  "starved", "fr-starved-12b", "en-fr-starved-23b", True),
    ("ar", "cross-script", "fair",    "ar-fair-12b",    "en-ar-fair-23b",    True),
    ("ar", "cross-script", "starved", "ar-starved-12b", "en-ar-starved-23b", True),
    ("zh", "cross-script", "fair",    "zh-fair-12b",    "en-zh-fair-23b",    True),
    ("zh", "cross-script", "starved", "zh-starved-12b", "en-zh-starved-23b", True),
    ("de", "same-script",  "fair",    "de-fair-15b",    "en-de-fair",        False),
    ("fr", "same-script",  "fair",    "fr-fair-15b",    "en-fr-fair",        False),
    ("fr", "same-script",  "starved", "fr-starved-15b", "en-fr-starved",     False),
    ("ar", "cross-script", "fair",    "ar-fair-15b",    "en-ar-fair",        False),
    ("ar", "cross-script", "starved", "ar-starved-15b", "en-ar-starved",     False),
    ("zh", "cross-script", "fair",    "zh-fair-15b",    "en-zh-fair",        False),
    ("zh", "cross-script", "starved", "zh-starved-15b", "en-zh-starved",     False),
]
# English anchor must match the bilingual's ENGLISH share and its LR state, so
# it is chosen per family: `en-*-12b` (~11.75B, mid-stable) against the -23b
# bilinguals' ~11.4B English share -- a near-exact match, unlike the ~14.76B
# `en-*-15b` this script previously had to use for zh.
EN_ANCHOR = {True:  {"fair": "en-fair-12b",  "starved": "en-starved-12b"},
             False: {"fair": "en-fair-15b",  "starved": "en-starved-15b"}}
# No remaining anchor is a stand-in: every cell now has a same-LR English
# monolingual within ~3% of the bilingual's English token count.
APPROX_EN_ANCHOR: set[str] = set()


def load(path: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(path.glob("*_final.json")):
        d = json.loads(f.read_text())
        if "error" in d:
            print(f"# skipping {f.name}: {d['error']}")
            continue
        out[d["run"]] = d
    return out


def score(data: dict, lang: str, task: str, metric: str):
    return data.get("metrics", {}).get(lang, {}).get(task, {}).get(metric)


def hits(data: dict, lang: str, task: str, metric: str):
    return data.get("correct", {}).get(lang, {}).get(task, {}).get(metric)


def paired_bootstrap_delta(a, b, rng):
    """delta = mean(b) - mean(a) with B resampled deltas. One resample of doc
    indices is applied to both, which is valid because both models score the
    identical fixed doc order (and for SIB-200 that order is identical across
    LANGUAGES too -- docs are sorted by FLORES sentence id)."""
    n = min(len(a), len(b))
    point = sum(b[:n]) / n - sum(a[:n]) / n
    reps = [sum(b[i] for i in idx) / n - sum(a[i] for i in idx) / n
            for idx in ([rng.randrange(n) for _ in range(n)] for _ in range(B))]
    return point, reps


def ci95(reps):
    s = sorted(reps)
    return s[int(0.025 * len(s))], s[int(0.975 * len(s)) - 1]


def fmt(point, lo, hi):
    star = "*" if lo > 0 or hi < 0 else " "
    return f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"


def tasks_present(models: dict) -> dict[str, list[str]]:
    """{lang: [task, ...]} actually present across the loaded results."""
    found = {lang: [] for lang in LANGS}
    for d in models.values():
        for lang, ts in d.get("metrics", {}).items():
            for t in ts:
                if t not in found.setdefault(lang, []):
                    found[lang].append(t)
    return {k: sorted(v) for k, v in found.items() if v}


def section_scoring(models, present):
    print("\n" + "=" * 78)
    print("1. SCORING CHOICE -- the three metrics do not agree")
    print("=" * 78)
    print(f"{'':<22}{'':<22}" + "".join(f"{m:>18}" for m in METRICS))
    for lang, ts in present.items():
        for t in ts:
            vals = {m: [score(d, lang, t, m) for d in models.values()] for m in METRICS}
            vals = {m: [v for v in vs if v is not None] for m, vs in vals.items()}
            if not any(vals.values()):
                continue
            cell = "".join(
                (f"{sum(vals[m]) / len(vals[m]):>18.3f}" if vals.get(m) else f"{'-':>18}")
                for m in METRICS)
            print(f"{lang:<22}{t:<22}{cell}")
    print("\nmean over all loaded models; '-' = metric not reported by that task")
    print(f"baselines -- SIB-200: chance {BASELINES['sib200']['chance']:.3f}, "
          f"MAJORITY CLASS {BASELINES['sib200']['majority']:.3f} "
          "(science/technology, 252/1004); HellaSwag: chance 0.250")
    print("acc_norm divides loglik by the answer's byte length, which favours the")
    print("LONGEST option -- for SIB-200 that is 'science and technology', which is")
    print("also the majority class, so a high acc_norm can be pure length bias.")


def gold_labels() -> dict[str, list[int]]:
    """{'sib200': [...], 'taxi1500': [...]} -- the gold class per doc, in the
    same order the tasks emit (index_id / verse_id ascending). Fetched over the
    network; returns {} if unavailable, in which case the degeneracy check is
    skipped rather than guessed at.

    The labels are identical across the five languages -- both benchmarks are
    parallel translations of one sentence set -- so one vector serves all.
    """
    import csv
    import io
    import urllib.request

    def get(url):
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read().decode("utf-8")

    out = {}
    try:
        cats = ["science/technology", "travel", "politics", "sports", "health",
                "entertainment", "geography"]
        rows = []
        base = ("https://huggingface.co/datasets/Davlan/sib200/resolve/main/"
                "data/eng_Latn/{}.tsv")
        for split in ("train", "dev", "test"):
            rdr = csv.DictReader(io.StringIO(get(base.format(split))),
                                 delimiter="\t", quoting=csv.QUOTE_NONE)
            rows += [(int(r["index_id"]), cats.index(r["category"])) for r in rdr]
        out["sib200"] = [lab for _, lab in sorted(rows)]
    except Exception as exc:
        print(f"# sib200 gold labels unavailable ({type(exc).__name__}), "
              "degeneracy check skipped for it")
    try:
        cats = ["Recommendation", "Faith", "Description", "Sin", "Grace", "Violence"]
        rows = []
        base = ("https://raw.githubusercontent.com/cisnlp/Taxi1500/main/"
                "eng_data/eng_{}.tsv")
        for split in ("train", "dev", "test"):
            for line in get(base.format(split)).splitlines():
                p = line.split("\t")
                if len(p) >= 3:
                    rows.append((p[0].strip(), cats.index(p[1].strip())))
        out["taxi1500"] = [lab for _, lab in sorted(rows)]
    except Exception as exc:
        print(f"# taxi1500 gold labels unavailable ({type(exc).__name__}), "
              "degeneracy check skipped for it")
    return out


def section_degeneracy(models, present, metric, gold):
    """Flag models whose hit vector is EXACTLY the indicator of one class.

    A model that ranks the same label first for every document scores the
    majority-class rate (0.251 on SIB-200, 0.261 on Taxi-1500) while having
    learned nothing about topics -- the surface-form-competition failure
    CLAUDE.md 6 documents for XNLI, where weak models collapse onto the
    highest-prior option. Raw accuracy cannot distinguish that from real
    signal; this can, exactly, because a constant prediction makes the 0/1 hit
    vector identical to `gold == c` for that class.
    """
    if not gold:
        return
    print("\n" + "=" * 78)
    print(f"1b. DEGENERACY CHECK -- constant-prediction collapse  ({metric})")
    print("=" * 78)
    flagged = 0
    total = 0
    for run in sorted(models):
        for lang, ts in present.items():
            for t in ts:
                fam = ("sib200" if t.startswith("sib200")
                       else "taxi1500" if t.startswith("taxi1500") else None)
                if fam is None or fam not in gold:
                    continue
                h = hits(models[run], lang, t, metric)
                if not h or len(h) != len(gold[fam]):
                    continue
                total += 1
                for c in range(max(gold[fam]) + 1):
                    if h == [int(g == c) for g in gold[fam]]:
                        print(f"  COLLAPSED  {run:<22}{lang:<4}{t:<24}"
                              f"always predicts class {c} (acc = "
                              f"{sum(h) / len(h):.3f})")
                        flagged += 1
                        break
    print(f"\n{flagged} of {total} (model x lang x task) cells collapsed to a "
          "constant prediction.")
    if not flagged:
        print("No cell is a constant prediction -- every score reflects at least")
        print("some input-dependent ranking (which is NOT the same as being above")
        print("chance; read this together with the baselines above).")


def section_per_model(models, present, metric):
    print("\n" + "=" * 78)
    print(f"2. PER-MODEL x PER-LANGUAGE  ({metric})")
    print("=" * 78)
    cols = [(lang, t) for lang in LANGS if lang in present for t in present[lang]]
    head = "".join(f"{lang}:{t.split('_')[0] if not t.startswith('sib200_enlab') else 'sib200enlab'}"[:15].rjust(16)
                   for lang, t in cols)
    print(f"{'model':<22}{head}")
    for run in sorted(models):
        d = models[run]
        row = ""
        for lang, t in cols:
            v = score(d, lang, t, metric)
            row += (f"{v:>16.3f}" if v is not None else f"{'-':>16}")
        print(f"{run:<22}{row}")
    print("\nEvery model is scored on every language -- off-training-language")
    print("columns are a zero-shot cross-lingual transfer readout, not a bug.")


def section_label_control(models, metric, rng):
    pairs = [(lang, f"sib200_{CODE[lang]}", f"sib200_enlab_{CODE[lang]}")
             for lang in ["de", "fr", "ar", "zh"]]
    rows = []
    for run in sorted(models):
        d = models[run]
        for lang, loc, eng in pairs:
            h_loc, h_eng = hits(d, lang, loc, metric), hits(d, lang, eng, metric)
            if h_loc and h_eng:
                point, reps = paired_bootstrap_delta(h_eng, h_loc, rng)
                rows.append((run, lang, point, *ci95(reps)))
    if not rows:
        return
    print("\n" + "=" * 78)
    print(f"3. SIB-200 LABEL LANGUAGE: localized - English labels  ({metric})")
    print("=" * 78)
    print("Same texts, same 7 topics; only the prompt word and the label words")
    print("differ. Positive = localizing the labels helps. '*' = CI excludes 0.")
    print(f"{'model':<22}{'lang':<6}{'delta [95% CI]':<30}")
    for run, lang, point, lo, hi in rows:
        print(f"{run:<22}{lang:<6}{fmt(point, lo, hi):<30}")
    for lang in ["de", "fr", "ar", "zh"]:
        vals = [r[2] for r in rows if r[1] == lang]
        if vals:
            print(f"{'  mean':<22}{lang:<6}{sum(vals) / len(vals):+.3f}"
                  f"   (over {len(vals)} models)")


def section_transfer(models, present, metric, rng):
    print("\n" + "=" * 78)
    print(f"4. TRANSFER DELTAS: bilingual - monolingual  ({metric})")
    print("=" * 78)
    print(f"{'partner':<9}{'script':<14}{'tok':<9}{'LR':<5}{'task':<22}"
          f"{'delta on partner':<30}{'delta on English':<30}")
    for partner, script, tok, mono, bi, lr_ok in TRANSFER_PAIRS:
        if mono not in models or bi not in models:
            print(f"{partner:<9}{script:<14}{tok:<9}{'ok' if lr_ok else 'BAD':<5}"
                  f"{'-- missing ' + (mono if mono not in models else bi):<82}")
            continue
        anchor = EN_ANCHOR[lr_ok][tok]
        for t in present.get(partner, []):
            hm, hb = hits(models[mono], partner, t, metric), hits(models[bi], partner, t, metric)
            if not hm or not hb:
                continue
            p, reps = paired_bootstrap_delta(hm, hb, rng)
            cell_p = fmt(p, *ci95(reps))
            cell_e = ""
            en_task = t.replace(CODE[partner], CODE["en"])
            if anchor in models and en_task in present.get("en", []):
                ha = hits(models[anchor], "en", en_task, metric)
                hbe = hits(models[bi], "en", en_task, metric)
                if ha and hbe:
                    pe, repse = paired_bootstrap_delta(ha, hbe, rng)
                    cell_e = ("~" if partner in APPROX_EN_ANCHOR else "") \
                        + fmt(pe, *ci95(repse))
            print(f"{partner:<9}{script:<14}{tok:<9}{'ok' if lr_ok else 'BAD':<5}"
                  f"{t:<22}{cell_p:<30}{cell_e:<30}")
    print("\nLR column:")
    print("  ok  = mono `*-12b` vs bilingual `*-23b`, BOTH mid-stable @3e-3, and")
    print("        the English anchor is `en-*-12b` (~11.75B vs the bilingual's")
    print("        ~11.4B English share). These are the quotable rows.")
    print("  BAD = mono `*-15b` (mid-stable @3e-3) vs the COOLED 30B bilingual")
    print("        final (@3e-4), which hands the bilingual an entire decay phase")
    print("        for free (CLAUDE.md 6). Biased positive; shown ONLY so the size")
    print("        of that bias is visible next to the matched row above it.")
    print("Compare the two rows for the same (partner, tok) to read the bias off")
    print("directly -- on zh HellaSwag it is an 8-9x inflation.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--metric", default="acc", choices=METRICS,
                    help="metric for sections 2-4 (default: acc -- the raw "
                         "loglikelihood ranking, the only one with no "
                         "length-bias or prior correction baked in)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-network", action="store_true",
                    help="skip the degeneracy check, which fetches the gold "
                         "label vectors (a few hundred KB) to compare against")
    args = ap.parse_args()

    models = load(args.results_dir)
    if not models:
        raise SystemExit(f"no usable *_final.json in {args.results_dir}")
    present = tasks_present(models)
    print(f"# {len(models)} model(s) from {args.results_dir}")
    rng = random.Random(args.seed)

    section_scoring(models, present)
    section_degeneracy(models, present, args.metric,
                       {} if args.no_network else gold_labels())
    section_per_model(models, present, args.metric)
    section_label_control(models, args.metric, rng)
    section_transfer(models, present, args.metric, rng)


if __name__ == "__main__":
    main()
