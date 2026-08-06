#!/usr/bin/env python
"""PolyFact results over the cooled 30B finals: estimator choice, then tables.

    python analyze_polyfact.py [--results DIR]

Three sections:

  1. ESTIMATOR CHOICE, made the way CLAUDE.md 6g makes it -- on discrimination
     over the per-cell EMPIRICAL null and on prediction entropy, never on gold
     accuracy. acc_cal is absent by construction: PolyFact's four candidates
     are per-document entity names, so choice index c denotes nothing in
     common across documents and `rawscores` withholds it (6e).

  2. PER-CELL TABLE with acc, the empirical null, acc-null and entropy.
     Never accuracy alone -- 6f.

  3. CONTRASTS. Every model here is a COOLED 30B final at LR 3.0e-4, so they
     are mutually comparable and must not be paired against any mid-stable
     checkpoint. Two contrasts:

       bilingual - monolingual   MATCHED-TOTAL: the bilingual has seen ~15B of
                                 the language against the monolingual's 30B,
                                 so a negative delta is expected and the
                                 interpretable part is the spread ACROSS
                                 languages, not the level.
       fair - starved            within-language, same budget.

     de/starved and zh have no 30B monolingual (6h: de-starved stops at 16.1B
     uncooled; no zh monolingual was ever trained to 30B), so those partner
     cells are structurally absent rather than missing.

PolyFact is a KNOWLEDGE instrument. Per 6f it is a separate panel and is NOT
pooled into the cross-language capability aggregate: its language ordering
tracks Wikidata entity coverage as much as model capability.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

ESTS = ("acc", "acc_norm", "acc_pmi")
PARTNERS = [("de", "same"), ("fr", "same"), ("ar", "cross"), ("zh", "cross")]


def load(results: Path):
    from xscript.eval import rawscores as rs
    cells = {}
    for f in sorted(glob.glob(str(results / "extra_bench" / "raw" / "*_raw.json"))):
        run = os.path.basename(f)[:-len("_raw.json")]
        for lang, ts in (json.loads(Path(f).read_text()).get("raw") or {}).items():
            for task, r in ts.items():
                if not task.startswith("polyfact_") or "encue" in task:
                    continue
                v = rs.score_variants(r)
                cells[(run, lang)] = {e: rs.prediction_profile(r, e)
                                      for e in ESTS if e in v}
                cells[(run, lang)]["_acc_cal_withheld"] = "acc_cal" not in v
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path("/mnt/scratch/xscript_bench/results"))
    ap.add_argument("--est", default=None, help="force an estimator")
    a = ap.parse_args()

    cells = load(a.results)
    if not cells:
        sys.exit("no PolyFact cells found")
    print(f"{len(cells)} PolyFact cells (n=2039 each, 4-way, nominal chance .250)")
    print(f"acc_cal withheld everywhere: "
          f"{all(c['_acc_cal_withheld'] for c in cells.values())}  "
          f"(required -- per-document entity choices, 6e)\n")

    print("=" * 72)
    print("1. ESTIMATOR CHOICE (6g criteria: over-null and entropy, not accuracy)")
    print("=" * 72)
    print(f"{'estimator':<12}{'mean acc':>10}{'mean over-null':>16}{'mean entropy':>14}"
          f"{'min over-null':>15}")
    best, bestv = None, -9
    for e in ESTS:
        ov = [c[e]["acc_over_null"] for c in cells.values() if e in c]
        ac = [c[e]["acc"] for c in cells.values() if e in c]
        en = [c[e]["pred_entropy"] for c in cells.values() if e in c]
        print(f"  {e:<10}{sum(ac)/len(ac):>10.4f}{sum(ov)/len(ov):>16.4f}"
              f"{sum(en)/len(en):>14.3f}{min(ov):>15.4f}")
        if sum(ov) / len(ov) > bestv:
            best, bestv = e, sum(ov) / len(ov)
    est = a.est or best
    print(f"\n  -> using {est}")

    print("\n" + "=" * 72)
    print(f"2. PER-CELL ({est})")
    print("=" * 72)
    print(f"{'model':<18}{'lang':<6}{'acc':>8}{'null':>8}{'acc-null':>10}{'entropy':>9}")
    for (run, lang) in sorted(cells, key=lambda k: (k[1], k[0])):
        p = cells[(run, lang)][est]
        print(f"  {run:<16}{lang:<6}{p['acc']:>8.4f}{p['null']:>8.4f}"
              f"{p['acc_over_null']:>+10.4f}{p['pred_entropy']:>9.3f}")

    g = lambda r, l: cells.get((r, l), {}).get(est, {}).get("acc")

    print("\n" + "=" * 72)
    print("3a. BILINGUAL - MONOLINGUAL  (MATCHED-TOTAL: bilingual saw ~half the")
    print("    language; a negative level is EXPECTED, read the spread not the level)")
    print("=" * 72)
    print(f"{'partner':<10}{'tok':<10}{'on partner':>13}{'on English':>13}")
    for lang, script in PARTNERS:
        for tok in ("fair", "starved"):
            bi, mono, enm = f"en-{lang}-{tok}", f"{lang}-{tok}", f"en-{tok}"
            dp = (g(bi, lang) - g(mono, lang)) if g(bi, lang) is not None and g(mono, lang) is not None else None
            de = (g(bi, "en") - g(enm, "en")) if g(bi, "en") is not None and g(enm, "en") is not None else None
            f = lambda v: f"{v:+.4f}" if v is not None else "  n/a"
            note = "" if dp is not None else "   <- no 30B monolingual"
            print(f"  {lang} ({script[0]}){'':<2}{tok:<10}{f(dp):>13}{f(de):>13}{note}")

    print("\n" + "=" * 72)
    print("3b. FAIR - STARVED (within language, same budget)")
    print("=" * 72)
    print(f"{'model family':<20}{'lang':<6}{'fair':>9}{'starved':>10}{'fair-starved':>14}")
    fams = [("en-{}", "en"), ("ar-{}", "ar"), ("fr-{}", "fr"),
            ("en-ar-{}", "ar"), ("en-ar-{}", "en"), ("en-de-{}", "de"),
            ("en-de-{}", "en"), ("en-fr-{}", "fr"), ("en-fr-{}", "en"),
            ("en-zh-{}", "zh"), ("en-zh-{}", "en")]
    diffs = []
    for pat, lang in fams:
        fa, st = g(pat.format("fair"), lang), g(pat.format("starved"), lang)
        if fa is None or st is None:
            continue
        diffs.append((pat.format("*"), lang, fa - st))
        print(f"  {pat.format('*'):<18}{lang:<6}{fa:>9.4f}{st:>10.4f}{fa-st:>+14.4f}")
    if diffs:
        pos = sum(1 for _, _, d in diffs if d > 0)
        print(f"\n  fair > starved in {pos}/{len(diffs)} cells; "
              f"mean {sum(d for _,_,d in diffs)/len(diffs):+.4f}")
        for grp, name in ((lambda l: l == "en", "English"), (lambda l: l != "en", "non-English")):
            v = [d for _, l, d in diffs if grp(l)]
            if v:
                print(f"    {name:<12} mean {sum(v)/len(v):+.4f}  (n={len(v)})")


if __name__ == "__main__":
    main()
