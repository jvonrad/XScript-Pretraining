#!/usr/bin/env python
"""Capability BTS split by MEASURED LANGUAGE and by TOKENIZER CONDITION.

CLAUDE.md 6g reports capability BTS (bilingual - monolingual on benchmarks)
split by which language is measured, and separately reports a same-vs-cross
transfer gap -- but it never crosses the two with the tokenizer condition, so
"is the cross-script penalty worse under a starved tokenizer?" is not
answerable from its tables. This does that crossing.

    python bts_split.py [--families ...] [--ci]

Pairing is MATCHED-LANG: mono X B/lang vs bilingual 2X B total, so both sides
have seen the same amount of the measured language. Every checkpoint used is
mid-stable at peak LR 3.0e-3 (decay starts at 24B), so all four tiers are
LR-matched by construction -- the confound 6/6d call their biggest weakness
does not apply. The cooled 30B finals are deliberately excluded: 6g measures
them gaining 4-5x more headroom per token, so they are not on the same curve.

Estimator per family is 6g's, chosen there on candidate structure and
confirmed on over-null / entropy / monotonicity:

    arceasy, bmlama  -> acc        (short fixed phrases)
    hellaswag, story -> acc_norm   (long free-form continuations)
    sib200, xnli     -> acc_cal    (shared label set; prior calibration)

acc_tokennorm is never used: it is tokenizer-dependent and this project's
whole contrast is a tokenizer (6g proves it favours the more fragmented
candidate at 100% of disagreements).

Three data sources, because no single one covers every checkpoint:

  1. results/mubench_sweep/per_cell_table.md  -- the 100-checkpoint sweep,
     already on the estimators above.
  2. results/recalibrated/{extra_bench,appendix_c5}/  -- per-example hit
     lists for the 41 calibrated checkpoints, incl. the 12b monolinguals and
     23b bilinguals that the sweep table carries MuBench-only.
  3. --new-results (the 2026-08-05 sweep) -- the nine *-15b monolinguals and
     the seven de-starved-*, whose acc_cal is re-derived from raw sidecars
     via rawscores.score_variants (pure CPU, no accelerator).

--ci adds a paired bootstrap over documents wherever BOTH sides of a pair
have per-example hit lists. That is only the mono-12B/bi-23B tier on
sib200/xnli: the low-budget series' hit lists died with the box that produced
them, and MuBench hit lists were never committed. Point estimates elsewhere.
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

MUB = ["mub_arceasy", "mub_storycloze", "mub_hellaswag", "mub_bmlama"]
SHARED = ["sib200", "xnli"]
EST = {"mub_arceasy": "acc", "mub_bmlama": "acc",
       "mub_storycloze": "acc_norm", "mub_hellaswag": "acc_norm",
       "sib200": "acc_cal", "xnli": "acc_cal"}
CODE = {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn",
        "ar": "arb_Arab", "zh": "zho_Hans"}
TIERS = [(1, 2), (5, 10), (8, 15), (12, 23)]
PARTNERS = [("de", "same"), ("fr", "same"), ("ar", "cross"), ("zh", "cross")]


def task_name(fam: str, lang: str) -> str:
    if fam == "sib200":
        return f"sib200_{CODE[lang]}"
    if fam == "xnli":
        return f"xnli_{lang}"
    return f"{fam}_{lang}"


def load_sweep_table(cells):
    """Source 1: the committed 100-checkpoint per-cell table."""
    sec = None
    path = REPO / "results" / "mubench_sweep" / "per_cell_table.md"
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            m = re.match(r"#+\s*(\S+)", line)
            sec = m.group(1) if m and m.group(1) in EST else None
            continue
        if line.startswith("| ") and "---" not in line and sec:
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) >= 9 and c[1].isdigit():
                cells.setdefault((c[0], c[2], sec), float(c[4]))


def load_recalibrated(cells, hits):
    """Source 2: committed hit lists for the 41 calibrated checkpoints."""
    base = REPO / "results" / "recalibrated"
    for sub, key in (("extra_bench", "correct"), ("appendix_c5", "correct_calibrated")):
        d = base / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*_final.json")):
            run = f.name[:-len("_final.json")]
            blob = json.loads(f.read_text()).get(key) or {}
            for lang, ts in blob.items():
                for task, per_metric in ts.items():
                    fam = ("sib200" if task.startswith("sib200_") and "enlab" not in task
                           else "xnli" if task.startswith("xnli_") else None)
                    if fam is None or not isinstance(per_metric, dict):
                        continue
                    hl = per_metric.get("acc_cal")
                    if not isinstance(hl, list) or not hl:
                        continue
                    cells.setdefault((run, lang, fam), sum(hl) / len(hl))
                    hits.setdefault((run, lang, fam), hl)


def load_new(cells, hits, newdir: Path):
    """Source 3: the 2026-08-05 sweep -- MuBench direct, acc_cal from raw."""
    from xscript.eval import rawscores as rs
    for sub in ("extra_bench", "appendix_c5"):
        d = newdir / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*_final.json")):
            run = f.name[:-len("_final.json")]
            blob = json.loads(f.read_text())
            for lang, ts in (blob.get("metrics") or {}).items():
                for task, m in ts.items():
                    for fam in MUB:
                        if task == f"{fam}_{lang}":
                            cells[(run, lang, fam)] = m[EST[fam]]
            raw_f = d / "raw" / f"{run}_raw.json"
            if not raw_f.exists():
                continue
            for lang, ts in (json.loads(raw_f.read_text()).get("raw") or {}).items():
                for task, raw in ts.items():
                    fam = ("sib200" if task.startswith("sib200_") and "enlab" not in task
                           else "xnli" if task.startswith("xnli_") else None)
                    if fam is None:
                        continue
                    hl = rs.score_variants(raw).get("acc_cal")
                    if not hl:
                        continue
                    cells[(run, lang, fam)] = sum(hl) / len(hl)
                    hits[(run, lang, fam)] = hl


def delta(cells, bi, mono, lang, fams):
    ds = [(f, cells[(bi, lang, f)] - cells[(mono, lang, f)])
          for f in fams if (bi, lang, f) in cells and (mono, lang, f) in cells]
    if not ds:
        return None, []
    return sum(v for _, v in ds) / len(ds), [f for f, _ in ds]


def boot_ci(cells, hits, bi, mono, lang, fams, b=2000, seed=0):
    """Paired bootstrap over documents; needs hit lists on BOTH sides.

    Resamples doc indices once per replicate and applies the same resample to
    both models, which is valid because both scored the identical fixed doc
    order (same estimator as bootstrap_transfer.py).
    """
    usable = [f for f in fams
              if (bi, lang, f) in hits and (mono, lang, f) in hits
              and len(hits[(bi, lang, f)]) == len(hits[(mono, lang, f)])]
    if not usable:
        return None
    rng = random.Random(seed)
    reps = []
    for _ in range(b):
        per_fam = []
        for f in usable:
            hb, hm = hits[(bi, lang, f)], hits[(mono, lang, f)]
            n = len(hb)
            idx = [rng.randrange(n) for _ in range(n)]
            per_fam.append(sum(hb[i] - hm[i] for i in idx) / n)
        reps.append(sum(per_fam) / len(per_fam))
    reps.sort()
    return reps[int(0.025 * b)], reps[int(0.975 * b)], usable


def headline(cells, hits, fams, ci):
    """The two views that stand on their own, each internally LR-consistent.

    MATCHED-LANG, mono 12B / bi 23B -- the largest tier where both sides are
    mid-stable at peak LR. Equal exposure to the measured language, so a
    positive delta is transfer.

    MATCHED-TOTAL, 30B -- mono and bilingual are BOTH cooled finals, so the
    pair is LR-consistent even though it cannot be compared against the
    mid-stable tiers (6g: the cooldown yields 4-5x more headroom per token).
    The bilingual has seen HALF the measured language, so deltas are expected
    negative; what is interpretable is the spread across languages and the
    fair-vs-starved interaction, not the level.
    """
    VIEWS = [
        ("MATCHED-LANG  mono 12B / bilingual 23B  (both mid-stable @3.0e-3)",
         lambda l, t: (f"{l}-{t}-12b", f"en-{l}-{t}-23b", f"en-{t}-12b")),
        ("MATCHED-TOTAL  30B mono / 30B bilingual  (both COOLED @3.0e-4)",
         lambda l, t: (f"{l}-{t}", f"en-{l}-{t}", f"en-{t}")),
    ]
    for title, pair in VIEWS:
        print("=" * 78)
        print(title)
        print("=" * 78)
        res = {}
        for lang, script in PARTNERS:
            for tok in ("fair", "starved"):
                mono, bi, enmono = pair(lang, tok)
                dp, fp = delta(cells, bi, mono, lang, fams)
                de_, fe = delta(cells, bi, enmono, "en", fams)
                res[(lang, tok)] = (dp, de_, len(fp), len(fe), mono, bi)
        print(f"{'partner':<11}{'tok':<10}{'on partner':>13}{'on English':>13}"
              f"{'nfam':>6}   pair")
        for lang, script in PARTNERS:
            for tok in ("fair", "starved"):
                dp, de_, nfp, nfe, mono, bi = res[(lang, tok)]
                f = lambda v: f"{v:+.4f}" if v is not None else "   n/a"
                note = "" if dp is not None else "  <- no such monolingual"
                print(f"  {lang} ({script[0]}){'':<3}{tok:<10}{f(dp):>13}{f(de_):>13}"
                      f"{max(nfp, nfe):>6}   {mono} vs {bi}{note}")

        # Like-for-like only: a script group may enter a contrast only if BOTH
        # tokenizer conditions exist for it -- 6's defect 3, which removed
        # 65-76% of the reported interaction when fixed.
        print("\n  group means (only languages having BOTH tokenizer conditions):")
        elig = {s: [l for l, sc in PARTNERS if sc == s
                    and res[(l, "fair")][0] is not None
                    and res[(l, "starved")][0] is not None] for s in ("same", "cross")}
        print(f"    eligible: same={elig['same'] or '-'}  cross={elig['cross'] or '-'}")
        for idx, what in ((0, "partner"), (1, "English")):
            g = {}
            for tok in ("fair", "starved"):
                for s in ("same", "cross"):
                    v = [res[(l, tok)][idx] for l in elig[s] if res[(l, tok)][idx] is not None]
                    g[(tok, s)] = sum(v) / len(v) if v else None
            line = f"    on {what:<8}"
            for tok in ("fair", "starved"):
                if g[(tok, "same")] is not None and g[(tok, "cross")] is not None:
                    line += (f" | {tok}: same {g[(tok,'same')]:+.4f} cross "
                             f"{g[(tok,'cross')]:+.4f} gap {g[(tok,'same')]-g[(tok,'cross')]:+.4f}")
            print(line)
            if all(g[k] is not None for k in g):
                inter = ((g[("starved", "same")] - g[("starved", "cross")])
                         - (g[("fair", "same")] - g[("fair", "cross")]))
                print(f"    {'':<11}   INTERACTION gap(starved) - gap(fair) = {inter:+.4f}")

        if ci:
            print("\n  paired bootstrap 95% CI (SIB-200 + XNLI, where hit lists exist):")
            shared = [f for f in fams if f in SHARED]
            for lang, script in PARTNERS:
                for tok in ("fair", "starved"):
                    mono, bi, enmono = pair(lang, tok)
                    for mo, meas in ((mono, lang), (enmono, "en")):
                        r = boot_ci(cells, hits, bi, mo, meas, shared)
                        if r:
                            lo, hi, used = r
                            d, _ = delta(cells, bi, mo, meas, used)
                            star = "*" if lo > 0 or hi < 0 else " "
                            print(f"    {lang}/{tok:<8} on {meas:<3} "
                                  f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-results", type=Path,
                    default=Path("/mnt/scratch/xscript_bench/results"))
    ap.add_argument("--families", nargs="*", default=MUB + SHARED, choices=MUB + SHARED)
    ap.add_argument("--ci", action="store_true", help="paired bootstrap where hit lists exist")
    ap.add_argument("--headline", action="store_true",
                    help="only the two publishable views: matched-lang at the largest "
                         "LR-matched tier (mono 12B / bi 23B) and matched-total at 30B "
                         "(both sides cooled)")
    a = ap.parse_args()

    cells, hits = {}, {}
    load_new(cells, hits, a.new_results)       # most specific first
    load_recalibrated(cells, hits)
    load_sweep_table(cells)
    fams = a.families

    print(f"families: {', '.join(fams)}")
    print(f"estimators: " + ", ".join(f"{f}->{EST[f]}" for f in fams))
    print(f"{len(cells)} (model, lang, family) cells; {len(hits)} with hit lists\n")

    if a.headline:
        headline(cells, hits, fams, a.ci)
        return

    rows, cov = {}, {}
    for lang, _ in PARTNERS:
        for tok in ("fair", "starved"):
            for mb, bb in TIERS:
                bi = f"en-{lang}-{tok}-{bb}b"
                dp, fp = delta(cells, bi, f"{lang}-{tok}-{mb}b", lang, fams)
                de_, fe = delta(cells, bi, f"en-{tok}-{mb}b", "en", fams)
                rows[(lang, tok, mb)] = (dp, de_)
                cov[(lang, tok, mb)] = (len(fp), len(fe))

    def fmt(v):
        return f"{v:+.3f}" if v is not None else "  n/a "

    for idx, name in ((0, "PARTNER LANGUAGE"), (1, "ENGLISH")):
        print(f"=== delta on {name} (bilingual - monolingual) ===")
        hdr = "".join(f"{f'mono{m}B/bi{b}B':>14}" for m, b in TIERS)
        print(f"{'partner':<10}{'tok':<9}{hdr}{'mean':>10}{'nfam':>7}")
        for lang, script in PARTNERS:
            for tok in ("fair", "starved"):
                vs = [rows[(lang, tok, m)][idx] for m, _ in TIERS]
                ok = [v for v in vs if v is not None]
                nf = max(cov[(lang, tok, m)][idx] for m, _ in TIERS)
                mean = f"{sum(ok)/len(ok):+.4f}" if ok else "n/a"
                print(f"  {lang} ({script[0]}){'':<2}{tok:<9}"
                      + "".join(f"{fmt(v):>14}" for v in vs)
                      + f"{mean:>10}{nf:>7}")
        print()

    print("=== group means over tiers ===")
    for idx, name in ((0, "partner"), (1, "English")):
        print(f"\n  -- delta on {name} --")
        gaps = {}
        for tok in ("fair", "starved"):
            g = {}
            for script in ("same", "cross"):
                langs = [l for l, s in PARTNERS if s == script]
                v = [rows[(l, tok, m)][idx] for l in langs for m, _ in TIERS
                     if rows[(l, tok, m)][idx] is not None]
                g[script] = sum(v) / len(v) if v else None
                print(f"    {tok:<8} {script:<6} {g[script]:+.4f}  (n={len(v)})")
            if None not in g.values():
                gaps[tok] = g["same"] - g["cross"]
                print(f"    {tok:<8} {'GAP':<6} {gaps[tok]:+.4f}")
        if len(gaps) == 2:
            print(f"    -> starvation changes the gap by {gaps['starved']-gaps['fair']:+.4f}")
        print()
        for lang, script in PARTNERS:
            per = {}
            for tok in ("fair", "starved"):
                v = [rows[(lang, tok, m)][idx] for m, _ in TIERS
                     if rows[(lang, tok, m)][idx] is not None]
                per[tok] = sum(v) / len(v) if v else None
            s = f"    {lang} ({script}): fair {fmt(per['fair'])}  starved {fmt(per['starved'])}"
            if None not in per.values():
                s += f"   fair-starved {per['fair']-per['starved']:+.4f}"
                s += f"   both {(per['fair']+per['starved'])/2:+.4f}"
            print(s)

    if a.ci:
        print("\n=== paired bootstrap 95% CI (only where BOTH sides have hit lists) ===")
        shared = [f for f in fams if f in SHARED]
        for lang, script in PARTNERS:
            for tok in ("fair", "starved"):
                for mb, bb in TIERS:
                    bi = f"en-{lang}-{tok}-{bb}b"
                    for mono, meas in ((f"{lang}-{tok}-{mb}b", lang), (f"en-{tok}-{mb}b", "en")):
                        r = boot_ci(cells, hits, bi, mono, meas, shared)
                        if r:
                            lo, hi, used = r
                            d, _ = delta(cells, bi, mono, meas, used)
                            star = "*" if lo > 0 or hi < 0 else " "
                            print(f"  {lang}/{tok:<8} mono{mb}B/bi{bb}B  on {meas:<3} "
                                  f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}  ({','.join(used)})")


if __name__ == "__main__":
    main()
