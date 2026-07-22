#!/usr/bin/env python
"""Matched-token BTS with paired bootstrap CIs, from run_bpb.py's output.

    BTS(p) = (BPB_mono(p) - BPB_bi(p)) / BPB_mono(p)     (+ve = bilingual helps)
    penalty(C) = mean BTS(same-script) - mean BTS(cross-script)  under tokenizer C
    interaction = penalty(starved) - penalty(destarved)

Four things this fixes relative to the committed `results/bts/*` (see
CLAUDE.md's BPB->BTS section):

1. TOKEN MATCHING IS REAL, NOT INTERPOLATED. The old code chose between
   `matched_total` (mono-final vs bi-final -- confounded: the mono saw ~2x the
   partner-language tokens) and `matched_lang` (mono at the checkpoint
   *nearest* total*mix_prob). Those disagreed in sign on the headline penalty
   in every cell. Here each bilingual is paired with a monolingual actually
   trained to the same per-language budget, so there is one number, not two.
   `matched_lang` also degenerated silently when a run had no checkpoint near
   the mark -- for zh it returned the final checkpoint, making its
   "matched_lang" BTS identical to its `matched_total` BTS while every other
   partner shifted by +0.04..+0.10.

2. ONE BUDGET FOR ALL PARTNERS. A same-script vs cross-script penalty is only
   like-for-like if every partner is compared at the same token budget. The
   "11.4B" budget below is the only one covering all four partners; the "15B"
   budget cannot include zh (no zh-*-15b monolingual exists) and is reported
   as a robustness check on de/fr/ar only.

3. LIKE-FOR-LIKE PARTNER SETS ACROSS TOKENIZERS. `penalty(C)` used to be
   averaged over whatever partners had data in condition C, so with
   `de-starved` missing the interaction subtracted a {de,fr} mean from a {fr}
   mean. Here the penalty/interaction use only partners present in BOTH
   tokenizer conditions, and the set is printed.

4. CONFIDENCE INTERVALS. BPB is a ratio of sums, so each replicate resamples
   FLORES sentence indices and recomputes sum(nll)/sum(bytes). One index draw
   is shared by every (model, language) in a replicate: FLORES is parallel and
   run_bpb.py scores the same aligned id list for every model, so sentence i is
   the same content everywhere. Per-partner BTS, penalties and the interaction
   are therefore all paired estimates from a single joint resample.

Pure stdlib.

    python bts_matched.py results/bpb/          # dir of <model>_bpb.json
"""
import json
import math
import random
import sys
from pathlib import Path

B = 2000
LN2 = math.log(2)

SAME_SCRIPT = {"de": True, "fr": True, "ar": False, "zh": False}
TOKS = ("destarved", "starved")
_T = {"destarved": "fair", "starved": "starved"}

# Bilingual mixing is token-level 50/50, so a bilingual trained to T total
# tokens saw ~T/2 of each language -- the budget its monolingual control must
# match. `de` has no *starved* monolingual at any budget (that run does not
# exist yet), so the de/starved cell is missing everywhere.
#
# LR-STATE MATCHING (the confound that decides which budget is quotable).
# base_main.yaml is WSD: warmup 1B, stable 23B, decay 6B -> decay starts at
# 24B and the run ends at 30B. So every `*-8b` / `*-12b` / `*-15b` / `*-23b`
# checkpoint is a mid-STABLE snapshot sitting at peak LR 3.0e-3, while an
# unsuffixed model is the fully COOLED 30B final at 3.0e-4. Pairing a mono
# intermediate against a cooled bilingual final hands the bilingual the whole
# decay phase for free and inflates BTS positive -- it measures the cooldown,
# not transfer. (The original bts.py's `matched_lang` did exactly this.)
# Only budgets whose mono and bilingual are BOTH mid-stable are quotable.
BUDGETS = {
    "7.5B": {"mono": "{p}-{t}-8b", "bi": "en-{p}-{t}-15b",
             "tokens": (7.75, 7.38), "lr_matched": True,
             "note": "both mid-stable @ peak LR -- LR-state matched"},
    "11.4B": {"mono": "{p}-{t}-12b", "bi": "en-{p}-{t}-23b",
              "tokens": (11.75, 11.38), "lr_matched": True,
              "note": "both mid-stable @ peak LR -- LR-state matched"},
    "15B": {"mono": "{p}-{t}-15b", "bi": "en-{p}-{t}",
            "tokens": (14.75, 15.0), "lr_matched": False,
            "note": "mono mid-stable @3e-3 vs bilingual COOLED @3e-4 -- "
                    "CONFOUNDED, BTS inflated by the cooldown"},
}


def load(path: Path) -> dict:
    out = {}
    for f in sorted(path.glob("*_bpb.json")):
        d = json.loads(f.read_text())
        if "error" not in d:
            out[d["run"]] = d
    return out


def bpb_of(d: dict, lang: str, idx=None) -> float:
    cell = d["per_lang"][lang]
    nll, byt = cell["nll_nats"], cell["bytes"]
    if idx is None:
        return sum(nll) / (LN2 * sum(byt))
    s_n = s_b = 0.0
    for i in idx:
        s_n += nll[i]
        s_b += byt[i]
    return s_n / (LN2 * s_b)


def ci(reps):
    s = sorted(reps)
    return s[int(0.025 * len(s))], s[int(0.975 * len(s)) - 1]


def analyse(data, budget_name, spec, draws, n):
    avail, names = {}, {}
    for p in SAME_SCRIPT:
        for t in TOKS:
            mono = spec["mono"].format(p=p, t=_T[t])
            bi = spec["bi"].format(p=p, t=_T[t])
            names[(p, t)] = (mono, bi)
            if mono in data and bi in data and p in data[mono]["per_lang"]:
                avail[(p, t)] = (mono, bi)

    bm, bb = spec["tokens"]
    flag = "" if spec["lr_matched"] else "  ⚠️ NOT LR-MATCHED"
    print(f"\n## Budget {budget_name} — mono ~{bm}B vs bilingual "
          f"~{bb}B/language{flag}\n")
    print(f"_{spec['note']}_\n")
    if not avail:
        print("_No complete mono/bilingual pairs at this budget._")
        return
    print("| partner | script | tok | mono | bilingual | BPB mono | BPB bi | BTS [95% CI] |")
    print("|---|---|---|---|---|---|---|---|")

    bts_pt, bts_reps = {}, {}
    for p in SAME_SCRIPT:
        for t in TOKS:
            mono, bi = names[(p, t)]
            script = "same" if SAME_SCRIPT[p] else "cross"
            if (p, t) not in avail:
                miss = [x for x in (mono, bi) if x not in data]
                print(f"| {p} | {script} | {t} | {mono} | {bi} | - | - "
                      f"| n/a (missing: {', '.join(miss)}) |")
                continue
            m_all, b_all = bpb_of(data[mono], p), bpb_of(data[bi], p)
            pt = (m_all - b_all) / m_all
            reps = []
            for idx in draws:
                mm, bb_ = bpb_of(data[mono], p, idx), bpb_of(data[bi], p, idx)
                reps.append((mm - bb_) / mm)
            bts_pt[(p, t)], bts_reps[(p, t)] = pt, reps
            lo, hi = ci(reps)
            star = "**" if lo * hi > 0 else ""
            print(f"| {p} | {script} | {t} | {mono} | {bi} | {m_all:.4f} "
                  f"| {b_all:.4f} | {star}{pt:+.4f} [{lo:+.4f}, {hi:+.4f}]{star} |")

    both = [p for p in SAME_SCRIPT
            if (p, "starved") in avail and (p, "destarved") in avail]
    same = [p for p in both if SAME_SCRIPT[p]]
    cross = [p for p in both if not SAME_SCRIPT[p]]
    excluded = sorted(set(SAME_SCRIPT) - set(both))
    print(f"\n**Penalty / interaction at {budget_name}** — over partners with "
          f"BOTH tokenizer conditions: same-script {same or 'none'}, "
          f"cross-script {cross or 'none'}"
          + (f" (excluded: {excluded})" if excluded else "") + "\n")
    if not same or not cross:
        # A one-tokenizer penalty is still meaningful even when the full
        # interaction is not: it is the same-script vs cross-script gap.
        for t in TOKS:
            s1 = [p for p in SAME_SCRIPT if SAME_SCRIPT[p] and (p, t) in avail]
            c1 = [p for p in SAME_SCRIPT if not SAME_SCRIPT[p] and (p, t) in avail]
            if s1 and c1:
                pt = (sum(bts_pt[(p, t)] for p in s1) / len(s1)
                      - sum(bts_pt[(p, t)] for p in c1) / len(c1))
                reps = [sum(bts_reps[(p, t)][k] for p in s1) / len(s1)
                        - sum(bts_reps[(p, t)][k] for p in c1) / len(c1)
                        for k in range(B)]
                lo, hi = ci(reps)
                star = "**" if lo * hi > 0 else ""
                print(f"- penalty({t}) over same={s1} vs cross={c1}: "
                      f"{star}{pt:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}")
        print("\n_Interaction not computable: needs a same-script AND a "
              "cross-script partner present in BOTH tokenizer conditions._")
        return

    def penalty(t):
        pt = (sum(bts_pt[(p, t)] for p in same) / len(same)
              - sum(bts_pt[(p, t)] for p in cross) / len(cross))
        reps = [sum(bts_reps[(p, t)][k] for p in same) / len(same)
                - sum(bts_reps[(p, t)][k] for p in cross) / len(cross)
                for k in range(B)]
        return pt, reps

    print("| quantity | value [95% CI] |")
    print("|---|---|")
    pens = {}
    for t in TOKS:
        pt, reps = penalty(t)
        pens[t] = (pt, reps)
        lo, hi = ci(reps)
        star = "**" if lo * hi > 0 else ""
        print(f"| penalty({t}) | {star}{pt:+.4f} [{lo:+.4f}, {hi:+.4f}]{star} |")
    ipt = pens["starved"][0] - pens["destarved"][0]
    ireps = [pens["starved"][1][k] - pens["destarved"][1][k] for k in range(B)]
    lo, hi = ci(ireps)
    star = "**" if lo * hi > 0 else ""
    print(f"| **interaction** (starved − destarved) | {star}{ipt:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]{star} |")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    data = load(Path(sys.argv[1]))
    if not data:
        sys.exit("no BPB results found (run run_bpb.py first)")

    n = min(len(d["per_lang"][l]["nll_nats"])
            for d in data.values() for l in d["per_lang"])
    rng = random.Random(0)
    draws = [[rng.randrange(n) for _ in range(n)] for _ in range(B)]

    print(f"# Matched-token BTS on FLORES+ (paired bootstrap, B={B}, 95% CI, "
          f"n={n} sentences)\n")
    print("BTS = (BPB_mono − BPB_bi) / BPB_mono; +ve = bilingual training helps "
          "the partner language.\nEach bilingual is paired with a monolingual "
          "trained to the same per-language token budget,\nso there is no "
          "matched-total vs matched-lang split to reconcile.\n")
    for name, spec in BUDGETS.items():
        analyse(data, name, spec, draws, n)
    print("\n(Bold = 95% CI excludes 0. penalty > 0 => cross-script transfers "
          "worse than same-script;\ninteraction > 0 => that gap shrinks under "
          "the de-starved tokenizer, i.e. was partly a\ntokenizer-starvation "
          "artifact.)")


if __name__ == "__main__":
    main()
