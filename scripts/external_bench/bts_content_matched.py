#!/usr/bin/env python
"""The repo's own BTS at a fixed budget, token-matched and content-matched.

    BTS = (BPB_mono - BPB_bi) / BPB_mono        (+ve = bilingual helps; null 0)

Bilingual mixing is token-level 50/50, so a bilingual at total T saw T/2 of
the partner language: the token-matched comparison is mono(X) vs bilingual(2X).

TOKEN-MATCHED vs CONTENT-MATCHED
--------------------------------
Within one tokenizer condition, mono and bilingual share a tokenizer, so
content-matching is only a relabelling of the x-axis and cannot change BTS.
It matters when comparing the two *conditions* to each other -- which is
exactly what the headline interaction does.

The starved tokenizer needs more tokens to encode the same text (fertility
ratios starved/fair, FLORES: ar 1.476, de 1.371, zh 1.304, fr 1.301, en 1.200),
so at an equal TOKEN budget a starved run has processed strictly LESS text.
Comparing penalty(starved) against penalty(destarved) at equal tokens
therefore conflates "worse tokenizer" with "less content seen" -- and the
conflation is largest for exactly the cross-script language the thesis cares
about (ar). Content-matching removes it by evaluating each condition at the
token count that corresponds to the SAME number of UTF-8 bytes of the partner
language:

    tokens_needed(cond) = target_bytes * fertility(cond, lang)

Curves come from the W&B training logs restricted to the stable-LR window, so
both tables are cooldown-clean (see bts_from_wandb.py).

    python bts_content_matched.py histories.json fertility.json
"""
import argparse
import json
import math
from pathlib import Path

import bts_from_wandb as W


def bts_at(mono, bi, x_lang):
    """(BPB_mono - BPB_bi)/BPB_mono comparing mono(x) with bilingual(2x)."""
    bm = W.interp_bpb(mono, x_lang)
    bb = W.interp_bpb(bi, 2 * x_lang)
    if bm is None or bb is None:
        return None, bm, bb
    return (bm - bb) / bm, bm, bb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("histories")
    ap.add_argument("fertility")
    ap.add_argument("--source", default="flores", choices=["flores", "holdout"])
    ap.add_argument("--per-lang-tokens", type=float, default=12.0,
                    help="token-matched budget per language, in B "
                         "(default 12 => bilingual total 24B)")
    args = ap.parse_args()

    data = W.load(args.histories, args.source)
    fert = json.loads(Path(args.fertility).read_text())

    cells = {}
    for p in ["de", "fr", "ar", "zh"]:
        for tok in W.TOKS:
            mono = W.stable(data.get((p, tok), {}).get(p, []))
            bi = W.stable(data.get((f"en-{p}", tok), {}).get(p, []))
            if len(mono) >= 3 and len(bi) >= 3:
                cells[(p, tok)] = (mono, bi)

    print(f"# Repo BTS at a fixed budget (source={args.source}, "
          f"stable-LR window {W.WARMUP_END_B}-{W.DECAY_START_B}B)\n")
    print("BTS = (BPB_mono − BPB_bi)/BPB_mono, mono(X) vs bilingual(2X). "
          "Null = 0.\n")

    # largest per-language budget each cell can support: mono must reach X and
    # the bilingual must reach 2X (it only spends half its tokens on `p`).
    maxX = {k: min(m[-1][0], b[-1][0] / 2) for k, (m, b) in cells.items()}

    # ---------- Table A: token-matched ----------
    Xreq = args.per_lang_tokens
    print(f"## A. Token-matched\n")
    print(f"Requested budget was {Xreq:.0f}B/language (bilingual {2*Xreq:.0f}B "
          f"total). Each partner is reported at the largest budget BOTH its\n"
          "tokenizer conditions support, so the fair-vs-starved gap is itself "
          "token-matched; `max feasible` shows how close that is to the\n"
          "request. Curve coverage, not method, is the binding constraint.\n")
    print("| cell | script | X/lang | BPB mono | BPB bi | BTS | vs requested |")
    print("|---|---|---|---|---|---|---|")
    tokA, XA = {}, {}
    for p in ["de", "fr", "ar", "zh"]:
        avail = [t for t in W.TOKS if (p, t) in cells]
        if not avail:
            continue
        X = min(maxX[(p, t)] for t in avail)
        XA[p] = X
        for t in avail:
            mono, bi = cells[(p, t)]
            v, bm, bb = bts_at(mono, bi, X)
            script = "same" if W.SAME_SCRIPT[p] else "cross"
            note = "as requested" if X >= Xreq * 0.98 else f"capped ({X:.2f}B < {Xreq:.0f}B)"
            if v is None:
                print(f"| {p}/{t} | {script} | {X:.2f}B | - | - | - | {note} |")
            else:
                tokA[(p, t)] = v
                print(f"| {p}/{t} | {script} | {X:.2f}B | {bm:.4f} | {bb:.4f} "
                      f"| **{v:+.4f}** | {note} |")

    # ---------- Table B: content-matched ----------
    print(f"\n## B. Content-matched — equal UTF-8 bytes of the partner language\n")
    print("Each condition is evaluated at the token count encoding the SAME "
          "amount of text: tokens = bytes x fertility(cond, lang).\n"
          "The target content is the most both conditions can reach inside the "
          "stable window, so the starved run (higher fertility)\nsits at a "
          "higher token count than the fair one for identical content.\n")
    print("| cell | script | fertility | X/lang | content | BPB mono | BPB bi | BTS |")
    print("|---|---|---|---|---|---|---|---|")
    conB = {}
    for p in ["de", "fr", "ar", "zh"]:
        avail = [t for t in W.TOKS if (p, t) in cells]
        if not avail:
            continue
        fc = {t: fert[p]["fair" if t == "destarved" else "starved"] for t in avail}
        # content each condition can reach; take the min so both are in range
        target_bytes = min(maxX[(p, t)] / fc[t] for t in avail)
        for t in avail:
            mono, bi = cells[(p, t)]
            x_lang = target_bytes * fc[t]
            v, bm, bb = bts_at(mono, bi, x_lang)
            script = "same" if W.SAME_SCRIPT[p] else "cross"
            if v is None:
                print(f"| {p}/{t} | {script} | {fc[t]:.5f} | {x_lang:.2f}B "
                      f"| {target_bytes:.1f}GB | - | - | - |")
            else:
                conB[(p, t)] = v
                print(f"| {p}/{t} | {script} | {fc[t]:.5f} | {x_lang:.2f}B "
                      f"| {target_bytes:.1f}GB | {bm:.4f} | {bb:.4f} "
                      f"| **{v:+.4f}** |")

    # ---------- penalty / interaction on each ----------
    print("\n## Penalty / interaction\n")
    for label, res in (("token-matched", tokA), ("content-matched", conB)):
        both = [p for p in W.SAME_SCRIPT
                if all((p, t) in res for t in W.TOKS)]
        same = [p for p in both if W.SAME_SCRIPT[p]]
        cross = [p for p in both if not W.SAME_SCRIPT[p]]
        if not same or not cross:
            print(f"- **{label}**: interaction not computable "
                  f"(same-script {same or 'none'}, cross-script "
                  f"{cross or 'none'} with both tokenizers)")
            continue
        pen = {t: (sum(res[(p, t)] for p in same) / len(same)
                   - sum(res[(p, t)] for p in cross) / len(cross))
               for t in W.TOKS}
        print(f"- **{label}** over same={same}, cross={cross}: "
              f"penalty(starved)={pen['starved']:+.4f}, "
              f"penalty(destarved)={pen['destarved']:+.4f}, "
              f"**interaction={pen['starved'] - pen['destarved']:+.4f}**")

    # cross-condition comparability is the whole point of Table B
    print("\n### Fair − starved BTS gap (only meaningful in Table B)\n")
    print("| partner | token-matched Δ | content-matched Δ |")
    print("|---|---|---|")
    for p in ["de", "fr", "ar", "zh"]:
        def gap(res):
            a, b = res.get((p, "destarved")), res.get((p, "starved"))
            return f"{a - b:+.4f}" if (a is not None and b is not None) else "-"
        print(f"| {p} | {gap(tokA)} | {gap(conB)} |")


if __name__ == "__main__":
    main()
