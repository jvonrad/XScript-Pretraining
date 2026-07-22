#!/usr/bin/env python
"""ATLAS-style BTS from the training-time BPB curves logged to W&B.

Why this rather than evaluating checkpoints (see CLAUDE.md's BPB->BTS section):
the trainer already logs `eval/{flores,holdout}_{lang}_bpb` against `tokens_b`
at every checkpoint interval, so the full loss-vs-tokens curve exists for every
run at zero compute -- denser than any checkpoint grid we could upload, and it
includes holdout (whose shards are not on the eval box at all).

Crucially it also makes the comparison **cooldown-clean by construction**: the
WSD schedule (warmup 1B / stable 23B / decay 6B) is flat at peak LR from 1B to
24B, so restricting to that window compares mono and bilingual at an identical
LR state. Pairing an uncooled monolingual against a cooled bilingual final --
what `results/bts/*`'s `matched_lang` did, and what a naive
"mono-15b vs en-X-fair" pairing does -- inflates BTS by handing the bilingual a
free decay phase.

Two estimators are reported, because the repo and ATLAS do NOT define BTS the
same way:

  repo (`src/xscript/eval/bts.py`):
      BTS = (BPB_mono - BPB_bi) / BPB_mono   at matched per-language tokens
      null (no transfer) = 0

  ATLAS (arXiv:2510.22037): "the relative training efficiency of a bilingual
  model (50%,50%) compared to the monolingual model t at reaching the same
  loss level" -- an iso-loss token-efficiency ratio:
      BTS = D_mono(L) / D_bi(L)              D = TOTAL tokens to reach loss L
      null (pure 50/50 dilution, zero transfer) = 0.5
      1.0 => the second language was free; >0.5 => real positive transfer

Curves are interpolated in log(tokens) vs BPB, which is near-linear over the
stable window. The iso-loss inversion is ill-conditioned wherever a curve is
flat, so `--anchor-frac` picks the target loss from the steeper early part of
the monolingual curve by default.

    python bts_from_wandb.py histories.json [--source flores] [--holdout]
"""
import argparse
import json
import math
from pathlib import Path

WARMUP_END_B = 1.0    # WSD: LR reaches peak here
DECAY_START_B = 24.0  # WSD: cooldown begins here -- everything after is
                      # annealed and NOT comparable to stable-phase points
SAME_SCRIPT = {"de": True, "fr": True, "ar": False, "zh": False}
TOKS = {"destarved": "destarved", "starved": "starved"}

# Runs excluded as known-bad. `de__unigram_starved` collapsed partway through
# (confirmed by the run's owner from live monitoring); the collapse is visible
# in the curve as an anchor BPB of ~1.72 where the destarved twin sits at
# ~1.06. It is being retrained -- re-include once the new run lands.
EXCLUDE_RUNS = {"de__unigram_starved"}


def load(path, source):
    """{(mix, tok): {lang: [(tokens_b, bpb), ...]}} for EN-anchored runs only."""
    raw = json.loads(Path(path).read_text())
    out = {}
    for key, r in raw.items():
        name = r["name"]
        if "__" not in name or name in EXCLUDE_RUNS:
            continue
        mix, tok = name.split("__", 1)
        tok = tok.replace("unigram_", "")
        if tok not in TOKS:
            continue
        # EN-anchored only: the non-English-anchor bilinguals never really ran.
        parts = mix.split("-")
        if len(parts) == 2 and parts[0] != "en":
            continue
        if len(parts) > 2:
            continue
        cur = out.setdefault((mix, tok), {})
        for rec in r["history"]:
            t = rec["tokens_b"]
            for k, v in rec.items():
                if not k.startswith("eval/") or not k.endswith("_bpb"):
                    continue
                src, lang = k[len("eval/"):-len("_bpb")].rsplit("_", 1)
                if src != source:
                    continue
                cur.setdefault(lang, []).append((t, v))
    # de-duplicate (repeated run names) keeping the lowest bpb per token point,
    # and sort
    for cell in out.values():
        for lang, pts in cell.items():
            best = {}
            for t, v in pts:
                if t not in best or v < best[t]:
                    best[t] = v
            cell[lang] = sorted(best.items())
    return out


def stable(pts):
    return [(t, v) for t, v in pts if WARMUP_END_B <= t <= DECAY_START_B]


def interp_bpb(pts, t):
    """BPB at token count t by linear interpolation in log-tokens."""
    xs = [math.log(p[0]) for p in pts]
    ys = [p[1] for p in pts]
    x = math.log(t)
    eps = 1e-9
    if x < xs[0] - eps or x > xs[-1] + eps:
        return None          # never extrapolate
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]        # exact-endpoint hit is valid, not a miss
    for i in range(1, len(xs)):
        if x <= xs[i]:
            f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return None


def tokens_to_reach(pts, target):
    """TOTAL tokens at which the curve first reaches `target` BPB (log-interp).

    Curves are noisy and not perfectly monotone, so we scan for the first
    bracketing segment rather than assuming monotonicity.
    """
    for i in range(1, len(pts)):
        y0, y1 = pts[i - 1][1], pts[i][1]
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
            x0, x1 = math.log(pts[i - 1][0]), math.log(pts[i][0])
            f = (target - y0) / (y1 - y0)
            return math.exp(x0 + f * (x1 - x0))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("histories")
    ap.add_argument("--source", default="flores", choices=["flores", "holdout"])
    ap.add_argument("--anchor-frac", type=float, default=0.5,
                    help="target loss = mono BPB at this fraction (log-space) "
                         "of its usable stable range; lower = steeper, better "
                         "conditioned inversion")
    args = ap.parse_args()
    data = load(args.histories, args.source)

    print(f"# BTS from W&B training curves (source={args.source})\n")
    print(f"Stable-LR window only: {WARMUP_END_B}B - {DECAY_START_B}B "
          f"(WSD peak-LR plateau). Points outside are dropped, so mono and\n"
          f"bilingual are compared at an identical LR state -- no cooldown "
          f"confound.\n")

    print("## Curve coverage (stable window)\n")
    print("| cell | mono pts (range) | bilingual pts (range) |")
    print("|---|---|---|")
    cells = {}
    for p in ["de", "fr", "ar", "zh"]:
        for tok in TOKS:
            mono = stable(data.get((p, tok), {}).get(p, []))
            bi = stable(data.get((f"en-{p}", tok), {}).get(p, []))
            ms = f"{len(mono)} ({mono[0][0]:.2f}-{mono[-1][0]:.2f}B)" if mono else "0"
            bs = f"{len(bi)} ({bi[0][0]:.2f}-{bi[-1][0]:.2f}B)" if bi else "0"
            print(f"| {p}/{tok} | {ms} | {bs} |")
            if len(mono) >= 3 and len(bi) >= 3:
                cells[(p, tok)] = (mono, bi)

    if not cells:
        print("\n_No cell has enough points in the stable window._")
        return

    print("\n## BTS\n")
    print("`repo BTS` = (BPB_mono − BPB_bi)/BPB_mono at matched PER-LANGUAGE "
          "tokens (null 0).\n`ATLAS BTS` = D_mono(L)/D_bi(L), total tokens to "
          "reach the same loss (null 0.5, 1.0 = free).\n")
    print("| cell | script | matched/lang | repo BTS | anchor L | D_mono | D_bi "
          "| ATLAS BTS | ATLAS range over anchors |")
    print("|---|---|---|---|---|---|---|---|---|")
    results = {}
    for (p, tok), (mono, bi) in sorted(cells.items()):
        script = "same" if SAME_SCRIPT[p] else "cross"
        lo_t, hi_t = max(mono[0][0], bi[0][0]), min(mono[-1][0], bi[-1][0])
        if hi_t <= lo_t:
            print(f"| {p}/{tok} | {script} | _no token overlap_ | | | | | | |")
            continue

        def atlas_at(frac):
            at = math.exp(math.log(lo_t) + frac * (math.log(hi_t) - math.log(lo_t)))
            L = interp_bpb(mono, at)
            if L is None:
                return None, None, None, None
            dm, db = tokens_to_reach(mono, L), tokens_to_reach(bi, L)
            return (dm / db if (dm and db) else None), L, dm, db

        atlas, L, d_mono, d_bi = atlas_at(args.anchor_frac)
        # anchor sensitivity: the iso-loss inversion is ill-conditioned where a
        # curve is flat, so a single anchor can badly misrepresent the cell.
        sweep = [a for a, *_ in (atlas_at(f) for f in
                                 (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) if a]
        rng = f"{min(sweep):.2f}–{max(sweep):.2f} (n={len(sweep)})" if sweep else "-"

        # repo-style: matched PER-LANGUAGE tokens. A bilingual at total T saw
        # T/2 of this language, so compare mono(x) with bi(2x).
        x = min(hi_t, mono[-1][0], bi[-1][0] / 2)
        rb = None
        if x >= mono[0][0]:
            bm, bb = interp_bpb(mono, x), interp_bpb(bi, 2 * x)
            if bm is not None and bb is not None:
                rb = (bm - bb) / bm
        results[(p, tok)] = {"repo": rb, "atlas": atlas}
        print(f"| {p}/{tok} | {script} | {x:.2f}B "
              f"| {f'{rb:+.4f}' if rb is not None else '-'} "
              f"| {f'{L:.4f}' if L else '-'} "
              f"| {f'{d_mono:.2f}B' if d_mono else '-'} "
              f"| {f'{d_bi:.2f}B' if d_bi else '-'} "
              f"| {f'**{atlas:.3f}**' if atlas else '-'} | {rng} |")

    # interaction on whichever estimator has both tokenizers for a same- and a
    # cross-script partner
    print("\n## Penalty / interaction\n")
    for est in ("repo", "atlas"):
        both = [p for p in SAME_SCRIPT
                if all((p, t) in results and results[(p, t)][est] is not None
                       for t in TOKS)]
        same = [p for p in both if SAME_SCRIPT[p]]
        cross = [p for p in both if not SAME_SCRIPT[p]]
        if not same or not cross:
            print(f"- **{est} BTS**: not computable "
                  f"(same-script {same or 'none'}, cross-script {cross or 'none'} "
                  f"with both tokenizers)")
            continue
        pen = {}
        for t in TOKS:
            pen[t] = (sum(results[(p, t)][est] for p in same) / len(same)
                      - sum(results[(p, t)][est] for p in cross) / len(cross))
        print(f"- **{est} BTS** over same={same}, cross={cross}: "
              f"penalty(starved)={pen['starved']:+.4f}, "
              f"penalty(destarved)={pen['destarved']:+.4f}, "
              f"**interaction={pen['starved'] - pen['destarved']:+.4f}**")


if __name__ == "__main__":
    main()
