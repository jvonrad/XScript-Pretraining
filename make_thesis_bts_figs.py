#!/usr/bin/env python
"""Thesis-formulation BTS figures (Chapter 4, "Cross-lingual Transfer").

⚠️ **This script did not previously exist in the repo, in git history, or
anywhere else findable on this machine** — written 2026-09-02, first without
a formula for `fig_I_bts_perf` (the task's initial spec gave equations for
BTS^data/BTS^loss but only a data source, no formula, for BTS^perf), then
completed once the thesis's own Eq. bts-perf and Eq. bts-loss text were
provided directly. All three formulas below are transcribed from that text,
not independently derived. BTS^data is additionally algebraically checked
against `bts_from_wandb.py`'s existing, already-used ATLAS estimator (see
`check_identity()` and the docstring on `bts_data_at` below) -- the one
formula here that IS an independent derivation from a paper (ATLAS,
arXiv:2510.22037) rather than a direct transcription, so it gets the extra
check; BTS^loss and BTS^perf are transcriptions of Eq. bts-loss/Eq. bts-perf
and are not independently re-derived.

WHAT THIS ADDS ON TOP OF `bts_from_wandb.py`
==============================================
`bts_from_wandb.py` already implements verified log-token interpolation
(`interp_bpb`) and first-bracketing-segment root finding (`tokens_to_reach`),
and already loads the merged W&B + checkpoint-fill curves (`load()`, which as
of 2026-09-02 also merges `bpb_curves_ckpt.csv` -- see that file's docstring
for why `de__unigram_starved` needs it). This script imports those primitives
rather than reimplementing them, and adds two things bts_from_wandb.py does
not: (1) the thesis's own BTS reparametrizations, swept over a grid of `d`
(mono per-language token budget) rather than evaluated at one anchor, and
(2) PDF figures.

  BTS^data(d) = -(sigma_bi(L_mono(d)) - 2d) / d       (Eq. bts-data)

      sigma_bi(L) = tokens_to_reach(bi, L): TOTAL bilingual tokens to first
      reach loss L. L_mono(d) = interp_bpb(mono, d): the mono run's loss at
      d per-language tokens.

      Algebraic identity to `bts_from_wandb.py`'s ATLAS BTS: for a
      monotonically-decreasing mono curve, tokens_to_reach(mono, L_mono(d))
      = d exactly (d is itself the first token count reaching that loss), so
      ATLAS_BTS(d) := D_mono(L)/D_bi(L) at L=L_mono(d) equals d/sigma_bi(L).
      Then 2 - 1/ATLAS_BTS(d) = 2 - sigma_bi(L)/d = -(sigma_bi(L)-2d)/d,
      which is BTS^data(d) exactly. `check_identity()` verifies this holds to
      float precision on real data before any figure is drawn -- if it does
      not, the two scripts disagree about what a "curve" is and nothing here
      should be trusted.

      null (pure 50/50 dilution, zero transfer) = 0. 1.0 = the second
      language was free (bilingual reaches L_mono(d) on d TOTAL tokens, same
      as mono alone). This is `bts_from_wandb.py`'s ATLAS BTS in [0,1] units
      with a natural zero, i.e. "horizontal" transfer: how much less total
      data the bilingual needed.

  BTS^loss(d) = (L^mono(d) - L^bi(2d)) / L^mono(d)    (Eq. bts-loss, matched
                                                       OWN-LANGUAGE EXPOSURE)

      Exactly `bts_from_wandb.py`'s "repo BTS" (`src/xscript/eval/bts.py`),
      evaluated as a function of d instead of at one x. A bilingual at TOTAL
      budget 2d has seen ~d tokens of this language (50/50 mixing), so this
      compares mono(d) against bi(2d): the SAME per-language exposure.
      null (no transfer) = 0. "Vertical": how much lower the loss is at a
      matched per-language budget. This is the variant plotted in fig_G and
      used as the primary quotable number, per CLAUDE.md's own established
      preference for content/lang-matched over token/total-matched BTS.

      The thesis text (Eq. bts-loss) also defines a MATCHED TOTAL BUDGET
      variant, L^mono(d) vs L^bi(d) (both see d total tokens, so the
      bilingual has seen only ~d/2 of this language) -- `bts_loss_at(...,
      matched="total")` implements it, reported in the summary table only
      (not plotted): it answers a different question (does source-language
      data compensate for DISPLACED target-language data?) from the plotted
      variant (does adding a source language help, holding target exposure
      fixed?), and the two should not be averaged together.

  BTS^perf(d) = (Abar'^bi(2d) - Abar'^mono(d)) / (1 - Abar'^mono(d))
                                                      (Eq. bts-perf)

      Abar'(d) = unweighted mean over benchmarks of chance-corrected accuracy
      A'_b = (A_b - c_b)/(1 - c_b), from `results/mubench_sweep/
      accuracy_table.md`. BENCHMARKS is the thesis's own-language suite
      (XNLI, ARC-Easy, XStoryCloze, HellaSwag, BMLAMA, PolyFact, X-CSQA) --
      SIB-200 is excluded, both because it is not in that suite and because
      the table's own header calls it saturated. null (no transfer) = 0; the
      denominator is the mono model's remaining headroom, so BTS^perf is "the
      fraction of remaining headroom the bilingual closes." Same matched-lang
      (2d, plotted in fig_I) / matched-total (d, summary table only) split as
      BTS^loss -- `bts_perf_at(..., matched=...)`.

      Points come straight from the table's discrete (model, B) rows, log-
      interpolated the same way `interp_bpb` interpolates BPB curves (that
      function is generic over any (x, y) pairs, so it is reused unchanged --
      see `_ACC_INTERP_NOTE` below). Only numeric-budget rows (`-Xb` suffix)
      are used; the bare 30B rows are the COOLED finals (accuracy_table.md's
      own header: "30B rows are COOLED... do not pair across that boundary")
      and are excluded exactly as `DECAY_START_B` excludes them on the BPB
      side -- this is a stable-window restriction applied by ROW SELECTION
      here rather than by a token-count comparison, because unlike the BPB
      curves this table has no points between 23B and 30B to accidentally
      admit.

Both BTS^data and BTS^loss use `d` = the MONO run's per-language token
budget; `2d` is the matched-own-language-exposure bilingual TOTAL budget.
Grid: log-spaced over the largest bracket where BOTH curves have data (mono's
own range, bilingual's range halved), intersected with the stable-LR window
[WARMUP_END_B, DECAY_START_B] -- exactly the same bracket `bts_from_wandb.py`'s
per-cell table uses, just swept instead of anchor-sampled. EN-anchored
bilinguals only (the non-EN-anchor mixes in W&B never actually ran --
`bts_from_wandb.py`'s own filter, reused unchanged for BTS^data/BTS^loss; the
accuracy table has no non-EN-anchor bilinguals to begin with).

    python make_thesis_bts_figs.py --tokenizer starved --source flores
    python make_thesis_bts_figs.py --tokenizer starved fair --source flores

`--tokenizer` accepts the thesis's fair/starved naming (CLAUDE.md) as an
alias for the repo's destarved/starved keys (`fair` -> `destarved`).
"""
import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts" / "external_bench"))
from bts_from_wandb import (  # noqa: E402
    load, stable, interp_bpb, tokens_to_reach,
    WARMUP_END_B, DECAY_START_B, SAME_SCRIPT,
)

DEFAULT_HISTORIES = REPO / "results" / "wandb_curves" / "histories.json"
DEFAULT_CKPT_CSV = REPO / "results" / "wandb_curves" / "bpb_curves_ckpt.csv"
DEFAULT_ACC_TABLE = REPO / "results" / "mubench_sweep" / "accuracy_table.md"
DEFAULT_OUT = REPO / "results" / "wandb_curves" / "figs"

# accuracy_table.md's own column order (after `model`, `B`). SIB-200 is
# EXCLUDED from BENCHMARKS -- it is not in the thesis's Table tab:benchmarks
# suite, and the table's own header separately calls it saturated ("do not
# put into benchmark aggregate"). Chance levels copied verbatim from that
# header; BMLAMA's is approximate there too ("~.10", ragged 2-10-way).
BENCHMARKS = ["XNLI", "ARC-E", "Story", "HSwag", "BMLAMA", "PolyFact", "X-CSQA"]
CHANCE = {"XNLI": 0.333, "ARC-E": 0.25, "Story": 0.50, "HSwag": 0.25,
         "BMLAMA": 0.10, "PolyFact": 0.25, "X-CSQA": 0.20}
_ACC_TABLE_COLS = ["SIB200"] + BENCHMARKS   # table's actual column order

PARTNERS = ["de", "fr", "ar", "zh"]          # fixed order -- see color table
TOK_ALIASES = {"fair": "destarved", "destarved": "destarved", "starved": "starved"}
TOK_LABEL = {"destarved": "fair", "starved": "starved"}   # thesis naming, for legends

# Categorical palette, dataviz skill reference/palette.md, slots 1-4, fixed
# order (never cycled/reassigned) -- validated on the adjacent-pairlist (the
# validation this palette documents as covering line charts) at CVD Delta E
# >=8.4 and normal-vision Delta E >=19.3 for every adjacent pair in this order.
COLOR = {
    "de": "#2a78d6",   # slot 1 blue    -- same-script
    "fr": "#eb6834",   # slot 2 orange  -- same-script
    "ar": "#1baf7a",   # slot 3 aqua    -- cross-script
    "zh": "#eda100",   # slot 4 yellow  -- cross-script
}
LINESTYLE = {"destarved": "-", "starved": "--"}   # tokenizer condition


def check_identity(mono, bi, d, atol=1e-9):
    """BTS^data(d) == 2 - 1/ATLAS_BTS(d) to float precision, or None/None if
    either side is undefined at this d (curve doesn't reach that far)."""
    Lm = interp_bpb(mono, d)
    if Lm is None:
        return None
    sigma_bi = tokens_to_reach(bi, Lm)
    dm = tokens_to_reach(mono, Lm)   # should equal d (mono is its own inverse here)
    if sigma_bi is None or dm is None or dm == 0:
        return None
    atlas = dm / sigma_bi
    bts_data = -(sigma_bi - 2 * d) / d
    identity = 2 - 1 / atlas
    assert abs(bts_data - identity) < atol or abs(dm - d) > 1e-6 * d, (
        f"BTS^data/ATLAS identity broken at d={d}: {bts_data} vs {identity} "
        f"(dm={dm} vs d={d}) -- do not trust the figures until this holds")
    return bts_data


def bts_data_at(mono, bi, d):
    Lm = interp_bpb(mono, d)
    if Lm is None:
        return None
    sigma_bi = tokens_to_reach(bi, Lm)
    if sigma_bi is None:
        return None
    return -(sigma_bi - 2 * d) / d


def bts_loss_at(mono, bi, d, matched="lang"):
    """Eq. bts-loss. `matched="lang"` (default, plotted in fig_G): bilingual
    at bi(2d), i.e. matched OWN-LANGUAGE exposure (bilingual's target-language
    share equals mono's d). `matched="total"`: bilingual at bi(d), i.e.
    matched TOTAL budget (both see d tokens overall) -- summary table only,
    per the thesis text's own budget-pair convention (see module docstring)."""
    Lm = interp_bpb(mono, d)
    if Lm is None:
        return None
    Lb = interp_bpb(bi, 2 * d if matched == "lang" else d)
    if Lb is None:
        return None
    return (Lm - Lb) / Lm


# --- accuracy_table.md: BTS^perf's data source --------------------------------

_ACC_ROW_RE = None   # compiled lazily; see parse_accuracy_table

def _acc_model_re():
    global _ACC_ROW_RE
    if _ACC_ROW_RE is None:
        import re
        _ACC_ROW_RE = re.compile(r"^\|\s*([\w-]+)\s*\|\s*(\d+)\s*\|(.+)\|\s*$")
    return _ACC_ROW_RE


def parse_accuracy_table(path):
    """`results/mubench_sweep/accuracy_table.md` -> {model_name: (B, {bench: acc})}.

    Ignores rows without a numeric `-Xb` budget suffix in the model name --
    i.e. the bare 30B cooled finals (`de-fair`, `en-de-fair`, ...) are
    excluded by construction, the stable-window restriction for this table
    (see module docstring). `–` cells (never scored) are dropped, not
    zero-filled -- `chance_corrected_mean` then averages over whatever
    benchmarks a model actually has.
    """
    row_re = _acc_model_re()
    out = {}
    for line in Path(path).read_text().splitlines():
        m = row_re.match(line)
        if not m or m.group(1) == "model":
            continue
        name, b, rest = m.group(1), int(m.group(2)), m.group(3)
        cells = [c.strip() for c in rest.split("|")]
        if len(cells) != len(_ACC_TABLE_COLS):
            continue   # header separator or malformed row
        vals = {}
        for col, cell in zip(_ACC_TABLE_COLS, cells):
            if cell in ("", "–", "-"):
                continue
            try:
                vals[col] = float(cell)
            except ValueError:
                continue
        out[name] = (b, vals)
    return out


def chance_corrected_mean(vals):
    """Abar'(model) = unweighted mean of (A_b - c_b)/(1 - c_b) over whatever
    of BENCHMARKS this model has a real (non-`–`) score for. None if none."""
    terms = [(vals[b] - CHANCE[b]) / (1 - CHANCE[b])
            for b in BENCHMARKS if b in vals]
    return sum(terms) / len(terms) if terms else None


def perf_points(acc, lang, tok, kind):
    """[(B, Abar'), ...] sorted, for `kind` in {"mono", "bi"}. `tok` is the
    repo's destarved/starved key; the table itself uses fair/starved names."""
    import re
    label = TOK_LABEL[tok]
    prefix = f"en-{lang}-" if kind == "bi" else f"{lang}-"
    pat = re.compile(rf"^{re.escape(prefix)}{label}-(\d+)b$")
    pts = []
    for name, (b, vals) in acc.items():
        if not pat.match(name):
            continue
        m = chance_corrected_mean(vals)
        if m is not None:
            pts.append((float(b), m))
    return sorted(pts)


def bts_perf_at(mono, bi, d, matched="lang"):
    """Eq. bts-perf. Same matched-lang(2d)/matched-total(d) split as
    bts_loss_at; `mono`/`bi` here are `perf_points(...)` series, log-
    interpolated with the SAME primitive as the BPB curves (`interp_bpb` is
    generic over any (x, y) pairs -- reused unchanged, not reimplemented)."""
    Am = interp_bpb(mono, d)
    if Am is None or Am >= 1.0:
        return None
    Ab = interp_bpb(bi, 2 * d if matched == "lang" else d)
    if Ab is None:
        return None
    return (Ab - Am) / (1 - Am)


def bracket(mono, bi):
    """[lo, hi] over d where mono(d) and bi(2d) are both defined, inside the
    stable-LR window. Same rule `bts_from_wandb.py`'s per-cell table uses."""
    if len(mono) < 2 or len(bi) < 2:
        return None
    lo = max(mono[0][0], WARMUP_END_B)
    hi = min(mono[-1][0], bi[-1][0] / 2, DECAY_START_B)
    if hi <= lo:
        return None
    return lo, hi


def log_grid(lo, hi, n=60):
    a, b = math.log(lo), math.log(hi)
    return [math.exp(a + i * (b - a) / (n - 1)) for i in range(n)]


def series(fn, mono, bi, lo, hi, n=60):
    pts = [(d, fn(mono, bi, d)) for d in log_grid(lo, hi, n)]
    return [(d, v) for d, v in pts if v is not None]


def bpb_cell_fn(data):
    """-> (partner, tok) -> (mono_pts, bi_pts), for BTS^data/BTS^loss."""
    def _cell(partner, tok):
        mono = stable(data.get((partner, tok), {}).get(partner, []))
        bi = stable(data.get((f"en-{partner}", tok), {}).get(partner, []))
        return mono, bi
    return _cell


def perf_cell_fn(acc):
    """-> (partner, tok) -> (mono_pts, bi_pts), for BTS^perf. Points are
    (B, Abar') from `perf_points`, already restricted to numeric-budget rows
    (the stable-window restriction for this table -- see module docstring)."""
    def _cell(partner, tok):
        return perf_points(acc, partner, tok, "mono"), perf_points(acc, partner, tok, "bi")
    return _cell


def plot_figure(out_path, title, ylabel, series_fn, cell_fn, toks, source, null_line=0.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=200)
    ax.axhline(null_line, color="#8a8a86", linewidth=1, linestyle=":",
              zorder=1, label=f"null ({null_line:g})")

    plotted = []
    for tok in toks:
        for p in PARTNERS:
            mono, bi = cell_fn(p, tok)
            br = bracket(mono, bi)
            if br is None:
                continue
            lo, hi = br
            pts = series(series_fn, mono, bi, lo, hi)
            if len(pts) < 2:
                continue
            xs = [d for d, _ in pts]   # already in B (tokens_b units throughout)
            ys = [v for _, v in pts]
            script = "same" if SAME_SCRIPT[p] else "cross"
            label = f"{p} ({script}-script, {TOK_LABEL[tok]})"
            ax.plot(xs, ys, color=COLOR[p], linestyle=LINESTYLE[tok],
                    linewidth=2, solid_capstyle="round", label=label,
                    marker="o", markersize=4, markevery=max(1, len(xs) // 10),
                    zorder=3)
            # honest markers at the mono run's OWN eval points inside the
            # bracket -- the smooth curve is interpolation, these are data.
            real_d = [t for t, _ in mono if lo <= t <= hi]
            real_y = [series_fn(mono, bi, t) for t in real_d]
            real = [(t, v) for t, v in zip(real_d, real_y) if v is not None]
            if real:
                ax.scatter([t for t, _ in real], [v for _, v in real],
                          color=COLOR[p], s=20, zorder=4, edgecolors="white",
                          linewidths=0.6)
            plotted.append((p, tok))

    if not plotted:
        plt.close(fig)
        print(f"[figs] SKIPPED {out_path.name}: no cell had >=2 points in the "
              f"stable-window bracket for tokenizer(s) {toks}")
        return False

    ax.set_xscale("log")
    ax.set_xlabel("d = monolingual per-language tokens (B)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (source={source})")
    # Legend BELOW the axes, not inside the plotted area -- "best" placement
    # collided with data in practice (a line's upswing at the right edge ran
    # straight through the legend text). Outside placement avoids that by
    # construction rather than by picking a corner that happens to be clear
    # for THIS run's curves.
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16),
             ncol=2, frameon=False, columnspacing=1.2, handletextpad=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="both", axis="both", color="#e5e5e2", linewidth=0.6,
           zorder=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[figs] wrote {out_path}  ({len(plotted)} (partner, tokenizer) lines)")
    return True


def summary_table(bpb_fn, perf_fn, toks, source):
    """Per-pair BTS^data/BTS^loss/BTS^perf at the LARGEST budget common to
    ALL FOUR partners AND BOTH data sources (the same "one budget for all
    partners" principle bts_matched.py documents, extended across BPB and
    accuracy since a thesis table quoting all three side by side needs one
    shared d) -- accuracy_table.md is much sparser than the BPB curves, so it
    is usually the binding constraint. Prints "NOT COMPUTABLE" rather than a
    number when no such d exists for a tokenizer -- never average over a
    partial partner or metric set.

    BTS^data has only the matched-own-language-exposure form (Eq. bts-data
    has no matched-total analog -- sigma_bi already searches the bilingual's
    full curve for whatever total tokens it needs, so there is nothing
    separate to "match"). BTS^loss and BTS^perf each report both budgets from
    the thesis text: matched-lang (plotted) and matched-total (table only).
    """
    print(f"\n## Matched-budget BTS summary (source={source})\n")
    for tok in toks:
        brackets = {}
        for p in PARTNERS:
            brackets[(p, "bpb")] = bracket(*bpb_fn(p, tok))
            brackets[(p, "perf")] = bracket(*perf_fn(p, tok))
        missing = [k for k, b in brackets.items() if b is None]
        print(f"### tokenizer={TOK_LABEL[tok]} ({tok})\n")
        if missing:
            print(f"NOT COMPUTABLE for a common budget: no bracket for "
                  f"{missing} (insufficient stable-window overlap between "
                  f"mono and bilingual curves for that partner/source).\n")
            for k, b in brackets.items():
                if b is not None:
                    print(f"  {k[0]}/{k[1]}: own bracket [{b[0]:.2f}, {b[1]:.2f}]B")
            print()
            continue
        d_common = min(hi for _, hi in brackets.values())
        lo_needed = max(lo for lo, _ in brackets.values())
        if d_common < lo_needed:
            print(f"NOT COMPUTABLE: common upper bound {d_common:.2f}B is "
                  f"below one (partner, source)'s own lower bound "
                  f"{lo_needed:.2f}B.\n")
            continue
        print(f"Common matched budget (BPB + accuracy_table.md): "
              f"d = {d_common:.3f}B (bilingual total = {2 * d_common:.3f}B)\n")
        cols = ["BTS^data", "BTS^loss(lang)", "BTS^loss(total)",
               "BTS^perf(lang)", "BTS^perf(total)"]
        print("| partner | script | " + " | ".join(cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        rows = {}
        for p in PARTNERS:
            mono_b, bi_b = bpb_fn(p, tok)
            mono_a, bi_a = perf_fn(p, tok)
            vals = (
                bts_data_at(mono_b, bi_b, d_common),
                bts_loss_at(mono_b, bi_b, d_common, matched="lang"),
                bts_loss_at(mono_b, bi_b, d_common, matched="total"),
                bts_perf_at(mono_a, bi_a, d_common, matched="lang"),
                bts_perf_at(mono_a, bi_a, d_common, matched="total"),
            )
            rows[p] = vals
            script = "same" if SAME_SCRIPT[p] else "cross"
            cells = " | ".join(f"{v:+.4f}" if v is not None else "-" for v in vals)
            print(f"| {p} | {script} | {cells} |")
        same = [rows[p] for p in ("de", "fr")]
        cross = [rows[p] for p in ("ar", "zh")]
        print()
        for i, metric in enumerate(cols):
            sv = [r[i] for r in same if r[i] is not None]
            cv = [r[i] for r in cross if r[i] is not None]
            if not sv or not cv:
                print(f"  {metric}: gap NOT COMPUTABLE ({len(sv)}/2 same-script, "
                      f"{len(cv)}/2 cross-script values available)")
                continue
            sm, cm = sum(sv) / len(sv), sum(cv) / len(cv)
            print(f"  {metric}: same-script mean = {sm:+.4f} (n={len(sv)}), "
                  f"cross-script mean = {cm:+.4f} (n={len(cv)}), "
                  f"gap (same - cross) = {sm - cm:+.4f}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--histories", type=Path, default=DEFAULT_HISTORIES)
    ap.add_argument("--ckpt-csv", type=Path, default=DEFAULT_CKPT_CSV)
    ap.add_argument("--acc-table", type=Path, default=DEFAULT_ACC_TABLE)
    ap.add_argument("--source", default="flores", choices=["flores", "holdout"])
    ap.add_argument("--tokenizer", nargs="+", default=["starved"],
                    choices=sorted(TOK_ALIASES))
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    toks = sorted({TOK_ALIASES[t] for t in args.tokenizer},
                  key=lambda t: ("destarved", "starved").index(t))
    ckpt = args.ckpt_csv if args.ckpt_csv.exists() else None
    data = load(args.histories, args.source, ckpt)
    acc = parse_accuracy_table(args.acc_table)
    print(f"[figs] parsed {len(acc)} model rows from {args.acc_table}")

    bpb_fn = bpb_cell_fn(data)
    perf_fn = perf_cell_fn(acc)

    # Verify the algebraic identity on real data before drawing anything --
    # this is what makes BTS^data trustworthy rather than a second,
    # independently-asserted formula living next to the repo's existing one.
    # (BTS^loss and BTS^perf are direct transcriptions of the thesis's own
    # Eq. bts-loss / Eq. bts-perf, so there is no second estimator to check
    # them against -- see the module docstring.)
    n_checked = 0
    for tok in toks:
        for p in PARTNERS:
            mono, bi = bpb_fn(p, tok)
            br = bracket(mono, bi)
            if br is None:
                continue
            lo, hi = br
            for d in log_grid(lo, hi, 5):
                if check_identity(mono, bi, d) is not None:
                    n_checked += 1
    print(f"[figs] BTS^data == 2 - 1/ATLAS_BTS identity verified at "
          f"{n_checked} (partner, tokenizer, d) points")

    tag = "-".join(TOK_LABEL[t] for t in toks)
    # fig_F/fig_G are BPB-derived, so they differ by --source and MUST carry it
    # in the filename -- without it a `--source holdout` run silently
    # overwrites the flores figures (it did, once). fig_I is NOT source-tagged
    # on purpose: BTS^perf reads accuracy_table.md, which has no flores/holdout
    # distinction, so both runs produce the identical figure.
    tag = f"{tag}_{args.source}"
    plot_figure(args.out_dir / f"fig_F_bts_data_horizontal_{tag}.pdf",
               "BTS$^{data}$ -- horizontal transfer (Eq. bts-data)",
               r"BTS$^{data}(d)$  (0 = dilution null, 1 = free)",
               bts_data_at, bpb_fn, toks, args.source, null_line=0.0)
    plot_figure(args.out_dir / f"fig_G_bts_loss_vertical_{tag}.pdf",
               "BTS$^{loss}$ -- vertical transfer, matched-lang (Eq. bts-loss)",
               r"BTS$^{loss}(d)$  (0 = no transfer)",
               bts_loss_at, bpb_fn, toks, args.source, null_line=0.0)
    perf_tag = "-".join(TOK_LABEL[t] for t in toks)   # no source: see above
    plot_figure(args.out_dir / f"fig_I_bts_perf_{perf_tag}.pdf",
               "BTS$^{perf}$ -- downstream transfer, matched-lang (Eq. bts-perf)",
               r"BTS$^{perf}(d)$  (0 = no transfer, headroom-closed fraction)",
               bts_perf_at, perf_fn, toks,
               f"accuracy_table.md ({len(BENCHMARKS)} benchmarks, chance-corrected)",
               null_line=0.0)

    summary_table(bpb_fn, perf_fn, toks, args.source)


if __name__ == "__main__":
    raise SystemExit(main())
