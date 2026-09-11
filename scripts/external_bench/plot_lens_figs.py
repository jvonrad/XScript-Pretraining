#!/usr/bin/env python
"""Two thesis figures for the logit-lens section (CLAUDE.md 6l).

  fig_lens_summary.{pdf,png}   three panels: (a) fair - starved commitment
        layer per partner at the 30B finals and at matched partner BPB;
        (b) latent-English trace on X->X repetition with the partner-language
        monolingual as control; (c) PolyFact 4-way accuracy by layer,
        partner-language prompts, fair vs starved.
  fig_lens_example.{pdf,png}   Wendler/Schut-style layer x position grid of
        the lens argmax for one en->ar translation prompt, fair vs starved.

    python plot_lens_figs.py [--word heart]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import analyze_logitlens as A                                   # noqa: E402
import run_logitlens as R                                       # noqa: E402
from xscript.tok.wrapper import Tok                              # noqa: E402
from xscript.eval.logitlens import LANGS, Lens, load_model, load_or_build_tokmap, encode_pair  # noqa: E402

FIGS = A.FIGS
COL = {"fair": "#1f77b4", "starved": "#ff7f0e", "mono": "#7f7f7f"}
PARTNERS = ("de", "fr", "ar", "zh")
MONO = {"de": "de-fair", "fr": "fr-fair", "ar": "ar-fair", "zh": "zh-fair-15b"}


def paired_commit(fr, sr, X, cond):
    ra, rb = A.load_rows(fr, X, cond), A.load_rows(sr, X, cond)
    ok = np.array([x["out"]["greedy_ok"] and y["out"]["greedy_ok"] for x, y in zip(ra, rb)])
    Pa = np.exp([x["out"]["full_lp"] for x in ra]); Pb = np.exp([x["out"]["full_lp"] for x in rb])
    return A.boot_ci(A.commit_layer(Pa)[ok] - A.commit_layer(Pb)[ok])


def summary_figure(runs, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report, summ = [], {}
    word = A.analyse_wordtasks(runs, models, report, summ)
    fact = A.analyse_factual(runs, models, report, summ)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # (a) commitment-layer deltas
    ax = axes[0]
    conds = [("rep_X", "repetition X→X", "o"), ("rep_en", "repetition en→en", "s"), ("fact_en", "factual recall, en prompt", "^")]
    xs = np.arange(len(PARTNERS))
    for ci, (cond, label, mk) in enumerate(conds):
        for tier, (fill, off) in {"30B": ("full", -0.25 + ci * 0.12), "matched": ("none", 0.1 + ci * 0.12)}.items():
            ms, los, his = [], [], []
            for X in PARTNERS:
                if tier == "30B":
                    fr, sr = f"en-{X}-fair", f"en-{X}-starved"
                else:
                    fr, sr = [p for p in A.MATCHED if p[0] == X][0][1:]
                if fr not in runs or sr not in runs:
                    ms.append(np.nan); los.append(np.nan); his.append(np.nan); continue
                if cond == "fact_en":
                    a, b = fact[fr]["en"]["settle"], fact[sr]["en"]["settle"]
                    ok = ~np.isnan(a) & ~np.isnan(b)
                    m, lo, hi = A.boot_ci(a[ok] - b[ok])
                else:
                    m, lo, hi = paired_commit(fr, sr, X, cond)
                ms.append(m); los.append(lo); his.append(hi)
            ms, los, his = map(np.array, (ms, los, his))
            ax.errorbar(xs + off, ms, yerr=[ms - los, his - ms], fmt=mk, color=COL["fair"] if tier == "30B" else "k",
                        mfc=(COL["fair"] if fill == "full" else "white"), mec=COL["fair"] if tier == "30B" else "k",
                        capsize=2, ms=6, lw=1, label=f"{label}, {'30B finals' if tier == '30B' else 'matched BPB'}")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(xs); ax.set_xticklabels([f"en-{X}\n({A.SCRIPT[X]})" for X in PARTNERS])
    ax.set_ylabel("commitment layer, fair − starved")
    ax.set_title("(a) fair decides the output language earlier")
    ax.legend(fontsize=6.5, loc="lower left", ncol=1)
    ax.grid(axis="y", alpha=.3)

    # (b) latent English trace on X->X repetition
    ax = axes[1]
    w = 0.26
    for k, (tag, off) in enumerate((("fair", -w), ("starved", 0), ("mono", w))):
        ms, los, his = [], [], []
        for X in PARTNERS:
            run = MONO[X] if tag == "mono" else f"en-{X}-{tag}"
            r = word.get(run, {}).get((X, "rep_X"))
            if not r:
                ms.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            m, lo, hi = A.boot_ci(r["latent_idx"])
            ms.append(m); los.append(lo); his.append(hi)
        ms, los, his = map(np.array, (ms, los, his))
        ax.bar(xs + off, ms, w, color=COL[tag], yerr=[ms - los, his - ms], capsize=2,
               label={"fair": "bilingual, fair", "starved": "bilingual, starved", "mono": "partner monolingual (never saw English)"}[tag])
    ax.set_xticks(xs); ax.set_xticklabels([f"en-{X}\n({A.SCRIPT[X]})" for X in PARTNERS])
    ax.set_ylabel("P(English translation) − P(other English word)\nmax over layers, X→X repetition")
    ax.set_title("(b) latent English is faint and same-script only")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=.3)

    # (c) factual accuracy by layer, partner prompts
    ax = axes[2]
    layers = np.arange(8, 17)
    for tag in ("fair", "starved"):
        curves = []
        for X in PARTNERS:
            f = fact.get(f"en-{X}-{tag}", {}).get(X)
            if f:
                c = f[f"acc_{X}"][8:17]; curves.append(c)
                ax.plot(layers, c, color=COL[tag], lw=0.8, alpha=.35)
        ax.plot(layers, np.mean(curves, 0), color=COL[tag], lw=2.5, label=f"{tag} (mean of 4 pairs)")
    ax.axhline(.25, color="k", lw=.8, ls=":")
    ax.set_xlabel("layer"); ax.set_ylabel("PolyFact accuracy")
    ax.set_title("(c) the interior knows more under fair")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_lens_summary.pdf"); fig.savefig(FIGS / "fig_lens_summary.png", dpi=150)
    plt.close(fig)


def _ar(s: str) -> str:
    """matplotlib (>= 3.7, DejaVu Sans) shapes and orders Arabic itself; do not reshape."""
    return s


def example_figure(word_en: str, ckpt_dir: Path, n_last: int = 8, layers=range(6, 17)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    items = json.load(open(A.RES / "items" / "translation.json"))["ar"]
    it = [x for x in items if x["en"] == word_en][0]
    demos = "".join(f"{R.cue('en')}{items[j]['en']} - {R.cue('ar')}{items[j]['x']}\n" for j in it["demos"])
    ctx = demos + f"{R.cue('en')}{it['en']} - {R.cue('ar').rstrip(' ')}"
    cont = R.sp("ar", it["x"])
    toks = {n: Tok(A.TOK_DIR / n) for n in ("unigram_destarved", "unigram_starved")}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, (tag, run, tn) in zip(axes, (("fair", "en-ar-fair", "unigram_destarved"), ("starved", "en-ar-starved", "unigram_starved"))):
        tok = toks[tn]
        P = load_or_build_tokmap(tok, A.RES / f"tokmap_{tn}.npz", Path("/mnt/scratch/xscript/holdout"))
        model = load_model(ckpt_dir / f"{run}.pt")
        lens = Lens(model, P)
        ids, k = encode_pair(tok, ctx, cont)
        prompt = ids[:-1]          # everything but the last answer piece, so the final column predicts the Arabic word itself
        with torch.no_grad():
            reps = lens.hidden(torch.tensor([prompt]))                  # [L+1, 1, T, D]
            T = len(prompt)
            cols = list(range(T - n_last, T))
            h = reps[:, 0, cols, :]                                     # [L+1, n, D]
            logits = lens.unembed(h)
            probs = torch.softmax(logits, -1)
            top = probs.max(-1)
        lang_of = P.argmax(1); conf = P.max(1)
        nL = len(list(layers))
        ans_from = cols.index(k - 1)          # the final ':' -- from here on the lens is predicting the answer
        for ri, l in enumerate(reversed(list(layers))):
            for ci, pos in enumerate(cols):
                t = int(top.indices[l, ci]); p = float(top.values[l, ci])
                lang = LANGS[lang_of[t]] if conf[t] > 0.5 else "other"
                color = {"en": COL["fair"], "ar": "#2ca02c"}.get(lang, "#9e9e9e")
                ax.add_patch(Rectangle((ci, ri), 1, 1, facecolor=color, alpha=0.15 + 0.75 * p, edgecolor="white"))
                s = tok.piece(t).replace("▁", "␣")
                if lang == "ar" or any("؀" <= ch <= "ۿ" for ch in s):
                    s = _ar(s)
                ax.text(ci + 0.5, ri + 0.5, s, ha="center", va="center", fontsize=8,
                        color="black" if p < 0.6 else "white")
        ax.add_patch(Rectangle((ans_from, 0), len(cols) - ans_from, nL, fill=False, edgecolor="black", lw=2.2))
        ax.set_xlim(0, len(cols)); ax.set_ylim(0, nL)
        ax.set_xticks(np.arange(len(cols)) + 0.5)
        xl = []
        for pos in cols:
            s = tok.piece(prompt[pos]).replace("▁", "␣")
            xl.append(_ar(s) if any("؀" <= ch <= "ۿ" for ch in s) else s)
        ax.set_xticklabels(xl, fontsize=8)
        ax.set_yticks(np.arange(nL) + 0.5)
        ax.set_yticklabels([f"L{l}" + (" (output)" if l == 16 else "") for l in reversed(list(layers))], fontsize=8)
        ax.set_xlabel("input token (prompt: 5 demos, then  English: heart - العربية:)".replace("heart", it["en"]))
        ax.set_title(f"en-ar {tag} tokenizer — lens argmax per layer; gold: {_ar(it['x'])} ({it['en']})")
        for sp in ax.spines.values():
            sp.set_visible(False)
        del lens, model, reps
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=COL["fair"], label="English token"), Patch(color="#2ca02c", label="Arabic token"),
                            Patch(color="#9e9e9e", label="other (punctuation / space / rare)")],
                   loc="upper left", fontsize=7, bbox_to_anchor=(0, -0.12), ncol=3)
    fig.suptitle("Logit lens on one translation prompt: colour = language of the top-1 token, shade = its probability; boxed = answer position(s)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_lens_example.pdf", bbox_inches="tight"); fig.savefig(FIGS / "fig_lens_example.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="heart")
    ap.add_argument("--ckpt-dir", default="/mnt/scratch/xscript_lens/ckpt")
    ap.add_argument("--threads", type=int, default=48)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    models = A.models_table()
    runs = A.runs_available()
    summary_figure(runs, models)
    example_figure(args.word, Path(args.ckpt_dir))
    print("wrote", FIGS / "fig_lens_summary.pdf", FIGS / "fig_lens_example.pdf")


if __name__ == "__main__" and not any(a.startswith("--") for a in sys.argv[1:] if a in ("--variants", "--capital", "--capital-partner", "--capitals", "--commit-delta", "--interior")):
    main()


# ---------------------------------------------------------------------------
# single-panel variants requested for the thesis
# ---------------------------------------------------------------------------

EXAMPLES = {"de": "day", "fr": "day", "ar": "door", "zh": "cloud"}   # single token under both tokenizers, both models correct,
                                                                     # commit layers close to the pair means


def _fonts():
    import matplotlib.font_manager as fm
    f = Path("/mnt/scratch/xscript_lens/fonts/NotoSansSC-Regular.otf")
    if f.exists():
        fm.fontManager.addfont(str(f))
    return ["DejaVu Sans", "Noto Sans CJK SC", "Droid Sans Fallback"]


def interior_figure(fact):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    layers = np.arange(8, 17)
    for tag in ("fair", "starved"):
        curves = []
        for X in PARTNERS:
            f = fact.get(f"en-{X}-{tag}", {}).get(X)
            if f:
                c = f[f"acc_{X}"][8:17]; curves.append(c)
                ax.plot(layers, c, color=COL[tag], lw=0.8, alpha=.35)
        ax.plot(layers, np.mean(curves, 0), color=COL[tag], lw=2.5, label=f"{tag} (mean of 4 pairs)")
    ax.axhline(.25, color="k", lw=.8, ls=":")
    ax.set_xlabel("layer"); ax.set_ylabel("PolyFact accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_lens_interior.pdf"); fig.savefig(FIGS / "fig_lens_interior.png", dpi=150); plt.close(fig)


def commit2_figure(word, fact, runs):
    import matplotlib.pyplot as plt
    pairs = A.bilingual_pairs(runs)
    panels = [("translation en→X", lambda run, X: word.get(run, {}).get((X, "tr_en2X"), {}).get("commit")),
              ("factual recall, X prompt", lambda run, X: (lambda f: f["settle"][~np.isnan(f["settle"])] if f else None)(fact.get(run, {}).get(X)))]
    # NOTE: fact[...]["settle"] is restricted to facts the layer-0-6 prior gets wrong (see analyze_logitlens)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, (title, get) in zip(axes, panels):
        xs = np.arange(len(pairs))
        for tag, off in (("fair", -0.18), ("starved", 0.18)):
            ms, los, his = [], [], []
            for X, fr, sr in pairs:
                v = get(fr if tag == "fair" else sr, X)
                m, lo, hi = A.boot_ci(np.asarray(v, float))
                ms.append(m); los.append(lo); his.append(hi)
            ms, los, his = map(np.array, (ms, los, his))
            ax.bar(xs + off, ms, 0.34, color=COL[tag], label=tag, yerr=[ms - los, his - ms], capsize=3)
        ax.set_xticks(xs); ax.set_xticklabels([f"en-{X}\n({A.SCRIPT[X]})" for X, _, _ in pairs])
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=.3)
        lo_y = min(b.get_height() for b in ax.patches)
        ax.set_ylim(max(0, lo_y - 3), None)
    axes[0].set_ylabel("commitment layer"); axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_lens_commit2.pdf"); fig.savefig(FIGS / "fig_lens_commit2.png", dpi=150); plt.close(fig)


def answer_example_figure(ckpt_dir: Path, layers=range(6, 17)):
    """One column per (pair, tokenizer): the lens argmax at the ANSWER position of
    an en->X translation prompt, layer by layer (output at the top)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    fam = _fonts()
    items_all = json.load(open(A.RES / "items" / "translation.json"))
    toks = {n: Tok(A.TOK_DIR / n) for n in ("unigram_destarved", "unigram_starved")}
    tokmaps = {n: load_or_build_tokmap(t, A.RES / f"tokmap_{n}.npz", Path("/mnt/scratch/xscript/holdout")) for n, t in toks.items()}
    L = list(layers); nL = len(L)
    cols = [(X, tag) for X in PARTNERS for tag in ("fair", "starved")]
    fig, ax = plt.subplots(figsize=(1.55 * len(cols) + 0.8, 0.5 * nL + 1.6))
    lang_col = {"en": COL["fair"], "de": "#2ca02c", "fr": "#2ca02c", "ar": "#2ca02c", "zh": "#2ca02c"}
    for ci, (X, tag) in enumerate(cols):
        tn = "unigram_destarved" if tag == "fair" else "unigram_starved"
        tok, P = toks[tn], tokmaps[tn]
        items = items_all[X]
        it = [x for x in items if x["en"] == EXAMPLES[X]][0]
        demos = "".join(f"{R.cue('en')}{items[j]['en']} - {R.cue(X)}{items[j]['x']}\n" for j in it["demos"])
        ctx = demos + f"{R.cue('en')}{it['en']} - {R.cue(X).rstrip(' ')}"
        ids, k = encode_pair(tok, ctx, R.sp(X, it["x"]))
        seq = ids[:-1]                     # last position predicts the answer word piece itself
        model = load_model(ckpt_dir / f"en-{X}-{tag}.pt")
        lens = Lens(model, P)
        with torch.no_grad():
            reps = lens.hidden(torch.tensor([seq]))
            probs = torch.softmax(lens.unembed(reps[:, 0, -1:, :]), -1)[:, 0]     # [L+1, V]
            top = probs.max(-1)
        lang_of, conf = P.argmax(1), P.max(1)
        for ri, l in enumerate(reversed(L)):
            t = int(top.indices[l]); p = float(top.values[l])
            lang = LANGS[lang_of[t]] if conf[t] > 0.5 else "other"
            color = lang_col.get(lang, "#9e9e9e")
            ax.add_patch(Rectangle((ci, ri), 1, 1, facecolor=color, alpha=0.15 + 0.75 * p, edgecolor="white"))
            ax.text(ci + 0.5, ri + 0.5, tok.piece(t).replace("▁", "␣"), ha="center", va="center", fontsize=8,
                    family=fam, color="black" if p < 0.6 else "white")
        ax.text(ci + 0.5, nL + 0.15, f"en-{X}\n{tag}\n{it['en']} → {it['x']}", ha="center", va="bottom", fontsize=8, family=fam)
        del lens, model, reps
    for ci in range(0, len(cols), 2):
        ax.add_patch(Rectangle((ci, 0), 2, nL, fill=False, edgecolor="black", lw=1.2))
    ax.set_xlim(0, len(cols)); ax.set_ylim(0, nL + 1.9)
    ax.set_xticks([])
    ax.set_yticks(np.arange(nL) + 0.5)
    ax.set_yticklabels([f"L{l}" + (" (output)" if l == 16 else "") for l in reversed(L)], fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(handles=[Patch(color=COL["fair"], label="English token"), Patch(color="#2ca02c", label="partner-language token"),
                       Patch(color="#9e9e9e", label="other (space / punctuation / rare)")],
              loc="upper center", fontsize=8, bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_lens_answer_examples.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_lens_answer_examples.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def main2():
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/mnt/scratch/xscript_lens/ckpt")
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--factual-tag", default="factual_all", choices=["factual", "factual_all"],
                    help="which factual sidecar to plot (default: all 2039 facts)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    A.FACT_TAG = args.factual_tag
    models, runs = A.models_table(), A.runs_available()
    report, summ = [], {}
    word = A.analyse_wordtasks(runs, models, report, summ)
    fact = A.analyse_factual(runs, models, report, summ)
    interior_figure(fact)
    commit2_figure(word, fact, runs)
    answer_example_figure(Path(args.ckpt_dir))
    print("wrote fig_lens_interior / fig_lens_commit2 / fig_lens_answer_examples")


if __name__ == "__main__" and "--variants" in sys.argv and "--interior" not in sys.argv:
    sys.argv.remove("--variants")
    main2()
    sys.exit(0)


EXAMPLE_Q = {
    "china": {"en": ("What is the capital of China?", "Beijing"),
              "de": ("Was ist die Hauptstadt von China?", "Peking"),
              "fr": ("Quelle est la capitale de la Chine ?", "Pékin"),
              "ar": ("ما هي عاصمة الصين؟", "بكين"),
              "zh": ("中国的首都是哪里？", "北京")},
}
CAPITAL_Q = {"en": ("What is the capital of France?", "Paris"),
             "de": ("Was ist die Hauptstadt von Frankreich?", "Paris"),
             "fr": ("Quelle est la capitale de la France ?", "Paris"),
             "ar": ("ما هي عاصمة فرنسا؟", "باريس"),
             "zh": ("法国的首都是哪里？", "巴黎")}


def capital_example_figure(ckpt_dir: Path, layers=range(6, 17)):
    """Lens argmax at the answer position of '<question>\\n<cue>' for the
    capital-of-France question, in English (top row) and in the partner
    language (bottom row), one column per (pair, tokenizer). The position shown
    predicts the FIRST piece of the gold answer (fair: 'Paris'/'باريس'/'巴黎';
    starved: 'Paris'/'بار'/'巴')."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from xscript.eval.logitlens import CUE
    fam = _fonts()
    toks = {n: Tok(A.TOK_DIR / n) for n in ("unigram_destarved", "unigram_starved")}
    tokmaps = {n: load_or_build_tokmap(t, A.RES / f"tokmap_{n}.npz", Path("/mnt/scratch/xscript/holdout")) for n, t in toks.items()}
    L = list(layers); nL = len(L)
    cols = [(X, tag) for X in PARTNERS for tag in ("fair", "starved")]
    fig, axes = plt.subplots(2, 1, figsize=(1.55 * len(cols) + 0.8, 2 * (0.5 * nL) + 3.2))
    lang_col = {"en": COL["fair"]}
    for ci, (X, tag) in enumerate(cols):
        tn = "unigram_destarved" if tag == "fair" else "unigram_starved"
        tok, P = toks[tn], tokmaps[tn]
        model = load_model(ckpt_dir / f"en-{X}-{tag}.pt")
        lens = Lens(model, P)
        lang_of, conf = P.argmax(1), P.max(1)
        for row, (ax, Lq) in enumerate(zip(axes, ("en", X))):
            q, gold = CAPITAL_Q[Lq]
            ids, k = encode_pair(tok, f"{q}\n{CUE[Lq]}", R.sp(Lq, gold))
            first = tok.piece(ids[k])
            with torch.no_grad():
                reps = lens.hidden(torch.tensor([ids[:k]]))
                probs = torch.softmax(lens.unembed(reps[:, 0, -1:, :]), -1)[:, 0]
                top = probs.max(-1)
                p_gold = probs[:, ids[k]]
            for ri, l in enumerate(reversed(L)):
                t = int(top.indices[l]); p = float(top.values[l])
                lang = LANGS[lang_of[t]] if conf[t] > 0.5 else "other"
                piece = tok.piece(t).replace("▁", "")
                latin = any("a" <= ch.lower() <= "z" for ch in piece)
                if latin and X in ("ar", "zh") and lang != X:
                    lang = "en"          # a Latin-script token cannot be the partner language in a cross-script pair
                color = lang_col.get(lang, "#2ca02c" if lang == X else "#9e9e9e")
                ax.add_patch(Rectangle((ci, ri), 1, 1, facecolor=color, alpha=0.15 + 0.75 * p, edgecolor="white"))
                ax.text(ci + 0.5, ri + 0.5, tok.piece(t).replace("▁", "␣"), ha="center", va="center", fontsize=8,
                        family=fam, color="black" if p < 0.6 else "white")
            head = f"en-{X}\n{tag}" + (f"\ngold piece: {first.replace('▁', '␣')}" if row == 1 else "")
            ax.text(ci + 0.5, nL + 0.15, head, ha="center", va="bottom", fontsize=8, family=fam)
        del lens, model
    for row, ax in enumerate(axes):
        for ci in range(0, len(cols), 2):
            ax.add_patch(Rectangle((ci, 0), 2, nL, fill=False, edgecolor="black", lw=1.2))
        ax.set_xlim(0, len(cols)); ax.set_ylim(0, nL + (1.4 if row == 0 else 2.2))
        ax.set_xticks([])
        ax.set_yticks(np.arange(nL) + 0.5)
        ax.set_yticklabels([f"L{l}" + (" (output)" if l == 16 else "") for l in reversed(L)], fontsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_title("prompt in English:  What is the capital of France?  Answer:", fontsize=10, loc="left")
    axes[1].set_title("prompt in the partner language (Hauptstadt von Frankreich / capitale de la France / عاصمة فرنسا / 法国的首都), localized cue",
                      fontsize=10, loc="left", family=fam)
    axes[1].legend(handles=[Patch(color=COL["fair"], label="English token (cross-script pairs: any Latin-script token)"),
                            Patch(color="#2ca02c", label="partner-language token"),
                            Patch(color="#9e9e9e", label="other / shared by en-de-fr (e.g. Paris) / punctuation")],
                   loc="upper center", fontsize=8, bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_lens_capital.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_lens_capital.png", dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__" and "--capital" in sys.argv and "--capital-partner" not in sys.argv and "--capitals" not in sys.argv and "--commit-delta" not in sys.argv and "--interior" not in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    torch.set_num_threads(48)
    capital_example_figure(Path("/mnt/scratch/xscript_lens/ckpt"))
    print("wrote fig_lens_capital")
    sys.exit(0)


LANG_NAME_EN = {"de": "German", "fr": "French", "ar": "Arabic", "zh": "Chinese"}


def capital_partner_cells(ckpt_dir: Path, layers=range(6, 17), Q=None):
    """Compute the answer-position argmax cells once (partner-language prompt only)."""
    from xscript.eval.logitlens import CUE
    Q = Q or CAPITAL_Q
    toks = {n: Tok(A.TOK_DIR / n) for n in ("unigram_destarved", "unigram_starved")}
    tokmaps = {n: load_or_build_tokmap(t, A.RES / f"tokmap_{n}.npz", Path("/mnt/scratch/xscript/holdout")) for n, t in toks.items()}
    cells = {}
    for X in PARTNERS:
        for tag in ("fair", "starved"):
            tn = "unigram_destarved" if tag == "fair" else "unigram_starved"
            tok, P = toks[tn], tokmaps[tn]
            lang_of, conf = P.argmax(1), P.max(1)
            model = load_model(ckpt_dir / f"en-{X}-{tag}.pt")
            lens = Lens(model, P)
            q, gold = Q[X]
            ids, k = encode_pair(tok, f"{q}\n{CUE[X]}", R.sp(X, gold))
            with torch.no_grad():
                reps = lens.hidden(torch.tensor([ids[:k]]))
                probs = torch.softmax(lens.unembed(reps[:, 0, -1:, :]), -1)[:, 0]
                top = probs.max(-1)
            out = []
            for l in layers:
                t = int(top.indices[l]); p = float(top.values[l])
                lang = LANGS[lang_of[t]] if conf[t] > 0.5 else "other"
                piece = tok.piece(t).replace("▁", "")
                if any("a" <= ch.lower() <= "z" for ch in piece) and X in ("ar", "zh") and lang != X:
                    lang = "en"
                out.append((tok.piece(t).replace("▁", "␣"), p, lang))
            cells[(X, tag)] = {"cells": out, "gold_piece": tok.piece(ids[k]).replace("▁", "␣")}
            del lens, model
    return cells


def capital_partner_figure(cells, font: str, suffix: str, layers=range(6, 17)):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    import matplotlib.font_manager as fm
    for f in Path("/usr/share/fonts/truetype/msttcorefonts").glob("*.ttf"):
        fm.fontManager.addfont(str(f))
    _fonts()
    fam = [font, "DejaVu Sans", "Noto Sans CJK SC"]
    plt.rcParams["font.family"] = fam
    L = list(layers); nL = len(L)
    cols = [(X, tag) for X in PARTNERS for tag in ("fair", "starved")]
    fig, ax = plt.subplots(figsize=(1.5 * len(cols) + 0.8, 0.48 * nL + 1.5))
    for ci, (X, tag) in enumerate(cols):
        for ri, (piece, p, lang) in enumerate(reversed(cells[(X, tag)]["cells"])):
            color = COL["fair"] if lang == "en" else ("#2ca02c" if lang == X else "#9e9e9e")
            ax.add_patch(Rectangle((ci, ri), 1, 1, facecolor=color, alpha=0.15 + 0.75 * p, edgecolor="white"))
            ax.text(ci + 0.5, ri + 0.5, piece, ha="center", va="center", fontsize=9, family=fam,
                    color="black" if p < 0.6 else "white")
        ax.text(ci + 0.5, nL + 0.12, tag, ha="center", va="bottom", fontsize=10, family=fam)
    for gi, X in enumerate(PARTNERS):
        ax.add_patch(Rectangle((2 * gi, 0), 2, nL, fill=False, edgecolor="black", lw=1.2))
        ax.text(2 * gi + 1, nL + 0.75, LANG_NAME_EN[X], ha="center", va="bottom", fontsize=11, family=fam)
    ax.set_xlim(0, len(cols)); ax.set_ylim(0, nL + 1.5)
    ax.set_xticks([])
    ax.set_yticks(np.arange(nL) + 0.5)
    ax.set_yticklabels([f"L{l}" + (" (output)" if l == 16 else "") for l in reversed(L)], fontsize=9, family=fam)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.legend(handles=[Patch(color=COL["fair"], label="English token"), Patch(color="#2ca02c", label="partner-language token"),
                       Patch(color="#9e9e9e", label="other (shared across languages / punctuation / rare)")],
              loc="upper center", fontsize=9, bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, prop={"family": fam, "size": 9})
    fig.tight_layout()
    fig.savefig(FIGS / f"fig_lens_capital_partner_{suffix}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"fig_lens_capital_partner_{suffix}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__" and "--capital-partner" in sys.argv and "--interior" not in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    torch.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 48)
    ex = sys.argv[sys.argv.index("--example") + 1] if "--example" in sys.argv else None
    cells = capital_partner_cells(Path("/mnt/scratch/xscript_lens/ckpt"), Q=EXAMPLE_Q.get(ex))
    fonts = (("Times New Roman", "times"), ("Arial", "arial"), ("DejaVu Sans", "dejavu")) if ex is None else (("DejaVu Sans", "dejavu"),)
    for font, suffix in fonts:
        capital_partner_figure(cells, font, suffix if ex is None else f"{ex}_{suffix}")
    print("wrote fig_lens_capital_partner_*")
    sys.exit(0)


EXAMPLE_Q.update({
    "paris": CAPITAL_Q,
    "berlin": {"en": ("What is the capital of Germany?", "Berlin"),
               "de": ("Was ist die Hauptstadt von Deutschland?", "Berlin"),
               "fr": ("Quelle est la capitale de l'Allemagne ?", "Berlin"),
               "ar": ("ما هي عاصمة ألمانيا؟", "برلين"),
               "zh": ("德国的首都是哪里？", "柏林")},
    "london": {"en": ("What is the capital of the United Kingdom?", "London"),
               "de": ("Was ist die Hauptstadt des Vereinigten Königreichs?", "London"),
               "fr": ("Quelle est la capitale du Royaume-Uni ?", "Londres"),
               "ar": ("ما هي عاصمة المملكة المتحدة؟", "لندن"),
               "zh": ("英国的首都是哪里？", "伦敦")},
    "cairo": {"en": ("What is the capital of Egypt?", "Cairo"),
              "de": ("Was ist die Hauptstadt von Ägypten?", "Kairo"),
              "fr": ("Quelle est la capitale de l'Égypte ?", "Le Caire"),
              "ar": ("ما هي عاصمة مصر؟", "القاهرة"),
              "zh": ("埃及的首都是哪里？", "开罗")},
})


def example_cells_multi(ckpt_dir: Path, specs, layers=range(6, 17)):
    """specs: list of (name, Q, prompt_lang) with prompt_lang 'partner' or 'en'.
    Loads each of the 8 bilinguals once; returns {name: cells-dict}."""
    from xscript.eval.logitlens import CUE
    toks = {n: Tok(A.TOK_DIR / n) for n in ("unigram_destarved", "unigram_starved")}
    tokmaps = {n: load_or_build_tokmap(t, A.RES / f"tokmap_{n}.npz", Path("/mnt/scratch/xscript/holdout")) for n, t in toks.items()}
    out = {name: {} for name, _, _ in specs}
    for X in PARTNERS:
        for tag in ("fair", "starved"):
            tn = "unigram_destarved" if tag == "fair" else "unigram_starved"
            tok, P = toks[tn], tokmaps[tn]
            lang_of, conf = P.argmax(1), P.max(1)
            model = load_model(ckpt_dir / f"en-{X}-{tag}.pt")
            lens = Lens(model, P)
            for name, Q, pl in specs:
                Lq = X if pl == "partner" else "en"
                q, gold = Q[Lq]
                ids, k = encode_pair(tok, f"{q}\n{CUE[Lq]}", R.sp(Lq, gold))
                with torch.no_grad():
                    reps = lens.hidden(torch.tensor([ids[:k]]))
                    probs = torch.softmax(lens.unembed(reps[:, 0, -1:, :]), -1)[:, 0]
                    top = probs.max(-1)
                cells = []
                for l in layers:
                    t = int(top.indices[l]); p = float(top.values[l])
                    lang = LANGS[lang_of[t]] if conf[t] > 0.5 else "other"
                    piece = tok.piece(t).replace("▁", "")
                    if any("a" <= ch.lower() <= "z" for ch in piece) and X in ("ar", "zh") and lang != X:
                        lang = "en"
                    cells.append((tok.piece(t).replace("▁", "␣"), p, lang))
                out[name][(X, tag)] = {"cells": cells, "gold_piece": tok.piece(ids[k]).replace("▁", "␣")}
            del lens, model
    return out


if __name__ == "__main__" and "--capitals" in sys.argv and "--interior" not in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    torch.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 48)
    specs = [(f"{n}_partner", EXAMPLE_Q[n], "partner") for n in ("berlin", "london", "cairo")] + \
            [(f"{n}_en", EXAMPLE_Q[n], "en") for n in ("paris", "china", "berlin", "london", "cairo")]
    allcells = example_cells_multi(Path("/mnt/scratch/xscript_lens/ckpt"), specs)
    for name, _, _ in specs:
        capital_partner_figure(allcells[name], "DejaVu Sans", f"{name}_dejavu")
    print("wrote", ", ".join(f"fig_lens_capital_partner_{n}_dejavu" for n, _, _ in specs))
    sys.exit(0)


def commit2_delta_figure(word, fact, runs):
    """Paired fair - starved commitment-layer differences with 95% CIs (the
    statistic the text quotes); companion to commit2_figure's marginal bars."""
    import matplotlib.pyplot as plt
    pairs = A.bilingual_pairs(runs)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for ax, title in zip(axes, ("translation en→X", "factual recall, X prompt")):
        ms, los, his = [], [], []
        for X, fr, sr in pairs:
            if title.startswith("translation"):
                m, lo, hi = paired_commit(fr, sr, X, "tr_en2X")
            else:
                a, b = fact[fr][X]["settle"], fact[sr][X]["settle"]
                ok = ~np.isnan(a) & ~np.isnan(b)
                m, lo, hi = A.boot_ci(a[ok] - b[ok])
            ms.append(m); los.append(lo); his.append(hi)
        ms, los, his = map(np.array, (ms, los, his))
        xs = np.arange(len(pairs))
        ax.bar(xs, ms, 0.5, color="#4c72b0", yerr=[ms - los, his - ms], capsize=4)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(xs); ax.set_xticklabels([f"en-{X}\n({A.SCRIPT[X]})" for X, _, _ in pairs])
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("commitment layer, fair − starved\n(paired over items, 95% CI)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_lens_commit2_delta.pdf"); fig.savefig(FIGS / "fig_lens_commit2_delta.png", dpi=150); plt.close(fig)


if __name__ == "__main__" and "--commit-delta" in sys.argv and "--interior" not in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    A.FACT_TAG = "factual_all"
    models, runs = A.models_table(), A.runs_available()
    report, summ = [], {}
    word = A.analyse_wordtasks(runs, models, report, summ)
    fact = A.analyse_factual(runs, models, report, summ)
    commit2_delta_figure(word, fact, runs)
    print("wrote fig_lens_commit2_delta")
    sys.exit(0)


def interior_figures(fact):
    """PolyFact accuracy by layer: 30B finals, BPB-matched pairs, and both side by side."""
    import matplotlib.pyplot as plt
    tiers = {"30B finals": [(X, f"en-{X}-fair", f"en-{X}-starved") for X in PARTNERS],
             "matched partner-language BPB": A.MATCHED}
    layers = np.arange(8, 17)

    def draw(ax, pairs, title=None):
        for tag in ("fair", "starved"):
            cs = []
            for X, fr, sr in pairs:
                run = fr if tag == "fair" else sr
                f = fact.get(run, {}).get(X)
                if f is None:
                    continue
                c = f[f"acc_{X}"][8:17]; cs.append(c)
                ax.plot(layers, c, color=COL[tag], lw=0.8, alpha=.35)
            ax.plot(layers, np.mean(cs, 0), color=COL[tag], lw=2.5, label=f"{tag} (mean of 4 pairs)")
        ax.axhline(.25, color="k", lw=.8, ls=":")
        ax.set_xlabel("layer"); ax.grid(alpha=.3)
        if title:
            ax.set_title(title, fontsize=10)

    for name, pairs, suffix in (("30B finals", tiers["30B finals"], ""),
                                ("matched BPB", tiers["matched partner-language BPB"], "_matched")):
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        draw(ax, pairs)
        ax.set_ylabel("PolyFact accuracy"); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / f"fig_lens_interior{suffix}.pdf"); fig.savefig(FIGS / f"fig_lens_interior{suffix}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, (name, pairs) in zip(axes, tiers.items()):
        draw(ax, pairs, name)
    axes[0].set_ylabel("PolyFact accuracy"); axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_lens_interior_both.pdf"); fig.savefig(FIGS / "fig_lens_interior_both.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__" and "--interior" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    A.FACT_TAG = "factual_all"
    models, runs = A.models_table(), A.runs_available()
    interior_figures(A.analyse_factual(runs, models, [], {}))
    print("wrote fig_lens_interior / _matched / _both")
    sys.exit(0)
