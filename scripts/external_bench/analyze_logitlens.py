#!/usr/bin/env python
"""Analyse the logit-lens sweep (CLAUDE.md 6l): curves, transition layers,
latent-language indices with matched controls, paired fair-vs-starved
bootstrap CIs, and figures.

    python analyze_logitlens.py [--scratch /mnt/scratch/xscript_lens/raw]

Reads  <scratch>/<run>_{translation,cloze,factual}.json and
       <scratch>/<run>_generation_<lang>.npz   (raw sidecars, not in git)
Writes results/logitlens/{report.md,summary.json,figs/*}

Quantities (all per layer l = 0..16; 0 = embeddings, 16 = model output):

  P_l(out), P_l(lat), P_l(ctl)   full-string probabilities of the output word,
        its translation in the other language, and the translation of a
        different word. LATENT INDEX = max_l [P_l(lat) - P_l(ctl)] over the
        interior layers, i.e. how much probability the network puts on the
        other-language rendering of the SAME concept, beyond what it puts on
        any other-language word.
  mass_l(lang)   probability mass per language from the token map.
        TRANSITION LAYER = first layer at which the output language's mass
        exceeds the other language's mass and stays above it to the output.
  factual: per-layer 4-way accuracy with prompt-language candidates and with
        other-language candidates (QID-aligned), the fraction of facts that
        were right at some interior layer but wrong at the output (Wang et
        al.'s "lost in transition"), and first-token rank curves.
  generation: at word-initial positions, split lexical / function, the share
        of layers' top-1 tokens that are (i) the token eventually emitted,
        (ii) any other token of the output language, (iii) an other-language
        token that is a MUSE translation of the word being emitted (hit),
        (iv) an other-language token that is not; plus the mass on the
        parallel other-language sentence's tokens minus a control sentence.

Fair - starved contrasts are paired over identical items; CIs are 95%
percentile bootstraps over items (B=2000).
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from xscript.tok.wrapper import Tok                                      # noqa: E402
from xscript.eval.logitlens import LANGS, OTHER, token_byte_spans         # noqa: E402

RES = REPO / "results" / "logitlens"
RAW = Path("/mnt/scratch/xscript_lens/raw")
FIGS = RES / "figs"
TOK_DIR = Path("/mnt/scratch/xscript/tokenizers")
DATA = Path("/mnt/scratch/xscript_lens/data")
PARTNERS = ("de", "fr", "ar", "zh")
SCRIPT = {"de": "same", "fr": "same", "ar": "cross", "zh": "cross"}
LI = {l: i for i, l in enumerate(LANGS)}
NL = 17
B = 2000
RNG = np.random.default_rng(0)

STOP = {
    "en": "the a an of to in and or is are was were be been being for on at by with from as that this these those it its he she they we you i his her their our your not no but if then than so do does did have has had will would can could may might shall should there here what which who whom whose when where why how all any some each every one two into over under out up down about after before between through during also very just more most much many such only own same other another".split(),
    "de": "der die das des dem den ein eine einer eines einem einen und oder ist sind war waren sein wird werden wurde wurden hat haben hatte hatten nicht kein keine zu in im an am auf aus bei mit nach von vom für über unter vor durch gegen ohne um bis als wie dass ob wenn weil da so auch noch nur schon sehr mehr er sie es wir ihr ich du man sich dieser diese dieses jener jene welche welcher was wer wo wann warum dort hier alle jeder jede jedes viele einige".split(),
    "fr": "le la les l un une des du de d et ou est sont était étaient être sera seront a ont avait avaient ne pas plus que qui quoi dont où quand comment pourquoi ce cet cette ces il elle ils elles on nous vous je tu me te se y en à au aux dans sur sous par pour avec sans chez vers entre depuis pendant après avant comme si mais donc car ni très aussi encore tout tous toute toutes chaque quelque quelques autre autres même".split(),
    "ar": "في من على إلى عن أن إن أو و ثم لا لم لن ما ماذا كيف متى أين لماذا هذا هذه ذلك تلك هو هي هم هن نحن أنا أنت التي الذي الذين كان كانت يكون تكون كانوا قد لقد بعد قبل بين حتى عند مع كل بعض غير ليس إذا لو كما مثل أيضا فقط جدا هناك هنا".split(),
    "zh": "的 了 是 在 和 与 及 或 也 都 就 而 但 并 被 把 从 对 向 以 为 于 之 其 这 那 我 你 他 她 它 我们 你们 他们 她们 它们 这个 那个 这些 那些 什么 怎么 为什么 哪 哪里 吗 呢 吧 啊 不 没 没有 有 会 能 可以 要 应该 已经 还 又 再 很 更 最 非常 一 一个 上 下 中 里 后 前 时 所 等 着 过 地 得 让 使 因为 所以 如果 虽然 但是 而且 然后".split(),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def boot_ci(x: np.ndarray, fn=np.mean, b: int = B) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(x), (b, len(x)))
    s = np.array([fn(x[i]) for i in idx])
    return float(fn(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def fmt(m, lo, hi, d=3):
    star = "*" if (lo > 0 or hi < 0) else ""
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]{star}"


def transition_layer(a: np.ndarray, b: np.ndarray, frac: float = 0.5) -> float:
    """First layer from which a > b AND a >= frac * a[output] both hold through
    to the output; nan if never. The floor stops the rule firing in early
    layers where both quantities are ~0."""
    ok = (a > b) & (a >= frac * a[-1])
    for l in range(len(a)):
        if ok[l:].all():
            return float(l)
    return float("nan")


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def models_table() -> dict:
    return json.load(open(REPO / "results" / "models.json"))


def runs_available() -> list[str]:
    return sorted({p.name.split("_")[0] for p in RAW.glob("*_translation.json") if "limit" not in p.name})


# BPB-matched intermediate pairs from CLAUDE.md 6k (|d bpb_partner| <= 0.01, both mid-stable):
# the fair checkpoint is the WEAKER-trained one, so a fair advantage that survives here is not capability.
MATCHED = [("de", "en-de-fair-5b", "en-de-starved-23b"), ("fr", "en-fr-fair-10b", "en-fr-starved-23b"),
           ("ar", "en-ar-fair-10b", "en-ar-starved-23b"), ("zh", "en-zh-fair-10b", "en-zh-starved-23b")]


def matched_pairs(runs):
    return [(X, f, s) for X, f, s in MATCHED if f in runs and s in runs]


def bilingual_pairs(runs):
    out = []
    for X in PARTNERS:
        f, s = f"en-{X}-fair", f"en-{X}-starved"
        if f in runs and s in runs:
            out.append((X, f, s))
    return out


# ---------------------------------------------------------------------------
# translation / cloze
# ---------------------------------------------------------------------------

def load_translation(run: str) -> dict:
    return json.load(open(RAW / f"{run}_translation.json"))["data"]


def load_cloze(run: str) -> dict:
    p = RAW / f"{run}_cloze.json"
    return json.load(open(p))["data"] if p.exists() else {}


_TOKS = {}


def single_both(word: str) -> bool:
    """One piece under BOTH tokenizers (recomputed here; the flag stored by
    early runs used a space-prefixed encoding that is never a single piece)."""
    from xscript.eval.logitlens import single_token
    if not _TOKS:
        for n in ("unigram_destarved", "unigram_starved"):
            _TOKS[n] = Tok(TOK_DIR / n)
    return all(single_token(t, word) for t in _TOKS.values())


def item_arrays(rows: list[dict], only_correct: bool, first_token: bool = False) -> dict:
    """Stack per-item per-layer probabilities for out/lat/ctl and language mass."""
    sel = [r for r in rows if (r["out"]["greedy_ok"] or not only_correct)]
    if first_token:
        sel = [r for r in sel if single_both(r["out"]["word"]) and single_both(r["lat"]["word"])]
    if not sel:
        return {}
    key = "first_lp" if first_token else "full_lp"
    P = {role: np.exp(np.array([r[role][key] for r in sel])) for role in ("out", "lat", "ctl")}
    mass = np.array([r["out"]["mass"] for r in sel])                     # [n, L, 6]
    return {"n": len(sel), "n_all": len(rows), **P, "mass": mass,
            "acc": np.mean([r["out"]["greedy_ok"] for r in rows])}


def commit_layer(P: np.ndarray, frac: float = 0.5, absolute: bool = False,
                 first_crossing: bool = False) -> np.ndarray:
    """Per item: first layer FROM WHICH P(output string) stays at or above `frac`
    of its OWN output-layer value through to the output (always defined, <= 16).

    Persistence, not first crossing: a curve that reaches the threshold at layer
    10, falls back at 12 and recovers at 14 has committed at 14, not at 10. This
    is the same rule the factual `settle` layer uses, so the two tasks in
    fig_lens_commit2 are measured the same way. Empirically the distinction is
    small (0-7% of items dip back, mean displacement <= 0.13 layers, deltas move
    by <= 0.08), but the persistent version is the claim actually being made.
    `first_crossing=True` restores the weaker rule.

    The earlier ABSOLUTE rule (P >= 0.5, censored to 17 when never reached)
    conflated deciding LATE with deciding with LOW final confidence -- 68% of
    en-ar-starved translations end below 0.5 -- and produced a phantom
    'layer 17'; `absolute=True` keeps it for comparison."""
    hit = (P >= frac) if absolute else (P >= frac * P[:, -1:])
    if absolute:
        if first_crossing:
            return np.where(hit.any(1), hit.argmax(1), P.shape[1]).astype(float)
        out = np.full(len(P), P.shape[1], float)
        for i in range(len(P)):
            if hit[i, -1]:
                l = P.shape[1] - 1
                while l > 0 and hit[i, l - 1]:
                    l -= 1
                out[i] = l
        return out
    if first_crossing:
        return hit.argmax(1).astype(float)
    out = np.full(len(P), P.shape[1] - 1, float)
    for i in range(len(P)):
        l = P.shape[1] - 1
        while l > 0 and hit[i, l - 1]:
            l -= 1
        out[i] = l
    return out


def lang_commit_layer(mass: np.ndarray, out_lang: str, oth_lang: str, frac: float = 0.5) -> np.ndarray:
    """LANGUAGE commitment, as opposed to output-STRING commitment: per item, the
    first layer from which the lens distribution's mass on the output language
    minus its mass on the other language stays at or above `frac` of that gap's
    output-layer value.

    Two things this fixes. (1) `commit_layer` measures P(the answer string), which
    bundles choosing the concept with choosing the language; this measures only
    the language of the whole next-token distribution. (2) A raw "first layer at
    which out-language mass exceeds other-language mass" rule is NOT usable: at
    layers 0-8 the distribution is a generic prior that barely depends on the
    prompt (identical to two decimals whether the model is asked to answer in X
    or in English), and where that prior sits differs by model (en-ar-fair holds
    0.38 English mass at L2, en-ar-starved 0.13), so the crossing layer partly
    measures the prior. Taking the midpoint of the GAP cancels a constant offset.

    mass: [n, n_layers+1, len(LANGS)+1] at the answer position.
    """
    g = mass[:, :, LI[out_lang]] - mass[:, :, LI[oth_lang]]
    hit = g >= frac * g[:, -1:]
    out = np.full(len(g), g.shape[1] - 1, float)
    for i in range(len(g)):
        l = g.shape[1] - 1
        while l > 0 and hit[i, l - 1]:
            l -= 1
        out[i] = l
    return out


def latent_index(P_lat: np.ndarray, P_ctl: np.ndarray, lo: int = 1, hi: int = 15) -> np.ndarray:
    """Per item: max over interior layers of P(lat)-P(ctl)."""
    d = P_lat - P_ctl
    return d[:, lo:hi + 1].max(1)


def analyse_wordtasks(runs, models, report, summary):
    """Translation + cloze curves, latent indices, transition layers, fair-starved deltas."""
    report.append("\n## Word tasks (Wendler et al.): translation, repetition, cloze\n")
    report.append("P = full-string probability under the layer-l lens, averaged over items the "
                  "model gets right at the output (greedy). `latent` = translation of the output "
                  "word in the other language; `ctl` = translation of a different word. "
                  "Latent index = max over layers 1-15 of P(latent) - P(ctl) per item, then mean. "
                  "Commit layer = first layer FROM WHICH P(output string) stays at or above half of its output-layer value.\n")
    per_model = {}
    for run in runs:
        langs = models[run]["langs"]
        d = load_translation(run)
        cz = load_cloze(run)
        per_model[run] = {}
        for X, block in d.items():
            for cond, rows in block["conds"].items():
                A = item_arrays(rows, only_correct=True)
                if not A:
                    continue
                out_lang = X if cond in ("tr_en2X", "rep_X") else "en"
                oth = "en" if out_lang == X else X
                m = A["mass"].mean(0)
                per_model[run][(X, cond)] = {
                    "n": A["n"], "n_all": A["n_all"], "acc": A["acc"],
                    "P_out": A["out"].mean(0), "P_lat": A["lat"].mean(0), "P_ctl": A["ctl"].mean(0),
                    "latent_idx": latent_index(A["lat"], A["ctl"]),
                    "latent_peak_layer": int(np.argmax((A["lat"] - A["ctl"]).mean(0)[1:16]) + 1),
                    "commit": commit_layer(A["out"]),
                    "lang_commit": lang_commit_layer(A["mass"], out_lang, oth),
                    "mass_out": m[:, LI[out_lang]], "mass_oth": m[:, LI[oth]], "mass_other": m[:, OTHER],
                    "share_out": m[:, LI[out_lang]] / np.maximum(1 - m[:, OTHER], 1e-6),
                    "share_oth": m[:, LI[oth]] / np.maximum(1 - m[:, OTHER], 1e-6),
                    "t_layer_mass": transition_layer(m[:, LI[out_lang]], m[:, LI[oth]]),
                    "t_layer_word": transition_layer(A["out"].mean(0), A["lat"].mean(0)),
                    "items": A,
                }
                # single-token replication (first-token probabilities, single under both tokenizers)
                A1 = item_arrays(rows, only_correct=True, first_token=True)
                if A1:
                    per_model[run][(X, cond)]["single"] = {"n": A1["n"], "P_out": A1["out"].mean(0),
                                                            "P_lat": A1["lat"].mean(0), "P_ctl": A1["ctl"].mean(0),
                                                            "latent_idx": latent_index(A1["lat"], A1["ctl"]),
                                                            "commit": commit_layer(A1["out"])}
        for key, block in cz.items():
            rows = block["items"]
            A = item_arrays(rows, only_correct=True)
            if not A:
                continue
            L, lat = block["lang"], block["lat_lang"]
            m = A["mass"].mean(0)
            X = L if L != "en" else lat
            per_model[run][(X, f"cloze_{L}")] = {
                "n": A["n"], "n_all": A["n_all"], "acc": A["acc"],
                "P_out": A["out"].mean(0), "P_lat": A["lat"].mean(0), "P_ctl": A["ctl"].mean(0),
                "latent_idx": latent_index(A["lat"], A["ctl"]),
                "latent_peak_layer": int(np.argmax((A["lat"] - A["ctl"]).mean(0)[1:16]) + 1),
                "commit": commit_layer(A["out"]),
                "lang_commit": lang_commit_layer(A["mass"], L, lat),
                "mass_out": m[:, LI[L]], "mass_oth": m[:, LI[lat]], "mass_other": m[:, OTHER],
                "share_out": m[:, LI[L]] / np.maximum(1 - m[:, OTHER], 1e-6),
                "share_oth": m[:, LI[lat]] / np.maximum(1 - m[:, OTHER], 1e-6),
                "t_layer_mass": transition_layer(m[:, LI[L]], m[:, LI[lat]]),
                "t_layer_word": transition_layer(A["out"].mean(0), A["lat"].mean(0)),
                "items": A,
            }

    conds = ["tr_en2X", "tr_X2en", "rep_X", "rep_en", "cloze_X", "cloze_en"]
    # --- table 1: accuracy + latent index + transition layers, bilinguals
    report.append("### Bilingual finals: accuracy, latent index, transition layer\n")
    report.append("| model | cond | n ok/all | acc | latent idx [95% CI] | peak L | P(lat) peak | commit L (stays >= half of final P) mean [CI] | transition L (mass) | transition L (word) |")
    report.append("|---|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        if not run.startswith("en-") or run.count("-") != 2:
            continue
        X = run.split("-")[1]
        for cond in conds:
            c = cond.replace("cloze_X", f"cloze_{X}")
            r = per_model[run].get((X, c))
            if not r:
                continue
            m, lo, hi = boot_ci(r["latent_idx"])
            cm, clo, chi = boot_ci(r["commit"])
            report.append(f"| {run} | {cond} | {r['n']}/{r['n_all']} | {r['acc']:.2f} | {fmt(m, lo, hi)} | "
                          f"{r['latent_peak_layer']} | {r['P_lat'].max():.3f} | {cm:.2f} [{clo:.2f}, {chi:.2f}] | {r['t_layer_mass']:.0f} | {r['t_layer_word']:.0f} |")
    # --- table 1b: Wendler-faithful first-token replication on words that are ONE token under BOTH tokenizers
    report.append("\n### First-token replication (Wendler-faithful): words that are a single token under BOTH tokenizers\n")
    report.append("| model | cond | n | P1(out) peak (L) | P1(lat) peak (L) | latent idx (first token) [CI] | commit L (stays >= half of final P1) mean |")
    report.append("|---|---|---|---|---|---|---|")
    for run in runs:
        for (X, c), r in sorted(per_model[run].items()):
            s1 = r.get("single")
            if not s1 or s1["n"] < 5:
                continue
            m, lo, hi = boot_ci(s1["latent_idx"])
            report.append(f"| {run} | {c} (en-{X}) | {s1['n']} | {s1['P_out'].max():.3f} ({s1['P_out'].argmax()}) | "
                          f"{s1['P_lat'].max():.3f} ({s1['P_lat'].argmax()}) | {fmt(m, lo, hi)} | {s1['commit'].mean():.2f} |")
    # --- table 2: controls (monolinguals)
    report.append("\n### Monolingual controls (the partner-language monolingual never saw English; the English monolingual never saw the partner)\n")
    report.append("| model | pair | cond | n ok/all | acc | latent idx [95% CI] | P(lat) peak | P(ctl) peak |")
    report.append("|---|---|---|---|---|---|---|---|")
    for run in runs:
        if run.startswith("en-") and run.count("-") == 2:
            continue
        for (X, c), r in sorted(per_model[run].items()):
            m, lo, hi = boot_ci(r["latent_idx"])
            report.append(f"| {run} | en-{X} | {c} | {r['n']}/{r['n_all']} | {r['acc']:.2f} | {fmt(m, lo, hi)} | "
                          f"{r['P_lat'].max():.3f} | {r['P_ctl'].max():.3f} |")
    # --- table 3: fair - starved, paired over items
    report.append("\n### Fair - starved (paired over identical items; both models' correct items)\n")
    report.append("`d commit L, output STRING` = the answer string reaches half its final probability; "
                  "`d commit L, output LANGUAGE` = the distribution's out-language-minus-other-language mass "
                  "reaches half its final value (concept choice removed).\n")
    report.append("| partner | script | cond | n | d latent idx [95% CI] | d commit L, output STRING [CI] | d commit L, output LANGUAGE [CI] | d commit L, single-token words [CI] (n) | d acc |")
    report.append("|---|---|---|---|---|---|---|---|---|")
    deltas = defaultdict(dict)
    for X, fr, sr in bilingual_pairs(runs):
        for cond in conds:
            c = cond.replace("cloze_X", f"cloze_{X}")
            a, b = per_model[fr].get((X, c)), per_model[sr].get((X, c))
            if not (a and b):
                continue
            # pair on item index within the ORIGINAL row order: rebuild with all items, mask correct-in-both
            ra = load_rows(fr, X, c); rb = load_rows(sr, X, c)
            ok = np.array([x["out"]["greedy_ok"] and y["out"]["greedy_ok"] for x, y in zip(ra, rb)])
            if ok.sum() < 5:
                continue
            la = latent_index(np.exp([x["lat"]["full_lp"] for x in ra]), np.exp([x["ctl"]["full_lp"] for x in ra]))[ok]
            lb = latent_index(np.exp([x["lat"]["full_lp"] for x in rb]), np.exp([x["ctl"]["full_lp"] for x in rb]))[ok]
            m, lo, hi = boot_ci(la - lb)
            dt = a["t_layer_mass"] - b["t_layer_mass"]
            Pa = np.exp([x["out"]["full_lp"] for x in ra]); Pb = np.exp([x["out"]["full_lp"] for x in rb])
            ca, cb = commit_layer(Pa)[ok], commit_layer(Pb)[ok]
            cm, clo, chi = boot_ci(ca - cb)
            sing = np.array([single_both(x["out"]["word"]) for x in ra]) & ok
            if sing.sum() >= 5:
                P1a = np.exp([x["out"]["first_lp"] for x in ra]); P1b = np.exp([x["out"]["first_lp"] for x in rb])
                sm, slo, shi = boot_ci(commit_layer(P1a)[sing] - commit_layer(P1b)[sing])
                sing_s = f"{fmt(sm, slo, shi, 2)} ({sing.sum()})"
            else:
                sing_s = "-"
            # language commitment must be recomputed over ALL rows so the paired mask applies
            out_lang = X if cond in ("tr_en2X", "rep_X") or cond == f"cloze_{X}" else "en"
            oth_lang = "en" if out_lang == X else X
            Ma = np.array([x["out"]["mass"] for x in ra]); Mb = np.array([y["out"]["mass"] for y in rb])
            lm, llo, lhi = boot_ci(lang_commit_layer(Ma, out_lang, oth_lang)[ok]
                                   - lang_commit_layer(Mb, out_lang, oth_lang)[ok])
            report.append(f"| {X} | {SCRIPT[X]} | {cond} | {ok.sum()} | {fmt(m, lo, hi)} | {fmt(cm, clo, chi, 2)} | {fmt(lm, llo, lhi, 2)} | {sing_s} | {a['acc']-b['acc']:+.2f} |")
            deltas[X][cond] = {"d_latent": [m, lo, hi], "d_commit": [cm, clo, chi], "d_lang_commit": [lm, llo, lhi], "d_t_mass": dt}
    if matched_pairs(runs):
        report.append("\n### Fair - starved at MATCHED partner-language BPB (6k pairs; fair is the less-trained checkpoint)\n")
        report.append("| partner | script | pair | cond | n | d latent idx [CI] | d commit L, full string [CI] | d commit L, single-token words [CI] (n) | d acc |")
        report.append("|---|---|---|---|---|---|---|---|---|")
        for X, fr, sr in matched_pairs(runs):
            for cond in ("tr_en2X", "tr_X2en", "rep_X", "rep_en"):
                a, b = per_model[fr].get((X, cond)), per_model[sr].get((X, cond))
                if not (a and b):
                    continue
                ra = load_rows(fr, X, cond); rb = load_rows(sr, X, cond)
                ok = np.array([x["out"]["greedy_ok"] and y["out"]["greedy_ok"] for x, y in zip(ra, rb)])
                if ok.sum() < 5:
                    continue
                la = latent_index(np.exp([x["lat"]["full_lp"] for x in ra]), np.exp([x["ctl"]["full_lp"] for x in ra]))[ok]
                lb = latent_index(np.exp([x["lat"]["full_lp"] for x in rb]), np.exp([x["ctl"]["full_lp"] for x in rb]))[ok]
                m, lo, hi = boot_ci(la - lb)
                Pa = np.exp([x["out"]["full_lp"] for x in ra]); Pb = np.exp([x["out"]["full_lp"] for x in rb])
                cm, clo, chi = boot_ci(commit_layer(Pa)[ok] - commit_layer(Pb)[ok])
                sing = np.array([single_both(x["out"]["word"]) for x in ra]) & ok
                if sing.sum() >= 5:
                    P1a = np.exp([x["out"]["first_lp"] for x in ra]); P1b = np.exp([x["out"]["first_lp"] for x in rb])
                    sm, slo, shi = boot_ci(commit_layer(P1a)[sing] - commit_layer(P1b)[sing])
                    sing_s = f"{fmt(sm, slo, shi, 2)} ({sing.sum()})"
                else:
                    sing_s = "-"
                report.append(f"| {X} | {SCRIPT[X]} | {fr} vs {sr} | {cond} | {ok.sum()} | {fmt(m, lo, hi)} | {fmt(cm, clo, chi, 2)} | {sing_s} | {a['acc']-b['acc']:+.2f} |")
    summary["wordtasks"] = {run: {f"{X}|{c}": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                              for k, v in r.items() if k not in ("items", "single")}
                                  for (X, c), r in pm.items()} for run, pm in per_model.items()}
    summary["wordtasks_deltas"] = deltas
    return per_model


def load_rows(run, X, cond):
    if cond.startswith("cloze_"):
        L = cond.split("_")[1]
        cz = load_cloze(run)
        key = f"{L}|lat={'en' if L != 'en' else X}"
        return cz[key]["items"]
    return load_translation(run)[X]["conds"][cond]


# ---------------------------------------------------------------------------
# factual
# ---------------------------------------------------------------------------

FACT_TAG = "factual"      # or "factual_all" (all 2039 PolyFact facts) via --factual-tag


def analyse_factual(runs, models, report, summary):
    report.append(f"\n## Factual recall (Wang et al.): PolyFact, {'all 2039' if FACT_TAG == 'factual_all' else '800'} facts, 4 QID-aligned candidates\n")
    report.append("`acc_L` = 4-way accuracy at layer L scoring the candidates written in the PROMPT language; "
                  "`acc_other` = the same four entities written in the OTHER language (English for a partner "
                  "prompt, the partner for an English prompt). `lost` = facts right at some layer 8-15 (prompt-"
                  "language candidates) but wrong at the output, as a share of facts right somewhere in 8-16. "
                  "`latent other` = mean over layers 8-15 of log P(gold English answer) - log P(mismatched-fact English answer), in nats. `settle L` = first layer from which the 4-way prediction stays right, over facts the layer-0-6 prior gets wrong.\n")
    per = {}
    for run in runs:
        p = RAW / f"{run}_{FACT_TAG}.json"
        if not p.exists():
            continue
        d = json.load(open(p))["data"]
        per[run] = {}
        for L, block in d.items():
            items = block["items"]
            gold = np.array([it["gold"] for it in items])
            res = {"n": len(items)}
            for lang in [L] + block["other_langs"]:
                lp = np.array([[c["full_lp"] for c in it["cands"][lang]] for it in items])   # [n, 4, L]
                pred = lp.argmax(1)                                                          # [n, L]
                correct = pred == gold[:, None]
                res[f"acc_{lang}"] = correct.mean(0)
                res[f"correct_{lang}"] = correct
                lp1 = np.array([[c["first_lp"] for c in it["cands"][lang]] for it in items])
                res[f"acc1_{lang}"] = (lp1.argmax(1) == gold[:, None]).mean(0)
                rk = np.array([it["cands"][lang][g]["first_rank"] for it, g in zip(items, gold)])
                res[f"rank_{lang}"] = np.median(rk, 0)
                res[f"logrank_{lang}"] = np.log10(rk + 1).mean(0)
            c = res[f"correct_{L}"]
            # settle layer: first layer from which the prompt-language 4-way prediction is right through to the
            # output -- restricted to facts the PRIOR gets wrong (wrong at every layer 0-6). In the early layers the
            # lens is a frequency prior over candidate names, so a fact whose gold name is the a-priori favourite
            # looks "settled" from layer 0 (25-41% of facts) and would swamp the retrieval signal (6e's label prior).
            ok_out = c[:, 16] & ~c[:, :7].any(1)
            settle = np.full(len(c), np.nan)
            for i in np.where(ok_out)[0]:
                l = 16
                while l > 0 and c[i, l - 1]:
                    l -= 1
                settle[i] = l
            res["settle"] = settle
            somewhere = c[:, 8:].any(1)
            res["lost"] = float((c[:, 8:16].any(1) & ~c[:, 16])[somewhere].mean()) if somewhere.any() else np.nan
            res["lost_items"] = (c[:, 8:16].any(1) & ~c[:, 16])[somewhere]
            # counterpart: wrong at EVERY interior layer 8-15 but right at the output
            res["gained"] = float((~c[:, 8:16].any(1) & c[:, 16])[somewhere].mean()) if somewhere.any() else np.nan
            oth = block["other_langs"][0]
            co = res[f"correct_{oth}"]
            res["lost_via_other"] = float((co[:, 8:16].any(1) & ~c[:, 16])[somewhere].mean()) if somewhere.any() else np.nan
            mass = np.array([it["mass"] for it in items]).mean(0)
            res["mass"] = mass
            res["share_out"] = mass[:, LI[L]] / np.maximum(1 - mass[:, OTHER], 1e-6)
            res["share_oth"] = mass[:, LI[oth]] / np.maximum(1 - mass[:, OTHER], 1e-6)
            res["t_layer_mass"] = transition_layer(mass[:, LI[L]], mass[:, LI[oth]])
            # latent English/other-language factual signal vs mismatched control
            if L != "en":
                pg = np.exp(np.array([it["cands"]["en"][g]["full_lp"] for it, g in zip(items, gold)]))
                pc = np.exp(np.array([it["ctl"]["full_lp"] for it in items]))
                lg = np.array([it["cands"]["en"][g]["full_lp"] for it, g in zip(items, gold)])
                lc = np.array([it["ctl"]["full_lp"] for it in items])
                res["latent_items"] = (lg - lc)[:, 8:16].mean(1)          # nats, gold-en minus mismatched-en
                res["P_gold_en"], res["P_ctl_en"] = pg.mean(0), pc.mean(0)
            per[run][L] = res
    report.append("| model | prompt | n | acc_out | acc_other_out | acc_L8 / L12 / L14 | acc_other L8 / L12 / L14 | lost | gained | lost-but-other-right | latent other (nats) [CI] | settle L mean |")
    report.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run, d in per.items():
        for L, r in d.items():
            oth = [k for k in r if k.startswith("acc_") and k != f"acc_{L}" and not k.startswith("acc_1")][0]
            a, o = r[f"acc_{L}"], r[oth]
            lat = fmt(*boot_ci(r["latent_items"])) if "latent_items" in r else "-"
            report.append(f"| {run} | {L} | {r['n']} | {a[16]:.3f} | {o[16]:.3f} | {a[8]:.3f} / {a[12]:.3f} / {a[14]:.3f} | "
                          f"{o[8]:.3f} / {o[12]:.3f} / {o[14]:.3f} | {r['lost']:.3f} | {r['gained']:.3f} | {r['lost_via_other']:.3f} | {lat} | {np.nanmean(r['settle']):.2f} |")
    report.append("\n### Fair - starved (paired over facts)\n")
    report.append("| partner | script | prompt | d acc_out | d best interior acc | d lost [CI] | d latent other [CI] | d transition L | d settle L [CI] (n) |")
    report.append("|---|---|---|---|---|---|---|---|---|")
    deltas = defaultdict(dict)
    for X, fr, sr in bilingual_pairs(runs):
        for L in (X, "en"):
            if fr not in per or sr not in per:
                continue
            a, b = per[fr][L], per[sr][L]
            ca, cb = a[f"correct_{L}"], b[f"correct_{L}"]
            dl = fmt(*boot_ci(a["lost_items"].astype(float))) if False else ""
            # lost: paired over facts right somewhere in both
            sa = ca[:, 8:].any(1); sb = cb[:, 8:].any(1); both = sa & sb
            la = (ca[:, 8:16].any(1) & ~ca[:, 16])[both].astype(float)
            lb = (cb[:, 8:16].any(1) & ~cb[:, 16])[both].astype(float)
            m, lo, hi = boot_ci(la - lb)
            if "latent_items" in a:
                m2, lo2, hi2 = boot_ci(a["latent_items"] - b["latent_items"])
                lat = fmt(m2, lo2, hi2)
            else:
                lat = "-"
            report.append(f"| {X} | {SCRIPT[X]} | {L} | {a[f'acc_{L}'][16]-b[f'acc_{L}'][16]:+.3f} | "
                          f"{a[f'acc_{L}'][8:16].max()-b[f'acc_{L}'][8:16].max():+.3f} | {fmt(m, lo, hi)} | {lat} | "
                          f"{a['t_layer_mass']-b['t_layer_mass']:+.0f} |")
            both_ok = ~np.isnan(a["settle"]) & ~np.isnan(b["settle"])
            sm, slo, shi = boot_ci(a["settle"][both_ok] - b["settle"][both_ok])
            report[-1] = report[-1] + f" {fmt(sm, slo, shi, 2)} ({both_ok.sum()}) |"
            deltas[X][L] = {"d_acc": float(a[f"acc_{L}"][16] - b[f"acc_{L}"][16]), "d_lost": [m, lo, hi],
                            "d_settle": [sm, slo, shi]}
    if matched_pairs(runs):
        report.append("\n### Fair - starved at MATCHED partner-language BPB (6k pairs)\n")
        report.append("| partner | script | pair | prompt | d acc_out | d acc_other_out | d best interior acc | d lost [CI] | d latent other [CI] | d settle L [CI] (n) |")
        report.append("|---|---|---|---|---|---|---|---|---|---|")
        for X, fr, sr in matched_pairs(runs):
            for L in (X, "en"):
                if fr not in per or sr not in per:
                    continue
                a, b = per[fr][L], per[sr][L]
                oth = "en" if L != "en" else X
                ca, cb = a[f"correct_{L}"], b[f"correct_{L}"]
                both = ca[:, 8:].any(1) & cb[:, 8:].any(1)
                la = (ca[:, 8:16].any(1) & ~ca[:, 16])[both].astype(float)
                lb = (cb[:, 8:16].any(1) & ~cb[:, 16])[both].astype(float)
                m, lo, hi = boot_ci(la - lb)
                lat = fmt(*boot_ci(a["latent_items"] - b["latent_items"])) if "latent_items" in a else "-"
                both_ok = ~np.isnan(a["settle"]) & ~np.isnan(b["settle"])
                sm, slo, shi = boot_ci(a["settle"][both_ok] - b["settle"][both_ok])
                report.append(f"| {X} | {SCRIPT[X]} | {fr} vs {sr} | {L} | {a[f'acc_{L}'][16]-b[f'acc_{L}'][16]:+.3f} | "
                              f"{a[f'acc_{oth}'][16]-b[f'acc_{oth}'][16]:+.3f} | {a[f'acc_{L}'][8:16].max()-b[f'acc_{L}'][8:16].max():+.3f} | "
                              f"{fmt(m, lo, hi)} | {lat} | {fmt(sm, slo, shi, 2)} ({both_ok.sum()}) |")
    summary["factual"] = {run: {L: {k: (v.tolist() if isinstance(v, np.ndarray) and v.ndim == 1 else v)
                                    for k, v in r.items() if not k.startswith("correct_") and k not in ("lost_items", "latent_items", "settle")}
                                for L, r in d.items()} for run, d in per.items()}
    summary["factual_deltas"] = deltas
    return per


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def muse_inverse(pair: str) -> dict[str, set[str]]:
    """x_word -> {en words} from MUSE en-x (and en -> {x words} via muse_forward)."""
    d: dict[str, set[str]] = {}
    for line in open(DATA / "muse" / f"{pair}.txt", encoding="utf-8"):
        parts = line.rstrip("\n").split("\t") if "\t" in line else line.split()
        if len(parts) >= 2:
            d.setdefault(parts[1].strip().lower(), set()).add(parts[0].strip().lower())
    return d


def muse_forward(pair: str) -> dict[str, set[str]]:
    d: dict[str, set[str]] = {}
    for line in open(DATA / "muse" / f"{pair}.txt", encoding="utf-8"):
        parts = line.rstrip("\n").split("\t") if "\t" in line else line.split()
        if len(parts) >= 2:
            d.setdefault(parts[0].strip().lower(), set()).add(parts[1].strip().lower())
    return d


def is_script(word: str, lang: str) -> bool:
    letters = [ch for ch in word if unicodedata.category(ch)[0] == "L"]
    if not letters:
        return False
    if lang == "ar":
        return all("ARABIC" in unicodedata.name(ch, "") for ch in letters)
    if lang == "zh":
        return all("CJK" in unicodedata.name(ch, "") for ch in letters)
    return all("LATIN" in unicodedata.name(ch, "") for ch in letters)


def words_with_spans(text: str, lang: str) -> list[tuple[int, int, str]]:
    """(byte_start, byte_end, word) over b' '+text, words stripped of punctuation."""
    base = b" " + text.encode("utf-8")
    out = []
    if lang == "zh":
        import jieba
        jieba.setLogLevel(60)
        pos = 1
        for w in jieba.cut(text):
            wb = w.encode("utf-8")
            core = w.strip(" \t\n　，。、；：？！“”‘’（）《》〈〉【】…—-")
            if core:
                out.append((pos, pos + len(wb), core))
            pos += len(wb)
        return out
    pos = 0
    for tokw in text.split(" "):
        start = pos + 1                       # +1 for the leading space of base
        wb = tokw.encode("utf-8")
        core = tokw.strip(".,;:!?\"'()[]{}«»“”‘’…-–—/")
        if core:
            lead = len(tokw) - len(tokw.lstrip(".,;:!?\"'()[]{}«»“”‘’…-–—/"))
            s = start + len(tokw[:lead].encode("utf-8"))
            out.append((s, s + len(core.encode("utf-8")), core))
        pos = start + len(wb)
    return out


def analyse_generation(runs, models, report, summary, scratch: Path):
    report.append("\n## Open-ended text (Schut et al.): FLORES+ devtest, teacher-forced and greedy continuations\n")
    report.append("At WORD-INITIAL positions (the position that predicts a word's first token), per layer: "
                  "`same` = top-1 is the token eventually emitted; `out-lang` = another token of the output "
                  "language; `xlat` = an other-language token that is a MUSE translation of the word being "
                  "emitted (prefix match, >= 3 chars); `oth-lang` = other-language token, not a translation. "
                  "`xlat - ctl` subtracts the same test against a different word's translations. `par - ctl` "
                  "= mass on the parallel other-language sentence's tokens minus a control sentence's, all "
                  "positions. Lexical = not in a function-word stoplist.\n")
    toks = {n: Tok(TOK_DIR / n) for n in ("unigram_destarved", "unigram_starved")}
    tokmaps = {n: np.load(RES / f"tokmap_{n}.npz")["P"] for n in toks}
    items = json.load(open(RES / "items" / "generation.json"))["items"]
    dicts = {}
    per = {}
    for run in runs:
        tn = models[run]["tok"]; tok = toks[tn]; P = tokmaps[tn]
        top_lang = P.argmax(1); top_conf = P.max(1)
        pieces = [tok.piece(i).replace("▁", "").lower() for i in range(tok.vocab_size)]
        per[run] = {}
        for L in models[run]["langs"]:
            p = scratch / f"{run}_generation_{L}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            set_langs = [s.split(":")[0] for s in z["set_langs"].tolist()][::2]
            oth = set_langs[0]
            if L != "en":
                dicts.setdefault(f"inv-{L}", muse_inverse(f"en-{L}"))
                xdict = dicts[f"inv-{L}"]
            else:
                dicts.setdefault(f"fwd-{oth}", muse_forward(f"en-{oth}"))
                xdict = dicts[f"fwd-{oth}"]
            stop = set(STOP[L])
            res = {}
            for pre in ("tf", "gen"):
                ids_all, ioff, offs, ks = z[f"{pre}_ids"], z[f"{pre}_id_offsets"], z[f"{pre}_offsets"], z[f"{pre}_k"]
                top1, mass, setm, lp = z[f"{pre}_top1"], z[f"{pre}_mass"].astype(np.float32), z[f"{pre}_setmass"].astype(np.float32), z[f"{pre}_lp"].astype(np.float32)
                cats = {"lex": defaultdict(list), "fun": defaultdict(list)}
                mass_by = {"lex": [], "fun": []}
                n_pos = 0
                for si in range(len(offs) - 1):
                    ids = ids_all[ioff[si]:ioff[si + 1]].tolist()
                    text = tok.decode(ids) if pre == "gen" else items[si]["text"][L]
                    if pre == "gen":
                        # positions from the generated part; decode whole sequence for word spans
                        text = tok.decode(ids)
                    spans = token_byte_spans(tok, ids)
                    words = words_with_spans(text, L)
                    wstart = {s: (e, w) for s, e, w in words}
                    wlist = [w for _, _, w in words]
                    k = int(ks[si])
                    for j, pos in enumerate(range(k - 1, len(ids) - 1)):     # pos predicts ids[pos+1]
                        col = offs[si] + j
                        t_next = ids[pos + 1]
                        s, e = spans[pos + 1]
                        pb = tok.piece_bytes(t_next)
                        s_core = s + (1 if pb[:1] == b" " else 0)
                        if s_core not in wstart:
                            continue
                        _, w = wstart[s_core]
                        if not is_script(w, L):
                            continue
                        kind = "fun" if w.lower() in stop or len(w) <= 1 else "lex"
                        n_pos += 1
                        trans = xdict.get(w.lower(), set())
                        # control: translations of another lexical word in the same sentence
                        others = [x for x in wlist if x.lower() != w.lower() and x.lower() in xdict]
                        ctl_trans = xdict.get(others[(wlist.index(w) + 3) % len(others)].lower(), set()) if others else set()
                        for l in range(NL):
                            t = int(top1[l, col])
                            same = t == t_next
                            lang_t = LANGS[top_lang[t]] if top_conf[t] > 0.5 else None
                            s_t = pieces[t]
                            def hit(tr):
                                if not s_t or not tr:
                                    return False
                                if s_t in tr:
                                    return True
                                return len(s_t) >= 3 and any(x.startswith(s_t) for x in tr)
                            cats[kind][("same", l)].append(same)
                            cats[kind][("outlang", l)].append((not same) and lang_t == L)
                            cats[kind][("xlat", l)].append((not same) and lang_t == oth and hit(trans))
                            cats[kind][("xlat_ctl", l)].append((not same) and lang_t == oth and hit(ctl_trans))
                            cats[kind][("othlang", l)].append((not same) and lang_t == oth and not hit(trans))
                        mass_by[kind].append(mass[:, col])
                r = {"n_pos": n_pos}
                for kind in ("lex", "fun"):
                    for c in ("same", "outlang", "xlat", "xlat_ctl", "othlang"):
                        r[f"{kind}_{c}"] = np.array([np.mean(cats[kind][(c, l)]) if cats[kind][(c, l)] else np.nan for l in range(NL)])
                    r[f"{kind}_n"] = len(cats[kind][("same", 0)])
                    if mass_by[kind]:
                        mm = np.mean(mass_by[kind], 0)
                        r[f"{kind}_mass_out"], r[f"{kind}_mass_oth"] = mm[:, LI[L]], mm[:, LI[oth]]
                mm = mass.mean(1)
                r["mass_out"], r["mass_oth"], r["mass_other"] = mm[:, LI[L]], mm[:, LI[oth]], mm[:, OTHER]
                r["share_out"] = mm[:, LI[L]] / np.maximum(1 - mm[:, OTHER], 1e-6)
                r["share_oth"] = mm[:, LI[oth]] / np.maximum(1 - mm[:, OTHER], 1e-6)
                r["t_layer_mass"] = transition_layer(mm[:, LI[L]], mm[:, LI[oth]])
                r["par_minus_ctl"] = (setm[0] - setm[1]).mean(1)
                r["par_minus_ctl_by_sent"] = np.array([(setm[0, :, offs[i]:offs[i + 1]] - setm[1, :, offs[i]:offs[i + 1]]).mean(1)
                                                       for i in range(len(offs) - 1)])   # [n_sent, L]
                tgt = np.concatenate([ids_all[ioff[i] + ks[i]:ioff[i + 1]] for i in range(len(offs) - 1)])
                r["top1_acc"] = np.array([(top1[l] == tgt).mean() for l in range(NL)])
                eq = top1 == tgt[None]                                            # [L, P]
                ok = eq[16]
                # settle = 16 - (number of consecutive correct layers ending at 16)
                run_len = np.zeros(eq.shape[1], int)
                alive = ok.copy()
                for l in range(16, -1, -1):
                    alive &= eq[l]
                    run_len += alive
                settle = np.where(ok, 17 - run_len, np.nan).astype(float)
                r["settle"] = settle
                r["settle_by_sent"] = np.array([np.nanmean(settle[offs[i]:offs[i + 1]]) if ok[offs[i]:offs[i + 1]].any() else np.nan
                                                for i in range(len(offs) - 1)])
                res[pre] = r
            per[run][L] = res
    report.append("| model | text | mode | n word-initial | lexical: xlat-ctl peak (L) | function: xlat-ctl peak (L) | lexical: oth-lang peak (L) | par-ctl peak (L) [CI] | transition L (mass) | top-1 acc out | settle L mean |")
    report.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for run, d in per.items():
        for L, res in d.items():
            for pre, r in res.items():
                lx = r["lex_xlat"] - r["lex_xlat_ctl"]; fx = r["fun_xlat"] - r["fun_xlat_ctl"]
                pk = int(np.nanargmax(r["par_minus_ctl"][1:16]) + 1)
                m, lo, hi = boot_ci(r["par_minus_ctl_by_sent"][:, pk])
                report.append(f"| {run} | {L} | {pre} | {r['n_pos']} | {np.nanmax(lx[1:16]):.3f} ({np.nanargmax(lx[1:16])+1}) | "
                              f"{np.nanmax(fx[1:16]):.3f} ({np.nanargmax(fx[1:16])+1}) | {np.nanmax(r['lex_othlang'][1:16]):.3f} ({np.nanargmax(r['lex_othlang'][1:16])+1}) | "
                              f"{fmt(m, lo, hi)} ({pk}) | {r['t_layer_mass']:.0f} | {r['top1_acc'][16]:.3f} | {np.nanmean(r['settle']):.2f} |")
    report.append("\n### Fair - starved (paired over sentences, par - ctl at the fair model's peak layer; lexical xlat-ctl peak difference)\n")
    report.append("| partner | script | text | mode | d par-ctl [CI] | d lexical xlat-ctl peak | d oth-lang peak | d transition L | d settle L [CI] (sentences) |")
    report.append("|---|---|---|---|---|---|---|---|---|")
    deltas = defaultdict(dict)
    for X, fr, sr in bilingual_pairs(runs):
        for L in (X, "en"):
            for pre in ("tf", "gen"):
                if fr not in per or sr not in per or L not in per[fr] or L not in per[sr]:
                    continue
                a, b = per[fr][L][pre], per[sr][L][pre]
                pk = int(np.nanargmax(a["par_minus_ctl"][1:16]) + 1)
                m, lo, hi = boot_ci(a["par_minus_ctl_by_sent"][:, pk] - b["par_minus_ctl_by_sent"][:, pk])
                la = np.nanmax((a["lex_xlat"] - a["lex_xlat_ctl"])[1:16]); lb = np.nanmax((b["lex_xlat"] - b["lex_xlat_ctl"])[1:16])
                oa = np.nanmax(a["lex_othlang"][1:16]); ob = np.nanmax(b["lex_othlang"][1:16])
                ds = a["settle_by_sent"] - b["settle_by_sent"]; ds = ds[~np.isnan(ds)]
                sm, slo, shi = boot_ci(ds)
                report.append(f"| {X} | {SCRIPT[X]} | {L} | {pre} | {fmt(m, lo, hi)} | {la-lb:+.3f} | {oa-ob:+.3f} | {a['t_layer_mass']-b['t_layer_mass']:+.0f} | {fmt(sm, slo, shi, 2)} |")
                deltas[X][f"{L}|{pre}"] = {"d_par_ctl": [m, lo, hi], "d_xlat": float(la - lb), "d_othlang": float(oa - ob), "d_settle": [sm, slo, shi]}
    summary["generation"] = {run: {L: {pre: {k: (v.tolist() if isinstance(v, np.ndarray) and v.ndim == 1 else v)
                                             for k, v in r.items() if k not in ("par_minus_ctl_by_sent", "settle", "settle_by_sent")}
                                       for pre, r in res.items()} for L, res in d.items()} for run, d in per.items()}
    summary["generation_deltas"] = deltas
    return per


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def figures(word, fact, gen, runs, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    layers = np.arange(NL)
    col = {"fair": "#1f77b4", "starved": "#ff7f0e"}
    pairs = bilingual_pairs(runs)

    # Fig 1: translation en->X and repetition X->X: P(out) / P(latent) / P(ctl) by layer, fair vs starved
    for cond, title in (("tr_en2X", "translation en→X"), ("rep_X", "repetition X→X"), ("tr_X2en", "translation X→en"), ("cloze_X", "cloze in X")):
        fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (X, fr, sr) in zip(axes, pairs):
            c = cond.replace("cloze_X", f"cloze_{X}")
            for tag, run in (("fair", fr), ("starved", sr)):
                r = word.get(run, {}).get((X, c))
                if not r:
                    continue
                ax.plot(layers, r["P_out"], color=col[tag], lw=2, label=f"{tag}: output word")
                ax.plot(layers, r["P_lat"], color=col[tag], lw=2, ls="--", label=f"{tag}: latent (other lang)")
                ax.plot(layers, r["P_ctl"], color=col[tag], lw=1, ls=":", label=f"{tag}: control")
            ax.set_title(f"en-{X} ({SCRIPT[X]}-script)"); ax.set_xlabel("layer"); ax.grid(alpha=.3)
        axes[0].set_ylabel("P (full string, lens)"); axes[0].legend(fontsize=7)
        fig.suptitle(f"{title}: probability of output word vs its other-language translation, by layer")
        fig.tight_layout(); fig.savefig(FIGS / f"fig_lens_{cond}.pdf"); fig.savefig(FIGS / f"fig_lens_{cond}.png", dpi=130); plt.close(fig)

    # Fig 2: language mass by layer (translation en->X, factual X, generation X)
    fig, axes = plt.subplots(3, len(pairs), figsize=(4 * len(pairs), 8.5), sharey=True)
    for ci, (X, fr, sr) in enumerate(pairs):
        for tag, run in (("fair", fr), ("starved", sr)):
            r = word.get(run, {}).get((X, "tr_en2X"))
            if r:
                axes[0, ci].plot(layers, r["mass_out"], color=col[tag], lw=2, label=f"{tag}: {X} mass")
                axes[0, ci].plot(layers, r["mass_oth"], color=col[tag], lw=2, ls="--", label=f"{tag}: en mass")
            f = fact.get(run, {}).get(X)
            if f:
                axes[1, ci].plot(layers, f["mass"][:, LI[X]], color=col[tag], lw=2)
                axes[1, ci].plot(layers, f["mass"][:, LI["en"]], color=col[tag], lw=2, ls="--")
            g = gen.get(run, {}).get(X, {}).get("tf")
            if g:
                axes[2, ci].plot(layers, g["mass_out"], color=col[tag], lw=2)
                axes[2, ci].plot(layers, g["mass_oth"], color=col[tag], lw=2, ls="--")
        axes[0, ci].set_title(f"en-{X}: translation en→{X}"); axes[1, ci].set_title(f"factual, {X} prompt"); axes[2, ci].set_title(f"FLORES {X} text")
        for a in axes[:, ci]:
            a.grid(alpha=.3)
        axes[2, ci].set_xlabel("layer")
    axes[0, 0].legend(fontsize=7); axes[0, 0].set_ylabel("language mass"); axes[1, 0].set_ylabel("language mass"); axes[2, 0].set_ylabel("language mass")
    fig.suptitle("Language mass of the lens distribution by layer (solid: output language, dashed: English)")
    fig.tight_layout(); fig.savefig(FIGS / "fig_lens_mass.pdf"); fig.savefig(FIGS / "fig_lens_mass.png", dpi=130); plt.close(fig)

    # Fig 5: commitment depth across tasks, fair vs starved, per partner
    panels = [("translation en→X", lambda run, X: word.get(run, {}).get((X, "tr_en2X"), {}).get("commit")),
              ("repetition X→X", lambda run, X: word.get(run, {}).get((X, "rep_X"), {}).get("commit")),
              ("PolyFact, X prompt", lambda run, X: (lambda f: f["settle"][~np.isnan(f["settle"])] if f else None)(fact.get(run, {}).get(X))),
              ("FLORES X text", lambda run, X: (lambda g: g["settle_by_sent"][~np.isnan(g["settle_by_sent"])] if g else None)(gen.get(run, {}).get(X, {}).get("tf")))]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.4), sharey=False)
    for ax, (title, get) in zip(axes, panels):
        xs = np.arange(len(pairs))
        for k, (tag, off) in enumerate((("fair", -0.18), ("starved", 0.18))):
            ms, los, his = [], [], []
            for X, fr, sr in pairs:
                v = get(fr if tag == "fair" else sr, X)
                if v is None or len(v) == 0:
                    ms.append(np.nan); los.append(np.nan); his.append(np.nan); continue
                m, lo, hi = boot_ci(np.asarray(v, float))
                ms.append(m); los.append(lo); his.append(hi)
            ms, los, his = map(np.array, (ms, los, his))
            ax.bar(xs + off, ms, 0.34, color=col[tag], label=tag, yerr=[ms - los, his - ms], capsize=3)
        ax.set_xticks(xs); ax.set_xticklabels([f"en-{X}\n({SCRIPT[X]})" for X, _, _ in pairs])
        ax.set_title(title); ax.grid(axis="y", alpha=.3)
        lo_y = np.nanmin([b.get_height() for b in ax.patches]) if ax.patches else 0
        ax.set_ylim(max(0, lo_y - 3), None)
    axes[0].set_ylabel("commitment layer (lower = earlier)"); axes[0].legend(fontsize=8)
    fig.suptitle("Layer from which the lens settles on the final answer, by task and partner")
    fig.tight_layout(); fig.savefig(FIGS / "fig_lens_commit.pdf"); fig.savefig(FIGS / "fig_lens_commit.png", dpi=130); plt.close(fig)

    # Fig 3: factual per-layer accuracy, prompt-language vs other-language candidates
    fig, axes = plt.subplots(2, len(pairs), figsize=(4 * len(pairs), 6), sharey=True)
    for ci, (X, fr, sr) in enumerate(pairs):
        for ri, L in enumerate((X, "en")):
            ax = axes[ri, ci]
            for tag, run in (("fair", fr), ("starved", sr)):
                f = fact.get(run, {}).get(L)
                if not f:
                    continue
                oth = "en" if L != "en" else X
                ax.plot(layers, f[f"acc_{L}"], color=col[tag], lw=2, label=f"{tag}: {L} candidates")
                ax.plot(layers, f[f"acc_{oth}"], color=col[tag], lw=2, ls="--", label=f"{tag}: {oth} candidates")
            ax.axhline(.25, color="k", lw=.8, ls=":"); ax.grid(alpha=.3)
            ax.set_title(f"en-{X}: prompt in {L}")
        axes[1, ci].set_xlabel("layer")
    axes[0, 0].set_ylabel("4-way accuracy"); axes[1, 0].set_ylabel("4-way accuracy"); axes[0, 0].legend(fontsize=7)
    fig.suptitle("PolyFact: per-layer 4-way accuracy, candidates in the prompt language vs the other language")
    fig.tight_layout(); fig.savefig(FIGS / "fig_lens_factual.pdf"); fig.savefig(FIGS / "fig_lens_factual.png", dpi=130); plt.close(fig)

    # Fig 4: generation, lexical positions: same / out-lang / xlat-ctl / oth-lang by layer
    fig, axes = plt.subplots(2, len(pairs), figsize=(4 * len(pairs), 6), sharey="row")
    for ci, (X, fr, sr) in enumerate(pairs):
        for tag, run in (("fair", fr), ("starved", sr)):
            g = gen.get(run, {}).get(X, {}).get("tf")
            if not g:
                continue
            axes[0, ci].plot(layers, g["lex_same"], color=col[tag], lw=2, label=f"{tag}: = emitted token")
            axes[0, ci].plot(layers, g["lex_outlang"], color=col[tag], lw=1.5, ls="--", label=f"{tag}: other {X} token")
            axes[0, ci].plot(layers, g["lex_othlang"] + g["lex_xlat"], color=col[tag], lw=1.5, ls=":", label=f"{tag}: English token")
            axes[1, ci].plot(layers, g["lex_xlat"] - g["lex_xlat_ctl"], color=col[tag], lw=2, label=f"{tag}: lexical")
            axes[1, ci].plot(layers, g["fun_xlat"] - g["fun_xlat_ctl"], color=col[tag], lw=1.5, ls="--", label=f"{tag}: function")
        axes[0, ci].set_title(f"en-{X}: {X} text, lexical word-initial positions"); axes[1, ci].set_xlabel("layer")
        for a in axes[:, ci]:
            a.grid(alpha=.3)
    axes[0, 0].set_ylabel("share of positions (top-1)"); axes[1, 0].set_ylabel("English-translation hit − control")
    axes[0, 0].legend(fontsize=7); axes[1, 0].legend(fontsize=7)
    fig.suptitle("FLORES text: what the lens top-1 token is at each layer")
    fig.tight_layout(); fig.savefig(FIGS / "fig_lens_generation.pdf"); fig.savefig(FIGS / "fig_lens_generation.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="/mnt/scratch/xscript_lens/raw")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--factual-tag", default="factual", choices=["factual", "factual_all"])
    args = ap.parse_args()
    global FACT_TAG
    FACT_TAG = args.factual_tag
    models = models_table()
    runs = runs_available()
    report = ["# Logit-lens analysis (CLAUDE.md 6l)\n", f"Runs: {', '.join(runs)}\n"]
    summary = {"runs": runs}
    word = analyse_wordtasks(runs, models, report, summary)
    fact = analyse_factual(runs, models, report, summary)
    gen = analyse_generation(runs, models, report, summary, Path(args.scratch))
    if not args.no_figs:
        figures(word, fact, gen, runs, models)
    (RES / "report.md").write_text("\n".join(report) + "\n")
    json.dump(to_py(summary), open(RES / "summary.json", "w"), indent=1)
    print("\n".join(report))


if __name__ == "__main__":
    main()
