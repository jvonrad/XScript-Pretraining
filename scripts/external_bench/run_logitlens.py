#!/usr/bin/env python
"""Logit-lens sweep over checkpoints (CLAUDE.md 6l). CPU only; never touches
the Neuron devices, so it can run while training occupies them.

    python run_logitlens.py --runs en-de-fair en-de-starved ... \
        --ckpt-dir /mnt/scratch/xscript_lens/ckpt --threads 48

Per run, four tasks over the fixed items from build_lens_items.py:

  translation  Wendler-style 5-shot word translation (en->X, X->en) and
               repetition (X->X, en->en). For each item three continuations
               are scored at every layer: the OUTPUT word, its translation in
               the other language (LATENT), and the translation of a different
               word (CONTROL).
  cloze        Wendler's masked-definition prompts in X (and in en, with the
               partner word as latent), same three continuations.
  factual      PolyFact question + localized cue; the four candidate entities
               in the prompt language, the same four in the OTHER language
               (aligned by QID), and a mismatched-fact English answer.
  generation   FLORES+ devtest sentences teacher-forced (lens at every
               position) and 12-token greedy continuations of their first 40%,
               with per-position mass on the tokens of the PARALLEL sentence in
               the other language vs a control sentence.

Outputs (raw sidecars on scratch -- ~20 MB per model, NOT in git; the analysis
script writes the summaries into results/logitlens/):
  <scratch>/raw/<run>_{translation,cloze,factual,generation}.json
  <scratch>/raw/<run>_generation_<lang>.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from xscript.tok.wrapper import Tok, BOS_ID                       # noqa: E402
from xscript.eval.logitlens import (                                  # noqa: E402
    CUE, LANGS, LANG_NAMES, Lens, Request, encode_pair, lens_self_check,
    load_model, load_or_build_tokmap, single_token)

ITEMS = REPO / "results" / "logitlens" / "items"
OUT_GIT = Path("/mnt/scratch/xscript_lens/raw")   # raw sidecars live on the box, not in git (6f)
TOK_DIR = Path("/mnt/scratch/xscript/tokenizers")
HOLDOUT = Path("/mnt/scratch/xscript/holdout")
PARTNERS = ("de", "fr", "ar", "zh")


def sp(lang: str, w: str) -> str:
    """Continuation form of a word: space-prefixed, except Chinese, which is
    written without a space after the (full-width) colon. With a space the
    first continuation token is the bare `▁` piece, which carries no language
    and makes the answer-position language mass meaningless for zh."""
    return w if lang == "zh" else " " + w


def cue(lang: str) -> str:
    """'{Name}:' with the language's own colon convention."""
    return f"{LANG_NAMES[lang]}：" if lang == "zh" else f"{LANG_NAMES[lang]}: "


def resp_summary(r, with_pos: bool = False) -> dict:
    d = {"full_lp": r.lp.sum(1).round(4).tolist(),
         "first_lp": r.lp[:, 0].round(4).tolist(),
         "first_rank": r.rank[:, 0].tolist(),
         "n_tok": int(r.lp.shape[1]),
         "greedy_ok": bool(r.rank[-1].max() == 0)}     # greedy decoding would emit exactly this string
    if with_pos:
        d["mass"] = r.mass[:, 0].round(5).tolist()          # [L+1, 6] at the answer position
        d["top1"] = r.top1[:, 0].tolist()
    return d


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------

def task_translation(lens, tok, toks_both, langs, limit):
    items_all = json.load(open(ITEMS / "translation.json"))
    partners = [x for x in PARTNERS if x in langs] or list(PARTNERS)
    out = {}
    for X in partners:
        items = items_all[X][:limit] if limit else items_all[X]
        n = len(items)
        conds = {
            "tr_en2X": ("en", X), "tr_X2en": (X, "en"),
            "rep_X": (X, X), "rep_en": ("en", "en"),
        }
        reqs, keys = [], []
        for ci, (cond, (src, dst)) in enumerate(conds.items()):
            wsrc = lambda it: it["en"] if src == "en" else it["x"]
            wdst = lambda it: it["en"] if dst == "en" else it["x"]
            lat_lang = "en" if dst != "en" else X
            wlat = lambda it: it["en"] if lat_lang == "en" else it["x"]
            for i, it in enumerate(items):
                demos = "".join(f"{cue(src)}{wsrc(items[j])} - {cue(dst)}{wdst(items[j])}\n"
                                for j in it["demos"] if j < n)
                ctx = demos + f"{cue(src)}{wsrc(it)} - {cue(dst).rstrip(' ')}"
                ctl_it = items[it["ctl"] % n]
                for role, w in (("out", wdst(it)), ("lat", wlat(it)), ("ctl", wlat(ctl_it))):
                    ids, k = encode_pair(tok, ctx, sp(dst, w))
                    reqs.append(Request(ids, k))
                    keys.append((cond, i, role, w))
        t0 = time.time()
        res = lens.run(reqs)
        rows = {}
        for (cond, i, role, w), r in zip(keys, res):
            d = rows.setdefault(cond, {}).setdefault(i, {"en": items[i]["en"], "x": items[i]["x"]})
            d[role] = resp_summary(r, with_pos=(role == "out"))
            d[role]["word"] = w
            d[role]["single_both"] = all(single_token(t, sp("", w)) for t in toks_both)
        out[X] = {"n": n, "conds": {c: [rows[c][i] for i in range(n)] for c in conds}}
        print(f"    translation en-{X}: {len(reqs)} requests in {time.time()-t0:.0f}s", flush=True)
    return out


def task_cloze(lens, tok, toks_both, langs, limit):
    items_all = json.load(open(ITEMS / "cloze.json"))
    tr = json.load(open(ITEMS / "translation.json"))
    partners = [x for x in PARTNERS if x in langs] or list(PARTNERS)
    out = {}
    for L in [l for l in ("de", "fr", "zh", "en") if l in langs]:
        items = items_all[L]
        if L == "en":
            # latent = the partner's word; one block per partner the model trained on
            blocks = [(X, {it["en"]: it["x"] for it in tr[X]}) for X in partners]
        else:
            blocks = [("en", {it["en"]: it["en"] for it in items})]
        for lat_lang, lat_map in blocks:
            sel = [it for it in items if it["en"] in lat_map]
            sel = sel[:limit] if limit else sel
            n = len(sel)
            reqs, keys = [], []
            for i, it in enumerate(sel):
                demos = "\n".join(items[j]["demo"] for j in it["demos"]) + "\n"
                ctx = demos + it["ctx"]
                ctl = sel[it["ctl"] % n]
                lat_w, ctl_w = lat_map[it["en"]], lat_map[ctl["en"]]
                cands = (("out", it["cont"]), ("lat", sp(lat_lang, lat_w)), ("ctl", sp(lat_lang, ctl_w)))
                for role, cont in cands:
                    ids, k = encode_pair(tok, ctx, cont)
                    reqs.append(Request(ids, k)); keys.append((i, role, cont))
            t0 = time.time()
            res = lens.run(reqs)
            rows = [{"en": it["en"], "x": it["x"]} for it in sel]
            for (i, role, cont), r in zip(keys, res):
                rows[i][role] = resp_summary(r, with_pos=(role == "out"))
                rows[i][role]["word"] = cont
                rows[i][role]["single_both"] = all(single_token(t, cont) for t in toks_both)
            out[f"{L}|lat={lat_lang}"] = {"lang": L, "lat_lang": lat_lang, "n": n, "items": rows}
            print(f"    cloze {L} (latent {lat_lang}): {len(reqs)} requests in {time.time()-t0:.0f}s", flush=True)
    return out


def task_factual(lens, tok, langs, limit, facts_file="factual.json"):
    facts = json.load(open(ITEMS / facts_file))["facts"]
    facts = facts[:limit] if limit else facts
    partners = [x for x in PARTNERS if x in langs] or list(PARTNERS)
    out = {}
    for L in langs:
        other_langs = ["en"] if L != "en" else partners
        reqs, keys = [], []
        for fi, f in enumerate(facts):
            ctx = f"{f['langs'][L]['question']}\n{CUE[L]}"
            for c, qid in enumerate(f["qids"]):
                ids, k = encode_pair(tok, ctx, sp(L, f["langs"][L]["options"][qid]))
                reqs.append(Request(ids, k)); keys.append((fi, L, c))
            for OL in other_langs:
                for c, qid in enumerate(f["qids"]):
                    ids, k = encode_pair(tok, ctx, sp(OL, f["langs"][OL]["options"][qid]))
                    reqs.append(Request(ids, k)); keys.append((fi, OL, c))
            ctl = facts[f["ctl"] % len(facts)]
            ids, k = encode_pair(tok, ctx, sp("en", ctl["langs"]["en"]["answer"]))
            reqs.append(Request(ids, k)); keys.append((fi, "ctl", 0))
        t0 = time.time()
        res = lens.run(reqs)
        rows = [{"fact_id": f["fact_id"], "relation": f["relation"], "gold": f["gold_idx"],
                 "cands": {}} for f in facts]
        for (fi, lang, c), r in zip(keys, res):
            row = rows[fi]
            if lang == "ctl":
                row["ctl"] = resp_summary(r)
                continue
            cl = row["cands"].setdefault(lang, [None] * 4)
            gold = (c == facts[fi]["gold_idx"])
            cl[c] = {"full_lp": r.lp.sum(1).round(4).tolist(), "first_lp": r.lp[:, 0].round(4).tolist(),
                     "n_tok": int(r.lp.shape[1])}
            if gold:
                cl[c]["first_rank"] = r.rank[:, 0].tolist()
                if lang == L:
                    row["mass"] = r.mass[:, 0].round(5).tolist()
                    row["top1"] = r.top1[:, 0].tolist()
        out[L] = {"n": len(facts), "other_langs": other_langs, "items": rows}
        print(f"    factual {L}: {len(reqs)} requests in {time.time()-t0:.0f}s", flush=True)
    return out


def _sent_token_set(tok, text: str) -> np.ndarray:
    ids = [i for i in tok.encode(text) if i > 3 and tok.piece(i) != "▁"]
    return np.array(sorted(set(ids)), np.int64)


def task_generation(lens, tok, langs, limit, scratch_out: Path, run: str, n_new: int = 12):
    items = json.load(open(ITEMS / "generation.json"))["items"]
    items = items[:limit] if limit else items
    partners = [x for x in PARTNERS if x in langs] or list(PARTNERS)
    summary = {}
    for L in langs:
        set_langs = ["en"] if L != "en" else partners
        t0 = time.time()
        # (a) teacher-forced full sentences
        reqs = []
        for it in items:
            ids = tok.encode(it["text"][L], bos=True)
            ctl = items[it["ctl"] % len(items)]
            sets = []
            for SL in set_langs:
                sets.append(_sent_token_set(tok, it["text"][SL]))
                sets.append(_sent_token_set(tok, ctl["text"][SL]))
            reqs.append(Request(ids, 1, sets))
        tf = lens.run(reqs)
        # (b) greedy continuations of the prompt prefix
        prompts = [tok.encode(it["prompt"][L], bos=True) for it in items]
        gens = lens.greedy(prompts, n_new)
        greqs = [Request(p + g, len(p), rq.sets) for p, g, rq in zip(prompts, gens, reqs)]
        gr = lens.run(greqs)

        def pack(rs, seqs, ks):
            offs = np.cumsum([0] + [r.lp.shape[1] for r in rs])
            return {
                "ids": np.concatenate([np.asarray(s, np.int32) for s in seqs]),
                "id_offsets": np.cumsum([0] + [len(s) for s in seqs]).astype(np.int64),
                "k": np.asarray(ks, np.int32),
                "offsets": offs.astype(np.int64),
                "lp": np.concatenate([r.lp for r in rs], 1).astype(np.float16),
                "top1": np.concatenate([r.top1 for r in rs], 1).astype(np.int32),
                "rank": np.concatenate([r.rank for r in rs], 1).astype(np.int32),
                "mass": np.concatenate([r.mass for r in rs], 1).astype(np.float16),
                "setmass": np.concatenate([r.setmass for r in rs], 2).astype(np.float16),
            }
        d = {}
        for pre, (rs, seqs, ks) in {"tf": (tf, [r.ids for r in reqs], [1] * len(reqs)),
                                     "gen": (gr, [r.ids for r in greqs], [r.k for r in greqs])}.items():
            for key, arr in pack(rs, seqs, ks).items():
                d[f"{pre}_{key}"] = arr
        d["sent_ids"] = np.asarray([it["id"] for it in items], np.int64)
        d["set_langs"] = np.asarray([f"{SL}:{kind}" for SL in set_langs for kind in ("par", "ctl")])
        scratch_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(scratch_out / f"{run}_generation_{L}.npz", **d)
        summary[L] = {"n": len(items), "set_langs": d["set_langs"].tolist(),
                      "gen_text": [tok.decode(g) for g in gens[:20]]}
        print(f"    generation {L}: {len(reqs)} sentences + {len(greqs)} continuations in {time.time()-t0:.0f}s", flush=True)
    return summary


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ckpt-dir", default="/mnt/scratch/xscript_lens/ckpt")
    ap.add_argument("--scratch-out", default="/mnt/scratch/xscript_lens/raw")
    ap.add_argument("--tasks", nargs="*", default=["translation", "cloze", "factual", "generation"])
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="items per task (smoke test)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--facts-file", default="factual.json",
                    help="items file for the factual task; 'factual_all.json' = all 2039 PolyFact facts, "
                         "written as <run>_factual_all.json")
    args = ap.parse_args()
    fact_tag = "factual" if args.facts_file == "factual.json" else "factual_" + args.facts_file.split("_", 1)[1].split(".")[0]
    torch.set_num_threads(args.threads)
    models = json.load(open(REPO / "results" / "models.json"))
    OUT_GIT.mkdir(parents=True, exist_ok=True)
    toks = {n: Tok(TOK_DIR / n) for n in ("unigram_destarved", "unigram_starved")}
    tokmaps = {n: load_or_build_tokmap(t, REPO / "results" / "logitlens" / f"tokmap_{n}.npz", HOLDOUT)
               for n, t in toks.items()}
    for run in args.runs:
        tok_name, langs = models[run]["tok"], models[run]["langs"]
        tag = "" if not args.limit else f"_limit{args.limit}"
        done = all((OUT_GIT / f"{run}{tag}_{(fact_tag if t == 'factual' else t)}.json").exists() for t in args.tasks if t != "generation") \
            and all((Path(args.scratch_out) / f"{run}_generation_{L}.npz").exists() for L in langs
                    if "generation" in args.tasks)
        if done and not args.force:
            print(f"[lens] {run}: done, skip", flush=True); continue
        t0 = time.time()
        model = load_model(Path(args.ckpt_dir) / f"{run}.pt")
        lens = Lens(model, tokmaps[tok_name])
        tok = toks[tok_name]
        err = lens_self_check(lens, tok)
        print(f"[lens] {run} ({tok_name}, {langs}) loaded in {time.time()-t0:.0f}s; "
              f"layer-16 lens vs model output max|dlogp| = {err:.4f}", flush=True)
        assert err < 0.1, "lens does not reproduce the model output"
        toks_both = list(toks.values())
        meta = {"run": run, "tok": tok_name, "langs": langs, "self_check": err, "n_layers": model.cfg.n_layers}
        for task in args.tasks:
            t1 = time.time()
            if task == "translation":
                res = task_translation(lens, tok, toks_both, langs, args.limit)
            elif task == "cloze":
                res = task_cloze(lens, tok, toks_both, langs, args.limit)
            elif task == "factual":
                res = task_factual(lens, tok, langs, args.limit, args.facts_file)
            elif task == "generation":
                res = task_generation(lens, tok, langs, args.limit, Path(args.scratch_out), run)
            else:
                raise SystemExit(f"unknown task {task}")
            out_name = fact_tag if task == "factual" else task
            json.dump({"meta": meta, "task": task, "seconds": time.time() - t1, "data": res},
                      open(OUT_GIT / f"{run}{tag}_{out_name}.json", "w"), ensure_ascii=False)
            print(f"[lens] {run} {task} done in {time.time()-t1:.0f}s", flush=True)
        del lens, model
    print("[lens] all done", flush=True)


if __name__ == "__main__":
    main()
