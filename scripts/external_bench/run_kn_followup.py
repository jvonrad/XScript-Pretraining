#!/usr/bin/env python
"""Follow-up ablation passes for the knowledge-neuron analysis (CLAUDE.md 6j).

Three forward-only passes per model, all reusing phase C's compiled
[24, width] graph (masks are data, not shape):

  1. SAME-RELATION controls (`<run>_ablation_samerel.json`): phase C's six
     conditions, but the different-fact baselines f' now share the RELATION
     of f (e.g. capital-of vs capital-of), so the subtraction also removes
     relation-level circuitry, isolating fact-identity-specific transfer.
     own_fact/cross_fact are re-run identically as an internal consistency
     check against the original phase C.

  2./3. INTERSECTION decomposition at K=100 and K=200
     (`<run>_intersect_k{K}.json`): per (fact, target language t, source s),
     six masks:
       none | I = topK_s(f) & topK_t(f) | D = topK_s(f) \\ I |
       two random |I|-subsets of topK_s(f) | I' from the same-relation f'
       (f''s own intersection first, filled from its top-K, truncated to |I|)
     This separates Story A (shared storage: damage concentrated in I, beyond
     size-matched random) from Story B (coupled-but-disjoint circuits: D
     carries it). Size-fairness across tokenizers: I is only ever compared
     to |I|-sized controls from the same model. Facts with |I| = 0 are
     recorded and skipped.

    python run_kn_followup.py --runs en-de-fair ... [--workdir ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent))

from run_knowneurons import fetch_checkpoint, load_polyfact, contexts_for  # noqa: E402

SR_CONDS = ("none", "own_fact", "cross_fact", "own_other_sr", "cross_other_sr", "random")
INT_CONDS = ("none", "inter", "disjoint", "rand_sub1", "rand_sub2", "inter_ctrl")


def same_rel_next(shared, relations):
    """For each position in `shared`, the position of the next fact with the
    same relation (wrapping); falls back to the next fact of any relation when
    the relation group has a single member. Returns (nxt, fallback_flags)."""
    groups = {}
    for i, fi in enumerate(shared):
        groups.setdefault(relations[fi], []).append(i)
    nxt = [(i + 1) % len(shared) for i in range(len(shared))]
    fb = [True] * len(shared)
    for g in groups.values():
        if len(g) < 2:
            continue
        for j, i in enumerate(g):
            nxt[i] = g[(j + 1) % len(g)]
            fb[i] = False
    return nxt, fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="jvonrad/xscript-eval")
    ap.add_argument("--workdir", default="/home/ubuntu/xscript_kn")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--device", default="xla")
    ap.add_argument("--k", type=int, default=100, help="K for the same-relation pass")
    ap.add_argument("--int-ks", type=int, nargs="*", default=[100, 200])
    ap.add_argument("--keep-checkpoints", action="store_true")
    args = ap.parse_args()

    import torch
    import datasets
    from huggingface_hub import list_repo_files
    from xscript.eval import knowneurons as kn
    from xscript.model import ModelConfig, Transformer
    from xscript.tok.wrapper import Tok

    is_xla = args.device == "xla"
    if is_xla:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device(args.device)

    work = Path(args.workdir).resolve()
    res = work / "results" / "knowneurons"
    relations = datasets.load_dataset("jvonrad/PolyFact", "en", split="test")["relation"]
    repo_files = list_repo_files(args.repo)

    for run in args.runs:
        t_run = time.time()
        sr_f = res / f"{run}_ablation_samerel.json"
        int_fs = {K: res / f"{run}_intersect_k{K}.json" for K in args.int_ks}
        if sr_f.exists() and all(f.exists() for f in int_fs.values()):
            print(f"[fu] {run}: already complete"); continue

        print(f"\n===== {run} =====")
        z = np.load(res / f"{run}_kn.npz")
        sel = json.loads((res / f"{run}_selection.json").read_text())
        shared = [int(v) for v in z["fact_ids"]]
        topk = z["topk_idx"]                 # [n, 2, 512]
        langs = [str(v) for v in z["langs"]]
        width = sel["width"]

        ckpt = fetch_checkpoint(args.repo, "model", repo_files, work,
                                f"runs/{run}/checkpoints")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        tok = Tok(work / "_repo" / "tokenizers" / ck["cfg"]["tok_name"])
        model = Transformer(ModelConfig(**ck["cfg"]["model"])).eval()
        model.load_state_dict(ck["model"])
        model = model.to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        max_seq = model.cfg.max_seq_len
        del ck

        data = load_polyfact(langs)
        nS = len(shared)
        nxt, fb = same_rel_next(shared, relations)
        n_neur = model.cfg.n_layers * model.cfg.ffn_dim
        K = args.k

        # ---------- pass 1: same-relation controls ----------
        if not sr_f.exists():
            t0 = time.time()
            out = {"run": run, "langs": langs, "k": K, "conditions": list(SR_CONDS),
                   "fact_ids": shared, "fallback": fb,
                   "ll": {lang: [] for lang in langs}}
            for ti, tlang in enumerate(langs):
                si = 1 - ti
                ctxs = contexts_for(data, tlang)
                for i, fi in enumerate(shared):
                    d = data[tlang][fi]
                    rng = np.random.default_rng([1001, fi, ti])
                    masks = [None,
                             topk[i, ti, :K],
                             topk[i, si, :K],
                             topk[nxt[i], ti, :K],
                             topk[nxt[i], si, :K],
                             rng.choice(n_neur, size=K, replace=False)]
                    out["ll"][tlang].append(kn.ablation_ll(
                        model, tok, device, ctxs[fi],
                        [" " + c for c in d["choices"]], masks, width, max_seq, is_xla))
                    if i % 200 == 0:
                        print(f"[fu] {run} samerel {tlang} {i}/{nS} ({time.time()-t0:.0f}s)")
            out["gold"] = {lang: [data[lang][fi]["label"] for fi in shared]
                           for lang in langs}
            kn.save_json(sr_f, out)
            print(f"[fu] {run} samerel done ({time.time()-t0:.0f}s)")

        # ---------- passes 2/3: intersection decomposition ----------
        for KI in args.int_ks:
            if int_fs[KI].exists():
                continue
            t0 = time.time()
            out = {"run": run, "langs": langs, "k": KI, "conditions": list(INT_CONDS),
                   "cells": {lang: [] for lang in langs}}
            for ti, tlang in enumerate(langs):
                si = 1 - ti
                ctxs = contexts_for(data, tlang)
                for i, fi in enumerate(shared):
                    src = topk[i, si, :KI]
                    tgt = set(topk[i, ti, :KI].tolist())
                    inter = np.array([v for v in src if v in tgt], dtype=np.int64)
                    disj = np.array([v for v in src if v not in tgt], dtype=np.int64)
                    nI = len(inter)
                    if nI == 0 or nI == KI:
                        out["cells"][tlang].append({"fact": fi, "nI": nI, "ll": None,
                                                    "fallback": fb[i]})
                        continue
                    rng = np.random.default_rng([1002, fi, ti, KI])
                    r1 = src[rng.choice(KI, size=nI, replace=False)]
                    r2 = src[rng.choice(KI, size=nI, replace=False)]
                    # control: same-relation f''s "most shared" |I| neurons of
                    # its own source top-K (its intersection first, then filled
                    # by attribution rank)
                    o_src = topk[nxt[i], si, :KI]
                    o_tgt = set(topk[nxt[i], ti, :KI].tolist())
                    o_first = [v for v in o_src if v in o_tgt]
                    o_rest = [v for v in o_src if v not in o_tgt]
                    ictl = np.array((o_first + o_rest)[:nI], dtype=np.int64)
                    d = data[tlang][fi]
                    lls = kn.ablation_ll(model, tok, device, ctxs[fi],
                                         [" " + c for c in d["choices"]],
                                         [None, inter, disj, r1, r2, ictl],
                                         width, max_seq, is_xla)
                    out["cells"][tlang].append({"fact": fi, "nI": nI, "ll": lls,
                                                "fallback": fb[i]})
                    if i % 200 == 0:
                        print(f"[fu] {run} intersect k{KI} {tlang} {i}/{nS} "
                              f"({time.time()-t0:.0f}s)")
            out["gold"] = {lang: [data[lang][fi]["label"] for fi in shared]
                           for lang in langs}
            kn.save_json(int_fs[KI], out)
            print(f"[fu] {run} intersect k{KI} done ({time.time()-t0:.0f}s)")

        print(f"[fu] {run} COMPLETE in {time.time()-t_run:.0f}s")
        if not args.keep_checkpoints:
            Path(ckpt).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
