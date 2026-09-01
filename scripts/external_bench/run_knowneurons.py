#!/usr/bin/env python
"""Cross-lingual knowledge-neuron sweep over the bilingual finals (CLAUDE.md 6j).

Per model, three phases (each resumable via its output file):

  A. SELECTION -- score all 4 PolyFact candidates in both training languages
     (raw loglikelihoods persisted, 6e's sidecar rule). A fact is "known" in a
     language if the gold candidate ranks first under plain `acc` (argmax of
     summed loglikelihood; PolyFact's estimator analysis found acc/acc_norm a
     dead tie, and the selection rule is identical across the conditions being
     contrasted). The attribution set is facts known in BOTH languages.

  B. ATTRIBUTION -- joint-path integrated gradients per (fact, language)
     (eval/knowneurons.py); stores the top-512 neurons + per-layer sums +
     completeness endpoints per cell.

  C. ABLATION -- per (fact, target language T, with S the other language),
     gold + distractor loglikelihoods under six masks:
       c0 none | c1 topK(T,f) | c2 topK(S,f) | c3 topK(T,f') | c4 topK(S,f')
       | c5 random-K
     f' is the next shared fact (deterministic derangement); random-K is
     seeded per (run, fact, T). c2-vs-c4 is the Ifergan-style functional
     transfer readout; c1-vs-c3 is specificity.

One fixed [batch, width] shape per phase, width snapped UP to a shared rung
over both languages so each phase compiles one weight-independent graph that
is reused for every model (facts longer than --max-width in EITHER language
are dropped in BOTH, keeping the sets aligned). All NEURON.md 4 rules are in
eval/knowneurons.py.

    python run_knowneurons.py --repo jvonrad/xscript-eval --device xla \
        --workdir /home/ubuntu/xscript_kn --runs en-de-fair en-de-starved ...
"""
import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

CONDITIONS = ("none", "own_fact", "cross_fact", "own_other", "cross_other", "random")


def fetch_checkpoint(repo, repo_type, repo_files, work, rel_dir: str) -> Path:
    """final.pt for one run, reassembling part-files (same as run_benchmarks)."""
    from huggingface_hub import hf_hub_download
    dl = dict(repo_id=repo, repo_type=repo_type,
              local_dir=str(work / "_repo"))
    whole = f"{rel_dir}/final.pt"
    if whole in repo_files:
        return Path(hf_hub_download(filename=whole, **dl))
    parts = sorted(f for f in repo_files if f.startswith(f"{rel_dir}/final.pt.part"))
    if not parts:
        sys.exit(f"no checkpoint under {rel_dir}")
    n_parts_f = f"{rel_dir}/n_parts.txt"
    if n_parts_f in repo_files:
        expected = int(Path(hf_hub_download(filename=n_parts_f, **dl)).read_text())
        if len(parts) != expected:
            sys.exit(f"{rel_dir}: expected {expected} parts, found {len(parts)}")
    out_dir = work / "_assembled" / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "final.pt"
    if not out.exists():
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as w:
            for p in parts:
                local = hf_hub_download(filename=p, **dl)
                with open(local, "rb") as r:
                    while chunk := r.read(64 * 1024 * 1024):
                        w.write(chunk)
                Path(local).unlink(missing_ok=True)
        tmp.rename(out)
    return out


def load_polyfact(langs):
    """{lang: list of {question, choices, label}} in aligned fact order."""
    from xscript.eval.c5_tasks.polyfact import utils as pf
    data = {}
    for lang in langs:
        ds = pf.build_dataset(lang=lang)["test"]
        data[lang] = [dict(question=d["question"], choices=d["choices"],
                           label=d["label"]) for d in ds]
    ns = {lang: len(v) for lang, v in data.items()}
    assert len(set(ns.values())) == 1, f"unaligned PolyFact sizes: {ns}"
    return data


def contexts_for(data, lang):
    from xscript.eval.c5_tasks.polyfact.utils import CUE
    return [f"{d['question']}\n{CUE[lang]}" for d in data[lang]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="jvonrad/xscript-eval")
    ap.add_argument("--repo-type", default="model")
    ap.add_argument("--workdir", default="/home/ubuntu/xscript_kn")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--device", default="xla")
    ap.add_argument("--batch-size", type=int, default=24,
                    help="rows per forward batch (phases A and C)")
    ap.add_argument("--k", type=int, default=100,
                    help="top-K neurons used for the ablation masks")
    ap.add_argument("--n-alpha", type=int, default=20)
    ap.add_argument("--max-facts", type=int, default=800,
                    help="cap on shared-correct facts per model (seeded sample)")
    ap.add_argument("--max-width", type=int, default=128,
                    help="facts longer than this in either language are dropped")
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--limit-facts", type=int, default=None,
                    help="debug: truncate the PolyFact roster to N facts")
    args = ap.parse_args()

    work = Path(args.workdir).resolve()
    res = work / "results" / "knowneurons"
    res.mkdir(parents=True, exist_ok=True)

    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from xscript.eval import knowneurons as kn
    from xscript.model import ModelConfig, Transformer
    from xscript.tok.wrapper import Tok

    is_xla = args.device == "xla"
    if is_xla:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device(args.device)

    dl = dict(repo_id=args.repo, repo_type=args.repo_type,
              local_dir=str(work / "_repo"))
    repo_files = list_repo_files(args.repo, repo_type=args.repo_type)
    for f in repo_files:
        if f.startswith("tokenizers/"):
            hf_hub_download(filename=f, **dl)
    models = json.loads(Path(hf_hub_download(filename="models.json", **dl)).read_text())

    for run in args.runs:
        t_run = time.time()
        if run not in models:
            print(f"[kn] {run}: not in models.json, skipping"); continue
        done = res / f"{run}_ablation.json"
        if done.exists():
            print(f"[kn] {run}: already complete"); continue

        print(f"\n===== {run} =====")
        ckpt = fetch_checkpoint(args.repo, args.repo_type, repo_files, work,
                                f"runs/{run}/checkpoints")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        tok_name = models[run]["tok"]
        assert tok_name == ck["cfg"]["tok_name"], (tok_name, ck["cfg"]["tok_name"])
        tok = Tok(work / "_repo" / "tokenizers" / tok_name)
        model = Transformer(ModelConfig(**ck["cfg"]["model"])).eval()
        model.load_state_dict(ck["model"])
        model = model.to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        langs = list(ck["cfg"]["langs"])
        assert len(langs) == 2, f"{run} is not bilingual: {langs}"
        del ck
        max_seq = model.cfg.max_seq_len
        print(f"[kn] langs={langs} tok={tok_name}")

        data = load_polyfact(langs)
        if args.limit_facts:
            data = {lang: v[:args.limit_facts] for lang, v in data.items()}
        n_facts = len(data[langs[0]])

        # ---- widths: drop facts too long in either language, one shared rung
        preps_len = {}
        for lang in langs:
            ctxs = contexts_for(data, lang)
            lens = []
            for i, d in enumerate(data[lang]):
                longest = 0
                for c in d["choices"]:
                    p = kn.prepare(tok, ctxs[i], " " + c, max_seq)
                    longest = max(longest, len(p.x))
                lens.append(longest)
            preps_len[lang] = lens
        keep = [i for i in range(n_facts)
                if all(preps_len[lang][i] <= args.max_width for lang in langs)]
        width = args.max_width
        print(f"[kn] width={width}, facts kept {len(keep)}/{n_facts}")

        # ---------------- phase A: selection ----------------
        sel_f = res / f"{run}_selection.json"
        sel = kn.load_json(sel_f)
        if sel is None:
            t0 = time.time()
            sel = {"run": run, "langs": langs, "width": width,
                   "kept_facts": keep, "ll": {}, "gold": {}}
            for lang in langs:
                ctxs = contexts_for(data, lang)
                flat_ctx, flat_cont = [], []
                for i in keep:
                    for c in data[lang][i]["choices"]:
                        flat_ctx.append(ctxs[i]); flat_cont.append(" " + c)
                lls = kn.score_candidates(model, tok, device, flat_ctx, flat_cont,
                                          width, args.batch_size, max_seq, is_xla)
                sel["ll"][lang] = [lls[4 * j:4 * j + 4] for j in range(len(keep))]
                sel["gold"][lang] = [data[lang][i]["label"] for i in keep]
                acc = float(np.mean([int(np.argmax(l4) == g) for l4, g in
                                     zip(sel["ll"][lang], sel["gold"][lang])]))
                print(f"[kn] {run} {lang}: acc={acc:.4f} (n={len(keep)}, "
                      f"{time.time()-t0:.0f}s)")
            kn.save_json(sel_f, sel)
        keep = sel["kept_facts"]          # resume: the stored mapping governs
        correct = {}
        for lang in langs:
            correct[lang] = {keep[j] for j in range(len(sel["kept_facts"]))
                             if int(np.argmax(sel["ll"][lang][j])) == sel["gold"][lang][j]}
        shared = sorted(correct[langs[0]] & correct[langs[1]])
        print(f"[kn] known: {langs[0]}={len(correct[langs[0]])} "
              f"{langs[1]}={len(correct[langs[1]])} shared={len(shared)}")
        if len(shared) > args.max_facts:
            shared = sorted(random.Random(f"kn:{run}").sample(shared, args.max_facts))

        # ---------------- phase B: attribution ----------------
        kn_f = res / f"{run}_kn.npz"
        if not kn_f.exists():
            t0 = time.time()
            L, Fd = model.cfg.n_layers, model.cfg.ffn_dim
            nS = len(shared)
            topk_idx = np.zeros((nS, 2, kn.TOPK_STORE), np.int32)
            topk_val = np.zeros((nS, 2, kn.TOPK_STORE), np.float32)
            layer_sum = np.zeros((nS, 2, L), np.float32)
            g1 = np.zeros((nS, 2), np.float32)
            g0 = np.zeros((nS, 2), np.float32)
            attr_tot = np.zeros((nS, 2), np.float32)
            for li, lang in enumerate(langs):
                ctxs = contexts_for(data, lang)
                for si, fi in enumerate(shared):
                    d = data[lang][fi]
                    gold = d["choices"][d["label"]]
                    prep = kn.prepare(tok, ctxs[fi], " " + gold, max_seq)
                    attr, a1, a0 = kn.attribute_fact(model, prep, device, width,
                                                     is_xla, args.n_alpha)
                    idx, val = kn.topk_flat(attr, kn.TOPK_STORE)
                    topk_idx[si, li], topk_val[si, li] = idx, val
                    layer_sum[si, li] = attr.reshape(L, Fd).sum(1).numpy()
                    g1[si, li], g0[si, li] = a1, a0
                    attr_tot[si, li] = float(attr.sum())
                    if si % 100 == 0:
                        print(f"[kn] {run} IG {lang} {si}/{nS} "
                              f"({time.time()-t0:.0f}s)")
            kn.save_npz(kn_f, fact_ids=np.array(shared, np.int32),
                        topk_idx=topk_idx, topk_val=topk_val,
                        layer_sum=layer_sum, g1=g1, g0=g0, attr_total=attr_tot,
                        langs=np.array(langs), k_store=np.array([kn.TOPK_STORE]))
            comp = np.abs(attr_tot - (g1 - g0)) / np.maximum(np.abs(g1 - g0), 1e-9)
            print(f"[kn] {run} IG done in {time.time()-t0:.0f}s; completeness "
                  f"median rel err {np.median(comp):.3f}")
        z = np.load(kn_f)
        shared = [int(v) for v in z["fact_ids"]]
        topk_idx = z["topk_idx"]

        # ---------------- phase C: ablation ----------------
        t0 = time.time()
        nS = len(shared)
        K = args.k
        n_neur = model.cfg.n_layers * model.cfg.ffn_dim
        abl = {"run": run, "langs": langs, "k": K, "conditions": list(CONDITIONS),
               "fact_ids": shared, "ll": {lang: [] for lang in langs}}
        for ti, tlang in enumerate(langs):
            si_other = 1 - ti
            ctxs = contexts_for(data, tlang)
            for si, fi in enumerate(shared):
                d = data[tlang][fi]
                nxt = (si + 1) % nS
                # zlib.crc32 not hash(): hash() is salted per process, and the
                # random-K control must be reproducible across resumes.
                rng = np.random.default_rng(
                    [zlib.crc32(run.encode()), fi, ti])
                masks = [None,
                         topk_idx[si, ti, :K],
                         topk_idx[si, si_other, :K],
                         topk_idx[nxt, ti, :K],
                         topk_idx[nxt, si_other, :K],
                         rng.choice(n_neur, size=K, replace=False)]
                lls = kn.ablation_ll(model, tok, device, ctxs[fi],
                                     [" " + c for c in d["choices"]],
                                     masks, width, max_seq, is_xla)
                abl["ll"][tlang].append(lls)
                if si % 200 == 0:
                    print(f"[kn] {run} ablation {tlang} {si}/{nS} "
                          f"({time.time()-t0:.0f}s)")
        abl["gold"] = {lang: [data[lang][fi]["label"] for fi in shared]
                       for lang in langs}
        kn.save_json(done, abl)
        print(f"[kn] {run} COMPLETE in {time.time()-t_run:.0f}s")

        if not args.keep_checkpoints:
            try:
                Path(ckpt).unlink(missing_ok=True)
            except OSError as e:
                print(f"[kn] cleanup warning: {e}")


if __name__ == "__main__":
    main()
