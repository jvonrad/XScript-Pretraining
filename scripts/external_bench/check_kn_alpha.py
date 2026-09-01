#!/usr/bin/env python
"""Robustness check: does the IG top-K depend on the alpha discretization?

Recomputes the joint-path IG with n_alpha=100 for a fact subsample of one
completed run and reports the top-K Jaccard against the stored n_alpha=20
maps. The real 1B model's completeness error (median .08-.37) is dominated by
path nonlinearity that a finer grid does not remove -- what matters for every
downstream statistic is whether the NEURON RANKING is stable. dKS is a
difference of Jaccards, so ranking stability is the operative guarantee.

    python check_kn_alpha.py --run en-de-fair [--n 30] [--n-alpha 100]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="en-de-fair")
    ap.add_argument("--workdir", default="/home/ubuntu/xscript_kn")
    ap.add_argument("--repo", default="jvonrad/xscript-eval")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-alpha", type=int, default=100)
    ap.add_argument("--device", default="xla")
    ap.add_argument("--ks", type=int, nargs="*", default=[50, 100, 200])
    a = ap.parse_args()

    import torch
    from huggingface_hub import list_repo_files
    from xscript.eval import knowneurons as kn
    from xscript.model import ModelConfig, Transformer
    from xscript.tok.wrapper import Tok
    from run_knowneurons import fetch_checkpoint, load_polyfact, contexts_for

    work = Path(a.workdir)
    res = work / "results" / "knowneurons"
    z = np.load(res / f"{a.run}_kn.npz")
    sel = json.loads((res / f"{a.run}_selection.json").read_text())
    langs = [str(v) for v in z["langs"]]
    width = sel["width"]

    is_xla = a.device == "xla"
    if is_xla:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device(a.device)

    repo_files = list_repo_files(a.repo)
    ckpt = fetch_checkpoint(a.repo, "model", repo_files, work,
                            f"runs/{a.run}/checkpoints")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    tok = Tok(work / "_repo" / "tokenizers" / ck["cfg"]["tok_name"])
    model = Transformer(ModelConfig(**ck["cfg"]["model"])).eval()
    model.load_state_dict(ck["model"])
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    max_seq = model.cfg.max_seq_len

    data = load_polyfact(langs)
    fact_ids = z["fact_ids"]
    pick = np.random.default_rng(0).choice(len(fact_ids), a.n, replace=False)

    jac = {k: [] for k in a.ks}
    comp20, compN = [], []
    for li, lang in enumerate(langs):
        ctxs = contexts_for(data, lang)
        for si in pick:
            fi = int(fact_ids[si])
            d = data[lang][fi]
            gold = d["choices"][d["label"]]
            prep = kn.prepare(tok, ctxs[fi], " " + gold, max_seq)
            # fine grid in chunks of 20 so every chunk reuses the fleet's
            # compiled [22, width] graph (a 102-row graph exceeds the
            # compiler's 5M-instruction limit, NCC_EBVF030)
            chunks = a.n_alpha // 20
            fine = [(k + 0.5) / a.n_alpha for k in range(a.n_alpha)]
            acc = None
            for c in range(chunks):
                attr, g1, g0 = kn.attribute_fact(
                    model, prep, device, width, is_xla, 20,
                    alpha_grid=fine[c * 20:(c + 1) * 20])
                acc = attr if acc is None else acc + attr
            attr = acc / chunks
            idx_new, _ = kn.topk_flat(attr, kn.TOPK_STORE)
            compN.append(abs(float(attr.sum()) - (g1 - g0)) / max(abs(g1 - g0), 1e-9))
            old = z["topk_idx"][si, li]
            comp20.append(abs(float(z["attr_total"][si, li])
                              - (z["g1"][si, li] - z["g0"][si, li]))
                          / max(abs(z["g1"][si, li] - z["g0"][si, li]), 1e-9))
            for k in a.ks:
                sa, sb = set(old[:k].tolist()), set(idx_new[:k].tolist())
                jac[k].append(len(sa & sb) / len(sa | sb))

    print(f"\n{a.run}: n_alpha 20 vs {a.n_alpha}, {a.n} facts x {langs}")
    for k in a.ks:
        v = np.array(jac[k])
        print(f"  top-{k:<4} Jaccard: mean {v.mean():.4f}  min {v.min():.4f}")
    print(f"  completeness rel err: median {np.median(comp20):.4f} (n=20) "
          f"-> {np.median(compN):.4f} (n={a.n_alpha})")


if __name__ == "__main__":
    main()
