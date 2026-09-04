#!/usr/bin/env python
"""Per-fact cross-lingual representation alignment on PolyFact questions --
the mediation test for CLAUDE.md 6k ("is the fair tokenizer's consistency
gain carried by more aligned representations rather than better recall?").

For each checkpoint and each of its two languages, embeds every PolyFact
question (the question text alone, no cue, no options) with the SAME
fixed-shape XLA path 6b's alignment sweep uses (`alignment._embed`:
mean-pooled, L2-normalised, every layer) and stores the
`(n_layers+1, n_facts, dim)` fp32 stack per language. analyze_fact_align.py
then joins these, fact-for-fact, with the stored candidate scores
(polyfact_traj/<run>.json) to relate per-fact alignment to per-fact answer
consistency.

    python run_fact_align.py --runs en-de-fair en-de-starved ...
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/mnt/scratch/xscript_rankc")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="xla")
    ap.add_argument("--width", type=int, default=64)
    args = ap.parse_args()
    work = Path(args.workdir)
    os.environ.setdefault("XSCRIPT_SCRATCH", "/mnt/scratch/xscript")
    models = json.loads((work / "_repo" / "models.json").read_text())
    out_dir = work / "results" / "fact_align"
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    import datasets
    from xscript.model import ModelConfig, Transformer
    from xscript.tok.wrapper import Tok
    from xscript.paths import tokenizer_dir
    from xscript.eval.alignment import _encode, _embed

    if args.device == "xla":
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device(args.device)

    q = {}
    for lang in ("en", "de", "fr", "ar", "zh"):
        ds = datasets.load_dataset("jvonrad/PolyFact", lang, split="test")
        q[lang] = (list(ds["question"]), list(ds["fact_id"]))
    for run in args.runs:
        out_path = out_dir / f"{run}.npz"
        if out_path.exists():
            print(f"[falign] {run}: exists, skip", flush=True); continue
        langs = models[run]["langs"]; tok_name = models[run]["tok"]
        ck_path = work / "_assembled" / "runs" / run / "checkpoints" / "final.pt"
        if not ck_path.exists():
            print(f"[falign] {run}: checkpoint missing, skip", flush=True); continue
        t0 = time.time()
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = Transformer(ModelConfig(**ck["cfg"]["model"])).to(device).eval()
        model.load_state_dict(ck["model"]); seq_len = ck["cfg"]["model"]["max_seq_len"]; del ck
        tok = Tok(tokenizer_dir(tok_name))
        arrays = {}
        for lang in langs:
            texts, fids = q[lang]
            seqs = _encode(tok, texts, seq_len)
            longest = max(len(s) for s in seqs)
            if longest > args.width:
                seqs = [s[:args.width] for s in seqs]   # truncate the rare long question
            with torch.no_grad():
                emb = _embed(model, seqs, device, batch=args.batch_size, fixed_width=args.width)["mean"]
            arrays[lang] = emb.astype(np.float32)
            print(f"[falign] {run} {lang}: {emb.shape} longest={longest} ({time.time()-t0:.0f}s)", flush=True)
        np.savez(out_path, fact_id=np.array(q[langs[0]][1]), langs=np.array(langs), **arrays)
        del model
    print("FALIGN_DONE", flush=True)


if __name__ == "__main__":
    main()
