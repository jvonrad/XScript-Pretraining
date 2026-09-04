#!/usr/bin/env python
"""PolyFact candidate scores + FLORES+ BPB for a list of checkpoints, so
cross-lingual consistency (RankC, analyze_polyfact_traj.py) can be compared
between the two tokenizer conditions at MATCHED validation loss rather than
at matched tokens -- i.e. "is the fair tokenizer's higher consistency a
property of the tokenizer or just of a better model?".

Per checkpoint, one JSON with:
  ll[lang][fact] = 4 summed loglikelihoods of " " + option, given
                   "<question>\\n<localized cue>"  (run_knowneurons phase A,
                   verbatim: same contexts, same continuation form)
  option_ids[lang][fact] = the 4 Wikidata QIDs in that language's option
                   order (PolyFact shuffles positions per language)
  gold[lang][fact], and
  bpb[lang] = FLORES+ dev+devtest bits-per-byte (sum NLL / sum UTF-8 bytes,
                   BOS-prefixed whole-sentence scoring, same convention as
                   run_bpb.py), plus per-sentence NLL/bytes for bootstrapping.

Two fixed-shape graphs (PolyFact width 64, FLORES width 144 -- both cover
every request under both tokenizers, measured). Checkpoints are the
reassembled `final.pt` files under <workdir>/_assembled/runs/<name>/.

    python run_polyfact_traj.py --runs en-de-fair-2b en-de-fair-5b ...
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "src"))

PF_WIDTH = 64
FL_WIDTH = 144


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="/mnt/scratch/xscript_rankc")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="xla")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-flores", action="store_true")
    args = ap.parse_args()

    work = Path(args.workdir)
    os.environ.setdefault("XSCRIPT_SCRATCH", "/mnt/scratch/xscript")
    models = json.loads((work / "_repo" / "models.json").read_text())
    out_dir = work / "results" / "polyfact_traj"
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import datasets
    from xscript.model import ModelConfig, Transformer
    from xscript.tok.wrapper import Tok
    from xscript.paths import tokenizer_dir
    from xscript.eval.c5_tasks.polyfact.utils import CUE
    from xscript.eval.knowneurons import prepare, batch_tensors, gold_loglik
    from xscript import flores

    if args.device == "xla":
        import torch_xla.core.xla_model as xm
        device = xm.xla_device(); is_xla = True
    else:
        device = torch.device(args.device); is_xla = False

    pf = {}
    for lang in ("en", "de", "fr", "ar", "zh"):
        ds = datasets.load_dataset("jvonrad/PolyFact", lang, split="test")
        rows = [dict(question=r["question"], choices=[r["option_a"], r["option_b"], r["option_c"], r["option_d"]],
                     label=int(r["answer_index"]), ids=list(r["option_ids"]), fid=r["fact_id"]) for r in ds]
        if args.limit:
            rows = rows[:args.limit]
        pf[lang] = rows
    fl = {}
    if not args.no_flores:
        for sp in ("dev", "devtest"):
            par = flores.load_parallel(["en", "de", "fr", "ar", "zh"], sp)
            for l, s in par.items():
                fl.setdefault(l, []).extend(s)
        if args.limit:
            fl = {l: s[:args.limit] for l, s in fl.items()}

    def full_logits(model, x):
        h = model.tok_emb(x)
        cos, sin = model._rope_for(x.shape[1], x.device, h.dtype)
        for layer in model.layers:
            h = layer(h, cos, sin)
        return model.lm_head(model.norm(h))

    @torch.no_grad()
    def score(model, tok, reqs, width, max_seq_len):
        preps = [prepare(tok, c, k, max_seq_len) for c, k in reqs]
        out = []
        bs = args.batch_size
        for i in range(0, len(preps), bs):
            chunk = preps[i:i + bs]
            x, y, mask = batch_tensors(chunk + [chunk[-1]] * (bs - len(chunk)), width)
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            ll = gold_loglik(full_logits(model, x), y, mask)
            if is_xla:
                xm.mark_step()
            ll = ll.cpu()
            out.extend(float(ll[j]) for j in range(len(chunk)))
        return out

    for run in args.runs:
        out_path = out_dir / f"{run}.json"
        if out_path.exists() and not args.limit:
            print(f"[traj] {run}: exists, skip", flush=True); continue
        tok_name = models[run]["tok"]; langs = models[run]["langs"]
        ck_path = work / "_assembled" / "runs" / run / "checkpoints" / "final.pt"
        if not ck_path.exists():
            print(f"[traj] {run}: checkpoint missing, skip", flush=True); continue
        t0 = time.time()
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = Transformer(ModelConfig(**ck["cfg"]["model"])).to(device).eval()
        model.load_state_dict(ck["model"])
        max_seq_len = ck["cfg"]["model"]["max_seq_len"]
        tokens = ck.get("tokens"); step = ck.get("step")
        del ck
        tok = Tok(tokenizer_dir(tok_name))
        rec = {"run": run, "tok": tok_name, "langs": langs, "tokens": tokens, "step": step,
               "ll": {}, "gold": {}, "option_ids": {}, "fact_id": [r["fid"] for r in pf[langs[0]]],
               "bpb": {}, "flores_nll": {}, "flores_bytes": {}}
        for lang in langs:
            reqs = [(f"{r['question']}\n{CUE[lang]}", " " + c) for r in pf[lang] for c in r["choices"]]
            ll = score(model, tok, reqs, PF_WIDTH, max_seq_len)
            rec["ll"][lang] = [ll[4 * i:4 * i + 4] for i in range(len(pf[lang]))]
            rec["gold"][lang] = [r["label"] for r in pf[lang]]
            rec["option_ids"][lang] = [r["ids"] for r in pf[lang]]
            if fl:
                nll = [-v for v in score(model, tok, [("", s) for s in fl[lang]], FL_WIDTH, max_seq_len)]
                nb = [len(s.encode("utf-8")) for s in fl[lang]]
                import math
                rec["flores_nll"][lang] = nll; rec["flores_bytes"][lang] = nb
                rec["bpb"][lang] = sum(nll) / sum(nb) / math.log(2)
        json.dump(rec, open(out_path, "w"))
        print(f"[traj] {run}: done in {time.time()-t0:.0f}s  bpb={ {l: round(v, 4) for l, v in rec['bpb'].items()} }",
              flush=True)
        del model
    print("TRAJ_DONE", flush=True)


if __name__ == "__main__":
    main()
