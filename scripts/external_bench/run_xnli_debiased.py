#!/usr/bin/env python
"""Debiased XNLI evaluation for XScript checkpoints (Neuron/XLA or CPU/CUDA).

Why this exists
---------------
lm-eval's XNLI is a cloze task: `doc_to_text` is empty and each of the three
choices is the whole ``{premise}, {Q}? {LABEL}, {hypothesis}`` string, differing
only by a one/two-token connective (Yes/Also/No). Scoring by raw summed
loglikelihood then rewards whichever connective has the highest *unconditional*
probability -- classic surface-form competition (Holtzman et al., 2021,
"Surface Form Competition"). Weak models collapse to always predicting the
entailment word, scoring exactly majority-class (~0.335).

Two concrete defects this fixes, both verified on the 30B-token 1B models:
  1. Arabic connectives in lm-eval are mistranslated -- CONTRADICTION is `رقم`
     ("number", not "no") and NEUTRAL is `لذا` ("therefore"). With the correct
     `لا` / `أيضا`, ar-fair goes 0.32 -> 0.45 under standard scoring.
  2. Surface-form competition hides real signal for Chinese: PMI (prior-
     normalized) scoring takes en-zh-fair from 0.33 -> 0.48.

So this runner reports BOTH scorings, with corrected connectives:
  * ``standard`` -- argmax of ll(premise, Q? L, hyp)       (the lm-eval metric)
  * ``pmi``      -- argmax of ll(premise, Q? L, hyp) - ll(Q? L, hyp), which
                    cancels the connective's prior (the premise-free "domain").

The model/tokenizer loading, Neuron fixed-shape scoring, and sharded-checkpoint
reassembly mirror run_benchmarks.py so numbers are directly comparable.

Usage:
    export HF_TOKEN=hf_...
    python run_xnli_debiased.py --repo jvonrad/xscript-eval --device xla --limit 200
"""
import argparse
import json
import os
import sys
from pathlib import Path

# XNLI connectives: (question_word, entailment, neutral, contradiction).
# en/de/fr/zh are lm-eval's originals; `ar` is CORRECTED (was لذا / رقم).
CONNECTIVES = {
    "en": ("right",   "Yes",  "Also",  "No"),
    "de": ("richtig", "Ja",   "Auch",  "Nein"),
    "fr": ("correct", "Oui",  "Aussi", "Non"),
    "ar": ("صحيح",    "نعم",  "أيضا",  "لا"),     # corrected: لا (no), أيضا (also)
    "zh": ("正确",     "是的", "所以",   "不是的"),
}
XNLI_LANG = {"en": "en", "de": "de", "fr": "fr", "ar": "ar", "zh": "zh"}


def fetch_checkpoint(rel_dir, repo_files, dl, work):
    """final.pt, reassembling final.pt.partNNN chunks (see upload_chunked.py)."""
    from huggingface_hub import hf_hub_download
    whole = f"{rel_dir}/final.pt"
    if whole in repo_files:
        return Path(hf_hub_download(filename=whole, **dl))
    parts = sorted(f for f in repo_files if f.startswith(f"{rel_dir}/final.pt.part"))
    if not parts:
        sys.exit(f"no checkpoint under {rel_dir}")
    out = work / "_assembled" / rel_dir / "final.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as w:
            for p in parts:
                local = hf_hub_download(filename=p, **dl)
                with open(local, "rb") as r:
                    while chunk := r.read(64 * 1024 * 1024):
                        w.write(chunk)
        tmp.rename(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--repo-type", default="model")
    ap.add_argument("--workdir", default="./xscript_bench")
    ap.add_argument("--device", default=None, help="cuda / cpu / xla")
    ap.add_argument("--limit", type=int, default=None, help="docs per language")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--keep-checkpoints", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files
    import torch
    import datasets

    work = Path(args.workdir).resolve()
    scratch = work / "xscript"
    (scratch / "tokenizers").mkdir(parents=True, exist_ok=True)
    os.environ["XSCRIPT_SCRATCH"] = str(scratch)
    os.environ.setdefault("XSCRIPT_RESULTS", str(work / "results"))
    dl = dict(repo_id=args.repo, repo_type=args.repo_type,
              local_dir=str(scratch.parent / "_repo"))
    repo_files = list_repo_files(args.repo, repo_type=args.repo_type)

    # bundled source + local-repo override (same as run_benchmarks.py)
    for f in repo_files:
        if f.startswith("src/xscript/"):
            hf_hub_download(filename=f, **dl)
    sys.path.insert(0, str(scratch.parent / "_repo" / "src"))
    _local = Path(__file__).resolve().parents[2] / "src"
    if (_local / "xscript" / "eval" / "bench.py").exists():
        sys.path.insert(0, str(_local))
    for f in repo_files:
        if f.startswith("tokenizers/"):
            local = hf_hub_download(filename=f, **dl)
            dest = scratch / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(local)

    from xscript.model import ModelConfig, Transformer
    from xscript.tok.wrapper import Tok, BOS_ID
    from xscript.paths import tokenizer_dir

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "xla":
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device(args.device)
    is_xla = getattr(device, "type", str(device)).startswith("xla")

    models = json.loads(Path(hf_hub_download(filename="models.json", **dl)).read_text())
    runs = args.runs or sorted(models)

    @torch.no_grad()
    def score_strings(model, tok, strings, fixed_width):
        """Total loglik of each full string (BOS-prefixed) under `model`.

        Fixed-shape on XLA (one graph per width); reuses the gather-free,
        host-clamped one-hot/logsumexp reduction proven correct in eval/bench.py.
        """
        import torch.nn.functional as F
        seqs = [[BOS_ID] + tok.encode(s, bos=False, eos=False) for s in strings]
        out = [0.0] * len(seqs)
        bs = args.batch_size
        for st in range(0, len(seqs), bs):
            chunk = seqs[st:st + bs]
            width = fixed_width if is_xla else max(len(s) - 1 for s in chunk)
            R = bs if is_xla else len(chunk)
            from xscript.tok.wrapper import PAD_ID
            x = torch.full((R, width), PAD_ID, dtype=torch.long)
            y = torch.full((R, width), -100, dtype=torch.long)
            lens = []
            for r, s in enumerate(chunk):
                m = len(s) - 1
                lens.append(m)
                x[r, :m] = torch.tensor(s[:-1])
                y[r, :m] = torch.tensor(s[1:])
            y_idx = y.clamp_min(0)  # clamp on host (on-device clamp -> one_hot OOB)
            x, y, y_idx = x.to(device), y.to(device), y_idx.to(device)
            logits, _ = model(x, y)
            logits = logits.float()
            onehot = F.one_hot(y_idx, logits.size(-1)).to(logits.dtype)
            tok_lp = (logits * onehot).sum(-1) - torch.logsumexp(logits, -1)  # [R,W]
            mask = (y != -100)
            tok_lp = torch.where(mask, tok_lp, torch.zeros_like(tok_lp))
            totals = tok_lp.sum(-1)
            if is_xla:
                xm.mark_step()
            totals = totals.cpu()
            for r in range(len(chunk)):
                out[st + r] = float(totals[r])
        return out

    summary = {}
    for ri, run in enumerate(runs, 1):
        tokname = models[run]["tok"]
        langs = [l for l in models[run]["langs"] if l in CONNECTIVES]
        if not langs:
            continue
        print(f"\n===== [{ri}/{len(runs)}] {run} (tok={tokname}) langs={langs} =====")
        ckpt = fetch_checkpoint(f"runs/{run}/checkpoints", repo_files, dl, work)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = Transformer(ModelConfig(**ck["cfg"]["model"])).to(device).eval()
        model.load_state_dict(ck["model"])
        tok = Tok(tokenizer_dir(tokname))
        res = {}
        for lang in langs:
            qw, ent, neu, con = CONNECTIVES[lang]
            conns = [ent, neu, con]
            ds = datasets.load_dataset("xnli", XNLI_LANG[lang], split="validation")
            n = len(ds) if args.limit is None else min(args.limit, len(ds))
            full_strs, null_strs, golds = [], [], []
            for i in range(n):
                d = ds[i]
                golds.append(d["label"])
                for c in conns:
                    full_strs.append(f"{d['premise']}, {qw}? {c}, {d['hypothesis']}")
                    null_strs.append(f"{qw}? {c}, {d['hypothesis']}")
            # one fixed width across full+null for this language (single XLA graph)
            fw = 1
            if is_xla:
                fw = max(len([BOS_ID] + tok.encode(s, bos=False, eos=False)) - 1
                         for s in full_strs + null_strs)
            full_ll = score_strings(model, tok, full_strs, fw)
            null_ll = score_strings(model, tok, null_strs, fw)
            std_ok = pmi_ok = 0
            for j, g in enumerate(golds):
                f3 = full_ll[3 * j:3 * j + 3]
                p3 = [f3[k] - null_ll[3 * j + k] for k in range(3)]
                std_ok += (max(range(3), key=lambda k: f3[k]) == g)
                pmi_ok += (max(range(3), key=lambda k: p3[k]) == g)
            res[lang] = {"n": n, "standard": std_ok / n, "pmi": pmi_ok / n}
            print(f"  xnli_{lang}: standard={std_ok/n:.4f}  pmi={pmi_ok/n:.4f}  (n={n})")
        summary[run] = res
        if not args.keep_checkpoints and "_assembled" in str(ckpt):
            try:
                Path(ckpt).unlink(missing_ok=True)
            except OSError:
                pass

    out = work / "results" / "xnli_debiased.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"limit": args.limit, "connectives": CONNECTIVES,
                               "scores": summary}, indent=2))
    print(f"\n[xnli] wrote {out}")


if __name__ == "__main__":
    main()
