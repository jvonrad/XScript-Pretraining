#!/usr/bin/env python
"""Score BMLAMA-17 / BMLAMA-53 candidates in a bilingual model's two
languages so RankC (Qi et al. 2023, arXiv:2310.10378, "Cross-Lingual
Consistency of Factual Knowledge in Multilingual Language Models") can be
computed on CPU afterwards (analyze_rankc.py). Only raw per-candidate
scores are persisted here (CLAUDE.md 6e's sidecar rule).

Data: `1_easyrun/BMLAMA{17,53}/<lang>.tsv` from github.com/Betswish/
Cross-Lingual-Consistency, columns Prompt / Ans / Candidates / Subject.
Rows are index-aligned across languages and candidates are POSITION-aligned
(verified: 0 candidate-count mismatches, gold always among the candidates,
exactly one <mask> per prompt). BMLAMA-17 covers en/fr/ar/zh (no German);
BMLAMA-53 covers all five, with 3070 instead of 6792 queries.

Two scores per (row, candidate, language), from one fixed-shape XLA graph
per width rung:

  whole  -- the reference implementation's causal-LM rule: substitute the
            candidate into the prompt and score the WHOLE sentence; the
            reference ranks by MEAN per-token cross-entropy (so we store the
            summed loglikelihood AND the token count; analyze_rankc.py
            derives both the faithful mean-CE ranking and a summed variant).
  cand   -- summed loglikelihood of the candidate tokens given the prompt
            prefix up to <mask> (suffix dropped): the per-candidate-only
            estimator this project uses everywhere else.

Fixed-shape conventions follow eval/knowneurons.py (host-built tensors,
one-hot target selection, float masks, even widths). Rows whose longest
request exceeds --max-width in EITHER language are dropped in BOTH.

    python run_rankc.py --workdir /mnt/scratch/xscript_rankc --device xla \
        --runs en-de-fair en-de-starved ... [--sets 17 53] [--limit N]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "src"))

BMLAMA_DIR = Path(os.environ.get("BMLAMA_DIR", "/mnt/scratch/bmlama"))
WIDTH_RUNGS = (32, 48, 64, 96, 128, 192, 256)


def load_mubench(lang: str):
    """MuBench's re-extracted BMLAMA (aialt/MuBench, `_id`-aligned across
    languages, 6016 items, cleaner candidate sets than the original -- see
    CLAUDE.md 6e's MuBench notes). Returned in the same row schema as
    load_tsv: `prompt` is the question line with MuBench's `_` blank rewritten
    as <mask> (so the `whole` variant substitutes exactly as for the
    original), and `stem` is the full cloze context ending in the localized
    answer cue for the `cand` variant."""
    from xscript.eval.c5_tasks.mubench.utils import _load
    rows = []
    for r in _load("bmlama", lang):
        q = r["stem"].split("\n")[0]
        rows.append({"_id": r["_id"], "prompt": q.replace("_", "<mask>", 1) if "_" in q else None,
                     "stem": r["stem"], "gold": [r["options"][r["label"]]], "cands": r["options"]})
    return rows


def load_tsv(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = line.rstrip("\n").split("\t")
            if d[0] == "Prompt" or len(d) <= 1:
                continue
            rows.append({"prompt": d[0], "gold": d[1].split(", "),
                         "cands": d[2].split(", ")})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="/mnt/scratch/xscript_rankc")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--sets", nargs="*", default=["17", "53"],
                    help="17 / 53 = original Qi et al. TSVs; mub = MuBench's BMLAMA")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-width", type=int, default=128)
    ap.add_argument("--device", default="xla")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (debug/warm)")
    ap.add_argument("--only-cand", action="store_true",
                    help="re-score only the `cand` variant and merge it into an "
                         "existing result file (used after the boundary fix)")
    ap.add_argument("--width", type=int, default=None,
                    help="pin the fixed width (one graph for every run/set; rows longer "
                         "than it are dropped). Measured: every BMLAMA request is <= 48 "
                         "tokens under both tokenizers, so 48 is the natural pin.")
    args = ap.parse_args()

    work = Path(args.workdir)
    os.environ.setdefault("XSCRIPT_SCRATCH", "/mnt/scratch/xscript")
    models = json.loads((work / "_repo" / "models.json").read_text())
    out_dir = work / "results" / "rankc"
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from xscript.model import ModelConfig, Transformer
    from xscript.tok.wrapper import Tok
    from xscript.paths import tokenizer_dir
    from xscript.eval.knowneurons import prepare, batch_tensors, gold_loglik, even_width

    if args.device == "xla":
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        is_xla = True
    else:
        device = torch.device(args.device)
        is_xla = False

    def full_logits(model, x):
        """Logits at EVERY position. Transformer.forward(idx) without targets
        returns only the last position (model.py), so walk the blocks here --
        the same path eval/knowneurons.forward_ffn_scaled takes."""
        h = model.tok_emb(x)
        cos, sin = model._rope_for(x.shape[1], x.device, h.dtype)
        for layer in model.layers:
            h = layer(h, cos, sin)
        return model.lm_head(model.norm(h))

    @torch.no_grad()
    def score(model, tok, reqs, width, max_seq_len):
        """reqs: list of (context, continuation) -> list of (sum_ll, n_cont)."""
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
            out.extend((float(ll[j]), chunk[j].n_cont) for j in range(len(chunk)))
        return out

    for run in args.runs:
        tok_name = models[run]["tok"]
        langs = models[run]["langs"]
        ck_path = work / "_assembled" / "runs" / run / "checkpoints" / "final.pt"
        if not ck_path.exists():
            sys.exit(f"{run}: checkpoint missing at {ck_path} (run fetch_ckpts.py)")
        t0 = time.time()
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = Transformer(ModelConfig(**ck["cfg"]["model"])).to(device).eval()
        model.load_state_dict(ck["model"])
        max_seq_len = ck["cfg"]["model"]["max_seq_len"]
        del ck
        tok = Tok(tokenizer_dir(tok_name))
        print(f"[rankc] {run} loaded ({tok_name}, langs {langs}) in {time.time()-t0:.0f}s", flush=True)

        for s in args.sets:
            if s == "17" and "de" in langs:
                continue
            out_path = out_dir / f"{run}_bmlama{s}.json"
            if out_path.exists() and not args.limit and not args.only_cand:
                print(f"[rankc] {run} bmlama{s}: exists, skip", flush=True)
                continue
            prev = json.load(open(out_path)) if args.only_cand and out_path.exists() else None
            if s == "mub":
                rows = {l: load_mubench(l) for l in langs}
                ids = [r["_id"] for r in rows[langs[0]]]
                assert all([r["_id"] for r in rows[l]] == ids for l in langs), "MuBench ids differ"
                # keep only rows whose question line carries the blank in every language
                ok = [i for i in range(len(ids)) if all(rows[l][i]["prompt"] for l in langs)]
                rows = {l: [rows[l][i] for i in ok] for l in langs}
                print(f"[rankc] mubench: {len(ok)}/{len(ids)} rows have a `_` blank in all langs", flush=True)
            else:
                rows = {l: load_tsv(BMLAMA_DIR / f"BMLAMA{s}" / f"{l}.tsv") for l in langs}
            n = len(rows[langs[0]])
            assert all(len(r) == n for r in rows.values())
            if args.limit:
                rows = {l: r[:args.limit] for l, r in rows.items()}
                n = args.limit
            # requests per language: whole-sentence and candidate-only
            reqs = {l: {"whole": [], "cand": []} for l in langs}
            row_len = [0] * n
            for i in range(n):
                for l in langs:
                    r = rows[l][i]
                    assert len(r["cands"]) == len(rows[langs[0]][i]["cands"])
                    pre, suf = r["prompt"].split("<mask>", 1)
                    # candidate-only: strip the prefix's trailing space and
                    # prepend it to the continuation, else SentencePiece
                    # merges it into the candidate's first piece and the
                    # continuation span collapses (lm-eval's _encode_pair
                    # convention; the first version scored at chance).
                    ctx = pre.rstrip(" ")
                    lead = pre[len(ctx):]
                    if "stem" in r:            # MuBench: candidate after the answer cue
                        ctx, lead = r["stem"], " "
                    for c in r["cands"]:
                        reqs[l]["whole"].append(("", pre + c + suf))
                        reqs[l]["cand"].append((ctx, lead + c))
                        row_len[i] = max(row_len[i], len(tok.encode(pre + c + suf)) + 1,
                                         len(tok.encode(ctx + lead + c)) + 1)
            max_w = args.width or args.max_width
            keep = [i for i in range(n) if row_len[i] <= max_w]
            width = args.width or next(w for w in WIDTH_RUNGS if w >= max(row_len[i] for i in keep))
            width = even_width(width)
            dropped = n - len(keep)
            keep_set = set(keep)
            # filter requests to kept rows (same candidate offsets per language)
            offs, k = [], 0
            for i in range(n):
                m = len(rows[langs[0]][i]["cands"])
                offs.append((k, m)); k += m
            sel = [j for i in keep for j in range(offs[i][0], offs[i][0] + offs[i][1])]
            res = {}
            t1 = time.time()
            variants = ("cand",) if args.only_cand else ("whole", "cand")
            for l in langs:
                for v in variants:
                    rq = [reqs[l][v][j] for j in sel]
                    res[(l, v)] = score(model, tok, rq, width, max_seq_len)
                print(f"[rankc] {run} bmlama{s} {l}: {len(sel)} x2 requests, width {width}, "
                      f"{time.time()-t1:.0f}s", flush=True)
            # assemble per-row records
            out_rows = []
            p = 0
            for i in keep:
                m = offs[i][1]
                rec = {"row": i, "n_cands": m,
                       "gold": [rows[langs[0]][i]["cands"].index(g) for g in rows[langs[0]][i]["gold"]],
                       **({"_id": rows[langs[0]][i]["_id"]} if "_id" in rows[langs[0]][i] else {})}
                for l in langs:
                    rec[l] = {v: res[(l, v)][p:p + m] for v in variants}
                    if prev is not None:   # keep the stored `whole` scores
                        old_rec = next(rr for rr in prev["rows"] if rr["row"] == i)
                        rec[l]["whole"] = old_rec[l]["whole"]
                    # gold index must agree across languages (position-aligned)
                    gl = [rows[l][i]["cands"].index(g) for g in rows[l][i]["gold"]]
                    if gl != rec["gold"]:
                        rec["gold_mismatch"] = True
                out_rows.append(rec)
                p += m
            json.dump({"run": run, "tok": tok_name, "langs": langs, "set": s, "width": width,
                       "n_rows_total": n, "n_dropped": dropped, "rows": out_rows},
                      open(out_path, "w"))
            print(f"[rankc] {run} bmlama{s}: wrote {len(out_rows)} rows "
                  f"({dropped} dropped > width {args.max_width}) in {time.time()-t1:.0f}s",
                  flush=True)
        del model
    print("RANKC_DONE", flush=True)


if __name__ == "__main__":
    main()
