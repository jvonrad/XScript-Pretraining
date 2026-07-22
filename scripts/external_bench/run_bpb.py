#!/usr/bin/env python
"""Recompute per-language BPB on FLORES+ for the uploaded checkpoints, with
per-sentence detail so BTS can carry bootstrap confidence intervals.

Why this exists (see CLAUDE.md "BPB -> BTS"): the committed `results/bts/*`
were produced on the training cluster from each run's `train.jsonl` eval
records, which are not reproducible here (`RUNS` points at a cluster path).
More importantly they mix token budgets -- the monolingual baselines are the
30B runs while a bilingual's per-language share is only ~15B -- and their
`bts_matched_lang` variant tries to correct for that by picking the mono
checkpoint nearest `total * mix_prob`, which silently degenerates to the
final checkpoint whenever a run has no checkpoint near that mark (this is
what happened to zh). The `-15b`/`-12b`/`-23b` checkpoints uploaded later are
the real matched-token design, so BTS can be computed directly with no
interpolation and no per-language special cases.

Scoring: each FLORES sentence is framed as a bare continuation (empty
context) and scored through `bench.XScriptLM._loglikelihood_tokens`, i.e. the
same verified fixed-shape Neuron path the downstream harness uses (including
the even-`fixed_width` NCC-5266 workaround). The returned loglik is the total
log P of the sentence's tokens given a leading BOS, so

    BPB = -sum(loglik) / (ln2 * total_utf8_bytes)

which is exactly `eval/bpb.py`'s definition (BOS never a target, EOS is a
target). `--verify-cpu` checks that equivalence against `bpb.score_texts`
directly rather than taking it on trust.

Per-sentence `(nll_nats, bytes)` are written out so `bts_matched.py` can
bootstrap the ratio-of-sums estimator over sentences, paired across models.

    python run_bpb.py --repo jvonrad/xscript-eval --runs ar-fair-15b \
        --device xla --workdir $WORK
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

FLORES_LANGS = ["en", "de", "fr", "ar", "zh"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--workdir", default="./xscript_bpb")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--langs", nargs="*", default=None, choices=FLORES_LANGS)
    ap.add_argument("--split", default="both", choices=["dev", "devtest", "both"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N sentences per language (debug)")
    ap.add_argument("--device", default=None, help="cuda / cpu / xla")
    ap.add_argument("--verify-cpu", action="store_true",
                    help="also score with eval/bpb.py's score_texts and assert "
                         "the two agree (CPU only, slow -- use with --limit)")
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--refresh-listing", action="store_true",
                    help="re-fetch the repo file listing instead of using the "
                         "cached _repo_files.json (needed after new uploads)")
    args = ap.parse_args()
    langs = args.langs or FLORES_LANGS

    from huggingface_hub import hf_hub_download, list_repo_files

    work = Path(args.workdir).resolve()
    scratch = work / "xscript"
    (scratch / "runs").mkdir(parents=True, exist_ok=True)
    (scratch / "tokenizers").mkdir(parents=True, exist_ok=True)
    os.environ["XSCRIPT_SCRATCH"] = str(scratch)
    os.environ["XSCRIPT_RESULTS"] = str(work / "results")

    dl = dict(repo_id=args.repo, repo_type=args.repo_type,
              local_dir=str(scratch.parent / "_repo"))

    # Cache the file listing: it is one API call per process, and a fan-out of
    # N processes issuing it simultaneously reliably 429s ("maximum queue size
    # reached") before any checkpoint transfer starts. Delete the file (or pass
    # --refresh-listing) after uploading new checkpoints.
    listing = work / "_repo_files.json"
    if listing.exists() and not args.refresh_listing:
        repo_files = json.loads(listing.read_text())
    else:
        import random as _rnd
        import time as _t
        for attempt in range(6):
            try:
                repo_files = list_repo_files(args.repo, repo_type=args.repo_type)
                break
            except Exception as exc:
                if attempt == 5:
                    raise
                wait = min(60, 2 ** attempt) + _rnd.uniform(0, 3)
                print(f"[bpb] {type(exc).__name__} listing repo; retry in {wait:.0f}s",
                      flush=True)
                _t.sleep(wait)
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text(json.dumps(repo_files))

    def _dl_retry(filename: str, tries: int = 6) -> Path:
        """hf_hub_download with backoff -- many of these run in parallel and
        the hub 429s ('maximum queue size reached') under fan-out."""
        import random
        import time
        for attempt in range(tries):
            try:
                return Path(hf_hub_download(filename=filename, **dl))
            except Exception as exc:
                if attempt == tries - 1:
                    raise
                wait = min(60, 2 ** attempt) + random.uniform(0, 3)
                print(f"[bpb] {type(exc).__name__} on {filename}; "
                      f"retry {attempt + 1}/{tries - 1} in {wait:.0f}s", flush=True)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def fetch_checkpoint(rel_dir: str) -> Path:
        whole = f"{rel_dir}/final.pt"
        if whole in repo_files:
            return _dl_retry(whole)
        parts = sorted(f for f in repo_files
                       if f.startswith(f"{rel_dir}/final.pt.part"))
        if not parts:
            sys.exit(f"no checkpoint under {rel_dir}")
        n_parts_f = f"{rel_dir}/n_parts.txt"
        if n_parts_f in repo_files:
            expected = int(_dl_retry(n_parts_f).read_text())
            if len(parts) != expected:
                sys.exit(f"{rel_dir}: expected {expected} parts, found {len(parts)}")
        assembled = work / "_assembled" / rel_dir
        assembled.mkdir(parents=True, exist_ok=True)
        out = assembled / "final.pt"
        if not out.exists():
            # Unique tmp per process + atomic replace: a shared "final.tmp" is
            # a race if two processes ever assemble the same checkpoint (one
            # renames it away, the other's rename then ENOENTs).
            tmp = out.with_suffix(f".tmp.{os.getpid()}")
            try:
                with open(tmp, "wb") as w:
                    for p in parts:
                        local = _dl_retry(p)
                        with open(local, "rb") as r:
                            while chunk := r.read(64 * 1024 * 1024):
                                w.write(chunk)
                        # NB: drops the cached part (4GB/model adds up), so a
                        # failed run re-downloads rather than resuming.
                        local.unlink(missing_ok=True)
                if out.exists():          # someone else won the race
                    tmp.unlink(missing_ok=True)
                else:
                    os.replace(tmp, out)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        return out

    # Prefer local assets over the HF copies: this script only ever runs from
    # inside the repo (it needs the local bench.py scoring path anyway), and
    # the tokenizers already live in the shared scratch tree. Downloading the
    # bundled src/ + tokenizers is then pure redundant HF traffic -- which
    # matters because anonymous/rate-limited access 429s well before the
    # checkpoints (the only genuinely remote asset) finish.
    src_root = scratch.parent / "_repo"
    _local_src = Path(__file__).resolve().parents[2] / "src"
    if (_local_src / "xscript" / "eval" / "bench.py").exists():
        sys.path.insert(0, str(_local_src))
    else:
        for f in repo_files:
            if f.startswith("src/xscript/"):
                hf_hub_download(filename=f, **dl)
        sys.path.insert(0, str(src_root / "src"))

    local_toks = Path(os.environ.get("XSCRIPT_TOKENIZERS",
                                     "/mnt/scratch/xscript/tokenizers"))
    for f in repo_files:
        if not f.startswith("tokenizers/"):
            continue
        dest = scratch / f
        if dest.exists():
            continue
        cand = local_toks / Path(f).relative_to("tokenizers")
        src_path = cand if cand.exists() else Path(hf_hub_download(filename=f, **dl))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(src_path)

    _mj = src_root / "models.json"
    if not _mj.exists():
        _mj = Path(hf_hub_download(filename="models.json", **dl))
    models = json.loads(_mj.read_text())
    runs = args.runs or sorted(models)
    missing = [r for r in runs if r not in models]
    if missing:
        sys.exit(f"models not in repo: {missing}")

    import torch
    from xscript.model import ModelConfig, Transformer
    from xscript.tok.wrapper import Tok
    from xscript.paths import tokenizer_dir, ensure
    from xscript.eval.bench import _make_lm
    from xscript import flores

    # FLORES lives in the shared scratch tree, not this workdir.
    flores_src = Path(os.environ.get("XSCRIPT_FLORES",
                                     "/mnt/scratch/xscript/flores_plus"))
    if not flores_src.exists():
        sys.exit(f"FLORES+ not found at {flores_src} (set XSCRIPT_FLORES)")
    link = scratch / "flores_plus"
    if not link.exists():
        link.symlink_to(flores_src)

    splits = ["dev", "devtest"] if args.split == "both" else [args.split]
    texts: dict[str, list[str]] = {l: [] for l in langs}
    for sp in splits:
        par = flores.load_parallel(langs, sp)
        for l in langs:
            texts[l].extend(par[l])
    if args.limit:
        texts = {l: t[:args.limit] for l, t in texts.items()}
    n_sent = len(texts[langs[0]])
    print(f"[bpb] {len(runs)} model(s) x {len(langs)} lang(s), "
          f"{n_sent} sentences/lang ({'+'.join(splits)})")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "xla":
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print("[bpb] using XLA/Neuron (fixed-shape scoring).")
    else:
        device = torch.device(args.device)

    out_dir = ensure(Path(work / "results" / "bpb"))
    for i, run in enumerate(runs, 1):
        tok_name = models[run]["tok"]
        print(f"\n===== [{i}/{len(runs)}] {run} (tok={tok_name}) =====", flush=True)
        local_ckpt = fetch_checkpoint(f"runs/{run}/checkpoints")
        dest = scratch / f"runs/{run}/checkpoints/final.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.symlink_to(local_ckpt)
        try:
            ck = torch.load(dest, map_location="cpu", weights_only=False)
            model = Transformer(ModelConfig(**ck["cfg"]["model"])).to(device).eval()
            model.load_state_dict(ck["model"])
            tok = Tok(tokenizer_dir(tok_name))
            adapter = _make_lm(model, tok, device, model.cfg.max_seq_len)
            adapter.batch_size = args.batch_size

            per_lang = {}
            for lang in langs:
                sents = texts[lang]
                # Bare continuation (empty context) => _prepare prepends exactly
                # one BOS and every real token is a scored target, matching
                # eval/bpb.py's convention (BOS not a target, EOS is).
                enc = [tok.encode(s, bos=False, eos=True) for s in sents]
                nbytes = [len(s.encode("utf-8")) for s in sents]
                reqs = [(None, [], e) for e in enc]
                scored = adapter._loglikelihood_tokens(reqs, disable_tqdm=True)
                nll = [-lp for lp, _ in scored]          # nats, per sentence
                tot_nll, tot_b = sum(nll), sum(nbytes)
                per_lang[lang] = {
                    "bpb": tot_nll / (math.log(2) * max(tot_b, 1)),
                    "nll_nats": nll,
                    "bytes": nbytes,
                    "n_tokens": sum(len(e) for e in enc),
                }
                print(f"[bpb] {run} / {lang}: bpb={per_lang[lang]['bpb']:.4f} "
                      f"(n={len(sents)}, {tot_b} bytes)", flush=True)

                if args.verify_cpu:
                    from xscript.eval.bpb import score_texts, bpb as _bpb
                    ref_nll, ref_b, _ = score_texts(model, tok, sents, device,
                                                    model.cfg.max_seq_len)
                    ref = _bpb(ref_nll, ref_b)
                    got = per_lang[lang]["bpb"]
                    print(f"[verify] {lang}: fixed-shape={got:.6f} "
                          f"score_texts={ref:.6f} |d|={abs(got-ref):.2e}")
                    assert abs(got - ref) < 1e-4, f"BPB mismatch for {lang}"
                    assert ref_b == tot_b, f"byte-count mismatch for {lang}"

            payload = {
                "run": run, "tokenizer": tok_name, "langs": langs,
                "split": args.split, "n_sentences": n_sent,
                "source": "flores", "per_lang": per_lang,
            }
            (out_dir / f"{run}_bpb.json").write_text(json.dumps(payload))
        except Exception as exc:
            print(f"[bpb] {run} FAILED: {type(exc).__name__}: {exc}")
            (out_dir / f"{run}_bpb.json").write_text(
                json.dumps({"run": run, "error": f"{type(exc).__name__}: {exc}"}))
        finally:
            if not args.keep_checkpoints:
                try:
                    real = Path(local_ckpt).resolve()
                    dest.unlink(missing_ok=True)
                    real.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"[bpb] cleanup warning for {run}: {exc}")

    print(f"\n[bpb] per-model JSON in {out_dir}")


if __name__ == "__main__":
    main()
