#!/usr/bin/env python
"""Cross-lingual representation alignment (MEXA-style) for XScript checkpoints,
on Neuron/XLA (``--device xla``), CUDA, or CPU. The login node can't hold a live
~1B model at all, so like benchmarking this has to run on an accelerator box.

**Every model is evaluated on every language pair by default**, not just the
pairs it was trained on. A bilingual EN-AR model's EN-AR retrieval score means
nothing on its own -- FLORES sentences share digits, Latin-script named
entities, punctuation and length, so even a model that never saw Arabic scores
well above 1/n. The monolingual models scored on the *same* pair are the
control that turns the number into evidence, and they also give the
zero-shot readout ("does EN-DE training buy you EN-AR alignment?"). Restrict
with ``--langs`` if you only want the trained cells.

    export HF_TOKEN=hf_...
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; source ~/neuron_venv/bin/activate
    python run_alignment.py --repo jvonrad/xscript-eval --device xla --workdir $WORK

Writes ``$WORK/results/alignment/<model>.json`` (+ ``.md``). Like
run_appendix_c5.py it deliberately writes **no shared summary.json**, so a
fan-out over logical core-pairs can't clobber itself (NEURON.md section 5);
aggregate with analyze_alignment.py.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ALL_LANGS = ["en", "de", "fr", "ar", "zh"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--workdir", default="./xscript_bench")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="subset of friendly model names (default: all in models.json)")
    ap.add_argument("--langs", nargs="*", default=ALL_LANGS,
                    help=f"languages to embed (default: {' '.join(ALL_LANGS)}). "
                         "All unordered pairs among these are scored.")
    ap.add_argument("--split", default="both", choices=["dev", "devtest", "both"],
                    help="'both' = dev+devtest (2009 sentences). Default, because "
                         "top-1 retrieval over dev alone (997) saturates above "
                         "0.95 even for monolingual controls.")
    ap.add_argument("--device", default=None, choices=["xla", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="truncate sentences to this many tokens; also caps the "
                         "fixed graph width on XLA")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N parallel sentences (smoke tests)")
    ap.add_argument("--emb-dir", default=None,
                    help="cache the pooled per-layer embeddings here (~1.4 GB "
                         "per model). Lets any new metric be recomputed on CPU "
                         "without re-downloading checkpoints or re-running the "
                         "forward pass (84%% of runtime).")
    ap.add_argument("--keep-checkpoints", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    work = Path(args.workdir).resolve()
    scratch = work / "xscript"
    (scratch / "runs").mkdir(parents=True, exist_ok=True)
    (scratch / "tokenizers").mkdir(parents=True, exist_ok=True)
    os.environ["XSCRIPT_SCRATCH"] = str(scratch)
    os.environ["XSCRIPT_RESULTS"] = str(work / "results")

    dl = dict(repo_id=args.repo, repo_type=args.repo_type,
              local_dir=str(scratch.parent / "_repo"))

    def _dl_retry(filename: str, tries: int = 8) -> Path:
        """hf_hub_download with backoff -- the hub 429s ('maximum queue size
        reached') when a fan-out starts N processes at once."""
        import random
        import time
        for attempt in range(tries):
            try:
                return Path(hf_hub_download(filename=filename, **dl))
            except Exception as exc:
                if attempt == tries - 1:
                    raise
                wait = min(60, 2 ** attempt) + random.uniform(0, 3)
                print(f"[align] {type(exc).__name__} on {filename}; "
                      f"retry {attempt + 1}/{tries - 1} in {wait:.0f}s", flush=True)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    # Repo listing + sizes, cached on disk: N parallel `list_repo_files` calls
    # 429 before anything transfers. The sizes let fetch_checkpoint validate an
    # already-assembled checkpoint with no network call at all.
    _listing = work / "_repo_files.json"
    sizes = {}
    if _listing.exists():
        try:
            sizes = json.loads(_listing.read_text())
        except json.JSONDecodeError:
            sizes = {}
    if not sizes:
        from huggingface_hub import HfApi
        _info = HfApi().repo_info(args.repo, repo_type=args.repo_type,
                                  files_metadata=True)
        sizes = {s.rfilename: (s.size or 0) for s in _info.siblings}
        _tmp = _listing.with_suffix(f".tmp.{os.getpid()}")
        _tmp.write_text(json.dumps(sizes))
        os.replace(_tmp, _listing)      # atomic: processes race to write it
    repo_files = list(sizes)

    def fetch_checkpoint(rel_dir: str) -> Path:
        parts = sorted(f for f in repo_files if f.startswith(f"{rel_dir}/final.pt.part"))
        # Already assembled (by prefetch_checkpoints.py, or a previous sweep
        # sharing this workdir)? Then touch the network for nothing -- not even
        # n_parts.txt, which is pure 429 bait under fan-out and raises outright
        # under HF_HUB_OFFLINE. Validate size only when the manifest gives one;
        # size 0 means offline metadata, so trust the existing assembly.
        out = work / "_assembled" / rel_dir / "final.pt"
        if parts and out.exists():
            want = sum(sizes.get(p, 0) for p in parts)
            if not want or out.stat().st_size == want:
                return out
            print(f"[align] {rel_dir}: size {out.stat().st_size} != {want}, refetching",
                  flush=True)
            out.unlink(missing_ok=True)

        whole = f"{rel_dir}/final.pt"
        if whole in repo_files:
            return _dl_retry(whole)
        if not parts:
            sys.exit(f"no checkpoint found under {rel_dir}")
        n_parts_rel = f"{rel_dir}/n_parts.txt"
        if n_parts_rel in repo_files:
            want_n = int(_dl_retry(n_parts_rel).read_text().strip())
            if want_n != len(parts):
                sys.exit(f"{rel_dir}: expected {want_n} parts, found {len(parts)}")
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            tmp = out.with_suffix(f".tmp.{os.getpid()}")   # shared name races
            try:
                with open(tmp, "wb") as w:
                    for p in parts:
                        local = _dl_retry(p)
                        with open(local, "rb") as r:
                            while chunk := r.read(64 * 1024 * 1024):
                                w.write(chunk)
                os.replace(tmp, out)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        return out

    # Prefer this repo's local src/ over the bundled HF export so local patches
    # to alignment.py take effect (same convention as run_benchmarks.py).
    local_src = Path(__file__).resolve().parents[2] / "src"
    src_root = scratch.parent / "_repo"
    for f in repo_files:
        if f.startswith("src/xscript/"):
            hf_hub_download(filename=f, **dl)
    sys.path.insert(0, str(src_root / "src"))
    if (local_src / "xscript" / "eval" / "alignment.py").exists():
        sys.path.insert(0, str(local_src))
        print(f"[align] using local src: {local_src}")
    for f in repo_files:
        if f.startswith("tokenizers/"):
            local = hf_hub_download(filename=f, **dl)
            dest = scratch / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(local)

    models = json.loads(Path(hf_hub_download(filename="models.json", **dl)).read_text())
    runs = args.runs or sorted(models)
    missing = [r for r in runs if r not in models]
    if missing:
        sys.exit(f"models not in repo: {missing}\navailable: {sorted(models)}")
    bad = [l for l in args.langs if l not in ALL_LANGS]
    if bad:
        sys.exit(f"unknown languages {bad}; choose from {ALL_LANGS}")
    print(f"[align] {len(runs)} model(s) x {len(args.langs)} langs "
          f"({args.langs}): {runs}")

    import torch
    from xscript import flores
    from xscript.eval import alignment

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[align] --device not given, using {device}"
              + ("" if device == "cuda" else " (slow; pass --device xla on Neuron)"))
    if device == "xla":
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
    else:
        dev = torch.device(device)

    splits = ("dev", "devtest") if args.split == "both" else (args.split,)
    flores.download(args.langs, splits=splits, token=os.environ.get("HF_TOKEN"))

    out_dir = work / "results" / "alignment"
    for i, run in enumerate(runs, 1):
        tok = models[run]["tok"]
        print(f"\n===== [{i}/{len(runs)}] {run} (tok={tok}, device={device}) =====",
              flush=True)
        ckpt_rel = f"runs/{run}/checkpoints/final.pt"
        local_ckpt = fetch_checkpoint(f"runs/{run}/checkpoints")
        dest = scratch / ckpt_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.symlink_to(local_ckpt)
        try:
            alignment.run(run, tok, split=args.split, out_dir=out_dir,
                          device=dev, langs=args.langs, batch=args.batch_size,
                          max_tokens=args.max_tokens, limit=args.limit,
                          emb_dir=args.emb_dir)
        except Exception as exc:
            import traceback
            print(f"[align] {run} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            if not args.keep_checkpoints:
                try:
                    real = Path(local_ckpt).resolve()
                    dest.unlink(missing_ok=True)
                    real.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"[align] cleanup warning for {run}: {exc}")

    print(f"\n[align] per-run JSON/MD in {out_dir}")


if __name__ == "__main__":
    main()
