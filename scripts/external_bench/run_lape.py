#!/usr/bin/env python
"""Record language-specific-neuron activation statistics (LAPE, arXiv
2402.16438) for XScript checkpoints on Neuron/XLA, CUDA, or CPU.

Every model is recorded on all five languages (FLORES+ dev+devtest, the same
parallel pool as the alignment sweep), so monolingual checkpoints provide the
controls and the token-budget series (``*-1b`` .. ``*-23b``) gives the
training-token axis. Identification/analysis is a separate pure-CPU pass
(``analyze_lape.py``) over the tiny per-run ``.npz`` outputs.

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; source ~/neuron_venv/bin/activate
    export HF_HOME=/mnt/scratch/hf NEURON_CC_FLAGS="--cache_dir=/mnt/scratch/neuron-cache"
    python run_lape.py --repo jvonrad/xscript-eval --device xla --workdir /mnt/scratch/xscript_lape

Writes ``$WORK/results/lape/<model>.npz`` (~4 MB each; over_zero counts + n).
No shared summary file, so a fan-out over core-pairs can't clobber itself
(NEURON.md section 5). Resumable: existing outputs are skipped.

Disk: checkpoints are ~4.4 GB each and there are 109; by default both the
assembled ``final.pt`` AND the downloaded part-blobs in the HF cache are
deleted after each model. ``--keep-checkpoints`` disables that.
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
    ap.add_argument("--workdir", default="./xscript_bench")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="subset of friendly model names (default: all in models.json)")
    ap.add_argument("--langs", nargs="*", default=ALL_LANGS)
    ap.add_argument("--split", default="both", choices=["dev", "devtest", "both"])
    ap.add_argument("--device", default=None, choices=["xla", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N sentences (smoke tests)")
    ap.add_argument("--keep-checkpoints", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    work = Path(args.workdir).resolve()
    scratch = work / "xscript"
    (scratch / "runs").mkdir(parents=True, exist_ok=True)
    (scratch / "tokenizers").mkdir(parents=True, exist_ok=True)
    os.environ["XSCRIPT_SCRATCH"] = str(scratch)
    os.environ["XSCRIPT_RESULTS"] = str(work / "results")

    dl = dict(repo_id=args.repo, local_dir=str(scratch.parent / "_repo"))

    def _dl_retry(filename: str, tries: int = 8) -> Path:
        import random
        import time
        for attempt in range(tries):
            try:
                return Path(hf_hub_download(filename=filename, **dl))
            except Exception as exc:
                if attempt == tries - 1:
                    raise
                wait = min(60, 2 ** attempt) + random.uniform(0, 3)
                print(f"[lape] {type(exc).__name__} on {filename}; "
                      f"retry {attempt + 1}/{tries - 1} in {wait:.0f}s", flush=True)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    # cached repo listing + sizes (parallel list_repo_files calls 429)
    _listing = work / "_repo_files.json"
    sizes = {}
    if _listing.exists():
        try:
            sizes = json.loads(_listing.read_text())
        except json.JSONDecodeError:
            sizes = {}
    if not sizes:
        from huggingface_hub import HfApi
        _info = HfApi().repo_info(args.repo, files_metadata=True)
        sizes = {s.rfilename: (s.size or 0) for s in _info.siblings}
        _tmp = _listing.with_suffix(f".tmp.{os.getpid()}")
        _tmp.write_text(json.dumps(sizes))
        os.replace(_tmp, _listing)
    repo_files = list(sizes)

    def fetch_checkpoint(rel_dir: str) -> tuple[Path, list[Path]]:
        """Returns (assembled final.pt, local part files for later cleanup)."""
        parts = sorted(f for f in repo_files if f.startswith(f"{rel_dir}/final.pt.part"))
        out = work / "_assembled" / rel_dir / "final.pt"
        if parts and out.exists():
            want = sum(sizes.get(p, 0) for p in parts)
            if not want or out.stat().st_size == want:
                return out, []
            out.unlink(missing_ok=True)
        whole = f"{rel_dir}/final.pt"
        if whole in repo_files:
            p = _dl_retry(whole)
            return p, [p]
        if not parts:
            sys.exit(f"no checkpoint found under {rel_dir}")
        n_parts_rel = f"{rel_dir}/n_parts.txt"
        if n_parts_rel in repo_files:
            want_n = int(_dl_retry(n_parts_rel).read_text().strip())
            if want_n != len(parts):
                sys.exit(f"{rel_dir}: expected {want_n} parts, found {len(parts)}")
        out.parent.mkdir(parents=True, exist_ok=True)
        locals_ = []
        tmp = out.with_suffix(f".tmp.{os.getpid()}")
        try:
            with open(tmp, "wb") as w:
                for p in parts:
                    local = _dl_retry(p)
                    locals_.append(local)
                    with open(local, "rb") as r:
                        while chunk := r.read(64 * 1024 * 1024):
                            w.write(chunk)
            os.replace(tmp, out)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return out, locals_

    # prefer local src/ (same convention as the other run_* scripts)
    local_src = Path(__file__).resolve().parents[2] / "src"
    if (local_src / "xscript" / "eval" / "neurons.py").exists():
        sys.path.insert(0, str(local_src))
        print(f"[lape] using local src: {local_src}")
    else:
        sys.exit("local src/xscript/eval/neurons.py not found; run from the repo")

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
        sys.exit(f"models not in repo: {missing}")

    # FLORES: reuse the local copy if present rather than the gated download
    flores_local = Path("/mnt/scratch/xscript_tokanalysis/flores_plus")
    flores_dest = scratch / "flores_plus"
    if flores_local.exists() and not flores_dest.exists():
        flores_dest.symlink_to(flores_local)

    import torch
    from xscript.eval import neurons

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "xla":
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
    else:
        dev = torch.device(device)

    out_dir = work / "results" / "lape"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in runs if not (out_dir / f"{r}.npz").exists()]
    print(f"[lape] {len(todo)}/{len(runs)} model(s) to do on {device}", flush=True)

    for i, run in enumerate(todo, 1):
        tok = models[run]["tok"]
        print(f"\n===== [{i}/{len(todo)}] {run} (tok={tok}) =====", flush=True)
        local_ckpt, part_files = fetch_checkpoint(f"runs/{run}/checkpoints")
        dest = scratch / "runs" / run / "checkpoints" / "final.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.unlink(missing_ok=True)          # may be a stale/broken symlink
        dest.symlink_to(local_ckpt)
        try:
            res = neurons.compute(run, tok, args.langs, split=args.split,
                                  device=dev, batch=args.batch_size,
                                  max_tokens=args.max_tokens, limit=args.limit)
            p = neurons.save(out_dir, res)
            print(f"[lape] wrote {p}", flush=True)
        except Exception as exc:
            import traceback
            print(f"[lape] {run} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            if not args.keep_checkpoints:
                try:
                    real = Path(local_ckpt).resolve()
                    dest.unlink(missing_ok=True)
                    real.unlink(missing_ok=True)
                    for p in part_files:
                        rp = Path(p).resolve()   # symlink into blobs/
                        Path(p).unlink(missing_ok=True)
                        rp.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"[lape] cleanup warning for {run}: {exc}")

    print(f"\n[lape] per-run npz in {out_dir}")


if __name__ == "__main__":
    main()
