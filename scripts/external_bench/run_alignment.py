#!/usr/bin/env python
"""Cross-lingual representation alignment (MEXA-style) for XScript checkpoints
on any GPU -- the login node can't hold a live ~1B model at all (confirmed:
even mmap-loaded checkpoints die the instant the real model is constructed),
so like benchmarking this has to run elsewhere.

Only the EN-anchored bilingual models have an EN-partner pair to align, so
this evaluates those 8 by default (mono runs are skipped -- no pair exists).

    pip install -r requirements.txt
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    export HF_TOKEN=hf_...
    python run_alignment.py --repo jvonrad/xscript-eval

Results land in ./xscript_bench/results/alignment/<model>.json (+ .md tables).
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--workdir", default="./xscript_bench")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="subset of friendly model names (default: all EN-partner bilinguals)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--keep-checkpoints", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    work = Path(args.workdir).resolve()
    scratch = work / "xscript"
    (scratch / "runs").mkdir(parents=True, exist_ok=True)
    (scratch / "tokenizers").mkdir(parents=True, exist_ok=True)
    os.environ["XSCRIPT_SCRATCH"] = str(scratch)
    os.environ["XSCRIPT_RESULTS"] = str(work / "results")

    dl = dict(repo_id=args.repo, repo_type=args.repo_type, local_dir=str(scratch.parent / "_repo"))
    repo_files = list_repo_files(args.repo, repo_type=args.repo_type)

    def fetch_checkpoint(rel_dir: str) -> Path:
        whole = f"{rel_dir}/final.pt"
        if whole in repo_files:
            return Path(hf_hub_download(filename=whole, **dl))
        parts = sorted(f for f in repo_files if f.startswith(f"{rel_dir}/final.pt.part"))
        if not parts:
            sys.exit(f"no checkpoint found under {rel_dir}")
        assembled_dir = work / "_assembled" / rel_dir
        assembled_dir.mkdir(parents=True, exist_ok=True)
        out = assembled_dir / "final.pt"
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

    src_root = scratch.parent / "_repo"
    for f in repo_files:
        if f.startswith("src/xscript/"):
            hf_hub_download(filename=f, **dl)
    sys.path.insert(0, str(src_root / "src"))
    for f in repo_files:
        if f.startswith("tokenizers/"):
            local = hf_hub_download(filename=f, **dl)
            dest = scratch / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(local)

    models = json.loads(Path(hf_hub_download(filename="models.json", **dl)).read_text())
    # only EN-anchored bilinguals actually have a pair to align
    bilinguals = sorted(m for m, meta in models.items() if len(meta["langs"]) > 1)
    runs = args.runs or bilinguals
    missing = [r for r in runs if r not in models]
    if missing:
        sys.exit(f"models not in repo: {missing}\navailable bilinguals: {bilinguals}")
    print(f"[align] {len(runs)} model(s): {runs}")

    import torch
    from xscript.eval import alignment
    if not torch.cuda.is_available():
        print("[align] WARNING: no CUDA device -- this will be slow on CPU.")

    for i, run in enumerate(runs, 1):
        tok = models[run]["tok"]
        print(f"\n===== [{i}/{len(runs)}] {run} (tok={tok}) =====")
        ckpt_rel = f"runs/{run}/checkpoints/final.pt"
        local_ckpt = fetch_checkpoint(f"runs/{run}/checkpoints")
        dest = scratch / ckpt_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.symlink_to(local_ckpt)
        try:
            alignment.run(run, tok, split=args.split)
        except Exception as exc:
            print(f"[align] {run} FAILED: {type(exc).__name__}: {exc}")
        finally:
            if not args.keep_checkpoints:
                try:
                    real = Path(local_ckpt).resolve()
                    dest.unlink(missing_ok=True)
                    real.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"[align] cleanup warning for {run}: {exc}")

    print(f"\n[align] per-run JSON/MD in {work / 'results' / 'alignment'}")


if __name__ == "__main__":
    main()
