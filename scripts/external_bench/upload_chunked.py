#!/usr/bin/env python
"""Chunked, resumable checkpoint upload for hosts where a single long-running
upload gets killed (this login node kills any single process after a few
minutes of sustained work, regardless of memory/CPU used -- confirmed by
testing plain upload_file, multi-worker upload_large_folder, and hf_transfer,
all killed at similar wall-clock points despite very different memory
profiles; a 900MB single-shot upload reliably completes in ~15s).

Splits each checkpoint into ~900MB parts (`final.pt.partNNN`) and uploads each
part as its own short HfApi call -- each comfortably finishes before any
per-process limit. Idempotent: skips parts already present with the right
size, so re-running after an interruption only uploads what's missing.

    python upload_chunked.py --repo jvonrad/xscript-eval                # all models
    python upload_chunked.py --repo jvonrad/xscript-eval --model en-fair
"""
import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

SCRATCH = Path("/scratch/u6jh/jvonrad.u6jh/xscript")
STAGING = SCRATCH / "_hf_staging"
CHUNK_BYTES = 900 * 1024 * 1024        # empirically validated safe size
PARTS_DIR = SCRATCH / "_hf_parts"


def fix_tls_env() -> None:
    ca = None
    try:
        import certifi
        ca = certifi.where()
    except Exception:
        pass
    if not (ca and os.path.exists(ca)):
        ca = "/etc/ssl/certs/ca-certificates.crt"
    if os.path.exists(ca):
        for var in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
            os.environ[var] = ca


def split_checkpoint(model: str) -> list[Path]:
    """Split (if not already split) into PARTS_DIR/<model>/final.pt.partNNN."""
    src = STAGING / "runs" / model / "checkpoints" / "final.pt"
    out_dir = PARTS_DIR / model
    n_expected = math.ceil(src.stat().st_size / CHUNK_BYTES)
    existing = sorted(out_dir.glob("final.pt.part*")) if out_dir.exists() else []
    if len(existing) == n_expected:
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("final.pt.part*"):
        f.unlink()
    prefix = str(out_dir / "final.pt.part")
    subprocess.run(["split", "-d", "-a", "3", f"--bytes={CHUNK_BYTES}",
                    str(src), prefix], check=True)
    return sorted(out_dir.glob("final.pt.part*"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--repo-type", default="model")
    ap.add_argument("--model", default=None, help="single friendly name (default: all)")
    args = ap.parse_args()
    fix_tls_env()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type=args.repo_type, private=True, exist_ok=True)

    models = [args.model] if args.model else sorted(
        d.name for d in (STAGING / "runs").iterdir() if d.is_dir())

    # what's already committed in the repo, so re-runs skip fast
    info = api.repo_info(args.repo, repo_type=args.repo_type, files_metadata=True)
    remote = {s.rfilename: (s.size or 0) for s in info.siblings}

    for model in models:
        parts = split_checkpoint(model)
        n_parts_file = STAGING / "runs" / model / "checkpoints" / "n_parts.txt"
        n_parts_file.write_text(str(len(parts)))
        rel_manifest = f"runs/{model}/checkpoints/n_parts.txt"
        if remote.get(rel_manifest) != n_parts_file.stat().st_size:
            api.upload_file(path_or_fileobj=str(n_parts_file), path_in_repo=rel_manifest,
                            repo_id=args.repo, repo_type=args.repo_type)

        for part in parts:
            rel = f"runs/{model}/checkpoints/{part.name}"
            local_size = part.stat().st_size
            if remote.get(rel) == local_size:
                print(f"[skip] {rel} already committed ({local_size/1e6:.0f}MB)")
                continue
            print(f"[upload] {rel} ({local_size/1e6:.0f}MB)", flush=True)
            api.upload_file(path_or_fileobj=str(part), path_in_repo=rel,
                            repo_id=args.repo, repo_type=args.repo_type)
            remote[rel] = local_size  # so a crash-resume in the SAME process still tracks it

    print("[upload_chunked] pass complete.")


if __name__ == "__main__":
    main()
