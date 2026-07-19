#!/usr/bin/env python
"""Upload exactly one checkpoint chunk, then exit.

Each invocation is a fresh, short-lived process -- this is deliberate. The
login node kills a process after ~2.5-3GB of cumulative transfer *within that
one process*, regardless of wall-clock time or chunk count (confirmed: a
python loop uploading 900MB chunks died after 3 chunks / ~2.8GB in 42s, while
isolated single-chunk processes each completed in 15s). So no chunk-uploading
loop may live inside one Python process; the looping has to happen at the
shell level, launching a new process per chunk. See run_chunked_upload_loop.sh.

    python upload_one_part.py --repo jvonrad/xscript-eval --model en-starved --index 3
"""
import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

SCRATCH = Path("/scratch/u6jh/jvonrad.u6jh/xscript")
STAGING = SCRATCH / "_hf_staging"
CHUNK_BYTES = 900 * 1024 * 1024
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


def ensure_split(model: str) -> list[Path]:
    src = STAGING / "runs" / model / "checkpoints" / "final.pt"
    out_dir = PARTS_DIR / model
    n_expected = math.ceil(src.stat().st_size / CHUNK_BYTES)
    existing = sorted(out_dir.glob("final.pt.part*")) if out_dir.exists() else []
    if len(existing) == n_expected:
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("final.pt.part*"):
        f.unlink()
    subprocess.run(["split", "-d", "-a", "3", f"--bytes={CHUNK_BYTES}",
                    str(src), str(out_dir / "final.pt.part")], check=True)
    return sorted(out_dir.glob("final.pt.part*"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--repo-type", default="model")
    ap.add_argument("--model", required=True)
    ap.add_argument("--index", type=int, required=True, help="0-based part index")
    args = ap.parse_args()
    fix_tls_env()

    from huggingface_hub import HfApi
    api = HfApi()

    parts = ensure_split(args.model)
    if args.index >= len(parts):
        print(f"[part] index {args.index} out of range (only {len(parts)} parts) -- done")
        return
    part = parts[args.index]
    rel = f"runs/{args.model}/checkpoints/{part.name}"
    local_size = part.stat().st_size

    info = api.repo_info(args.repo, repo_type=args.repo_type, files_metadata=True)
    remote_size = next((s.size for s in info.siblings if s.rfilename == rel), None)
    if remote_size == local_size:
        print(f"[part] {rel} already committed ({local_size/1e6:.0f}MB) -- skip")
        return

    print(f"[part] uploading {rel} ({local_size/1e6:.0f}MB) "
          f"[{args.index + 1}/{len(parts)} for {args.model}]", flush=True)
    api.upload_file(path_or_fileobj=str(part), path_in_repo=rel,
                    repo_id=args.repo, repo_type=args.repo_type)

    n_parts_local = STAGING / "runs" / args.model / "checkpoints" / "n_parts.txt"
    n_parts_local.write_text(str(len(parts)))
    n_parts_rel = f"runs/{args.model}/checkpoints/n_parts.txt"
    n_remote = next((s.size for s in info.siblings if s.rfilename == n_parts_rel), None)
    if n_remote != n_parts_local.stat().st_size:
        api.upload_file(path_or_fileobj=str(n_parts_local), path_in_repo=n_parts_rel,
                        repo_id=args.repo, repo_type=args.repo_type)
    print(f"[part] {rel} committed.")


if __name__ == "__main__":
    main()
