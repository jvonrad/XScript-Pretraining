#!/usr/bin/env python
"""Upload exactly one checkpoint via plain (single-stream, low-memory)
upload_file -- no multi-worker hashing, unlike upload_large_folder, which
appears to exceed the login node's per-session memory cap on this 65GB job.

    python upload_one.py --repo jvonrad/xscript-eval --model en-starved
"""
import argparse
import os
import sys
from pathlib import Path

SCRATCH = Path("/scratch/u6jh/jvonrad.u6jh/xscript")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", required=True, help="friendly name, e.g. en-starved")
    ap.add_argument("--repo-type", default="model")
    args = ap.parse_args()
    fix_tls_env()

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type=args.repo_type, private=True, exist_ok=True)

    src = SCRATCH / "_hf_staging" / "runs" / args.model / "checkpoints" / "final.pt"
    if not src.exists():
        sys.exit(f"not staged: {src} (run upload_to_hf.py --dry-run first to stage)")

    rel = f"runs/{args.model}/checkpoints/final.pt"
    print(f"[upload-one] {src.stat().st_size/1e9:.2f}GB -> {rel}", flush=True)
    api.upload_file(path_or_fileobj=str(src), path_in_repo=rel,
                    repo_id=args.repo, repo_type=args.repo_type)

    info = api.repo_info(args.repo, repo_type=args.repo_type, files_metadata=True)
    match = [s for s in info.siblings if s.rfilename == rel]
    if match and (match[0].size or 0) > 4e9:
        print(f"[upload-one] VERIFIED committed: {match[0].size/1e9:.2f}GB")
    else:
        sys.exit(f"[upload-one] NOT verified in repo after upload: {match}")


if __name__ == "__main__":
    main()
