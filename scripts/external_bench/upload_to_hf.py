#!/usr/bin/env python
"""Export final checkpoints + tokenizers + runner to a private HF repo.

Runs on the Isambard login node. Streams each file from Lustre straight to the
Hub (no torch load), so it stays under the login memory cap. Idempotent: files
already present with the same hash are skipped, so it is safe to re-run/resume.

Models are published under friendly names: "<mixture>-<starved|fair>", e.g.
en-fair, en-ar-starved. A models.json manifest maps each friendly name to its
real tokenizer (unigram_starved/destarved) and original run name, so the runner
loads the correct tokenizer even though the friendly name hides it.

    export PYTHONPATH=$XSCRIPT_RUNTIME
    export HF_TOKEN=hf_...            # needs repo.write
    python upload_to_hf.py --repo jvonrad/xscript-eval
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

SCRATCH = Path("/scratch/u6jh/jvonrad.u6jh/xscript")
REPO_DIR = Path("/home/u6jh/jvonrad.u6jh/XScript-Pretraining")
BENCH_DIR = REPO_DIR / "scripts" / "external_bench"
STAGING = SCRATCH / "_hf_staging"
TOKENIZERS = ["unigram_starved", "unigram_destarved"]
# tokenizer-condition suffix in the original run name -> friendly suffix
COND = {"starved": "starved", "destarved": "fair"}


def friendly(orig: str) -> tuple[str, str, str]:
    """'en-ar__unigram_destarved' -> ('en-ar-fair', 'unigram_destarved', 'en-ar')."""
    mix, tok = orig.split("__", 1)              # 'en-ar', 'unigram_destarved'
    cond = tok.split("_", 1)[1]                 # 'destarved'
    return f"{mix}-{COND[cond]}", tok, mix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--repo-type", default="model")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel upload workers (keep modest on the login node)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # The login shell may export REQUESTS_CA_BUNDLE/SSL_CERT_FILE to host paths
    # that don't exist inside the container; point every TLS var at a real bundle
    # (certifi's, falling back to the container's system bundle) before any
    # requests/huggingface_hub call, or create_repo dies on cert_verify.
    import os
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

    from huggingface_hub import HfApi
    api = HfApi()

    # friendly-name manifest from the completed runs -------------------------
    origs = sorted(d.parent.parent.name
                   for d in SCRATCH.glob("runs/*/checkpoints/final.pt"))
    models: dict[str, dict] = {}
    for orig in origs:
        name, tok, mix = friendly(orig)
        models[name] = {"orig_run": orig, "tok": tok, "langs": mix.split("-")}

    print(f"[upload] {len(models)} models -> staging at {STAGING}")
    for name, meta in models.items():
        print(f"    {name:<16} <- {meta['orig_run']}  (tok={meta['tok']})")
    if args.dry_run:
        return

    # ---- build a staging tree mirroring the repo layout --------------------
    # Small files are copied; the 4GB checkpoints are HARD-LINKED (same Lustre
    # filesystem => zero extra bytes) under their friendly names.
    def stage_copy(src: Path, rel: str) -> None:
        dst = STAGING / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

    def stage_link(src: Path, rel: str) -> None:
        dst = STAGING / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.hardlink_to(src)          # same-FS hard link, no copy

    for p in sorted((REPO_DIR / "src" / "xscript").rglob("*.py")):
        stage_copy(p, str(p.relative_to(REPO_DIR)))
    for name in ("run_benchmarks.py", "requirements.txt", "README.md"):
        if (BENCH_DIR / name).exists():
            stage_copy(BENCH_DIR / name, name)
    for tok in TOKENIZERS:                # real names: must match checkpoint cfg
        for fname in ("sp.model", "meta.json"):
            stage_copy(SCRATCH / "tokenizers" / tok / fname, f"tokenizers/{tok}/{fname}")
    for name, meta in models.items():
        src = SCRATCH / "runs" / meta["orig_run"] / "checkpoints" / "final.pt"
        if not src.exists():
            sys.exit(f"missing checkpoint: {src}")
        stage_link(src, f"runs/{name}/checkpoints/final.pt")
    (STAGING / "models.json").write_text(json.dumps(models, indent=2))

    total_gb = sum(f.stat().st_size for f in STAGING.rglob("*") if f.is_file()) / 1e9
    print(f"[upload] staged {total_gb:.1f} GB; starting resumable upload_large_folder")

    # ---- resumable, multi-worker upload (survives/retries interruptions) ----
    api.create_repo(args.repo, repo_type=args.repo_type, private=True, exist_ok=True)
    api.upload_large_folder(repo_id=args.repo, repo_type=args.repo_type,
                            folder_path=str(STAGING), num_workers=args.workers,
                            print_report=True, print_report_every=30)

    # verify every checkpoint actually committed (>4GB); exit non-zero otherwise
    # so an `until ...; do ...; done` restart loop resumes rather than stopping.
    info = api.repo_info(args.repo, repo_type=args.repo_type, files_metadata=True)
    done = {s.rfilename for s in info.siblings
            if s.rfilename.endswith("final.pt") and (s.size or 0) > 4e9}
    want = {f"runs/{name}/checkpoints/final.pt" for name in models}
    missing = sorted(want - done)
    if missing:
        print(f"[upload] INCOMPLETE: {len(done)}/{len(want)} checkpoints committed; "
              f"missing {missing[:3]}{'...' if len(missing) > 3 else ''}")
        sys.exit(1)
    print(f"[upload] done. all {len(want)} checkpoints committed.")


if __name__ == "__main__":
    main()
