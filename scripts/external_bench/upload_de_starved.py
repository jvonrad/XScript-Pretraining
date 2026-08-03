#!/usr/bin/env python
"""Upload the `de__unigram_starved` retrain (§6h) to jvonrad/xscript-eval.

Mirrors de-fair's uploaded roster exactly, so every `de-fair-Xb` gains a
`de-starved-Xb` counterpart at the *same step and token count* — six of the
seven are step-for-step identical to de-fair's checkpoint names, which is what
makes the same-script starved-vs-fair contrast LR-matched by construction
rather than by interpolation.

⛔ THE NAMING RULE THIS SCRIPT ENFORCES. The retrain stops at 16.1B with **no
cooldown**, so its `final.pt` is a mid-stable checkpoint at peak LR 3.0e-3.
Every other model's `final.pt` in this repo is a **cooled 30B final at
3.0e-4**. Uploading it as bare `de-starved` would silently pit a cooled 30B
model against an uncooled 16B one in any "de-fair vs de-starved finals"
comparison — exactly the LR-state confound CLAUDE.md §6 spends its length
warning about. It therefore goes up as `de-starved-16b`, and this script
refuses to create a bare `de-starved` entry.

Layout matches the existing 109 model dirs:
    runs/<friendly>/checkpoints/final.pt.part000..N
    runs/<friendly>/checkpoints/n_parts.txt

Idempotent: a part already present at the right size is skipped, so re-running
after an interruption uploads only what is missing.

    python upload_de_starved.py --repo jvonrad/xscript-eval \
        --run-dir $XSCRIPT_SCRATCH/runs/de__unigram_starved/checkpoints \
        --staging $XSCRIPT_SCRATCH/_hf_parts_de_starved [--dry-run]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

CHUNK_BYTES = 900 * 1024 * 1024   # what the existing dirs were built with

# friendly name -> (local checkpoint file, de-fair counterpart, note)
# Six of seven are step-for-step identical to de-fair's own checkpoint names.
ROSTER = [
    ("de-starved-1b",  "step1092_1001M.pt",   "de-fair-1b",  "exact step+token match"),
    ("de-starved-2b",  "step2456_2253M.pt",   "de-fair-2b",  "exact step+token match"),
    ("de-starved-5b",  "step5181_4753M.pt",   "de-fair-5b",  "exact step+token match"),
    ("de-starved-8b",  "step8451_7753M.pt",   "de-fair-8b",  "exact step+token match"),
    ("de-starved-12b", "step12811_11754M.pt", "de-fair-12b", "exact step+token match"),
    ("de-starved-15b", "step16081_14754M.pt", "de-fair-15b", "14754M vs 14755M (0.007%)"),
    ("de-starved-16b", "final.pt",            None,          "content-match of de-fair-12b (x1.371); MID-STABLE, not cooled"),
]

ORIG_RUN = "de__unigram_starved"
TOK = "unigram_starved"
LANGS = ["de"]


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


def split(src: Path, out_dir: Path) -> list[Path]:
    n = math.ceil(src.stat().st_size / CHUNK_BYTES)
    existing = sorted(out_dir.glob("final.pt.part*")) if out_dir.exists() else []
    if len(existing) == n and all(p.stat().st_size > 0 for p in existing):
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in existing:
        p.unlink()
    parts = []
    with open(src, "rb") as r:
        for i in range(n):
            dst = out_dir / f"final.pt.part{i:03d}"
            remaining = CHUNK_BYTES
            with open(dst, "wb") as w:
                while remaining > 0:
                    buf = r.read(min(64 << 20, remaining))
                    if not buf:
                        break
                    w.write(buf)
                    remaining -= len(buf)
            parts.append(dst)
    total = sum(p.stat().st_size for p in parts)
    if total != src.stat().st_size:
        sys.exit(f"split mismatch for {src}: {total} != {src.stat().st_size}")
    return parts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--staging", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-parts", action="store_true",
                    help="do not delete staged parts after a successful upload")
    a = ap.parse_args()

    fix_tls_env()
    from huggingface_hub import HfApi
    api = HfApi()
    run_dir, staging = Path(a.run_dir), Path(a.staging)

    missing = [f for _, f, _, _ in ROSTER if not (run_dir / f).exists()]
    if missing:
        sys.exit(f"missing local checkpoints: {missing}")

    remote = set(api.list_repo_files(a.repo))
    sizes = {}
    if not a.dry_run:
        for s in api.repo_info(a.repo, files_metadata=True).siblings:
            sizes[s.rfilename] = s.size

    print(f"repo={a.repo}  roster={len(ROSTER)}  dry_run={a.dry_run}\n")
    for friendly, fname, counterpart, note in ROSTER:
        src = run_dir / fname
        rel = f"runs/{friendly}/checkpoints"
        gb = src.stat().st_size / 1e9
        print(f"== {friendly:15} <- {fname:22} {gb:5.2f} GB  [{note}]")
        if a.dry_run:
            print(f"   would upload {math.ceil(src.stat().st_size/CHUNK_BYTES)} parts to {rel}/")
            continue
        parts = split(src, staging / friendly)
        for p in parts:
            key = f"{rel}/{p.name}"
            if key in remote and sizes.get(key) == p.stat().st_size:
                print(f"   skip {p.name} (already present, {p.stat().st_size} B)")
                continue
            api.upload_file(path_or_fileobj=str(p), path_in_repo=key,
                            repo_id=a.repo, commit_message=f"{friendly}: {p.name}")
            print(f"   up   {p.name} ({p.stat().st_size/1e9:.2f} GB)")
        api.upload_file(path_or_fileobj=str(len(parts)).encode(),
                        path_in_repo=f"{rel}/n_parts.txt", repo_id=a.repo,
                        commit_message=f"{friendly}: n_parts")
        print(f"   up   n_parts.txt = {len(parts)}")
        if not a.keep_parts:
            for p in parts:
                p.unlink()
            print("   staged parts removed")

    # ---- models.json ----
    from huggingface_hub import hf_hub_download
    mj = json.loads(Path(hf_hub_download(a.repo, "models.json")).read_text())
    before = len(mj)
    for friendly, fname, _, _ in ROSTER:
        stem = fname[:-3]                      # drop ".pt"
        orig = ORIG_RUN if stem == "final" else f"{ORIG_RUN}__{stem}"
        mj[friendly] = {"orig_run": orig, "tok": TOK, "langs": LANGS}
    if "de-starved" in mj:                     # see the naming rule in the docstring
        sys.exit("refusing to proceed: a bare `de-starved` entry exists; "
                 "this run has no cooled 30B final and must not claim that name")
    print(f"\nmodels.json: {before} -> {len(mj)} entries")
    if a.dry_run:
        for friendly, *_ in ROSTER:
            print(f"   + {friendly}: {json.dumps(mj[friendly])}")
        return 0
    # Preserve the file's existing shape exactly: indent=2, INSERTION order
    # (it is not sorted — it groups the 15 originals before the intermediates),
    # and no trailing newline. Sorting or reformatting here would turn a
    # 7-entry addition into a 109-entry diff.
    out = staging / "models.json"
    out.write_text(json.dumps(mj, indent=2))
    api.upload_file(path_or_fileobj=str(out), path_in_repo="models.json",
                    repo_id=a.repo, commit_message="add de-starved-{1,2,5,8,12,15,16}b (§6h retrain)")
    print("models.json uploaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
