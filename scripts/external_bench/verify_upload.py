#!/usr/bin/env python
"""Verify every model's uploaded chunk total matches its original checkpoint
size exactly. Ground-truth check against the repo, not the upload log."""
import glob
import os

import certifi

os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from huggingface_hub import HfApi

REPO = "jvonrad/xscript-eval"


def main() -> None:
    api = HfApi()
    info = api.repo_info(REPO, repo_type="model", files_metadata=True)
    by_model: dict[str, int] = {}
    for s in info.siblings:
        if "/checkpoints/final.pt.part" in s.rfilename:
            model = s.rfilename.split("/")[1]
            by_model[model] = by_model.get(model, 0) + (s.size or 0)

    originals = {}
    for p in glob.glob("/scratch/u6jh/jvonrad.u6jh/xscript/_hf_staging/runs/*/checkpoints/final.pt"):
        model = p.split("/")[-3]
        originals[model] = os.path.getsize(p)

    print(f"{'model':<16} {'remote GB':>10} {'orig GB':>10}  match")
    bad = []
    for m in sorted(originals):
        r, o = by_model.get(m, 0), originals[m]
        ok = r == o
        if not ok:
            bad.append(m)
        print(f"{m:<16} {r/1e9:10.4f} {o/1e9:10.4f}  {'OK' if ok else 'MISMATCH'}")

    missing = sorted(set(originals) - set(by_model))
    print()
    print("models missing entirely:", missing or "none")
    print("BAD:", bad or "none -- all models verified byte-exact")


if __name__ == "__main__":
    main()
