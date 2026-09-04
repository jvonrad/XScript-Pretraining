#!/usr/bin/env python
"""Rebuild a language's in-domain eval holdout shard from the public corpus.

WHY THIS IS A RECONSTRUCTION, NOT A PROXY
=========================================
The trainer logs `eval/holdout_{lang}_bpb` against 500 documents reserved from
the training pool (`train.py`/`train_neuron.py`:
`load_holdout(l, cfg["train"]["eval_docs"])`, eval_docs=500). Those shards
lived on the Isambard-AI allocation's scratch, which has been torn down -- so
`bts_from_wandb.py --source holdout` has no way to extend any curve, and
`bpb_fill_from_checkpoints.py` is FLORES-only for exactly that reason.

But the holdout is not a random sample that was lost: it is a deterministic
function of the PUBLIC corpus, and every step of that function is still in
this repo (`xscript.data.fineweb`):

    sources    = _sources_for(lang)              # primary (repo, subdir)
    files      = _list_parquets(repo, subdir)    # HfApi tree listing, sorted()
    holdout    = _iter_texts(repo, files[0])     # FIRST file only, text column
                 until HOLDOUT_BYTES (30 MiB)
    pool       = files[1:]                       # holdout NEVER enters the pool

`_list_parquets` sorts, `_iter_texts` walks row groups in file order, and
`_PoolWriter` writes documents in the order received. The corpus is a fixed
published release. So re-running those same functions reproduces the SAME
documents in the SAME order -- and since `load_holdout` then takes the first
500 in file order, the 500 documents scored here are the 500 the trainer
scored. CLAUDE.md 6h asserts this equivalence for the German pool rebuild
("the manifest is sorted(), the holdout is the first parquet file's first
30MB, and the pool is files[1:], so the holdout ... comes out byte-identical")
-- this script is that assertion applied to the holdout alone.

⚠️ **It is still an assertion until a control passes.** Reconstructing the
text is not the same as proving the reconstruction is right. Score a
checkpoint whose `eval/holdout_{lang}_bpb` W&B ALREADY holds and check it
reproduces (`run_bpb.py --source holdout` + `bpb_fill_from_checkpoints.py`'s
control gate, exactly as the FLORES fill did at 8.94e-06). If that control
fails, this text is not what the model was evaluated on and nothing derived
from it should be spliced onto a curve.

This imports the pool builder's own functions rather than reimplementing the
recipe -- if `fineweb.py` ever changes how it reserves a holdout, this script
changes with it instead of silently diverging.

    python rebuild_holdout.py --lang de --workdir $WORK
    # writes $WORK/xscript/holdout/de_00000.jsonl.zst, the exact path+format
    # `xscript.eval.bpb.load_holdout` globs for.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--workdir", required=True, type=Path,
                    help="same --workdir as run_bpb.py; the shard lands in "
                         "<workdir>/xscript/holdout, which is where "
                         "load_holdout() looks once XSCRIPT_SCRATCH is set")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if a shard for this language exists")
    args = ap.parse_args()

    scratch = (args.workdir / "xscript").resolve()
    os.environ["XSCRIPT_SCRATCH"] = str(scratch)
    sys.path.insert(0, str(REPO / "src"))

    from xscript.data.fineweb import (_sources_for, _list_parquets, _iter_texts,
                                      _PoolWriter, HOLDOUT_BYTES)
    from xscript.paths import HOLDOUT

    existing = sorted(HOLDOUT.glob(f"{args.lang}_*.jsonl.zst"))
    if existing and not args.force:
        sys.exit(f"holdout already present: {[p.name for p in existing]} "
                 f"(pass --force to rebuild)")
    for p in existing:
        p.unlink()

    (repo, subdir) = _sources_for(args.lang)[0]
    files = _list_parquets(repo, subdir)
    if not files:
        sys.exit(f"no parquet files for {args.lang} under {repo}/{subdir}")
    first = files[0]
    print(f"[holdout] {args.lang}: source {repo}/{subdir}, {len(files)} files")
    print(f"[holdout] reserved file (files[0]): {first}")
    print(f"[holdout] target {HOLDOUT_BYTES / (1 << 20):.0f} MiB of text")

    w = _PoolWriter(HOLDOUT, prefix=args.lang)
    got, n_docs = 0, 0
    for t in _iter_texts(repo, first):
        w.write(t)
        got += len(t.encode("utf-8"))
        n_docs += 1
        if got >= HOLDOUT_BYTES:
            break
    w.close()

    shards = sorted(HOLDOUT.glob(f"{args.lang}_*.jsonl.zst"))
    meta = {
        "lang": args.lang, "repo": repo, "subdir": subdir,
        "n_files_in_manifest": len(files), "holdout_file": first,
        "holdout_bytes_target": HOLDOUT_BYTES, "text_bytes": got,
        "docs": n_docs,
        "shards": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in shards},
    }
    (HOLDOUT / f"{args.lang}_rebuild.json").write_text(json.dumps(meta, indent=2))
    print(f"[holdout] wrote {got / 1e6:.2f} MB of text over {n_docs} docs "
          f"to {[p.name for p in shards]}")
    print(f"[holdout] provenance -> {HOLDOUT / f'{args.lang}_rebuild.json'}")
    print("[holdout] ⚠️ NOT yet validated -- run the control before use "
          "(see the module docstring).")


if __name__ == "__main__":
    raise SystemExit(main())
