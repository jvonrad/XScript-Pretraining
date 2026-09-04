#!/usr/bin/env python
"""Whole-file parallel pool builder -- the successor to `fast_pool.py`.

Why a second tool (measured on a trn2.48xlarge, 2026-09-04, same file):

| method | per-stream text throughput |
|---|---|
| `HfFileSystem` + per-row-group `read_row_group(columns=["text"])` (what `build_pool` and `fast_pool.py` do) | **7.4 MB/s** |
| direct CDN URL + `pre_buffer=True` batched row groups | 9-10 MB/s |
| raw single-connection GET of the whole file | 110 MB/s (raw bytes) |
| `hf_transfer` whole-file download | **282 MB/s** raw -> ~70 MB/s of text |

FineWeb(2)-HQ parquet files are 1.8-2.2 GB with 128-329 row groups of
~7-14 MB, of which only ~25% is the `text` column (the rest is the 1024-d
`embeddings` column). Column-pruned range reads therefore issue hundreds of
small (~3.6 MB) range requests per file, each paying a ~0.3 s round trip, and
top out near 10 MB/s per stream no matter how the reads are batched. Pulling
the whole file with `hf_transfer` (multi-connection, one resolver hit per
file) moves 4x the bytes but is ~7x faster per stream in text terms, and the
local `text` extraction is ~700 MB/s. Disk churn is bounded: each worker
holds one file and deletes it as soon as it is extracted.

Design: a process pool of `workers` per language. Job i = (source file,
shard index i). Each worker downloads its file to `<pool>/tmp/`, extracts
`text`, writes ONE `pool_<i:05d>.jsonl.zst` shard (exactly `_PoolWriter`'s
line format, so `xscript pack` reads it unchanged; shards are ~0.4-0.5 GB
instead of ~1 GB, which the packer and loader do not care about) and
deletes the parquet. The parent tracks totals and writes a `stats.json` with
the same schema as `build_pool` (files_consumed tagged "repo::path",
holdout from the primary source's first file, `exhausted` flag), so
`pack()`'s pool-done check and every downstream reader work unmodified.

Not resumable across a kill (jobs in flight are lost, at most `workers`
files); re-running skips shards that already exist on disk and were
recorded in stats.json, so a restart costs only the in-flight files.

Usage:
    python scripts/neuron_train/fast_pool2.py --lang de --workers 4
    python scripts/neuron_train/fast_pool2.py --lang ar --workers 4 --gb 233
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from xscript.data.fineweb import (_sources_for, _list_parquets, _iter_texts,  # noqa: E402
                                  _PoolWriter, HOLDOUT_BYTES)
from xscript.paths import pool_dir, HOLDOUT, ensure  # noqa: E402


def _extract_one(args):
    """Worker: download file -> extract text -> write one shard -> delete."""
    repo, path, shard_path, tmp_dir = args
    import pyarrow.parquet as pq
    import zstandard
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    local = None
    for attempt in range(6):
        try:
            local = hf_hub_download(repo, path, repo_type="dataset",
                                    local_dir=tmp_dir, force_download=True)
            break
        except Exception as exc:  # 429 / transient network: back off, retry
            wait = min(30 * (1.6 ** attempt), 300)
            print(f"[fast-pool2] retry {attempt+1}/6 {repo}/{path}: {exc} "
                  f"(sleep {wait:.0f}s)", flush=True)
            time.sleep(wait)
    if local is None:
        return path, 0, 0, time.time() - t0, "download failed"
    n_bytes = n_docs = 0
    tmp_shard = Path(str(shard_path) + ".tmp")
    try:
        pf = pq.ParquetFile(local)
        with open(tmp_shard, "wb") as raw:
            w = zstandard.ZstdCompressor(level=3).stream_writer(raw, closefd=False)
            for rg in range(pf.num_row_groups):
                col = pf.read_row_group(rg, columns=["text"]).column("text")
                for t in col.to_pylist():
                    if not t:
                        continue
                    line = json.dumps({"text": t}, ensure_ascii=False) + "\n"
                    w.write(line.encode("utf-8"))
                    n_bytes += len(t.encode("utf-8"))
                    n_docs += 1
            w.close()
        os.replace(tmp_shard, shard_path)
    except Exception as exc:
        tmp_shard.unlink(missing_ok=True)
        return path, 0, 0, time.time() - t0, f"extract failed: {exc}"
    finally:
        try:
            os.remove(local)
        except OSError:
            pass
    return path, n_bytes, n_docs, time.time() - t0, ""


def build(lang: str, budget_bytes: float, workers: int = 4) -> dict:
    sources = _sources_for(lang)
    primary_repo, primary_subdir = sources[0]
    first_files = _list_parquets(primary_repo, primary_subdir)
    if not first_files:
        raise RuntimeError(f"no parquet files for {lang}")
    out = ensure(pool_dir(lang))
    tmp_dir = ensure(out / "tmp")
    stats_path = out / "stats.json"

    used: list[str] = []
    total_bytes = total_docs = 0
    holdout_got = 0
    if stats_path.exists():
        st = json.loads(stats_path.read_text())
        if st["text_bytes"] >= budget_bytes * 0.99:
            print(f"[fast-pool2] {lang}: cached ({st['text_bytes']/1e9:.1f}GB)")
            return st
        used = list(st.get("files_consumed", []))
        total_bytes, total_docs = st["text_bytes"], st["docs"]
        holdout_got = st.get("holdout_bytes", 0)
        print(f"[fast-pool2] {lang}: resuming ({total_bytes/1e9:.1f}/"
              f"{budget_bytes/1e9:.1f}GB, {len(used)} files done)")
    if holdout_got == 0:
        hw = _PoolWriter(HOLDOUT, prefix=lang)
        for t in _iter_texts(primary_repo, first_files[0]):
            hw.write(t)
            holdout_got += len(t.encode("utf-8"))
            if holdout_got >= HOLDOUT_BYTES:
                break
        hw.close()
        print(f"[fast-pool2] {lang}: holdout done ({holdout_got/1e6:.1f}MB)", flush=True)

    # job list in manifest order; shard index = position in the full job list
    # so names are stable across restarts.
    jobs = []
    used_set = set(used)
    idx = 0
    for i, (repo, subdir) in enumerate(sources):
        files = first_files if i == 0 else _list_parquets(repo, subdir)
        pool_files = files[1:] if i == 0 else files
        for f in pool_files:
            tag = f"{repo}::{f}"
            shard = out / f"pool_{idx:05d}.jsonl.zst"
            idx += 1
            if tag in used_set:
                continue
            if shard.exists():
                shard.unlink()  # unrecorded -> possibly truncated; redo
            jobs.append((repo, f, tag, shard))
    print(f"[fast-pool2] {lang}: {len(jobs)} files to fetch, {workers} workers, "
          f"budget {budget_bytes/1e9:.1f}GB", flush=True)

    def _write_stats(exhausted=False):
        st = {"lang": lang, "budget_bytes": budget_bytes, "text_bytes": total_bytes,
              "docs": total_docs, "holdout_bytes": holdout_got,
              "holdout_file": first_files[0], "files_consumed": used,
              "shard_idx": max([int(p.name[5:10]) for p in out.glob("pool_*.jsonl.zst")]
                               or [-1]),
              "exhausted": exhausted, "builder": "fast_pool2"}
        stats_path.write_text(json.dumps(st, indent=2))

    t0 = time.time()
    done = 0
    stopped = False
    with ProcessPoolExecutor(max_workers=workers) as ex:
        pending = {}
        it = iter(jobs)
        # keep at most `workers` + 2 jobs queued so a budget stop cancels the rest
        def _submit():
            for repo, f, tag, shard in it:
                fut = ex.submit(_extract_one, (repo, f, str(shard), str(tmp_dir)))
                pending[fut] = (tag, shard)
                return True
            return False
        for _ in range(workers + 2):
            _submit()
        while pending:
            fut = next(as_completed(list(pending)))
            tag, shard = pending.pop(fut)
            path, nb, nd, dt, err = fut.result()
            done += 1
            if err:
                print(f"[fast-pool2] WARN {tag}: {err}", flush=True)
            else:
                used.append(tag)
                total_bytes += nb
                total_docs += nd
            if done % 4 == 0 or total_bytes >= budget_bytes:
                el = time.time() - t0
                print(f"[fast-pool2] {lang}: {total_bytes/1e9:.2f}/{budget_bytes/1e9:.1f}GB "
                      f"({done}/{len(jobs)} files, {(total_bytes)/el/1e6:.0f}MB/s text "
                      f"this session incl. resumed)", flush=True)
                _write_stats()
            if total_bytes >= budget_bytes:
                stopped = True
                for f in list(pending):
                    f.cancel()
                break
            if not stopped:
                _submit()
    exhausted = total_bytes < budget_bytes * 0.99
    _write_stats(exhausted)
    for p in tmp_dir.glob("**/*"):
        if p.is_file():
            p.unlink()
    if exhausted:
        print(f"[fast-pool2] WARNING {lang}: corpus exhausted at "
              f"{total_bytes/1e9:.1f}GB < budget {budget_bytes/1e9:.1f}GB")
    print(f"[fast-pool2] {lang}: DONE {total_bytes/1e9:.2f}GB text, {total_docs} docs, "
          f"{(time.time()-t0)/60:.1f} min")
    return json.loads(stats_path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--gb", type=float, default=None, help="override byte budget (GB)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.gb:
        budget = args.gb * 1e9
    else:
        from xscript.data.fineweb import plan_budgets
        budget = plan_budgets()[args.lang]
    build(args.lang, budget, workers=args.workers)


if __name__ == "__main__":
    main()
