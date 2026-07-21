#!/usr/bin/env python
"""Parallel pool downloader — fetches multiple FineWeb2-HQ parquet files
concurrently instead of `xscript.data.fineweb.build_pool`'s one-file-at-a-time
loop. Empirically ~10-15x faster wall-clock on this box: a single `_iter_texts`
stream sustains only ~4-5MB/s, but 8 concurrent streams hit ~60MB/s aggregate
(measured directly against FineWeb2-HQ from this host) — the bottleneck is
per-HTTP-connection throughput, not the link or local disk/CPU.

Produces byte-for-byte-COMPATIBLE output: identical `pool_NNNNN.jsonl.zst`
shard format and `stats.json` schema to `xscript.data.fineweb.build_pool`, so
`xscript pack` (and, if ever needed, resuming via the standard `xscript pool`
command) work completely unmodified against pools built by this script. This
file only *imports* from `xscript.data.fineweb` (`_sources_for`,
`_list_parquets`, `_iter_texts`, `_PoolWriter`) — it does not change it, so
the existing sequential CLI path is untouched and still usable as-is.

Simplification vs. `build_pool`: on an interrupted/crashed run, this resumes
by skipping already-checkpointed files and continuing the writer from the
last checkpointed shard index, but (unlike `build_pool`) does not revalidate
whether that last shard was a clean checkpoint boundary or a mid-write
truncation — acceptable for this one-off, closely-watched download (worst
case: redo one ~1GB shard), not a general-purpose replacement.

CAUTION -- HF resolver rate limit: HF enforces a per-account quota of 5000
"resolver" requests / 5 minutes. Running two languages simultaneously at 16
workers each (32 combined concurrent file-opens) tripped this within ~4
minutes, after which every further file-open 429'd and the run falsely
concluded "corpus exhausted" well short of budget. `fetch_one` now retries
429s with backoff instead of giving up, but concurrency should still stay
modest (default 8/language; keep combined concurrency across simultaneously-
running language processes well under ~15-20 to leave headroom).

Usage:
    python scripts/neuron_train/fast_pool.py --lang de --workers 8
    python scripts/neuron_train/fast_pool.py --lang zh --workers 4 --gb 140
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from xscript.data.fineweb import (_sources_for, _list_parquets, _iter_texts,  # noqa: E402
                                  _PoolWriter, HOLDOUT_BYTES)
from xscript.paths import pool_dir, HOLDOUT  # noqa: E402


def fast_build_pool(lang: str, budget_bytes: float, workers: int = 8,
                    checkpoint_every: int = 16) -> dict:
    sources = _sources_for(lang)
    primary_repo, primary_subdir = sources[0]
    first_files = _list_parquets(primary_repo, primary_subdir)
    if not first_files:
        raise RuntimeError(f"no parquet files for {lang}")
    out = pool_dir(lang)
    stats_path = out / "stats.json"

    resume = None
    if stats_path.exists():
        st = json.loads(stats_path.read_text())
        if st["text_bytes"] >= budget_bytes * 0.99:
            print(f"[fast-pool] {lang}: cached ({st['text_bytes']/1e9:.1f}GB)")
            return st
        resume = st
        print(f"[fast-pool] {lang}: resuming ({st['text_bytes']/1e9:.1f}/"
              f"{budget_bytes/1e9:.1f}GB, {len(st['files_consumed'])} files done)")

    if resume is None:
        hw = _PoolWriter(HOLDOUT, prefix=lang)
        got = 0
        for t in _iter_texts(primary_repo, first_files[0]):
            hw.write(t)
            got += len(t.encode("utf-8"))
            if got >= HOLDOUT_BYTES:
                break
        hw.close()
        print(f"[fast-pool] {lang}: holdout done ({got/1e6:.1f}MB)")
        used: list[str] = []
        pw = _PoolWriter(out)
    else:
        got = resume["holdout_bytes"]
        used = list(resume["files_consumed"])
        pw = _PoolWriter(out, start_idx=resume["shard_idx"],
                         total_bytes=resume["text_bytes"], total_docs=resume["docs"])

    lock = threading.Lock()
    stop = threading.Event()
    used_set = set(used)

    all_jobs = []
    for i, (repo, subdir) in enumerate(sources):
        files = first_files if i == 0 else _list_parquets(repo, subdir)
        pool_files = files[1:] if i == 0 else files
        for f in pool_files:
            tag = f"{repo}::{f}"
            if tag not in used_set:
                all_jobs.append((repo, f, tag))

    def fetch_one(repo, f, tag):
        if stop.is_set():
            return None
        backoff = 30.0
        for attempt in range(10):
            if stop.is_set():
                return None
            try:
                for t in _iter_texts(repo, f):
                    if stop.is_set():
                        break
                    with lock:
                        pw.write(t)
                return tag
            except Exception as exc:
                msg = str(exc).lower()
                # HF's per-account "resolver requests / 5 min" quota (429) --
                # transient, resets on a rolling window. Too much concurrency
                # across BOTH languages' processes trips this almost
                # instantly (observed: 32 combined workers exhausted the
                # 5000-req/5min quota within ~4 minutes). Back off and retry
                # rather than treating it as a real per-file failure --
                # otherwise every remaining file in the job list fails fast
                # and the run falsely concludes "corpus exhausted".
                if ("429" in msg or "rate limit" in msg or
                        "too many requests" in msg) and attempt < 9:
                    print(f"[fast-pool] rate-limited on {tag}, backing off "
                          f"{backoff:.0f}s (attempt {attempt+1}/10)")
                    time.sleep(backoff)
                    backoff = min(backoff * 1.4, 180.0)
                    continue
                print(f"[fast-pool] WARN {tag}: {exc}")
                return None
        return None

    t0 = time.time()
    done_count = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_one, repo, f, tag): tag for repo, f, tag in all_jobs}
        for fut in as_completed(futures):
            tag = fut.result()
            done_count += 1
            if tag:
                used.append(tag)
            with lock:
                total, docs, idx = pw.total_bytes, pw.total_docs, pw.idx
            if total >= budget_bytes:
                stop.set()
            if done_count % checkpoint_every == 0 or stop.is_set():
                dt = time.time() - t0
                rate = total / dt / 1e6 if dt > 0 else 0
                print(f"[fast-pool] {lang}: {total/1e9:.2f}/{budget_bytes/1e9:.1f}GB "
                      f"({done_count}/{len(all_jobs)} files this session, "
                      f"{rate:.1f}MB/s avg)")
                with lock:
                    pw._roll()
                    st = {"lang": lang, "budget_bytes": budget_bytes,
                          "text_bytes": pw.total_bytes, "docs": pw.total_docs,
                          "holdout_bytes": got, "holdout_file": first_files[0],
                          "files_consumed": list(used), "shard_idx": pw.idx,
                          "exhausted": False}
                    stats_path.write_text(json.dumps(st, indent=2))
            if stop.is_set():
                break
        for f in futures:
            f.cancel()

    with lock:
        pw.close()
        exhausted = pw.total_bytes < budget_bytes * 0.99
        st = {"lang": lang, "budget_bytes": budget_bytes, "text_bytes": pw.total_bytes,
              "docs": pw.total_docs, "holdout_bytes": got, "holdout_file": first_files[0],
              "files_consumed": used, "shard_idx": pw.idx, "exhausted": exhausted}
        stats_path.write_text(json.dumps(st, indent=2))
    if exhausted:
        print(f"[fast-pool] WARNING {lang}: corpus exhausted at "
              f"{pw.total_bytes/1e9:.1f}GB < budget {budget_bytes/1e9:.1f}GB")
    print(f"[fast-pool] {lang}: DONE {pw.total_bytes/1e9:.2f}GB text, {pw.total_docs} docs")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--gb", type=float, default=None, help="override byte budget (GB)")
    ap.add_argument("--workers", type=int, default=8,
                    help="keep modest -- 32 combined workers across two "
                         "simultaneous language runs tripped HF's 5000-req/"
                         "5min resolver quota within ~4 minutes")
    args = ap.parse_args()
    if args.gb:
        budget = args.gb * 1e9
    else:
        from xscript.data.fineweb import plan_budgets
        budget = plan_budgets()[args.lang]
    fast_build_pool(args.lang, budget, workers=args.workers)


if __name__ == "__main__":
    main()
