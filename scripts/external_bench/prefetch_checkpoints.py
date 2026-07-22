#!/usr/bin/env python
"""Pre-download + reassemble every checkpoint in the HF export, in parallel.

`run_appendix_c5.py`'s `fetch_checkpoint()` already reassembles the
`final.pt.partNNN` chunks, but it does so serially inside each eval process, so
a fan-out spends its first minutes with 24 Neuron core-pairs idle waiting on the
network. This does the whole fetch up front with a thread pool, writing to the
exact layout the eval scripts expect:

    <workdir>/_assembled/runs/<run>/checkpoints/final.pt

Parts are appended and unlinked one at a time, so peak disk per in-flight run is
one assembled checkpoint plus one part, not two full copies.

Runs that already have a result JSON are skipped by default (--skip-done), as
are runs whose assembled `final.pt` is already the expected size.

    python prefetch_checkpoints.py --repo jvonrad/xscript-eval \
        --workdir /mnt/scratch/xscript_c5 --workers 12
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

# NB: plain Lock, so nothing that already holds it may call log() -- use
# _log_done() for the completion messages, which does the counter bump and the
# print in ONE locked block rather than nesting two acquisitions.
_print_lock = threading.Lock()
_done = 0


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def _log_done(total, msg):
    global _done
    with _print_lock:
        _done += 1
        print(f"[{_done}/{total}] {msg}", flush=True)


def _dl(filename, dl, tries=8):
    """hf_hub_download with backoff, hardened against two failure modes.

    1. The hub 429s ('maximum queue size reached') under fan-out -- same issue
       run_bpb.py's `_dl_retry` handles. Concurrency here is bounded by
       --workers, but hf_transfer also parallelises chunks WITHIN each file, so
       the request rate is a multiple of the worker count and 429s appear well
       before the network saturates.

    2. A 429 on the metadata HEAD makes hf_hub_download fall back to the local
       cache. We unlink each part as soon as it is folded into the assembled
       checkpoint, so the local_dir bookkeeping still claims the part is
       present and the call RETURNS A PATH THAT DOES NOT EXIST rather than
       raising. Treat that as retryable and force a real download.
    """
    import random
    for attempt in range(tries):
        try:
            p = Path(hf_hub_download(filename=filename,
                                     force_download=attempt > 0, **dl))
            if p.exists():
                return p
            raise FileNotFoundError(f"hub returned missing path {p}")
        except Exception as exc:
            if attempt == tries - 1:
                raise
            wait = min(60, 2 ** attempt) + random.uniform(0, 3)
            log(f"   {type(exc).__name__} on {filename}; "
                f"retry {attempt + 1}/{tries - 1} in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_one(run, repo, repo_type, work, sizes, total):
    """Download+assemble one run's checkpoint. Returns (run, bytes, seconds)."""
    t0 = time.time()
    rel = f"runs/{run}/checkpoints"
    parts = sorted(f for f in sizes if f.startswith(f"{rel}/final.pt.part"))
    if not parts:
        raise RuntimeError(f"{run}: no final.pt.part* in repo")

    expect = sum(sizes[p] for p in parts)
    out = work / "_assembled" / rel / "final.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size == expect:
        _log_done(total, f"{run}: already complete ({expect/1e9:.2f} GB)")
        return run, 0, 0.0

    # unique tmp per process AND per thread: several eval processes may race here
    tmp = out.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
    dl = dict(repo_id=repo, repo_type=repo_type,
              local_dir=str(work / "_repo"))
    try:
        with open(tmp, "wb") as w:
            for p in parts:
                local = _dl(p, dl)
                with open(local, "rb") as r:
                    while chunk := r.read(64 * 1024 * 1024):
                        w.write(chunk)
                local.unlink(missing_ok=True)   # folded in; reclaim the space now
        got = tmp.stat().st_size
        if got != expect:
            raise RuntimeError(f"{run}: assembled {got} bytes, expected {expect}")
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    dt = time.time() - t0
    _log_done(total, f"{run}: {expect/1e9:.2f} GB in {dt:.0f}s "
                     f"({expect/1e6/max(dt,1e-9):.0f} MB/s)")
    return run, expect, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--workers", type=int, default=10,
                    help="concurrent runs; hf_transfer parallelises within each "
                         "file too, so this need not be large (HF 429s above ~16)")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--results-subdir", default="results/appendix_c5",
                    help="used by --skip-done to detect already-evaluated runs")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip runs that already have a <run>_final.json result")
    args = ap.parse_args()

    work = Path(args.workdir).resolve()
    api = HfApi()
    info = api.repo_info(args.repo, repo_type=args.repo_type, files_metadata=True)
    sizes = {s.rfilename: (s.size or 0) for s in info.siblings}

    models = json.loads(Path(hf_hub_download(
        args.repo, "models.json", repo_type=args.repo_type)).read_text())
    all_runs = sorted(models) if isinstance(models, dict) else sorted(models)

    runs = args.runs or all_runs
    unknown = [r for r in runs if r not in all_runs]
    if unknown:
        sys.exit(f"unknown runs: {unknown}")

    if args.skip_done:
        rdir = work / args.results_subdir
        # a FAILED eval also leaves a <run>_final.json, but with an "error" key
        # -- treat those as not-done so a resume refetches and retries them.
        have = set()
        if rdir.is_dir():
            for f in rdir.glob("*_final.json"):
                try:
                    if "error" not in json.loads(f.read_text()):
                        have.add(f.name.removesuffix("_final.json"))
                except (OSError, json.JSONDecodeError):
                    pass
        skipped = [r for r in runs if r in have]
        runs = [r for r in runs if r not in have]
        if skipped:
            print(f"skipping {len(skipped)} already-evaluated run(s)")

    # Reap partial assemblies orphaned by an interrupted run. The in-flight
    # cleanup is an `except BaseException` in fetch_one, which SIGKILL skips --
    # so a hard kill strands one ~4 GB .tmp per worker. Only reap tmps whose
    # owning PID is gone: another prefetch may legitimately be running.
    stale = 0
    for t in (work / "_assembled").glob("runs/*/checkpoints/final.pt.tmp.*"):
        try:
            pid = int(t.name.split(".tmp.")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        try:
            os.kill(pid, 0)          # owner still alive -> leave it alone
        except ProcessLookupError:
            stale += t.stat().st_size
            t.unlink(missing_ok=True)
        except PermissionError:
            pass
    if stale:
        print(f"reaped {stale/1e9:.1f} GB of orphaned .tmp assemblies")

    want = sum(sizes[f] for r in runs for f in sizes
               if f.startswith(f"runs/{r}/checkpoints/final.pt.part"))
    print(f"fetching {len(runs)} run(s), {want/1e9:.1f} GB, "
          f"{args.workers} workers -> {work/'_assembled'}")

    t0 = time.time()
    errs = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, r, args.repo, args.repo_type, work,
                          sizes, len(runs)): r for r in runs}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                errs.append((futs[f], repr(e)))
                log(f"!! {futs[f]}: {e}")
    dt = time.time() - t0
    print(f"\ndone in {dt/60:.1f} min ({want/1e6/max(dt,1e-9):.0f} MB/s aggregate)")
    if errs:
        print(f"{len(errs)} FAILED:")
        for r, e in errs:
            print(f"  {r}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
