#!/usr/bin/env python
"""One-glance status of all training runs.

Usage (activate the venv first: `source $XSCRIPT_SCRATCH/venv/bin/activate`):

  python scripts/train_status.py            # all runs under $XSCRIPT_SCRATCH/runs
  python scripts/train_status.py en-ar       # only runs whose name contains this

For each run, reads the last few lines of its train.jsonl (see Trainer._log in
src/xscript/train.py: per-step records have loss/tok_per_s, checkpoint-interval
records have eval/eval_final) and reports latest step, tokens, loss, tok/s,
most recent eval BPB, and how stale the log is (no new line in a while usually
means the job finished, is queued, or crashed -- cross-check with squeue).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _runs_root() -> Path:
    scratch = os.environ.get("XSCRIPT_SCRATCH")
    if scratch:
        return Path(scratch) / "runs"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from xscript.paths import RUNS
    return RUNS


def _tail_json_records(path: Path, n: int = 400) -> list[dict]:
    """Last n lines of a jsonl file, parsed (skips any malformed trailing line
    from a process killed mid-write)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = min(size, 1 << 20)   # last 1MB is plenty for `n` short lines
            f.seek(size - block)
            lines = f.read().decode("utf-8", errors="ignore").splitlines()
    except FileNotFoundError:
        return []
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _squeue_states() -> dict[str, str]:
    """{job_name: state} for this user's current jobs (RUNNING/PENDING/...)."""
    try:
        out = subprocess.run(["squeue", "--me", "-h", "-o", "%j %T"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {}
    states = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            states[parts[0]] = parts[1]
    return states


def _fmt_age(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 7200:
        return f"{seconds/60:.0f}m"
    return f"{seconds/3600:.1f}h"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Status of all training runs.")
    ap.add_argument("filter", nargs="?", default="", help="substring to filter run names")
    a = ap.parse_args(argv)

    root = _runs_root()
    if not root.exists():
        print(f"no runs dir at {root} yet")
        return
    run_dirs = sorted(d for d in root.iterdir()
                      if d.is_dir() and a.filter in d.name)
    if not run_dirs:
        print(f"no runs found under {root}" + (f" matching '{a.filter}'" if a.filter else ""))
        return

    jobs = _squeue_states()
    now = time.time()

    hdr = f"{'run':38} {'job':10} {'step':>8} {'tokens':>9} {'loss':>7} {'tok/s':>8} {'eval bpb':>28} {'log age':>8}"
    print(hdr)
    print("-" * len(hdr))
    for d in run_dirs:
        name = d.name
        log_path = d / "train.jsonl"
        recs = _tail_json_records(log_path)
        last_step = next((r for r in reversed(recs) if "loss" in r), None)
        last_eval = next((r for r in reversed(recs)
                          if "eval" in r or "eval_final" in r), None)

        job_state = jobs.get(name, "-")
        step = f"{last_step['step']}" if last_step else "-"
        tokens = f"{last_step['tokens']/1e9:.2f}B" if last_step else "-"
        loss = f"{last_step['loss']:.4f}" if last_step else "-"
        tps = f"{last_step['tok_per_s']/1e3:.0f}k" if last_step else "-"
        if last_eval:
            ev = last_eval.get("eval") or last_eval.get("eval_final") or {}
            bpb = " ".join(f"{k}={v['bpb']:.3f}" for k, v in list(ev.items())[:3])
        else:
            bpb = "-"
        age = _fmt_age(now - log_path.stat().st_mtime) if log_path.exists() else "no log"

        print(f"{name:38} {job_state:10} {step:>8} {tokens:>9} {loss:>7} {tps:>8} {bpb:>28} {age:>8}")


if __name__ == "__main__":
    main()
