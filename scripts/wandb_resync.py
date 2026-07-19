#!/usr/bin/env python3
"""Catch up wandb runs whose local background sync process died mid-training.

train.jsonl (written directly by the training loop) is always the authoritative
record; this script never touches a running training process. It just opens a
resumed wandb session for a run and replays any train.jsonl records with more
tokens than wandb's current summary already has -- safe to run against runs
that are still training normally (it just resumes+finishes without missing
anything) or ones whose sync silently died (it catches the gap up).

Usage: python scripts/wandb_resync.py [run_name ...]   (default: all runs)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xscript.paths import RUNS


def resync(name: str) -> None:
    import wandb
    log = RUNS / name / "train.jsonl"
    if not log.exists():
        print(f"[resync] {name}: no train.jsonl, skipping")
        return

    api = wandb.Api()
    try:
        run = api.run(f"XScript-Pretraining/{name}")
        last_tokens = run.summary.get("tokens", -1)
        if run.state == "running":
            # a live training process may still hold its own wandb session;
            # don't race a second writer against it -- only catch up runs
            # whose background sync has actually died server-side.
            print(f"[resync] {name}: wandb still 'running' (at {last_tokens/1e9:.2f}B), skipping")
            return
    except Exception:
        last_tokens = -1  # no wandb run yet -- replay everything

    records = []
    for line in log.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "tokens" in rec and rec["tokens"] > last_tokens:
            records.append(rec)
    if not records:
        print(f"[resync] {name}: already caught up (wandb at {last_tokens/1e9:.2f}B tokens)")
        return

    wb = wandb.init(project="XScript-Pretraining", name=name, id=name, resume="allow")
    n_step, n_eval = 0, 0
    for rec in records:
        step = rec["step"]
        if "loss" in rec:
            wb.log({**rec, "tokens_b": rec["tokens"] / 1e9}, step=step)
            n_step += 1
        ev = rec.get("eval") or rec.get("eval_final")
        if ev:
            key = "eval_final" if "eval_final" in rec else "eval"
            wb.log({f"{key}/{k}_bpb": v["bpb"] for k, v in ev.items()} |
                   {f"{key}/{k}_ppl": v["ppl_token"] for k, v in ev.items()}, step=step)
            n_eval += 1
    wb.finish()
    print(f"[resync] {name}: replayed {n_step} step records + {n_eval} eval records "
          f"(was stuck at {last_tokens/1e9:.2f}B, now at {records[-1]['tokens']/1e9:.2f}B)")


def main():
    names = sys.argv[1:]
    if not names:
        names = sorted(d.name for d in RUNS.iterdir() if d.is_dir())
    for name in names:
        try:
            resync(name)
        except Exception as exc:
            print(f"[resync] {name}: FAILED ({exc})")


if __name__ == "__main__":
    main()
