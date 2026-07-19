#!/usr/bin/env python
"""Pool download progress + live ETA per language.

Usage (activate the venv first: `source $XSCRIPT_SCRATCH/venv/bin/activate`):

  python scripts/pool_status.py
  watch -n 30 'python scripts/pool_status.py'   # live-refreshing, shows ETA

Rate/ETA needs two data points, so the first run shows '-' for both; every
run after that compares against a cached snapshot (a small state file under
$XSCRIPT_SCRATCH) to report a real measured MB/s and time-to-completion per
language. Reads each pool's own stats.json checkpoint when present (see
xscript.data.fineweb.build_pool) -- exact, and survives the writer being
mid-file. Falls back to summing on-disk shard sizes if a language hasn't
checkpointed yet.
"""
import json
import os
import sys
import time
from pathlib import Path

LANGS = ["en", "de", "fr", "ar", "zh"]


def _scratch() -> Path:
    scratch = os.environ.get("XSCRIPT_SCRATCH")
    if scratch:
        return Path(scratch)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from xscript.paths import SCRATCH
    return SCRATCH


def _budgets(scratch: Path) -> dict[str, float]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    try:
        from xscript.data.fineweb import plan_budgets
        return plan_budgets()
    except Exception as exc:
        print(f"[pool_status] couldn't compute real budgets ({exc}); "
              f"showing bytes downloaded only, no % / ETA")
        return {}


def main() -> None:
    scratch = _scratch()
    root = scratch / "pools"
    state_path = scratch / "_pool_status_state.json"
    prev = json.loads(state_path.read_text()) if state_path.exists() else {}
    now = time.time()
    budgets = _budgets(scratch)

    cur = {}
    print(f"{'lang':4} {'GB':>8} {'target':>8} {'%':>6} {'rate':>10} {'ETA':>8}")
    for lang in LANGS:
        d = root / lang
        stats = d / "stats.json"
        if stats.exists():
            st = json.loads(stats.read_text())
            b = st["text_bytes"]
            budget = st.get("budget_bytes") or budgets.get(lang, 0)
        else:
            b = sum(f.stat().st_size for f in d.glob("*.zst")) if d.exists() else 0
            budget = budgets.get(lang, 0)
        cur[lang] = {"bytes": b, "t": now}

        pct = f"{100*b/budget:5.1f}%" if budget else "  -"
        rate_str = eta_str = "-"
        if lang in prev:
            dt = now - prev[lang]["t"]
            db = b - prev[lang]["bytes"]
            if dt > 2:
                rate = db / dt
                # stats.json only updates every CHECKPOINT_EVERY_N_FILES source
                # files (see fineweb.py) -- a 0 reading between two close-together
                # checks is normal checkpoint granularity, not necessarily stalled
                rate_str = f"{rate/1e6:.2f}MB/s" if rate > 0 else "0MB/s"
                remaining = budget - b
                if remaining <= 0:
                    eta_str = "done"
                elif rate > 0:
                    eta_h = remaining / rate / 3600
                    eta_str = f"{eta_h:.1f}h"
        print(f"{lang:4} {b/1e9:8.1f} {budget/1e9:8.1f} {pct:>6} {rate_str:>10} {eta_str:>8}")

    state_path.write_text(json.dumps(cur))


if __name__ == "__main__":
    main()
