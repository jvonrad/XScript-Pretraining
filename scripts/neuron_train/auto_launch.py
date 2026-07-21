#!/usr/bin/env python
"""Orchestrate the 3 unfinished models end-to-end: wait for each language's
pool download to finish -> pack only the tokenizer condition(s) actually
needed -> wait for free NeuronCore capacity (respecting whatever else is
running on this shared box, e.g. eval jobs) -> launch training pinned to
that free range.

Needed work (see scripts/neuron_train/README.md):
  de: pack unigram_starved only (unigram_destarved/"de-fair" already trained)
  zh: pack BOTH unigram_starved and unigram_destarved (no zh model exists yet)
  -> 3 training runs: de__unigram_starved, zh__unigram_starved,
     zh__unigram_destarved

Never touches cores already in use by another process (checked via
free_cores.free_runs(), which parses `neuron-ls --json-output
--show-all-procs`) -- if nothing is free, it polls and waits rather than
launching. Each launch claims its own disjoint core range (tracked locally
too, to avoid two of this script's own launches racing onto the same range
before neuron-ls reflects the first one's PID).

Run under nohup; logs everything to stdout (redirect to a file by the
caller). Safe to leave running unattended -- it only ever starts subprocess
jobs, never kills or touches anything it didn't start itself.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from free_cores import free_runs  # noqa: E402

XSCRIPT_SCRATCH = "/mnt/scratch/xscript"
WORK = "/mnt/scratch/xscript_train"
LOG_DIR = Path(XSCRIPT_SCRATCH) / "logs"

# (lang, [tok_conditions to pack], [run names to train once packed])
PLAN = [
    ("de", ["unigram_starved"], ["de__unigram_starved"]),
    ("zh", ["unigram_starved", "unigram_destarved"],
          ["zh__unigram_starved", "zh__unigram_destarved"]),
]

POLL_S = 60
MIN_CORES_TO_START = 2      # at least one free device (1 logical core pair... actually
                            # one device = 2 logical cores at logical_neuroncore_config=2)
TARGET_CORES_PER_JOB = 16   # cap so one job doesn't grab the whole box


def log(msg):
    print(f"[auto-launch] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def _base_env():
    import os
    env = dict(os.environ)
    assert env.get("HF_TOKEN"), "HF_TOKEN not set -- export it in ~/.bashrc"
    env["XSCRIPT_SCRATCH"] = XSCRIPT_SCRATCH
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = f"{Path.home()}/.local/bin:" + env.get("PATH", "")
    env["PYTHONPATH"] = str(_ROOT / "src")
    return env


def wait_for_pid(pid_file: Path, label: str):
    pid = int(pid_file.read_text().strip())
    log(f"waiting for {label} (pid {pid}) to finish downloading...")
    while True:
        try:
            subprocess.run(["kill", "-0", str(pid)], check=True,
                           capture_output=True)
        except subprocess.CalledProcessError:
            log(f"{label} download finished")
            return
        time.sleep(30)


def pack(lang: str, tok: str):
    log(f"packing {lang}/{tok} ...")
    env = _base_env()
    cmd = ["bash", "-lc",
          f"source ~/neuron_venv/bin/activate 2>/dev/null; "
          f"python3 -m xscript.cli pack --lang {lang} --tok {tok} --workers 32"]
    r = subprocess.run(cmd, cwd=str(_ROOT), env=env,
                       capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).splitlines()[-10:])
    log(f"pack {lang}/{tok} done (rc={r.returncode}):\n{tail}")
    if r.returncode != 0:
        raise RuntimeError(f"pack failed for {lang}/{tok}")


def launch_training(run_name: str, claimed: list):
    while True:
        runs = free_runs()
        usable = [r for r in runs
                 if not any(not (r[1] < c[0] or r[0] > c[1]) for c in claimed)]
        usable.sort(key=lambda r: -r[2])
        if usable and usable[0][2] >= MIN_CORES_TO_START:
            lo, hi, n = usable[0]
            n_use = min(n, TARGET_CORES_PER_JOB)
            core_hi = lo + (n_use * 2) - 1
            claimed.append((lo, core_hi, n_use))
            log(f"launching {run_name} on cores {lo}-{core_hi} "
               f"({n_use} logical cores)")
            env = _base_env()
            env["NPROC"] = str(n_use)
            env["NEURON_RT_VISIBLE_CORES"] = f"{lo}-{core_hi}"
            env["WORK"] = WORK
            env["RUN"] = run_name
            log_path = LOG_DIR / f"train_{run_name}.log"
            with open(log_path, "w") as lf:
                subprocess.Popen(
                    ["bash", "scripts/neuron_train/launch.sh"],
                    cwd=str(_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                )
            log(f"{run_name} launched (log: {log_path})")
            return
        log(f"{run_name}: no free cores yet (need >={MIN_CORES_TO_START} "
           f"logical, free={[r[2] for r in usable]}); waiting {POLL_S}s")
        time.sleep(POLL_S)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    claimed: list = []
    for lang, toks, run_names in PLAN:
        pid_file = LOG_DIR / f"fastpool_{lang}.pid"
        wait_for_pid(pid_file, lang)
        for tok in toks:
            pack(lang, tok)
        for run_name in run_names:
            launch_training(run_name, claimed)
    log("ALL 3 TRAINING RUNS LAUNCHED")


if __name__ == "__main__":
    main()
