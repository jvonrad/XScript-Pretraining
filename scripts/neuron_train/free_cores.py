#!/usr/bin/env python
"""Report currently-idle NeuronCore capacity via `neuron-ls --json-output
--show-all-procs` (machine-readable, not table-scraping).

A device counts busy if ANY process other than `neuron-ls` itself is
attached (--show-all-procs makes shared-device sub-tenants visible too).
Groups consecutive free devices (by `neuron_device` index, which is
core-id-contiguous on this instance) into maximal contiguous blocks, since a
single `NEURON_RT_VISIBLE_CORES` range for one `torchrun` launch must be
contiguous.

`free_runs()` -> list of (first_core_id, last_core_id, n_logical_cores),
largest first. Run standalone to print them.
"""
import json
import subprocess


def free_runs() -> list[tuple[int, int, int]]:
    raw = subprocess.run(["neuron-ls", "--json-output", "--show-all-procs"],
                         capture_output=True, text=True, check=True).stdout
    devices = sorted(json.loads(raw), key=lambda d: d["neuron_device"])
    blocks: list[list[dict]] = []
    cur: list[dict] = []
    for d in devices:
        procs = [p for p in d.get("neuron_processes", [])
                if not p.get("command", "").startswith("neuron-ls")]
        if not procs:
            cur.append(d)
        else:
            if cur:
                blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    out = []
    for block in blocks:
        ids = [i for d in block for i in d["neuroncore_ids"]]
        n_logical = sum(d["nc_count"] // d["logical_neuroncore_config"] for d in block)
        out.append((min(ids), max(ids), n_logical))
    out.sort(key=lambda r: -r[2])
    return out


if __name__ == "__main__":
    runs = free_runs()
    if not runs:
        print("(no free devices)")
    for lo, hi, n in runs:
        print(f"{lo}-{hi}  ({n} logical cores)")
