#!/usr/bin/env python3
"""Short real-model GPU pilot using production data and training code."""
import os

from xscript import _yaml, runmatrix, train


base = _yaml.load(os.environ.get("PILOT_BASE", "configs/base_main.yaml"))
name = os.environ.get("PILOT_NAME", "_gpu_pilot_en-ar_unigram_destarved")
cfg = runmatrix._stamp(base, name, "en-ar", "unigram_destarved")

# About six optimizer steps at the production ~1M-token global batch.  Frequent
# checkpoints and tiny eval slices deliberately exercise more than just fwd/bwd.
cfg["schedule"] = {
    **cfg["schedule"],
    "warmup_tokens": 1.0e6,
    "stable_tokens": 3.0e6,
    "decay_tokens": 1.0e6,
}
cfg["train"] = {
    **cfg["train"],
    "log_every": 1,
    "eval_docs": 8,
    "ckpt_schedule": [[1.0e15, 2.0e6]],
}

train.run_from_config(cfg)
