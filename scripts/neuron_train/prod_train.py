"""Production Neuron training entry for one XScript model (xmp.spawn).

Loads the real 30B WSD config from runmatrix and forces the PROVEN, robust
Neuron config: fp32 params, micro_batch=2, ZeRO-1 (auto when world>1), full
cross-entropy. (bf16 forward params + the fused chunked-CE unlock bigger
micro_batch but are slower to compile / less battle-tested; not used for the
unattended run.) Warm-starts the zh models from their 12B checkpoints; de trains
from scratch. wandb is enabled via WANDB_API_KEY in the env (the trainer wraps
wandb.init in try/except and silently continues if it fails).

env in:  PROD_MODEL (run name), PROD_WARM (warm-start checkpoint path or "")
"""
import os
import sys

sys.path.insert(0, "/home/ubuntu/XScript-Pretraining/src")
import torch_xla.distributed.xla_multiprocessing as xmp
from xscript import runmatrix, train_neuron

MODEL = os.environ["PROD_MODEL"]
WARM = os.environ.get("PROD_WARM", "")
TARGET_B = float(os.environ.get("PROD_TARGET_B", "30"))  # stop at this many B tokens


def _mp_fn(index):
    cfg = runmatrix.get_run("configs/base_main.yaml", "unigram", MODEL, True)
    cfg["train"] = {
        **cfg["train"],
        "micro_batch_size": 2,     # proven-to-fit with ZeRO at seq_len=2048
        "bf16_params": False,      # fp32: most comparable to the 15 CUDA models
        "fused_ce_chunk": 0,       # full cross-entropy (proven correct on Neuron)
        "eval_in_loop": False,     # BPB eval post-hoc (variable-len -> recompiles)
    }
    # Override the WSD total so the run STOPS at TARGET_B (e.g. 15B). warmup=1B
    # (already passed at the 11.75B resume), then constant peak-lr stable to the
    # target, no decay -> a raw 15B checkpoint at stable lr (matches how the 12B
    # partials look; a cooldown is a separate cheap branch run later if wanted).
    # The lr in the 11.75B->15B range is peak either way, so this doesn't change
    # the trajectory vs the original 30B schedule -- it just terminates at 15B.
    sc = dict(cfg["schedule"])
    sc["warmup_tokens"] = 1_000_000_000
    sc["stable_tokens"] = int(TARGET_B * 1e9) - 1_000_000_000
    sc["decay_tokens"] = 0
    cfg["schedule"] = sc
    # Dense checkpoints (every 250M tokens) so an occasional collective hang
    # only costs <250M tokens on restart, instead of thrashing all the way back
    # to the 11.76B warm-start (the default 1B interval > the typical time
    # between hangs, which would prevent ever saving a resumable checkpoint).
    cfg["train"]["ckpt_schedule"] = [[1e15, 250_000_000]]
    # Fresh wandb run id for the Neuron runs so we don't collide with stale
    # runs of the same name (a prior de run sat at step 8381 -> wandb rejects
    # our lower steps). Stable across restarts/resumes of THIS run.
    cfg["wandb_id"] = MODEL + "__neuron"
    if WARM:
        cfg["warm_start"] = {"from": WARM}
    train_neuron.run_from_config(cfg)


if __name__ == "__main__":
    xmp.spawn(_mp_fn, args=())
    print(f"PROD_{MODEL}_DONE")
