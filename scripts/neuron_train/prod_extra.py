"""Production Neuron training entry for the corroboration-tokenizer runs
(unigram_50lang / unigram_bi_<X> bilinguals), launched via xmp.spawn by
`run_extra.sh` and supervised by `orchestrate_extra.sh`.

Differences from `prod_train.py` (the de/zh finish-up entry): the FULL 30B
WSD schedule from `configs/base_main.yaml` is kept (warmup 1B, stable 23B,
decay 6B -> a COOLED final at 3.0e-4, which is what CLAUDE.md 6j's
cooled-vs-cooled comparison needs), no warm start, and a resumable `last.pt`
every 1B tokens after 5B (base_main saves only every 2B late in training,
which at ~48k tok/s is ~11h of work to lose on a collective hang).

Everything else is the PROVEN Neuron config (NEURON.md 9 / STATUS.md):
fp32 params, micro_batch=2, ZeRO-1, full cross-entropy, in-loop eval off.

env in:  PROD_MODEL (run name)
"""
import os
import sys

sys.path.insert(0, "/home/ubuntu/XScript-Pretraining/src")
import torch_xla.distributed.xla_multiprocessing as xmp  # noqa: E402
from xscript import runmatrix, train_neuron  # noqa: E402

MODEL = os.environ["PROD_MODEL"]


def _mp_fn(index):
    cfg = runmatrix.get_run("configs/base_main.yaml", "unigram", MODEL, True)
    cfg["train"] = {
        **cfg["train"],
        "micro_batch_size": 2,     # proven-to-fit with ZeRO at seq_len=2048
        "bf16_params": False,      # fp32: most comparable to the 15 CUDA models
        "fused_ce_chunk": 0,       # full cross-entropy (proven correct on Neuron)
        "eval_in_loop": False,     # BPB eval post-hoc (variable-len -> recompiles)
        # superset of base_main's log-spaced grid: 250M to 2B, 500M to 5B,
        # then 1B (base_main: 1B to 15B, 2B after) -- every original budget
        # (1,2,5,8,10,12,15,23B) is still hit, and a hang costs <= 1B tokens.
        "ckpt_schedule": [[2e9, 250e6], [5e9, 500e6], [1e15, 1e9]],
    }
    cfg["wandb_id"] = MODEL + "__neuron"
    train_neuron.run_from_config(cfg)


if __name__ == "__main__":
    xmp.spawn(_mp_fn, args=())
    print(f"PROD_{MODEL}_DONE")
