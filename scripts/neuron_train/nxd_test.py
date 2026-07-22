"""Does AWS's neuronx-distributed (NxD) run OUR model across many cores?

The decisive test of the thing we got wrong: NxD's `initialize_parallel_model`
wraps an ARBITRARY nn.Module, so we keep `xscript.model.Transformer` exactly as
trained on Isambard (interleaved RoPE, our init) and only borrow AWS's tested
parallelism + collectives.

TP=1 (pure data parallel) + ZeRO-1: our unmodified model uses plain nn.Linear, so
tensor-parallel sharding would require editing model.py; data parallelism does
not. This directly tests the wall we hit -- our hand-rolled ZeRO died at world=16
with `NRT_INVALID ... invalid send/recv targets`, capping us to 8 of 64 cores.

Synthetic data on purpose: this measures collectives + throughput, not data.
"""
import os
import time

import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend  # noqa: F401

import sys
sys.path.insert(0, "/home/ubuntu/XScript-Pretraining/src")
from xscript.model import Block, ModelConfig, Transformer  # noqa: E402

from neuronx_distributed.trainer import (  # noqa: E402
    initialize_parallel_model,
    initialize_parallel_optimizer,
    neuronx_distributed_config,
)

MB = int(os.environ.get("NXD_MB", "2"))
STEPS = int(os.environ.get("NXD_STEPS", "12"))
TP = int(os.environ.get("NXD_TP", "1"))

# The real experiment config (configs/base_main.yaml)
MCFG = dict(vocab_size=65536, dim=2048, n_layers=16, n_heads=16, n_kv_heads=16,
            ffn_dim=5632, max_seq_len=2048, rope_theta=10000.0, norm_eps=1e-5)

# NXD_TINY: same code path, ~3M params -- isolates "do world>8 ZeRO collectives
# work" from "does the 1B model fit in 24GB", which are separate questions.
if os.environ.get("NXD_TINY") == "1":
    MCFG.update(vocab_size=4096, dim=256, n_layers=2, n_heads=4, n_kv_heads=4,
                ffn_dim=704, max_seq_len=512)


def model_fn():
    return Transformer(ModelConfig(**MCFG))


def main():
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    import torch_xla.runtime as xr
    rank, world = xr.global_ordinal(), xr.world_size()
    if rank == 0:
        print(f"[nxd] world={world} TP={TP} micro_batch={MB} seq={MCFG['max_seq_len']}", flush=True)

    nxd_config = neuronx_distributed_config(
        tensor_parallel_size=TP,
        optimizer_config={"zero_one_enabled": True, "grad_clipping": True, "max_grad_norm": 1.0},
        # checkpoint each transformer Block (our model is not HF/nxdt, so "full" is unavailable)
        activation_checkpoint_config=Block if os.environ.get("NXD_ACT_CKPT", "1") == "1" else None,
        sequence_parallel=False,
    )

    model = initialize_parallel_model(nxd_config, model_fn)
    optimizer = initialize_parallel_optimizer(
        nxd_config, torch.optim.AdamW, model.parameters(),
        lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8,
    )
    device = xm.xla_device()
    seq = MCFG["max_seq_len"]
    tok_per_step = MB * seq * world

    t0 = None
    for step in range(1, STEPS + 1):
        x = torch.randint(0, MCFG["vocab_size"], (MB, seq), dtype=torch.int64).to(device)
        y = torch.randint(0, MCFG["vocab_size"], (MB, seq), dtype=torch.int64).to(device)
        optimizer.zero_grad()
        with torch.autocast("xla", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        xm.mark_step()
        if step == 2:                      # start timing after compile/warmup
            xm.wait_device_ops(); t0 = time.time(); n0 = step
        if step % 2 == 0 and rank == 0:
            print(f"[nxd] step {step}", flush=True)
    xm.wait_device_ops()
    if rank == 0 and t0:
        dt = time.time() - t0
        steps = STEPS - n0
        print(f"[nxd] RESULT world={world} TP={TP} mb={MB} "
              f"{steps} steps in {dt:.1f}s -> {steps*tok_per_step/dt:,.0f} tok/s "
              f"({steps*tok_per_step/dt/world:,.0f} tok/s/core)", flush=True)
    print("NXD_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
