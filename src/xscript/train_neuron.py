"""Pretraining loop for AWS Trainium (Neuron / torch-xla).

This is a **Neuron/XLA port of `train.py`** and is deliberately kept as a
separate module so the CUDA path (`xscript.train`) stays byte-for-byte
unchanged and usable as-is on the GH200/Nvidia cluster. Everything about the
*experiment* is identical: same `model.py`, same deterministic `loader.py`,
same `schedule.py`, same run configs from `runmatrix.py`, same checkpoint
layout and cadence. Only the execution mechanics differ, because Trainium is
an XLA device, not CUDA:

  * device            -> `xm.xla_device()` (PJRT_DEVICE=NEURON), not `cuda:N`
  * data parallelism  -> XLA replicas + `xm.reduce_gradients` all-reduce,
                         NOT torch `DistributedDataParallel` (there is no NCCL).
                         Requires `dist.init_process_group('xla', init_method=
                         'xla://')` early in __init__ -- without it,
                         `torch_xla.runtime.world_size()`/`global_ordinal()`
                         silently report 1/0 on EVERY rank even under a
                         correct multi-process torchrun launch (confirmed by
                         testing directly against the installed torch_xla
                         2.9 + torch_neuronx build: the PJRT env vars
                         (NEURON_PJRT_WORLD_SIZE etc.) get set correctly by
                         torch_neuronx's torchrun-detection at import time,
                         but `_xla_get_replication_devices_count()` -- what
                         `world_size()` actually reads -- stays 0 until the
                         process group is explicitly initialized). Getting
                         this wrong doesn't crash loudly in every case: only
                         non-zero ranks see it (rank 0 silently processes the
                         FULL global batch alone; every other rank's
                         batch-selection filter divides by the wrong world
                         size and comes up empty), so a single-process smoke
                         test cannot catch it -- validate multi-process
                         (`--nproc_per_node>1`) specifically before trusting
                         a real run.
  * step boundary     -> an explicit `xm.mark_step()` after EVERY micro-batch
                         (not just once per optimizer step -- fusing all
                         `grad_accum` micro-batches into a single graph blew
                         the 24GB HBM budget at real model scale, since the
                         compiler then has to keep every micro-batch's
                         activations live at once; confirmed on real
                         hardware: NCC_EVRF009, 89GB needed). Shapes are
                         fixed (constant `micro_batch_size` x `seq_len`), so
                         despite the extra mark_step()s this is still exactly
                         TWO compiled graphs total (one micro-batch fwd+bwd
                         shape, one optimizer-step shape), each cached and
                         reused for the rest of the run.
  * mixed precision   -> `torch.autocast("xla", bf16)` instead of the CUDA
                         autocast; behaviourally the same bf16 compute.
  * activation memory -> per-block activation checkpointing (`_checkpointed_
                         forward`, below), reimplementing `Transformer.
                         forward`'s loop over `model.layers` with
                         `torch.utils.checkpoint.checkpoint` around each
                         Block instead of calling `model.forward()` directly.
                         Confirmed on real hardware that the full model
                         (16 layers, dim=2048, ffn=5632, seq_len=2048) does
                         NOT fit in a NeuronCore's 24GB HBM without this --
                         even at micro_batch_size=1 it needed ~24GB just for
                         one micro-batch's fwd+bwd (dominated by ~13GB of
                         scratch/spill), confirmed structural rather than a
                         compiler-flag artifact (byte-identical across
                         `--optlevel=1` vs `2`, and across AdamW's `foreach`
                         True/False -- XLA traces to the same fused graph
                         either way, so `foreach` only matters for eager
                         execution). Recomputing each block's activations
                         during backward instead of storing all 16 layers'
                         at once keeps this to one (or a few) layers' worth
                         of live activations, at the cost of one extra
                         forward pass per layer during backward (~20-30%
                         slower steps) -- preserves the exact same model
                         config (seq_len=2048 etc.) as the 15 already-trained
                         CUDA models, so this is a training-mechanics
                         difference only, not an experimental-design one.
                         `model.py` itself is NOT modified -- this
                         reimplements its forward loop here instead.
  * loss/metric fetch -> deferred to log steps via `xm.add_step_closure`, so we
                         never stall the async XLA pipeline with a per-step
                         `.item()` (the #1 way to make XLA training crawl).
  * checkpoint save   -> `xm.save` (moves tensors host-side, writes on the
                         master ordinal only, rendezvous-safe).
  * weight sync       -> none needed: every replica seeds identically and, on
                         resume, every replica loads the same checkpoint file
                         from shared disk. The loader is already world-size-
                         independent by construction.

Launch with **`xmp.spawn`, NOT `torchrun`** -- torchrun cannot pin a job to a
subset of cores (torch_neuronx overwrites `NEURON_RT_VISIBLE_CORES` with
`LOCAL_RANK`), and concurrent jobs must each get a distinct `MASTER_PORT` or the
second one hangs forever at process-group init. See NEURON.md 9 for the full
recipe and `/home/ubuntu/xscript_prod/` for a working launcher. Cooldown
branching, WSD schedule, resume, log-spaced checkpoints and the run-name matrix
all behave exactly as on CUDA.

The one behavioural knob added here: in-loop BPB eval is OFF by default on
Neuron (`train.eval_in_loop: false`) because it feeds variable-length docs to
the model and would trigger endless XLA recompiles mid-training. BPB/BTS are
computed post-hoc from checkpoints with the existing fixed-shape eval path
(`src/xscript/eval/bench.py`, `--device xla`) exactly as documented in
NEURON.md 5. Set `train.eval_in_loop: true` to force it back on.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend  # noqa: F401 -- registers the 'xla' torch.distributed backend

from .model import ModelConfig, Transformer
from .data.loader import MixedStream
from .schedule import lr_at, ckpt_interval, stable_end_tokens, total_tokens
from .paths import run_dir, ensure


def _xla_world():
    """(rank, world) over XLA replicas, resilient across torch-xla versions.

    torch_xla >= 2.5 moved this to `torch_xla.runtime`; `xm.get_ordinal()` /
    `xm.xrt_world_size()` are gone entirely on 2.9 (verified on this box's
    torch_xla 2.9.0). Older builds only have the xm.* names.
    """
    import torch_xla.runtime as xr
    if hasattr(xr, "global_ordinal"):
        return xr.global_ordinal(), xr.world_size()
    return xm.get_ordinal(), xm.xrt_world_size()


def _log(rank, path, rec):
    if rank == 0:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


# torch.utils.checkpoint (both _get_device_module and, via torch.random.
# fork_rng, the RNG-fork path) unconditionally does `getattr(torch, "xla")`
# to find the device's runtime module -- "xla" isn't exposed there by this
# torch_xla build (confirmed on real hardware: AttributeError from
# `_get_device_module`, then `RuntimeError: torch has no module of xla` from
# `fork_rng`, which checks for the module's existence BEFORE it even looks at
# `enabled=preserve_rng_state`). `torch._register_device_module` is the
# public, intended mechanism for exactly this (any accelerator backend not
# built into core torch). We pass `preserve_rng_state=False` below (our
# model has no dropout/randomness inside a checkpointed Block, so there's no
# RNG state to save/restore), which makes `fork_rng` short-circuit as soon as
# it confirms the module exists -- so the registered module's *identity*
# (any non-None object) is all that's actually needed, not its interface.
import torch_xla  # noqa: E402
torch._register_device_module("xla", torch_xla)


class _ChunkedLMHeadCE(torch.autograd.Function):
    """Memory-efficient lm_head + cross-entropy over a large vocab.

    Computing full logits `(mb*seq, vocab)` then cross_entropy is the DOMINANT
    HBM cost at larger micro_batch on Neuron (observed: 23GB compiler scratchpad
    at mb=8, seq=2048, vocab=65536 -> NCC_EOOM002 29GB) because it materializes
    several `(mb*seq, vocab)`-sized fp32 tensors at once (logits, log_softmax,
    grad_logits). This chunks the FLATTENED token dimension so only `chunk` rows'
    logits exist at a time, and RECOMPUTES each chunk's logits in backward (nested
    autograd) so no full-size tensor is held across the backward pass. Peak logits
    drops from `(mb*seq, vocab)` to `(chunk, vocab)`.

    Exactness: mean-reduced cross-entropy is sum(per-token loss)/n_valid, and the
    token sum is associative, so chunking gives the SAME loss and gradients as the
    non-chunked `F.cross_entropy(..., ignore_index=-100)` (mean). Uses
    F.cross_entropy per chunk -- the same op the non-chunked path uses, already
    verified correct on this Neuron build -- so it introduces no new gather/one_hot
    Neuron gotchas (NEURON.md 4). `chunk` divides `mb*seq` evenly and both are
    static, so the loop unrolls to a fixed XLA graph.
    """

    @staticmethod
    def forward(ctx, x, weight, targets, chunk):
        # x: (N, dim) hidden states; weight: (vocab, dim); targets: (N,) w/ -100.
        ctx.save_for_backward(x, weight, targets)
        ctx.chunk = chunk
        # Count valid (non-ignored) targets IN FP32. A bool `.sum()` silently
        # mis-lowers on this Neuron build (returns -1 instead of the count),
        # which negates and un-normalizes the loss (verified: n_valid=-1 ->
        # loss=-sum). Casting to float32 BEFORE the reduction avoids the bad
        # bool-reduction lowering. (CPU never sees this; another NEURON.md-4-class
        # silent Neuron numeric bug.)
        n_valid = (targets != -100).to(torch.float32).sum().clamp(min=1.0)
        ctx.n_valid = n_valid
        N = x.shape[0]
        total = x.new_zeros((), dtype=torch.float32)
        for i in range(0, N, chunk):
            logits_c = (x[i:i + chunk] @ weight.t()).float()
            total = total + F.cross_entropy(
                logits_c, targets[i:i + chunk], ignore_index=-100, reduction="sum")
        return total / n_valid

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, targets = ctx.saved_tensors
        chunk, n_valid = ctx.chunk, ctx.n_valid
        N = x.shape[0]
        # Recompute each chunk's logits with grad enabled and reuse
        # F.cross_entropy's (Neuron-correct) backward via nested autograd.
        grad_x_parts = []
        grad_w = torch.zeros_like(weight, dtype=torch.float32)
        # backward() runs with grad disabled by default; re-enable so the
        # per-chunk recompute builds a graph for the nested autograd.grad.
        with torch.enable_grad():
            for i in range(0, N, chunk):
                xc = x[i:i + chunk].detach().requires_grad_(True)
                wc = weight.detach().requires_grad_(True)
                logits_c = (xc @ wc.t()).float()
                loss_c = F.cross_entropy(
                    logits_c, targets[i:i + chunk], ignore_index=-100,
                    reduction="sum") / n_valid
                gx, gw = torch.autograd.grad(loss_c, (xc, wc))
                grad_x_parts.append(gx)
                grad_w = grad_w + gw.float()
        # Build grad_x by cat (never per-row in-place scatter on an XLA tensor --
        # that trips NRT_EXEC_OOB, NEURON.md 4).
        grad_x = torch.cat(grad_x_parts, dim=0)
        return (grad_x * grad_out).to(x.dtype), (grad_w * grad_out).to(weight.dtype), None, None


def _checkpointed_forward(model: Transformer, idx: torch.Tensor, targets: torch.Tensor,
                          ce_chunk: int = 0):
    """Reimplements `Transformer.forward`'s loop over `model.layers`, with
    each Block wrapped in `torch.utils.checkpoint.checkpoint` so its
    activations are recomputed during backward instead of held live for all
    16 layers simultaneously. See the module docstring's "activation memory"
    bullet for why this is necessary on real hardware. Does not touch
    `model.py` -- calls straight through to the same `model.tok_emb`,
    `model._rope_for`, `model.layers`, `model.norm`, `model.lm_head`.
    """
    B, T = idx.shape
    x = model.tok_emb(idx)
    cos, sin = model._rope_for(T, idx.device, x.dtype)
    for layer in model.layers:
        # preserve_rng_state=False: no dropout/randomness inside a Block, so
        # there's no RNG state that needs saving/restoring across recompute.
        x = ckpt.checkpoint(layer, x, cos, sin, use_reentrant=False,
                           preserve_rng_state=False)
    x = model.norm(x)
    tflat = targets.reshape(-1)
    if ce_chunk and ce_chunk > 0:
        # Memory-efficient path: never materialize the full (N, vocab) logits.
        # Returns no logits (training only needs the loss).
        loss = _ChunkedLMHeadCE.apply(
            x.reshape(-1, x.size(-1)), model.lm_head.weight, tflat, ce_chunk)
        return None, loss
    logits = model.lm_head(x)
    # Loss in fp32 regardless of param/compute dtype: cross_entropy's softmax is
    # precision-sensitive, and the CUDA trainer computes it in fp32 too (autocast
    # keeps cross_entropy on its fp32 list). `.float()` is a no-op when logits are
    # already fp32, and upcasts under the bf16-params path.
    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                           tflat, ignore_index=-100)
    return logits, loss


class NeuronTrainer:
    """XLA/Neuron twin of `xscript.train.Trainer`. Same config schema."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        # Must run before _xla_world()/xm.reduce_gradients: without an
        # explicit XLA process group, torch_xla's replication device count
        # stays 0 and world_size()/global_ordinal() silently report 1/0 on
        # every rank even under a correct multi-process torchrun launch. See
        # the data-parallelism bullet in this module's docstring.
        if not dist.is_initialized():
            dist.init_process_group("xla", init_method="xla://")
        self.rank, self.world = _xla_world()
        self.device = xm.xla_device()
        # Seed identically on every replica so fresh-init weights match without
        # a DDP broadcast (there is no DDP here).
        torch.manual_seed(cfg.get("seed", 0))
        np.random.seed(cfg.get("seed", 0))

        mc = cfg["model"]
        self.mcfg = ModelConfig(**mc)
        self.seq_len = self.mcfg.max_seq_len
        self.raw_model = Transformer(self.mcfg).to(self.device)
        # ---- optional bf16 forward params (memory -> bigger micro_batch) ------
        # Storing the model's forward params in bf16 halves the fwd/bwd graph's
        # weight+grad HBM (~10GB fp32 -> ~4GB bf16), which is the binding limit on
        # micro_batch_size (see [[neuron-oom-optimizer-state-not-activations]]).
        # This is bf16-compute / FP32-MASTER mixed precision, NOT full bf16: ZeRO
        # keeps the master weights + Adam state in fp32 (optimizer_dtype defaults
        # to float32) and casts back to the bf16 model params after each step, so
        # the optimizer math matches the 15 fp32-master CUDA models. Only real
        # deviations vs the autocast path: grads are accumulated in bf16, and RoPE
        # cos/sin are bf16 (position encoding slightly lower precision). Requires
        # ZeRO for the fp32 master -- with plain AdamW this would be full bf16.
        self.bf16_params = bool(cfg["train"].get("bf16_params", False))
        if self.bf16_params:
            self.raw_model = self.raw_model.to(torch.bfloat16)
        # Chunk size (in tokens) for the memory-efficient lm_head+cross_entropy;
        # 0 disables (full-logits path). See _ChunkedLMHeadCE. Must divide
        # micro_batch*seq_len evenly.
        self.ce_chunk = int(cfg["train"].get("fused_ce_chunk", 0))

        # ---- optional warm-start from a foreign model-only checkpoint ---------
        # Resume the zh models from their 12B partial checkpoints (which hold
        # model weights ONLY -- no optimizer state). Loaded HERE, before the
        # optimizer is built, so ZeRO's fp32 master is sharded from the LOADED
        # weights, not the random init. Only applies on the very first launch
        # (no local last.pt yet); later restarts resume from last.pt with the
        # real optimizer state via maybe_resume(). The optimizer restarts fresh
        # (Adam moments = 0), which is fine mid-stable-phase (lr is constant,
        # no warmup needed). Sets the token/step counter so the WSD schedule
        # continues from where the checkpoint left off.
        self._warm_tokens = 0
        self._warm_step = 0
        ws = cfg.get("warm_start")
        if ws and not (run_dir(cfg["name"]) / "checkpoints" / "last.pt").exists():
            ck = torch.load(ws["from"], map_location="cpu", weights_only=False)
            self.raw_model.load_state_dict(ck["model"])
            xm.mark_step()  # push loaded weights onto the device
            self._warm_tokens = int(ck.get("tokens", ws.get("at_tokens", 0)))
            self._warm_step = int(ck.get("step", 0))
            if self.rank == 0:
                print(f"[train-neuron] warm-start from {ws['from']} @ "
                      f"{self._warm_tokens/1e9:.3f}B tokens (step {self._warm_step})")

        # No torch.compile / DDP wrapper: XLA *is* the compiler, and data
        # parallelism is handled by xm.reduce_gradients below.
        self.model = self.raw_model
        if self.rank == 0:
            print(f"[train-neuron] params: "
                  f"{self.raw_model.num_params(False)/1e6:.1f}M non-embedding, "
                  f"{self.raw_model.num_params(True)/1e6:.1f}M total")

        opt = cfg.get("optim", {})
        adamw_kwargs = dict(
            lr=cfg["schedule"]["peak_lr"],
            betas=tuple(opt.get("betas", (0.9, 0.95))),
            weight_decay=opt.get("weight_decay", 0.1),
            eps=opt.get("eps", 1e-8),
        )
        self.grad_clip = opt.get("grad_clip", 1.0)
        # ---- optimizer: ZeRO-1 (sharded optimizer state) ----------------------
        # A 1.09B-param model with a full fp32 AdamW state does NOT fit a
        # NeuronCore's 24GB HBM: measured from Neuron's own allocation ledger,
        # the persistent "tensor" bucket (fp32 master weights + fp32 grads +
        # AdamW m + AdamW v + bf16 autocast copies) is ~19.5GB, and the
        # compiler's shared scratchpad another ~8.25GB -> ~28GB peak. That
        # 19.5GB is entirely BATCH-INDEPENDENT (activations, the only
        # batch-scaling bucket, are just ~0.08GB once checkpointed) -- which is
        # exactly why shrinking micro_batch_size from 8 down to 1 barely moved
        # the number and never fit. The real lever is the fp32 optimizer state
        # (16 bytes/param = 17.4GB), so we shard it across data-parallel
        # replicas with ZeRO-1: m/v and the fp32 master weights are split
        # 1/world per rank, cutting the fixed tensor cost enough to fit (world=4
        # -> ~13.2GB tensor + 8.25 scratch ~= 21.5GB peak). ZeRO-1 is
        # numerically identical to plain AdamW (same update math, just sharded
        # storage), so it preserves exact comparability with the 15 CUDA models,
        # unlike bf16 optimizer states. Bonus: real data parallelism raises
        # aggregate tok/s per model. `grad_clipping=True/max_norm` and the grad
        # reduce-scatter are done INSIDE ZeRO.step() -- see the train loop, which
        # drops the manual xm.reduce_gradients + clip_grad_norm_ accordingly.
        # save_master_weights=True is required for correct resume (the fp32
        # master shards must be persisted, not just the bf16 model weights).
        self.use_zero = bool(cfg["train"].get("zero", self.world > 1))
        if self.use_zero:
            from torch_xla.distributed.zero_redundancy_optimizer import (
                ZeroRedundancyOptimizer)
            self.optim = ZeroRedundancyOptimizer(
                self.raw_model.parameters(),
                optimizer_class=torch.optim.AdamW,
                grad_clipping=True,
                max_norm=self.grad_clip,
                save_master_weights=True,
                **adamw_kwargs,
            )
        else:
            self.optim = torch.optim.AdamW(self.raw_model.parameters(), **adamw_kwargs)

        # ---- global batch bookkeeping (identical math to the CUDA trainer) ----
        self.micro_bsz = cfg["train"]["micro_batch_size"]
        gbt = cfg["train"]["global_batch_tokens"]
        per_step_windows = max(1, round(gbt / self.seq_len))
        unit = self.micro_bsz * self.world
        self.global_windows = max(unit, (per_step_windows // unit) * unit)
        self.grad_accum = self.global_windows // unit
        self.tokens_per_step = self.global_windows * self.seq_len
        if self.rank == 0:
            print(f"[train-neuron] global batch: {self.global_windows} windows "
                  f"({self.tokens_per_step/1e6:.3f}M tokens), "
                  f"grad_accum={self.grad_accum}, replicas={self.world}")

        # ---- schedule (branch collapses warmup/stable) ----
        self.sched = dict(cfg["schedule"])
        self.branch = cfg.get("branch")
        if self.branch:
            self.sched["warmup_tokens"] = 0
            self.sched["stable_tokens"] = 0
        self.target_tokens = total_tokens(self.sched)
        self.ckpt_table = cfg["train"].get("ckpt_schedule",
                                           [[2e9, 250e6], [5e9, 500e6],
                                            [15e9, 1e9], [1e15, 2e9]])
        self.stable_marks = sorted(cfg["train"].get("stable_marks", []))
        self.marks_done = set()

        # ---- data mixer (world-size independent, deterministic) ----
        self.mixer = MixedStream(cfg["langs"], cfg["tok_name"], self.seq_len,
                                 seed=cfg.get("data_seed", 1234),
                                 probs=cfg.get("probs"))

        self.rdir = ensure(run_dir(cfg["name"]))
        self.log_path = self.rdir / "train.jsonl"
        self.tokens = self._warm_tokens
        self.step = self._warm_step
        self.last_ckpt_tokens = self._warm_tokens
        self.saved_stable = self._warm_tokens >= stable_end_tokens(self.sched)
        self.eval_in_loop = bool(cfg["train"].get("eval_in_loop", False))

        self.wandb = None
        if self.rank == 0:
            try:
                import wandb
                self.wandb = wandb.init(
                    project="XScript-Pretraining", name=cfg["name"],
                    id=cfg.get("wandb_id", cfg["name"]),
                    resume="allow", config=cfg,
                )
                self.wandb.summary["params_total_M"] = self.raw_model.num_params(True) / 1e6
                self.wandb.summary["params_non_embed_M"] = self.raw_model.num_params(False) / 1e6
            except Exception as exc:
                print(f"[train-neuron] wandb disabled ({exc})")

    # ---- checkpoint io ----
    def _ckpt_path(self, tag):
        return ensure(self.rdir / "checkpoints") / f"{tag}.pt"

    def _optim_shard_path(self, tag, rank):
        return ensure(self.rdir / "checkpoints") / f"{tag}.optim.rank{rank}.pt"

    def save(self, tag, resumable=True):
        """Every replica must call this: xm.save rendezvouses and writes once.

        The MODEL payload (full, allgathered fp32 weights on every rank) is
        byte-compatible with the CUDA trainer's checkpoints (same keys), so a
        Neuron-trained checkpoint reassembles/uploads and evaluates through the
        exact same tooling. Under ZeRO the OPTIMIZER state is sharded per rank,
        and xm.save writes only the master ordinal -- so each rank persists its
        own optimizer shard to a rank-tagged sidecar file instead (plain AdamW
        keeps the single-file layout). model-only saves (resumable=False) carry
        no optimizer state, so they stay identical either way.
        """
        payload = {
            "model": self.raw_model.state_dict(),
            "step": self.step, "tokens": self.tokens,
            "cfg": self.cfg,
        }
        if resumable:
            payload.update({
                "mixer": self.mixer.state_dict(),
                "last_ckpt_tokens": self.last_ckpt_tokens,
                "saved_stable": self.saved_stable,
                "torch_rng": torch.get_rng_state(),
                "world": self.world,
            })
            if not self.use_zero:
                payload["optim"] = self.optim.state_dict()
        # xm.save moves XLA tensors to CPU and writes only on ordinal 0.
        xm.save(payload, str(self._ckpt_path(tag)))
        if resumable and self.use_zero:
            # ZeRO shard: every rank writes its own slice (host-side), so no
            # optimizer state is silently dropped on resume. Rendezvous first so
            # all ranks have finished the xm.save above before sidecar writes.
            xm.rendezvous(f"save_optim_{tag}")
            shard = xm._maybe_convert_to_cpu(self.optim.state_dict())
            torch.save(shard, str(self._optim_shard_path(tag, self.rank)))
        if self.rank == 0:
            kind = "full" if resumable else "model-only"
            print(f"[train-neuron] saved {tag} ({kind}) @ {self.tokens/1e9:.3f}B tokens")

    def maybe_resume(self):
        """Load on EVERY replica (shared disk) so weights match with no broadcast."""
        last = self._ckpt_path("last")
        if self.branch and not last.exists():
            ck = torch.load(self.branch["from"], map_location="cpu", weights_only=False)
            self.raw_model.load_state_dict(ck["model"])
            # A branch source is a full (unsharded) checkpoint from the trunk;
            # its optimizer state is not in ZeRO's sharded format. Cooldown
            # branches normally reset the optimizer anyway, so under ZeRO we skip
            # optim-load rather than mis-load a full state into sharded slots.
            if self.branch.get("load_optim", True) and not self.use_zero:
                self.optim.load_state_dict(ck["optim"])
            xm.mark_step()  # push loaded weights onto the device
            if self.rank == 0:
                print(f"[train-neuron] branched from {self.branch['from']} "
                      f"@ {ck['tokens']/1e9:.3f}B (cooldown {self.target_tokens/1e9:.1f}B)")
            return
        if last.exists():
            ck = torch.load(last, map_location="cpu", weights_only=False)
            self.raw_model.load_state_dict(ck["model"])
            if self.use_zero:
                # Each rank reloads its own optimizer shard. Requires the same
                # world size as the saving run (shards are world-specific).
                saved_world = ck.get("world", self.world)
                if saved_world != self.world:
                    raise RuntimeError(
                        f"ZeRO resume needs the same world size: checkpoint was "
                        f"saved with world={saved_world}, this run has "
                        f"world={self.world}. Relaunch with the original core count.")
                shard_path = self._optim_shard_path("last", self.rank)
                self.optim.load_state_dict(
                    torch.load(str(shard_path), map_location="cpu", weights_only=False))
            else:
                self.optim.load_state_dict(ck["optim"])
            self.mixer.load_state_dict(ck["mixer"])
            self.step = ck["step"]; self.tokens = ck["tokens"]
            self.last_ckpt_tokens = ck["last_ckpt_tokens"]
            self.saved_stable = ck.get("saved_stable", False)
            self.marks_done = {m for m in self.stable_marks if m <= self.tokens}
            torch.set_rng_state(ck["torch_rng"])
            xm.mark_step()
            if self.rank == 0:
                print(f"[train-neuron] resumed @ step {self.step}, "
                      f"{self.tokens/1e9:.3f}B tokens")

    # ---- data ----
    def _next_micro_batches(self):
        """grad_accum micro-batches of (x, y) on device for this replica.

        Build the whole rank batch on the host and move to device ONCE (a
        Neuron requirement: per-row in-place scatter on an XLA tensor trips
        NRT_EXEC_OOB -- see NEURON.md §4).
        """
        arr, counts = self.mixer.rank_batch(self.global_windows, self.rank, self.world)
        t = torch.from_numpy(arr.astype(np.int64))
        x = t[:, :-1].to(self.device)
        y = t[:, 1:].to(self.device)
        micros = [(x[i:i + self.micro_bsz], y[i:i + self.micro_bsz])
                  for i in range(0, x.size(0), self.micro_bsz)]
        return micros, counts

    # ---- eval (post-hoc by default on Neuron; see module docstring) ----
    def _eval_sources(self):
        srcs = {}
        for l in self.cfg["langs"]:
            try:
                from .eval.bpb import load_holdout
                h = load_holdout(l, self.cfg["train"].get("eval_docs", 500))
                if h:
                    srcs[f"holdout_{l}"] = h
            except Exception:
                pass
        try:
            from . import flores
            par = flores.load_parallel(list(self.cfg["langs"]), "dev")
            for l, sents in par.items():
                srcs[f"flores_{l}"] = sents
        except Exception as e:
            if self.rank == 0:
                print(f"[train-neuron] flores eval skipped: {e}")
        return srcs

    def evaluate(self):
        if self.rank != 0 or not self.eval_in_loop:
            return {}
        from .eval.bpb import eval_sources
        from .tok.wrapper import Tok
        from .paths import tokenizer_dir
        tok = Tok(tokenizer_dir(self.cfg["tok_name"]))
        res = eval_sources(self.raw_model, tok, self._eval_sources(),
                           self.device, self.seq_len)
        self.model.train()
        return res

    # ---- loop ----
    def train(self):
        self.maybe_resume()
        self.model.train()
        t0 = time.time()
        log_every = self.cfg["train"].get("log_every", 20)
        while self.tokens < self.target_tokens:
            lr = lr_at(self.tokens, self.sched)
            for g in self.optim.param_groups:
                g["lr"] = lr

            micros, counts = self._next_micro_batches()
            self.optim.zero_grad(set_to_none=True)
            # Accumulate the loss ON DEVICE; never .item() inside the step.
            loss_acc = torch.zeros((), device=self.device)
            n = len(micros)
            for x, y in micros:
                with torch.autocast("xla", dtype=torch.bfloat16):
                    _, loss = _checkpointed_forward(self.raw_model, x, y, self.ce_chunk)
                (loss / n).backward()
                loss_acc = loss_acc + loss.detach() / n
                # Execute+free THIS micro-batch's graph before starting the
                # next. Fusing all `grad_accum` micro-batches into one graph
                # (no mark_step until the end) forces the compiler to keep
                # every micro-batch's activations live simultaneously --
                # confirmed on real hardware at real model scale (1.09B
                # params, grad_accum=61): NCC_EVRF009, 89GB needed vs 24GB
                # HBM available. Grad accumulation still works: .backward()
                # adds into the same .grad tensors across calls regardless of
                # intervening mark_step()s, and every micro-batch shares one
                # graph shape, so this is still exactly one compiled graph
                # (cached, reused `grad_accum` times per step), not `n`
                # distinct compiles.
                xm.mark_step()

            # Cross-replica gradient average + clip + parameter update. Under
            # ZeRO, step() ITSELF reduce-scatters the grads and clips (see the
            # optimizer construction above), so we must NOT also call
            # xm.reduce_gradients / clip_grad_norm_ (that would double-reduce).
            # Both paths are one compiled graph, reused every step thereafter.
            if self.use_zero:
                self.optim.step()
            else:
                xm.reduce_gradients(self.optim)
                torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.grad_clip)
                self.optim.step()
            xm.mark_step()

            self.tokens += self.tokens_per_step
            self.step += 1

            if self.step % log_every == 0:
                dt = time.time() - t0
                tps = self.tokens_per_step * log_every / dt if dt > 0 else 0
                # add_step_closure fetches the device scalar without a hard sync,
                # keeping the pipeline asynchronous.
                self._log_async(loss_acc, lr, round(tps), self.mixer.stats())
                t0 = time.time()

            # stable checkpoint exactly once, at the trunk's decay boundary
            if (not self.branch and not self.saved_stable
                    and self.tokens >= stable_end_tokens(self.sched)):
                self.save("stable")
                self.saved_stable = True

            # named branch points for cooldown extensions (100B runs)
            for mark in self.stable_marks:
                if mark not in self.marks_done and self.tokens >= mark:
                    self.save(f"stable_{int(mark/1e6)}M")
                    self.marks_done.add(mark)

            # log-spaced checkpoint (+ optional in-loop eval)
            if self.tokens - self.last_ckpt_tokens >= ckpt_interval(self.tokens, self.ckpt_table):
                self.last_ckpt_tokens = self.tokens
                self.save("last")
                self.save(f"step{self.step}_{int(self.tokens/1e6)}M", resumable=False)
                res = self.evaluate()
                if res:
                    _log(self.rank, self.log_path,
                         {"step": self.step, "tokens": self.tokens, "eval": res})
                    if self.wandb:
                        self.wandb.log({f"eval/{k}_bpb": v["bpb"] for k, v in res.items()} |
                                       {f"eval/{k}_ppl": v["ppl_token"] for k, v in res.items()},
                                       step=self.step)
                    if self.rank == 0:
                        brief = {k: round(v["bpb"], 4) for k, v in res.items()}
                        print(f"[eval] {self.tokens/1e9:.2f}B: {brief}")

        self.save("last")
        self.save("final", resumable=False)
        res = self.evaluate()
        if res:
            _log(self.rank, self.log_path,
                 {"step": self.step, "tokens": self.tokens, "eval_final": res})
            if self.wandb:
                self.wandb.log({f"eval_final/{k}_bpb": v["bpb"] for k, v in res.items()} |
                               {f"eval_final/{k}_ppl": v["ppl_token"] for k, v in res.items()},
                               step=self.step)
        if self.wandb:
            self.wandb.finish()
        if self.rank == 0:
            print(f"[train-neuron] DONE {self.cfg['name']} @ {self.tokens/1e9:.2f}B tokens")

    def _log_async(self, loss_acc, lr, tps, mix):
        step, tokens, log_path, rank = self.step, self.tokens, self.log_path, self.rank
        wandb = self.wandb

        def _closure(loss_val):
            loss_val = float(loss_val)
            rec = {"step": step, "tokens": tokens, "lr": lr,
                   "loss": loss_val, "tok_per_s": tps, "mix": mix}
            _log(rank, log_path, rec)
            if wandb:
                # Never let a wandb hiccup (network blip) crash training.
                try:
                    wandb.log({**rec, "tokens_b": tokens / 1e9}, step=step)
                except Exception:
                    pass
            if rank == 0:
                print(f"[train-neuron] step {step} | {tokens/1e9:.2f}B | "
                      f"loss {loss_val:.4f} | lr {lr:.2e} | {tps/1e3:.0f}k tok/s")

        xm.add_step_closure(_closure, args=(loss_acc,))


def run_from_config(cfg: dict):
    NeuronTrainer(cfg).train()
