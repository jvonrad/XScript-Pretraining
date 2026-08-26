"""AWS nkilib fused cross-entropy (online logsumexp over bf16 logits, in-place
gradient) wrapped as an autograd Function usable inside torch.compile.

Forward: per-position loss + lse from bf16 logits [N, V] (no fp32 upcast, no
log_softmax materialization). Backward: grad_logits = (softmax - onehot) *
grad_scale written IN PLACE over the logits buffer (bf16). Value: mean CE.
Numerics: online-LSE reordering, bf16 internal accumulation option -> NOT
bitwise vs F.cross_entropy; dtype=fp32 keeps reductions in fp32.
"""
import torch
import nki.language as nl
from nkilib.experimental.loss.cross_entropy import cross_entropy_forward, cross_entropy_backward
from torch_neuronx import nki_op, wrap_nki

_fwd = wrap_nki(cross_entropy_forward)
_bwd = wrap_nki(cross_entropy_backward)

PPB = 128          # positions per batch (128 recommended for throughput)
CHUNK = 16384      # fp32 limit: chunk*4B*2 <= 229,376 per partition (V=65536 -> 4 chunks)


@nki_op("xscript::nkice_fwd", mutates_args={})
def nkice_fwd(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _fwd(logits, targets, positions_per_batch=PPB, chunk_size=CHUNK, dtype=nl.float32)


@nki_op("xscript::nkice_bwd", mutates_args={})
def nkice_bwd(logits: torch.Tensor, targets: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    return _bwd(logits, targets, lse, reduction="mean", positions_per_batch=PPB,
                chunk_size=CHUNK, dtype=nl.float32, inplace=False)


class _NKICE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        loss, lse = nkice_fwd(logits, targets)
        ctx.save_for_backward(logits, targets, lse)
        return loss.float().mean()

    @staticmethod
    def backward(ctx, g):
        logits, targets, lse = ctx.saved_tensors
        grad = nkice_bwd(logits, targets, lse)           # already scaled by 1/N (mean)
        return (grad * g).to(logits.dtype), None


def nki_cross_entropy_mean(logits_bf16, targets_i32):
    """logits [N, V] bf16 (matmul output), targets [N] int32 -> scalar mean CE."""
    return _NKICE.apply(logits_bf16, targets_i32)


if __name__ == "__main__":
    import os, sys, time, torch_neuronx
    import torch.nn.functional as F
    dev = torch.device("neuron")
    N, D, V = 2048, 2048, 65536
    torch.manual_seed(0)
    w = torch.randn(V, D, device=dev, requires_grad=True)
    hx = torch.randn(N, D, device=dev, requires_grad=True)
    tgt = torch.randint(0, V, (N,), dtype=torch.int32, device=dev)

    # correctness vs F.cross_entropy on the same bf16 logits
    with torch.autocast("neuron", dtype=torch.bfloat16):
        lg = hx @ w.t()
    ref = F.cross_entropy(lg.float(), tgt.long())
    ref.backward(); g_ref = hx.grad.clone(); hx.grad = None; w.grad = None
    with torch.autocast("neuron", dtype=torch.bfloat16):
        lg2 = hx @ w.t()
    new = nki_cross_entropy_mean(lg2, tgt)
    new.backward(); g_new = hx.grad.clone(); hx.grad = None; w.grad = None
    print(f"loss ref {float(ref):.6f} nki {float(new):.6f} | dX max rel diff "
          f"{((g_ref-g_new).abs().max()/g_ref.abs().max()).item():.3e}", flush=True)

    def sync(): torch_neuronx.synchronize()
    def timeit(fn, iters=10):
        for _ in range(3): fn()
        sync(); t0 = time.time()
        for _ in range(iters): fn()
        sync(); return (time.time()-t0)/iters*1e3
    def head_lse(a, W, t):
        lgx = (a @ W.t()).float()
        return (torch.logsumexp(lgx, -1) - lgx.gather(1, t.long().unsqueeze(1)).squeeze(1)).mean()
    def head_nki(a, W, t):
        return nki_cross_entropy_mean(a @ W.t(), t)
    for name, fn in (("lse-gather head (current)", head_lse), ("nkilib CE head", head_nki)):
        c = torch.compile(fn, backend="neuron", dynamic=False)
        def run():
            with torch.autocast("neuron", dtype=torch.bfloat16):
                loss = c(hx, w, tgt)
            loss.backward(); hx.grad = None; w.grad = None
        try:
            print(f"{name:32s} {timeit(run):7.2f} ms fwd+bwd (compiled)", flush=True)
        except Exception as e:
            print(f"{name}: FAILED {repr(e)[:300]}", flush=True)
    print("done", flush=True)
