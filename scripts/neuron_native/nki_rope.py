"""NKI kernel for interleaved (GPT-NeoX) rope, fwd + bwd.

Reference (model.py _apply_rope):
  o1 = x1*cos - x2*sin ; o2 = x1*sin + x2*cos   (x1=x[..., ::2], x2=x[..., 1::2])
Backward is the same rotation with sin negated:
  g1 = go1*cos + go2*sin ; g2 = -go1*sin + go2*cos
so ONE kernel serves both, called with sin or -sin.

Layout: x viewed as (R, D) with R = B*H*T, D = head_dim (128); T % 128 == 0,
tile = 128 rows of one head, so cos/sin tiles depend only on t-block index.
"""
import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch_neuronx import nki_op, wrap_nki

TILE = 128


@wrap_nki
@nki.jit
def _rope_rot_kernel(x, cos, sin):
    # x: (R, hd2, 2) — pairs on the last axis; cos/sin: (T, hd2); R = n_bh * T
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    R = x.shape[0]
    hd2 = x.shape[1]
    T = cos.shape[0]
    n_bh = R // T
    n_t = T // TILE
    for j in nl.affine_range(n_t):          # cos/sin tile reused across all heads
        c = nl.load(cos[nl.ds(j * TILE, TILE), :], dtype=nl.float32)
        s = nl.load(sin[nl.ds(j * TILE, TILE), :], dtype=nl.float32)
        for bh in nl.affine_range(n_bh):
            base = bh * T + j * TILE
            # ONE contiguous 128x(hd2*2) load; even/odd sliced in SBUF where
            # the compute engines read strided patterns natively (a per-pair
            # strided HBM load would issue 256B DMA bursts and be DMA-bound).
            xt = nl.load(x[nl.ds(base, TILE), :, :], dtype=nl.float32)
            rt = nl.ndarray((TILE, hd2, 2), dtype=x.dtype, buffer=nl.sbuf)
            t1 = nl.multiply(xt[:, :, 0], c)
            t2 = nl.multiply(xt[:, :, 1], s)
            nisa.tensor_tensor(dst=rt[:, :, 0], data1=t1, data2=t2, op=nl.subtract)
            t3 = nl.multiply(xt[:, :, 0], s)
            t4 = nl.multiply(xt[:, :, 1], c)
            nisa.tensor_tensor(dst=rt[:, :, 1], data1=t3, data2=t4, op=nl.add)
            nl.store(out[nl.ds(base, TILE), :, :], value=rt)
    return out


@nki_op("xscript::rope_rot", mutates_args={})
def rope_rot(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return _rope_rot_kernel(x, cos, sin)


class _NKIRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, neg_sin):
        B, H, T, D = x.shape
        ctx.save_for_backward(cos, neg_sin)
        ctx.shape = (B, H, T, D)
        out = rope_rot(x.reshape(B * H * T, D // 2, 2), cos, sin)
        return out.view(B, H, T, D)

    @staticmethod
    def backward(ctx, g):
        cos, neg_sin = ctx.saved_tensors
        B, H, T, D = ctx.shape
        gx = rope_rot(g.contiguous().reshape(B * H * T, D // 2, 2), cos, neg_sin)
        return gx.view(B, H, T, D), None, None, None


class RopeNKI:
    """Caches fp32 (cos, sin, -sin) per (T, device) and applies the kernel."""
    def __init__(self):
        self._cache = {}

    def __call__(self, x, cos, sin):
        key = (cos.shape[0], cos.dtype)
        if key not in self._cache:
            c = cos.contiguous().float()
            s = sin.contiguous().float()
            self._cache[key] = (c, s, (-s).contiguous())
        c, s, ns = self._cache[key]
        return _NKIRope.apply(x, c, s, ns)


apply_rope_nki = RopeNKI()


if __name__ == "__main__":
    import os, sys, time
    import torch_neuronx
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
    from xscript.model import _apply_rope, _rope_cache

    dev = torch.device("neuron")
    T, H, HD = 2048, 16, 128
    cos, sin = _rope_cache(T, HD, 10000.0, dev, torch.float32)

    x = torch.randn(1, H, T, HD, device=dev)
    ref = _apply_rope(x, cos, sin).cpu()
    new = apply_rope_nki(x.requires_grad_(False), cos, sin).cpu()
    print("fwd max diff:", (ref - new).abs().max().item(), flush=True)

    # backward check vs autograd of reference
    xg = torch.randn(1, H, T, HD, device=dev, requires_grad=True)
    go = torch.randn(1, H, T, HD, device=dev)
    _apply_rope(xg, cos, sin).backward(go)
    g_ref = xg.grad.detach().cpu(); xg.grad = None
    apply_rope_nki(xg, cos, sin).backward(go)
    g_new = xg.grad.detach().cpu()
    print("bwd max diff:", (g_ref - g_new).abs().max().item(), flush=True)

    # bf16 path (as under autocast)
    xb = torch.randn(1, H, T, HD, device=dev, dtype=torch.bfloat16)
    refb = _apply_rope(xb, cos, sin).float().cpu()
    newb = apply_rope_nki(xb, cos, sin).float().cpu()
    print("bf16 fwd max diff:", (refb - newb).abs().max().item(), flush=True)

    def sync(): torch_neuronx.synchronize()
    q = torch.randn(1, H, T, HD, device=dev, requires_grad=True)
    def run():
        apply_rope_nki(q, cos, sin).float().sum().backward()
        q.grad = None
    for _ in range(3): run()
    sync(); t0 = time.time()
    for _ in range(10): run()
    sync()
    print(f"NKI rope fwd+bwd (eager): {(time.time()-t0)/10*1e3:.2f} ms (view-based compiled: 1.62)", flush=True)

    # inside torch.compile
    c = torch.compile(lambda a: apply_rope_nki(a, cos, sin), backend="neuron", dynamic=False)
    def runc():
        c(q).float().sum().backward()
        q.grad = None
    try:
        for _ in range(3): runc()
        sync(); t0 = time.time()
        for _ in range(10): runc()
        sync()
        print(f"NKI rope fwd+bwd (compiled): {(time.time()-t0)/10*1e3:.2f} ms", flush=True)
    except Exception as e:
        print("compiled path FAILED:", repr(e)[:300], flush=True)
    print("done", flush=True)
