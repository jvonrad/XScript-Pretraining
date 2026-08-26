"""Throughput-oriented drop-in replacements for model.py's Attention / SwiGLU /
RMSNorm. All are MATHEMATICALLY identical to the originals (same dot
products, same rotation), NOT bitwise: matmul tiling / accumulation order
differ, and q/k live in a permuted head_dim basis (attention scores are
invariant to a shared permutation of the head_dim axis).

FusedAttention:
  * one [3D, D] qkv projection (real parameter, no per-step cat)
  * wq/wk rows permuted per head to [evens | odds] so rope is rotate-half
    over contiguous halves, applied to q and k in ONE call
  * one .contiguous() for all of q/k/v via a [3,B,H,T,Dh] permute
FusedSwiGLU: one [2*FF, D] projection for w1|w3.
rmsnorm_fn: F.rms_norm in fp32 (same formula as model.py RMSNorm).

`from_original(...)` builders construct fused weights from an original
module so equivalence can be tested; `rope_perm(head_dim)` is the row
permutation (apply to a checkpoint's wq/wk to convert losslessly).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_perm(head_dim):
    """Row permutation within one head: [0,2,4,...,1,3,5,...]."""
    return torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])


def _perm_rows(w, n_heads, head_dim):
    # w: [n_heads*head_dim, in]; permute the rows of every head block
    p = rope_perm(head_dim)
    idx = (torch.arange(n_heads)[:, None] * head_dim + p[None, :]).reshape(-1)
    return w[idx.to(w.device)]


def rope_rotate_half(x, cosf, sinf):
    # x: (N, H, T, Dh) in [evens|odds] basis; cosf/sinf: (T, Dh) = cat([c,c]), cat([s,s])
    h = x.shape[-1] // 2
    rot = torch.cat((-x[..., h:], x[..., :h]), dim=-1)
    return x * cosf[None, None, :, :] + rot * sinf[None, None, :, :]


class FusedAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.kv_heads == cfg.n_heads, "GQA not handled in the fused path"
        self.n_heads, self.head_dim, self.dim = cfg.n_heads, cfg.head_dim, cfg.dim
        self.wqkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self._rope = None

    @classmethod
    def from_original(cls, attn, cfg):
        m = cls(cfg)
        with torch.no_grad():
            wq = _perm_rows(attn.wq.weight, cfg.n_heads, cfg.head_dim)
            wk = _perm_rows(attn.wk.weight, cfg.n_heads, cfg.head_dim)
            m.wqkv.weight.copy_(torch.cat([wq, wk, attn.wv.weight], dim=0))
            m.wo.weight.copy_(attn.wo.weight)
        return m.to(attn.wo.weight.device)

    def _rope_full(self, cos, sin):
        # cache (T, Dh) full-width tables from model.py's (T, Dh/2) ones
        if self._rope is None or self._rope[0].shape[0] != cos.shape[0] or self._rope[0].device != cos.device:
            self._rope = (torch.cat([cos, cos], -1).contiguous(), torch.cat([sin, sin], -1).contiguous())
        return self._rope

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.head_dim
        qkv = self.wqkv(x).view(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4).contiguous()  # [3,B,H,T,Dh]
        cosf, sinf = self._rope_full(cos, sin)
        cosf = cosf.to(qkv.dtype); sinf = sinf.to(qkv.dtype)
        qk = rope_rotate_half(qkv[:2].reshape(2 * B, H, T, Dh), cosf, sinf)
        q, k = qk[:B], qk[B:]
        v = qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class FusedSwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ffn_dim = cfg.ffn_dim
        self.w13 = nn.Linear(cfg.dim, 2 * cfg.ffn_dim, bias=False)
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    @classmethod
    def from_original(cls, ffn, cfg):
        m = cls(cfg)
        with torch.no_grad():
            m.w13.weight.copy_(torch.cat([ffn.w1.weight, ffn.w3.weight], dim=0))
            m.w2.weight.copy_(ffn.w2.weight)
        return m.to(ffn.w2.weight.device)

    def forward(self, x):
        g, u = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(g) * u)


def rmsnorm_fn_forward(self, x):
    dt = x.dtype
    return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(dt)


def fuse_model(model, cfg, attn=True, ffn=True, norm=True):
    """In-place: swap Blocks' submodules for fused versions built from the
    originals (weights converted, so the function computed is unchanged)."""
    import xscript.model as _xm
    for blk in model.layers:
        if attn:
            blk.attn = FusedAttention.from_original(blk.attn, cfg)
        if ffn:
            blk.ffn = FusedSwiGLU.from_original(blk.ffn, cfg)
    if norm:
        _xm.RMSNorm.forward = rmsnorm_fn_forward
    return model


if __name__ == "__main__":
    import os, sys, time, torch_neuronx
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "XScript-Pretraining", "src"))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
    from xscript.model import ModelConfig, Attention, SwiGLU, RMSNorm, _rope_cache
    from rope_fix import apply_rope_viewbased
    import xscript.model as _xm
    _xm._apply_rope = apply_rope_viewbased

    dev = torch.device("neuron")
    cfg = ModelConfig()
    T = cfg.max_seq_len
    torch.manual_seed(0)

    # --- equivalence on CPU (fp32) ---
    attn = Attention(cfg); fattn = FusedAttention.from_original(attn, cfg)
    x = torch.randn(1, 256, cfg.dim)
    cos, sin = _rope_cache(256, cfg.head_dim, cfg.rope_theta, torch.device("cpu"), torch.float32)
    # need contiguous-transposed original forward (bench monkeypatch) -> emulate original math directly
    d = (attn(x, cos, sin) - fattn(x, cos, sin)).abs().max().item()
    print(f"FusedAttention vs Attention (cpu fp32) max diff: {d:.2e}", flush=True)
    ffn = SwiGLU(cfg); fffn = FusedSwiGLU.from_original(ffn, cfg)
    print(f"FusedSwiGLU vs SwiGLU (cpu fp32) max diff: {(ffn(x) - fffn(x)).abs().max().item():.2e}", flush=True)
    n = RMSNorm(cfg.dim, cfg.norm_eps)
    with torch.no_grad(): n.weight.add_(torch.randn(cfg.dim))
    ref = n(x); alt = rmsnorm_fn_forward(n, x)
    print(f"F.rms_norm vs RMSNorm (cpu fp32) max diff: {(ref - alt).abs().max().item():.2e}", flush=True)

    # --- compiled fwd+bwd timing on neuron, mb=1, T=2048 ---
    def sync(): torch_neuronx.synchronize()
    def timeit(fn, iters=10, warm=3):
        for _ in range(warm): fn()
        sync(); t0 = time.time()
        for _ in range(iters): fn()
        sync(); return (time.time() - t0) / iters * 1e3
    def fb(name, mod, *inputs):
        c = torch.compile(mod, backend="neuron", dynamic=False)
        def run():
            with torch.autocast("neuron", dtype=torch.bfloat16):
                out = c(*inputs)
            out.float().sum().backward()
            for t in inputs:
                if isinstance(t, torch.Tensor) and t.grad is not None: t.grad = None
        print(f"{name:34s} {timeit(run):7.2f} ms fwd+bwd", flush=True)

    xd = torch.randn(1, T, cfg.dim, device=dev, requires_grad=True)
    cos, sin = _rope_cache(T, cfg.head_dim, cfg.rope_theta, dev, torch.float32)
    from bench_train import _attn_forward_contig  # contiguous+viewrope original
    a0 = Attention(cfg).to(dev); a0.forward = _attn_forward_contig.__get__(a0)
    fb("Attention (contig + view rope)", a0, xd, cos, sin)
    fb("FusedAttention", FusedAttention(cfg).to(dev), xd, cos, sin)
    fb("SwiGLU", SwiGLU(cfg).to(dev), xd)
    fb("FusedSwiGLU", FusedSwiGLU(cfg).to(dev), xd)
    n0 = RMSNorm(cfg.dim, cfg.norm_eps).to(dev)
    fb("RMSNorm (module)", n0, xd)
    n1 = RMSNorm(cfg.dim, cfg.norm_eps).to(dev); n1.forward = rmsnorm_fn_forward.__get__(n1)
    fb("RMSNorm (F.rms_norm)", n1, xd)
    print("done", flush=True)
