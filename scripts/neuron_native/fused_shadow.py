"""Fused-weight bf16 shadow modules.

The bf16 shadow weights are refreshed once per optimizer step, so the fused
[3D, D] qkv and [2FF, D] gate|up weights can be MATERIALIZED ONCE PER STEP
instead of per micro -- which is what made an earlier in-forward `torch.cat`
attempt a loss. Wider matmuls are more efficient on this backend (roofline:
N=2048 -> 43% of peak, N=5632 -> 71%).

Math: identical to model.py -- each output element is the same dot product;
only the matmul tiling/accumulation order can differ. Gradients are split
back into the fp32 master wq/wk/wv (w1/w3) params, so the optimizer state and
the checkpoint format are unchanged.
"""
import torch
import torch.nn.functional as F

import xscript.model as _xm


class ShadowAttn(torch.nn.Module):
    """Attention with one fused [3*dim, dim] qkv weight."""

    def __init__(self, attn, cfg, device):
        super().__init__()
        self.n_heads, self.kv_heads, self.head_dim = attn.n_heads, attn.kv_heads, attn.head_dim
        self.dim = cfg.dim
        nq, nkv = cfg.n_heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim
        self.splits = (nq, nkv, nkv)
        self.wqkv = torch.nn.Parameter(
            torch.empty(nq + 2 * nkv, cfg.dim, dtype=torch.bfloat16, device=device))
        self.wo = torch.nn.Parameter(attn.wo.weight.data.to(torch.bfloat16))

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        qkv = F.linear(x, self.wqkv)
        q, k, v = qkv.split(self.splits, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()
        q = _xm._apply_rope(q, cos, sin)
        k = _xm._apply_rope(k, cos, sin)
        if self.kv_heads != self.n_heads:
            rep = self.n_heads // self.kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return F.linear(out, self.wo)


class ShadowFFN(torch.nn.Module):
    """SwiGLU with one fused [2*ffn, dim] gate|up weight."""

    def __init__(self, ffn, cfg, device):
        super().__init__()
        self.ffn_dim = cfg.ffn_dim
        self.w13 = torch.nn.Parameter(
            torch.empty(2 * cfg.ffn_dim, cfg.dim, dtype=torch.bfloat16, device=device))
        self.w2 = torch.nn.Parameter(ffn.w2.weight.data.to(torch.bfloat16))

    def forward(self, x):
        g, u = F.linear(x, self.w13).split((self.ffn_dim, self.ffn_dim), dim=-1)
        return F.linear(F.silu(g) * u, self.w2)


def build(shadow_model, raw_model, cfg, device):
    """Swap the shadow model's attn/ffn for fused versions.

    Returns (refresh_specs, hook_specs):
      refresh_specs: [(fused_param, [master fp32 params in order])]  -- copied
                     into the fused buffer once per optimizer step
      hook_specs:    same pairs; the fused param's bf16 grad is split and added
                     into each master's fp32 .grad
    """
    specs = []
    for s_blk, m_blk in zip(shadow_model.layers, raw_model.layers):
        sa = ShadowAttn(m_blk.attn, cfg, device)
        sf = ShadowFFN(m_blk.ffn, cfg, device)
        specs.append((sa.wqkv, [m_blk.attn.wq.weight, m_blk.attn.wk.weight, m_blk.attn.wv.weight]))
        specs.append((sa.wo, [m_blk.attn.wo.weight]))
        specs.append((sf.w13, [m_blk.ffn.w1.weight, m_blk.ffn.w3.weight]))
        specs.append((sf.w2, [m_blk.ffn.w2.weight]))
        s_blk.attn = sa
        s_blk.ffn = sf
    return specs


@torch.no_grad()
def refresh(specs):
    """fused_bf16 <- cat(master fp32 rows). Once per optimizer step."""
    for fused, masters in specs:
        off = 0
        for m in masters:
            n = m.shape[0]
            fused.data[off:off + n].copy_(m.data)
            off += n


def make_hook(fused, masters):
    """Split the fused bf16 grad into the master fp32 .grad buffers."""
    def _h(param):
        g = param.grad
        if g is None:
            return
        off = 0
        for m in masters:
            n = m.shape[0]
            gs = g[off:off + n]
            if m.grad is None:
                m.grad = gs.float()
            else:
                m.grad.add_(gs)
            off += n
        param.grad = None
    return _h
