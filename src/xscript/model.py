"""Llama-style decoder-only transformer.

RMSNorm, rotary position embeddings, SwiGLU MLP, untied input/output
embeddings. Deliberately small and dependency-light (just torch) so it runs
unchanged on the GH200 nodes and on a login-node CPU smoke test.

Configs live in configs/*.yaml; ModelConfig mirrors the yaml `model:` block.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 65536
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int | None = None      # None -> = n_heads (no GQA)
    ffn_dim: int = 5632
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def _rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)                       # (T, head_dim/2)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def _apply_rope(x, cos, sin):
    # x: (B, H, T, D). split even/odd halves (rotate-half convention)
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., ::2] = o1
    out[..., 1::2] = o2
    return out


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if self.kv_heads != self.n_heads:
            rep = self.n_heads // self.kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)   # gate
        self.w3 = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)   # up
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)  # untied
        self._rope = None
        self.apply(self._init)
        # scale residual-projection inits by depth (GPT-2/Llama convention)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope_for(self, T, device, dtype):
        if self._rope is None or self._rope[0].shape[0] < T or self._rope[0].device != device:
            self._rope = _rope_cache(self.cfg.max_seq_len, self.cfg.head_dim,
                                     self.cfg.rope_theta, device, dtype)
        cos, sin = self._rope
        return cos[:T], sin[:T]

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos, sin = self._rope_for(T, idx.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        if targets is None:
            return self.lm_head(x[:, -1:, :])
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def layer_reps(self, idx):
        """Per-layer hidden states for representation analysis (MEXA).

        Returns a tensor (n_layers+1, B, T, dim): index 0 is the embedding
        output, index i>=1 is the output of block i. Causal attention means
        right-padding never contaminates real positions, so callers can pool
        over a length mask safely.
        """
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos, sin = self._rope_for(T, idx.device, x.dtype)
        reps = [x]
        for layer in self.layers:
            x = layer(x, cos, sin)
            reps.append(x)
        return torch.stack(reps, dim=0)

    def num_params(self, embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not embedding:
            n -= self.tok_emb.weight.numel() + self.lm_head.weight.numel()
        return n
