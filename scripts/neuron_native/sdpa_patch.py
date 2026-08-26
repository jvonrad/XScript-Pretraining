"""Patch the SDPA->NKI flash-attention invocation parameters that
torch_neuronx hardcodes. Select with SDPA_PATCH=bwd_bf16|mm_bf16|both.

  bwd_bf16 : attention_bwd(mixed_precision=False) -- bf16 intermediates in the
             backward instead of fp32 (decompositions.py hardcodes True).
  mm_bf16  : attention_cte(mm_out_dtype=bfloat16) -- PE matmul output dtype in
             the forward (kernel allows bf16 on gen4/Trn2; default fp32).

BOTH CHANGE NUMERICS (softmax/PE accumulation precision). Measure the loss.
"""
import math
import os

import torch
import torch_neuronx.neuron_dynamo_backend.decompositions as D
from torch_neuronx.utils import get_logical_neuron_cores

MODE = os.environ.get("SDPA_PATCH", "")


def _bwd_bf16(grad_out, query, key, value, out, logsumexp, is_causal, scale):
    """Copy of D._compute_attention_backward_nki with mixed_precision=False."""
    B, q_heads, L = query.shape[:3]
    kv_heads = key.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    q_nki = query.transpose(-2, -1)
    k_nki = key.transpose(-2, -1)
    v_nki = value.transpose(-2, -1)
    out_nki = out.transpose(-2, -1) if out is not None else None
    grad_out_nki = grad_out.transpose(-2, -1)
    if logsumexp.ndim == 3:
        tile_size = 128
        n_tiles = L // tile_size
        if L % tile_size != 0:
            raise ValueError(f"seq {L} not divisible by {tile_size}")
        lse_4d = logsumexp.view(B, q_heads, n_tiles, tile_size).transpose(-2, -1)
    else:
        lse_4d = logsumexp
    lnc = int(get_logical_neuron_cores())
    grid = (lnc,) if B * kv_heads % lnc == 0 else (1,)
    gq, gk, gv = D.wrapped_flash_bwd[grid](
        q_nki, k_nki, v_nki, out_nki, grad_out_nki, lse_4d,
        use_causal_mask=is_causal,
        mixed_precision=False,          # <-- the patch
        softmax_scale=scale,
    )
    return gq.transpose(-2, -1), gk.transpose(-2, -1), gv.transpose(-2, -1)


def _fwd_mm_bf16(query, key, value, is_causal, scale, training):
    """Copy of D._compute_attention_nki with mm_out_dtype=bfloat16."""
    import nki.language as nl
    lnc = int(get_logical_neuron_cores())
    grid = (lnc,)
    B, H, L, E = query.shape
    _, H_kv, S, Dh = key.shape
    q_nki = query.reshape(B * H, L, E)
    k_nki = key.reshape(B * H_kv, S, Dh)
    v_nki = value.reshape(B * H_kv, S, Dh)
    if scale is None:
        scale = 1.0 / math.sqrt(E)
    if training:
        output, neg_max, out_sum_recip = D.wrapped_flash_fwd[grid](
            q_nki, k_nki, v_nki, tp_q=True, tp_k=True, scale=scale,
            causal_mask=is_causal, cache_softmax=True,
            mm_out_dtype=nl.bfloat16,   # <-- the patch
        )
        logsumexp = -(neg_max + torch.log(out_sum_recip))
        logsumexp = logsumexp.to(torch.float32).reshape(B, H, *logsumexp.shape[-2:])
        logsumexp = logsumexp.transpose(-2, -1).reshape(B, H, -1)
        return output.reshape(B, H, L, Dh), logsumexp
    output = D.wrapped_flash_fwd[grid](
        q_nki, k_nki, v_nki, tp_q=True, tp_k=True, scale=scale,
        causal_mask=is_causal, cache_softmax=False, mm_out_dtype=nl.bfloat16)
    return output.reshape(B, H, L, Dh)


def apply():
    if MODE in ("bwd_bf16", "both"):
        D._compute_attention_backward_nki = _bwd_bf16
    if MODE in ("mm_bf16", "both"):
        D._compute_attention_nki = _fwd_mm_bf16
    return MODE or "none"
