"""Cross-lingual knowledge neurons: attribution, overlap, and ablation transfer.

Answers one question at the mechanism level: when a model knows the SAME fact
in two languages, does it use the same FFN neurons for both -- and does the
fair (destarved) tokenizer increase that parametric sharing relative to the
starved one? (CLAUDE.md 6j.)

Method lineage, and where this deviates:

* Dai et al. 2022 / Chen et al. 2024 (AMIG, arXiv:2308.13198): knowledge
  neurons via integrated gradients over FFN intermediate activations;
  language-independent knowledge neurons = the cross-language intersection of
  per-language knowledge-neuron sets; verified by suppress/enhance.
* Ifergan et al. 2025 (NAACL Findings, "Beneath the Surface of Consistency"):
  sharing != consistency; sharing must be measured FUNCTIONALLY (they use
  cross-lingual edit transfer via ROME/MEMIT). Here the functional readout is
  cross-lingual ABLATION transfer instead: zero the neurons attributed to a
  fact in language A and measure the damage to the same fact in language B,
  against different-fact and random-neuron controls. This avoids porting
  ROME/EasyEdit (per-model covariance stats, HF-specific code) to a custom
  1B architecture while measuring the same construct.

Attribution: joint-path integrated gradients over the FFN multipliers.

  Let G(s) be the summed gold-answer loglikelihood (teacher-forced over ALL
  answer tokens, per Fable's plan -- never first-token-only, which is not
  comparable across scripts) when every SwiGLU intermediate activation
  h[l, pos, j] is multiplied by s[l, j] (broadcast over positions). IG along
  the diagonal path s = alpha * 1, alpha in (0, 1]:

      Attr[l, j] = integral_0^1 dG/ds[l,j] (alpha * 1) d alpha
                 ~ (1/N) sum_k dG/ds[l,j] (alpha_k * 1),   alpha_k midpoints.

  Because s is broadcast over positions, one backward against s yields the
  position-summed attribution map [n_layers, ffn_dim] directly -- the chain
  rule multiplies each position's gradient by that position's activation, so
  no separate "grad x activation" product is needed. Completeness holds:
  sum Attr ~ G(1) - G(0), where G(0) is the FFN-free (attention-only) model
  -- the standard zero/suppression baseline. `test_knowneurons.py` asserts
  this numerically.

  DEVIATION from AMIG, deliberate: AMIG integrates one layer at a time with
  per-word <eos> baselines (16 x more forward passes, m x more for the word
  decomposition). The joint path treats all layers symmetrically in a single
  well-defined attribution, costs one batched fwd+bwd per (fact, language),
  and any bias it introduces is identical across the fair/starved conditions
  being contrasted. The ablation phase provides causal validation that does
  not depend on the attribution flavor.

Overlap: fixed top-K (no per-language thresholds -- Chen et al.'s dynamic
thresholds are a free parameter that could differ between tokenizer
conditions; a fixed K cannot). The sharing statistic is

    dKS = mean_f J(topK_A(f), topK_B(f)) - mean_f J(topK_A(f), topK_B(f')),

  same-fact cross-language Jaccard minus mismatched-fact cross-language
  Jaccard (deterministic derangement f' = next fact), so "neurons that always
  show up" cancel. Raw overlap is NOT comparable across tokenizers; dKS and
  the ablation transfer ratio are the quotable quantities.

XLA/Neuron rules obeyed throughout (NEURON.md 4): fixed [batch, width]
shapes, even widths, tensors built on the host and moved once, no
torch.gather over the vocab dim (one-hot multiply + logsumexp), masks kept
in float, results reduced on device and moved back small.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ..tok.wrapper import BOS_ID, PAD_ID

N_ALPHA = 20          # Riemann midpoint steps for the IG path
TOPK_STORE = 512      # top neurons persisted per (fact, lang); analysis K <= this


# ---------------------------------------------------------------------------
# forward with FFN multipliers
# ---------------------------------------------------------------------------

def forward_ffn_scaled(model, idx: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Full-position logits with every SwiGLU intermediate scaled.

    `scale` is [B, n_layers, ffn_dim] (per-row multipliers -- rows carry the
    IG alpha steps or per-row ablation masks). Mirrors Transformer.forward /
    Block.forward exactly apart from the multiply; parity with the stock
    forward at scale=1 is asserted in test_knowneurons.py.
    """
    B, T = idx.shape
    x = model.tok_emb(idx)
    cos, sin = model._rope_for(T, idx.device, x.dtype)
    for li, layer in enumerate(model.layers):
        x = x + layer.attn(layer.attn_norm(x), cos, sin)
        h = layer.ffn_norm(x)
        a = F.silu(layer.ffn.w1(h)) * layer.ffn.w3(h)          # [B, T, ffn]
        a = a * scale[:, li].unsqueeze(1)                       # broadcast over T
        x = x + layer.ffn.w2(a)
    return model.lm_head(model.norm(x))


def gold_loglik(logits: torch.Tensor, y_idx: torch.Tensor,
                cont_mask: torch.Tensor) -> torch.Tensor:
    """Per-row summed loglikelihood of the target over continuation positions.

    y_idx must be host-clamped to valid ids before reaching the device
    (NEURON.md 4 trap 2); cont_mask is float (bool-sum trap). One-hot
    multiply instead of gather (trap 1).
    """
    logits = logits.float()
    onehot = F.one_hot(y_idx, logits.size(-1)).to(logits.dtype)
    target_logit = (logits * onehot).sum(-1)                    # [B, W]
    token_lp = target_logit - torch.logsumexp(logits, dim=-1)   # [B, W]
    return (token_lp * cont_mask).sum(-1)                       # [B]


# ---------------------------------------------------------------------------
# request preparation (lm-eval conventions, matching bench.py)
# ---------------------------------------------------------------------------

@dataclass
class Prepared:
    """One scoring request: x/y token rows plus the continuation mask."""
    x: list[int]          # seq[:-1]
    y: list[int]          # seq[1:]
    cont_from: int        # first position in y that belongs to the continuation
    n_cont: int


def prepare(tok, context: str, continuation: str, max_seq_len: int) -> Prepared:
    """lm-eval's _encode_pair convention: continuation tokens are the whole
    encoding minus the context encoding's length, so any context/continuation
    boundary re-tokenization lands inside the continuation. BOS prepended
    once, exactly as bench.XScriptLM._prepare does."""
    ctx_enc = tok.encode(context, bos=False, eos=False)
    whole_enc = tok.encode(context + continuation, bos=False, eos=False)
    n_cont = max(1, len(whole_enc) - len(ctx_enc))
    seq = [BOS_ID] + whole_enc
    if len(seq) > max_seq_len:
        raise ValueError(f"sequence of {len(seq)} tokens exceeds {max_seq_len}")
    m = len(seq) - 1
    return Prepared(x=seq[:-1], y=seq[1:], cont_from=m - n_cont, n_cont=n_cont)


def batch_tensors(preps: list[Prepared], width: int):
    """Host-built fixed-shape tensors for a batch of requests.

    Returns (x, y_idx, cont_mask): x [B, W] long, y_idx [B, W] long (pads
    clamped to 0 ON THE HOST), cont_mask [B, W] float32.
    """
    B = len(preps)
    x = torch.full((B, width), PAD_ID, dtype=torch.long)
    y = torch.full((B, width), 0, dtype=torch.long)
    mask = torch.zeros((B, width), dtype=torch.float32)
    for r, p in enumerate(preps):
        m = len(p.x)
        if m > width:
            raise ValueError(f"row of {m} tokens exceeds width {width}")
        x[r, :m] = torch.tensor(p.x)
        y[r, :m] = torch.tensor(p.y)
        mask[r, p.cont_from:p.cont_from + p.n_cont] = 1.0
    return x, y, mask


def even_width(n: int) -> int:
    """Round up to even (NCC-5266: odd widths fail compilation)."""
    return n + (n % 2)


# ---------------------------------------------------------------------------
# phase A: candidate scoring (fact selection)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_candidates(model, tok, device, contexts: list[str],
                     continuations: list[str], width: int,
                     batch_size: int, max_seq_len: int,
                     is_xla: bool) -> list[float]:
    """Summed loglikelihood for each (context, continuation) pair.

    Fixed [batch_size, width] shape on XLA so the whole pass hits one graph.
    """
    preps = [prepare(tok, c, k, max_seq_len) for c, k in zip(contexts, continuations)]
    out: list[float] = []
    ones = torch.ones((batch_size, model.cfg.n_layers, model.cfg.ffn_dim),
                      device=device)
    for i in range(0, len(preps), batch_size):
        chunk = preps[i:i + batch_size]
        pad_rows = batch_size - len(chunk)
        x, y, mask = batch_tensors(chunk + [chunk[-1]] * pad_rows, width)
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits = forward_ffn_scaled(model, x, ones)
        ll = gold_loglik(logits, y, mask)
        if is_xla:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        ll = ll.cpu()
        out.extend(float(v) for v in ll[:len(chunk)])
    return out


# ---------------------------------------------------------------------------
# phase B: integrated-gradients attribution
# ---------------------------------------------------------------------------

def attribute_fact(model, prep: Prepared, device, width: int,
                   is_xla: bool, n_alpha: int = N_ALPHA,
                   alpha_grid: "list[float] | None" = None):
    """Joint-path IG attribution map for one prepared request.

    Returns (attr [n_layers, ffn_dim] float32 cpu, G1, G0) where G1/G0 are
    the full-model / FFN-off gold loglikelihoods (completeness endpoints).
    Batch layout: rows 0..n_alpha-1 are the midpoint alphas, row n_alpha is
    alpha=1 (G1), row n_alpha+1 is alpha=0 (G0) -- one fwd+bwd per fact.

    `alpha_grid` overrides the midpoint grid (len must equal n_alpha) so a
    finer integration can be run as several chunks that all reuse the one
    compiled [n_alpha+2, width] graph -- alpha values are data, not shape.
    Each chunk returns (1/n_alpha) * sum over its own grid; average C chunks
    covering a C*n_alpha-point grid to get the fine-grid attribution.
    """
    L, Fd = model.cfg.n_layers, model.cfg.ffn_dim
    B = n_alpha + 2
    x, y, mask = batch_tensors([prep] * B, width)
    x, y, mask = x.to(device), y.to(device), mask.to(device)

    grid = (alpha_grid if alpha_grid is not None
            else [(k + 0.5) / n_alpha for k in range(n_alpha)])
    assert len(grid) == n_alpha
    alphas = torch.tensor(list(grid) + [1.0, 0.0], dtype=torch.float32)
    scale = (alphas.view(B, 1, 1).expand(B, L, Fd).clone()
             .to(device).requires_grad_(True))
    logits = forward_ffn_scaled(model, x, scale)
    ll = gold_loglik(logits, y, mask)                          # [B]
    grad_out = torch.zeros(B, dtype=torch.float32)
    grad_out[:n_alpha] = 1.0 / n_alpha                          # the IG mean
    grad_out = grad_out.to(device)
    # grad row b = w_b * d ll_b / d scale_b (rows are independent), so summing
    # rows yields (1/N) sum_k d ll / d s at alpha_k; endpoint rows have w=0.
    (grad,) = torch.autograd.grad(ll, scale, grad_outputs=grad_out)
    attr = grad.sum(0)
    if is_xla:
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    attr = attr.cpu().float()
    ll = ll.detach().cpu()
    return attr, float(ll[n_alpha]), float(ll[n_alpha + 1])


def topk_flat(attr: torch.Tensor, k: int = TOPK_STORE):
    """Top-k neuron ids (layer * ffn_dim + j) and values, descending."""
    flat = attr.reshape(-1)
    val, idx = torch.topk(flat, k)
    return idx.to(torch.int32).numpy(), val.numpy()


# ---------------------------------------------------------------------------
# phase C: ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def ablation_ll(model, tok, device, context: str, continuations: list[str],
                masks_flat: list, width: int, max_seq_len: int,
                is_xla: bool) -> "list[list[float]]":
    """Gold loglikelihoods for each candidate under each ablation mask.

    `masks_flat` is a list of neuron-id arrays (flattened layer*ffn+j); entry
    None means no ablation. Returns [n_masks][n_cands] summed loglikelihoods.
    Rows = n_masks x n_cands in one fixed-shape batch.
    """
    L, Fd = model.cfg.n_layers, model.cfg.ffn_dim
    preps = [prepare(tok, context, c, max_seq_len) for c in continuations]
    nC, nM = len(preps), len(masks_flat)
    B = even_width(nM * nC)
    rows = preps * nM + [preps[-1]] * (B - nM * nC)
    x, y, mask = batch_tensors(rows, width)

    scale = torch.ones((B, L, Fd), dtype=torch.float32)
    for mi, ids in enumerate(masks_flat):
        if ids is None:
            continue
        t = torch.as_tensor(ids, dtype=torch.long)
        flat = scale[mi * nC:(mi + 1) * nC].reshape(nC, L * Fd)
        flat[:, t] = 0.0                                        # host-side scatter
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    scale = scale.to(device)
    logits = forward_ffn_scaled(model, x, scale)
    ll = gold_loglik(logits, y, mask)
    if is_xla:
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    ll = ll.cpu()
    return [[float(ll[mi * nC + ci]) for ci in range(nC)] for mi in range(nM)]


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def save_npz(path: Path, **arrays):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(path)


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.rename(path)
