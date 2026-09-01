#!/usr/bin/env python
"""Offline correctness tests for eval/knowneurons.py -- no checkpoint, no
accelerator, a few seconds on CPU. Run after touching knowneurons.py.

Asserts, on a small random Transformer:

  1. forward_ffn_scaled at scale=1 reproduces the stock forward bit-for-bit;
  2. scale=0 removes every FFN contribution (equals a hand-built
     attention-only forward);
  3. gold_loglik (one-hot + logsumexp, the Neuron-safe path) matches a
     gather-based log_softmax computation to fp32 precision;
  4. IG completeness: sum(Attr) -> G(1) - G(0) as n_alpha grows, and the
     midpoint rule at n_alpha=20 is within 2% of n_alpha=200 -- the numeric
     check that the joint-path IG is what the docstring claims;
  5. ablation_ll with an empty mask reproduces phase-A scores, and ablating
     ALL neurons reproduces G(0);
  6. prepare() puts any boundary re-tokenization inside the continuation and
     never loses the target.
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from xscript.model import ModelConfig, Transformer  # noqa: E402
from xscript.eval import knowneurons as kn          # noqa: E402


class TinyTok:
    """Deterministic stand-in tokenizer: one token per character."""
    def encode(self, text, bos=False, eos=False):
        ids = [10 + (ord(c) % 50) for c in text]
        return ([1] if bos else []) + ids + ([2] if eos else [])


def main():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=128, dim=64, n_layers=4, n_heads=4,
                      ffn_dim=96, max_seq_len=64)
    model = Transformer(cfg).eval()
    tok = TinyTok()
    dev = torch.device("cpu")

    ctx, cont = "the capital of tanzania is", " dodoma"
    prep = kn.prepare(tok, ctx, cont, cfg.max_seq_len)
    width = kn.even_width(len(prep.x))
    x, y, mask = kn.batch_tensors([prep], width)

    # 1. parity at scale=1
    ones = torch.ones((1, cfg.n_layers, cfg.ffn_dim))
    with torch.no_grad():
        logits_scaled = kn.forward_ffn_scaled(model, x, ones)
        logits_ref, _ = model(x, y.masked_fill(mask == 0, -100))
    assert torch.equal(logits_scaled, logits_ref), "scale=1 != stock forward"
    print("1. scale=1 parity: OK")

    # 2. scale=0 == attention-only model
    with torch.no_grad():
        logits0 = kn.forward_ffn_scaled(model, x, torch.zeros_like(ones))
        h = model.tok_emb(x)
        cos, sin = model._rope_for(x.shape[1], dev, h.dtype)
        for layer in model.layers:
            h = h + layer.attn(layer.attn_norm(h), cos, sin)
        ref0 = model.lm_head(model.norm(h))
    assert torch.equal(logits0, ref0), "scale=0 != attention-only forward"
    print("2. scale=0 removes FFNs: OK")

    # 3. one-hot scoring == gather-based log_softmax
    with torch.no_grad():
        ll = kn.gold_loglik(logits_ref, y, mask)
        lp = torch.log_softmax(logits_ref.float(), dim=-1)
        ll_ref = (lp.gather(2, y.unsqueeze(2)).squeeze(2) * mask).sum(-1)
    assert torch.allclose(ll, ll_ref, atol=1e-5), f"{ll} vs {ll_ref}"
    print(f"3. one-hot scoring == gather ({float(ll):.4f}): OK")

    # 4. IG completeness and step-count stability
    sums = {}
    for n in (20, 200):
        attr, g1, g0 = kn.attribute_fact(model, prep, dev, width,
                                         is_xla=False, n_alpha=n)
        sums[n] = float(attr.sum())
    gap = g1 - g0
    rel200 = abs(sums[200] - gap) / abs(gap)
    rel_steps = abs(sums[20] - sums[200]) / abs(sums[200])
    assert rel200 < 0.02, f"completeness off by {rel200:.1%} at n=200"
    assert rel_steps < 0.02, f"n=20 vs n=200 differ by {rel_steps:.1%}"
    print(f"4. completeness: sum(Attr)={sums[200]:.4f} vs G1-G0={gap:.4f} "
          f"(rel err {rel200:.2%}; n=20 vs n=200 {rel_steps:.2%}): OK")

    # 5. ablation consistency
    all_ids = list(range(cfg.n_layers * cfg.ffn_dim))
    res = kn.ablation_ll(model, tok, dev, ctx, [cont, " paris"],
                         [None, all_ids], width, cfg.max_seq_len, is_xla=False)
    assert abs(res[0][0] - float(ll)) < 1e-4, "no-mask ablation != phase A"
    assert abs(res[1][0] - g0) < 1e-4, "all-neuron ablation != G(0)"
    print("5. ablation endpoints: OK")

    # 6. boundary re-tokenization safety
    p = kn.prepare(tok, "abc", "def", 64)
    assert p.n_cont >= 3 and len(p.x) == len(p.y)
    assert p.cont_from + p.n_cont == len(p.y)
    print("6. prepare boundary: OK")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
