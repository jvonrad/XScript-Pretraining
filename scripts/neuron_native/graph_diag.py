"""Are the per-Block compiled graphs actually ONE graph? And are the
.contiguous() calls (added for the OLD strided rope) still needed?"""
import os, sys, time, torch, torch_neuronx
import torch.nn.functional as F
sys.path.insert(0, "/bench"); sys.path.insert(0, "/repo/src")
import bench_train as bt          # installs view-based rope + contig attention
import xscript.model as _xm
from xscript.model import ModelConfig, Block, _rope_cache

dev = torch.device("neuron"); cfg = ModelConfig(**bt.MODEL)
T, D = cfg.max_seq_len, cfg.dim
cos, sin = _rope_cache(T, cfg.head_dim, cfg.rope_theta, dev, torch.float32)

def attn_no_contig(self, x, cos, sin):
    """Same as bench_train._attn_forward_contig but WITHOUT the three
    .contiguous() copies -- valid now that rope is view-based."""
    B, T_, _ = x.shape
    q = self.wq(x).view(B, T_, self.n_heads, self.head_dim).transpose(1, 2)
    k = self.wk(x).view(B, T_, self.kv_heads, self.head_dim).transpose(1, 2)
    v = self.wv(x).view(B, T_, self.kv_heads, self.head_dim).transpose(1, 2)
    q = _xm._apply_rope(q, cos, sin)
    k = _xm._apply_rope(k, cos, sin)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return self.wo(out.transpose(1, 2).contiguous().view(B, T_, -1))

def attn_qk_contig(self, x, cos, sin):
    """contiguous() only on q,k (rope inputs); v left as a view."""
    B, T_, _ = x.shape
    q = self.wq(x).view(B, T_, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
    k = self.wk(x).view(B, T_, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()
    v = self.wv(x).view(B, T_, self.kv_heads, self.head_dim).transpose(1, 2)
    q = _xm._apply_rope(q, cos, sin)
    k = _xm._apply_rope(k, cos, sin)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return self.wo(out.transpose(1, 2).contiguous().view(B, T_, -1))

# ---- 1. graph-break census on one Block ----
blk = Block(cfg).to(dev)
h = torch.randn(2, T, D, device=dev, requires_grad=True)
try:
    exp = torch._dynamo.explain(blk)(h, cos, sin)
    print(f"[explain] graph_count={exp.graph_count} break_count={exp.graph_break_count} "
          f"op_count={exp.op_count}", flush=True)
    for i, r in enumerate(getattr(exp, "break_reasons", [])[:6]):
        print(f"  break {i}: {str(r)[:180]}", flush=True)
except Exception as e:
    print("explain failed:", repr(e)[:200], flush=True)
torch._dynamo.reset()

# ---- 2. timing: contiguous variants, and fullgraph ----
def timeit(mod, mb=2, iters=8, fullgraph=False):
    c = torch.compile(mod, backend="neuron", dynamic=False, fullgraph=fullgraph)
    x = torch.randn(mb, T, D, device=dev, requires_grad=True)
    def run():
        with torch.autocast("neuron", dtype=torch.bfloat16):
            out = c(x, cos, sin)
        out.float().sum().backward(); x.grad = None
    for _ in range(3): run()
    torch_neuronx.synchronize(); t0 = time.time()
    for _ in range(iters): run()
    torch_neuronx.synchronize()
    return (time.time() - t0) / iters * 1e3

import types
for name, fn, fg in (("contig q,k,v (current)", bt._attn_forward_contig, False),
                     ("contig q,k only",        attn_qk_contig,          False),
                     ("no contig",              attn_no_contig,          False),
                     ("contig q,k,v FULLGRAPH", bt._attn_forward_contig, True)):
    torch._dynamo.reset()
    b = Block(cfg).to(dev)
    b.attn.forward = types.MethodType(fn, b.attn)
    try:
        ms = timeit(b, fullgraph=fg)
        print(f"{name:26s} Block mb=2 fwd+bwd {ms:7.2f} ms -> {ms/2:.2f} ms/2048tok", flush=True)
    except Exception as e:
        print(f"{name:26s} FAILED {repr(e)[:150]}", flush=True)

# ---- 3. numerics check: no-contig must match contig ----
torch._dynamo.reset()
b1 = Block(cfg).to(dev)
b2 = Block(cfg).to(dev); b2.load_state_dict(b1.state_dict())
b2.attn.forward = types.MethodType(attn_no_contig, b2.attn)
xx = torch.randn(2, T, D, device=dev)
with torch.no_grad():
    d = (b1(xx, cos, sin) - b2(xx, cos, sin)).abs().max().item()
print(f"no-contig vs contig max diff: {d:.3e}", flush=True)
print("done", flush=True)
