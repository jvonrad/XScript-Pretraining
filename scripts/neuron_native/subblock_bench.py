"""Compile granularity BELOW one Block: does splitting attn/ffn into separate
compiled graphs reduce SBUF spilling (profile shows 3.57 GB spill/reload)?"""
import sys, time, torch, torch_neuronx
sys.path.insert(0, "/bench"); sys.path.insert(0, "/repo/src")
import bench_train as bt
from xscript.model import ModelConfig, Block, _rope_cache
dev = torch.device("neuron"); cfg = ModelConfig(**bt.MODEL)
T, D = cfg.max_seq_len, cfg.dim
cos, sin = _rope_cache(T, cfg.head_dim, cfg.rope_theta, dev, torch.float32)
def sync(): torch_neuronx.synchronize()

def timeit(fn, mb, iters=8):
    x = torch.randn(mb, T, D, device=dev, requires_grad=True)
    def run():
        with torch.autocast("neuron", dtype=torch.bfloat16):
            out = fn(x)
        out.float().sum().backward(); x.grad = None
    for _ in range(3): run()
    sync(); t0 = time.time()
    for _ in range(iters): run()
    sync(); return (time.time() - t0) / iters * 1e3

mb = int(sys.argv[1]) if len(sys.argv) > 1 else 2

# A: whole Block as one graph (current)
torch._dynamo.reset()
b = Block(cfg).to(dev); b.compile(backend="neuron", dynamic=False)
ms = timeit(lambda x: b(x, cos, sin), mb)
print(f"whole Block 1 graph      mb={mb}: {ms:7.2f} ms -> {ms/mb:.2f} ms/2048tok", flush=True)

# B: attn and ffn compiled separately (2 graphs), norms eager
torch._dynamo.reset()
b2 = Block(cfg).to(dev)
b2.attn.compile(backend="neuron", dynamic=False)
b2.ffn.compile(backend="neuron", dynamic=False)
def fwd_split(x):
    x = x + b2.attn(b2.attn_norm(x), cos, sin)
    return x + b2.ffn(b2.ffn_norm(x))
ms = timeit(fwd_split, mb)
print(f"attn|ffn 2 graphs        mb={mb}: {ms:7.2f} ms -> {ms/mb:.2f} ms/2048tok", flush=True)

# C: norm+attn and norm+ffn as 2 graphs (norms inside)
torch._dynamo.reset()
b3 = Block(cfg).to(dev)
def half1(x): return x + b3.attn(b3.attn_norm(x), cos, sin)
def half2(x): return x + b3.ffn(b3.ffn_norm(x))
c1 = torch.compile(half1, backend="neuron", dynamic=False)
c2 = torch.compile(half2, backend="neuron", dynamic=False)
ms = timeit(lambda x: c2(c1(x)), mb)
print(f"norm+attn|norm+ffn 2 gr  mb={mb}: {ms:7.2f} ms -> {ms/mb:.2f} ms/2048tok", flush=True)

# D: four graphs (qkv+sdpa, wo, w13+act, w2) -- maximum splitting
torch._dynamo.reset()
b4 = Block(cfg).to(dev)
import torch.nn.functional as F
def d_attn_core(x):
    B, T_, _ = x.shape
    a = b4.attn
    q = a.wq(x).view(B, T_, a.n_heads, a.head_dim).transpose(1, 2).contiguous()
    k = a.wk(x).view(B, T_, a.kv_heads, a.head_dim).transpose(1, 2).contiguous()
    v = a.wv(x).view(B, T_, a.kv_heads, a.head_dim).transpose(1, 2).contiguous()
    import xscript.model as _xm
    q = _xm._apply_rope(q, cos, sin); k = _xm._apply_rope(k, cos, sin)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return o.transpose(1, 2).contiguous().view(B, T_, -1)
d1 = torch.compile(d_attn_core, backend="neuron", dynamic=False)
d2 = torch.compile(lambda o: b4.attn.wo(o), backend="neuron", dynamic=False)
d3 = torch.compile(lambda x: F.silu(b4.ffn.w1(x)) * b4.ffn.w3(x), backend="neuron", dynamic=False)
d4 = torch.compile(lambda g: b4.ffn.w2(g), backend="neuron", dynamic=False)
def fwd4(x):
    x = x + d2(d1(b4.attn_norm(x)))
    return x + d4(d3(b4.ffn_norm(x)))
ms = timeit(fwd4, mb)
print(f"4 graphs (max split)     mb={mb}: {ms:7.2f} ms -> {ms/mb:.2f} ms/2048tok", flush=True)
print("done", flush=True)
