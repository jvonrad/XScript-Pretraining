"""Component-level timing + matmul roofline for the whole-graph training step.

Answers two questions:
1. Where do the 397ms/micro (mb=1, per core) go — blocks, attention, ffn, head+CE?
2. What matmul utilization can this runtime even reach (the MFU ceiling)?

Run single-core: python3 comp_bench.py
"""
import sys, time, torch, torch_neuronx
import torch.nn.functional as F
import os as _os_, sys as _sys_
_HERE = _os_.path.dirname(_os_.path.abspath(__file__))
_REPO_SRC = _os_.path.join(_os_.path.dirname(_os_.path.dirname(_HERE)), "src")
_sys_.path.insert(0, _REPO_SRC)
_sys_.path.insert(0, _HERE)
from bench_train import MODEL, forward_loss  # noqa (imports the contiguous-rope patch too)
from xscript.model import ModelConfig, Transformer, Block, _rope_cache

dev = torch.device("neuron")
torch.manual_seed(1)
cfg = ModelConfig(**MODEL)
T, D, FF, V = cfg.max_seq_len, cfg.dim, cfg.ffn_dim, cfg.vocab_size

def sync(): torch_neuronx.synchronize()

def timeit(fn, iters=10, warm=3):
    for _ in range(warm): fn()
    sync(); t0 = time.time()
    for _ in range(iters): fn()
    sync()
    return (time.time() - t0) / iters

def report(name, secs, gflop):
    tfs = gflop / secs / 1e3
    print(f"{name:34s} {secs*1e3:8.1f} ms  {tfs:7.1f} TF/s  ({tfs/166.75*100:5.1f}% of core peak)", flush=True)

print("=== pure matmul roofline (bf16, fwd only) ===", flush=True)
for (m, k, n, label) in [
    (T, D, D, "attn proj  2048x2048x2048"),
    (T, D, FF, "ffn up     2048x2048x5632"),
    (T, FF, D, "ffn down   2048x5632x2048"),
    (T, D, V, "lm_head    2048x2048x65536"),
    (8*T, D, D, "8x tokens  16384x2048x2048"),
]:
    a = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    b = torch.randn(k, n, dtype=torch.bfloat16, device=dev)
    f = torch.compile(lambda x, y: x @ y, backend="neuron", dynamic=False)
    s = timeit(lambda: f(a, b))
    report(label, s, 2 * m * k * n / 1e9)

print("=== compiled components, fwd+bwd, mb=1 ===", flush=True)
mb = 1
x_tok = torch.randint(0, V, (mb, T), dtype=torch.int32, device=dev)
y_tok = torch.randint(0, V, (mb, T), dtype=torch.int32, device=dev)

blk = Block(cfg).to(dev)
blk.compile(backend="neuron", dynamic=False)
cos, sin = _rope_cache(T, cfg.head_dim, cfg.rope_theta, dev, torch.float32)
h = torch.randn(mb, T, D, device=dev, requires_grad=True)

def block_fb():
    with torch.autocast("neuron", dtype=torch.bfloat16):
        out = blk(h, cos, sin)
    out.sum().backward()
    h.grad = None
s = timeit(block_fb)
# per-layer fwd+bwd flops: (qkvo 4 + ffn 3 mats + attn) * tokens
gf_layer = (4*2*D*D + 2*D*FF*3 + 4*T*D) * T * 3 / 1e9
report("Block fwd+bwd", s, gf_layer)
print(f"  -> x16 layers = {16*s*1e3:.0f} ms", flush=True)

w_head = torch.randn(V, D, dtype=torch.float32, device=dev, requires_grad=True)
hx = torch.randn(T, D, device=dev, requires_grad=True)
tgt = torch.randint(0, V, (T,), dtype=torch.int32, device=dev)
head_fn = torch.compile(lambda a, w, t: F.cross_entropy((a @ w.t()).float(), t), backend="neuron", dynamic=False)

def head_fb():
    with torch.autocast("neuron", dtype=torch.bfloat16):
        loss = head_fn(hx, w_head, tgt)
    loss.backward()
    hx.grad = None; w_head.grad = None
s = timeit(head_fb)
report("lm_head+CE fwd+bwd", s, 2*T*D*V*3/1e9)

model = Transformer(cfg).to(dev)
def full(x, y):
    return forward_loss(model, x, y, False, 0, False)
fullc = torch.compile(full, backend="neuron", dynamic=False)

def full_fb():
    with torch.autocast("neuron", dtype=torch.bfloat16):
        loss = fullc(x_tok, y_tok)
    loss.backward()
    for p in model.parameters(): p.grad = None
s = timeit(full_fb, iters=5)
report("FULL graph fwd+bwd (per micro)", s, 6.54*T/1e0/1e0 * 1.0)  # 6.54 GF/tok * 2048 tok
print("done", flush=True)
