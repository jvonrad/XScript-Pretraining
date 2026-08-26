"""Profile shows 3.57 GB spill/reload. The FFN is per-token, so computing it
in token chunks is mathematically identical and halves live intermediates.
Does that reduce spilling enough to win?"""
import sys, time, torch, torch_neuronx
import torch.nn.functional as F
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

def make_block(nchunk):
    b = Block(cfg).to(dev)
    if nchunk > 1:
        orig_ffn = b.ffn
        def chunked_ffn(x, _f=orig_ffn, _n=nchunk):
            B, T_, Dd = x.shape
            xs = x.reshape(B * T_, Dd).chunk(_n, dim=0)
            return torch.cat([_f(c) for c in xs], dim=0).view(B, T_, Dd)
        b.ffn_call = chunked_ffn
        def fwd(x, cos, sin, _b=b):
            x = x + _b.attn(_b.attn_norm(x), cos, sin)
            return x + _b.ffn_call(_b.ffn_norm(x))
        return b, fwd
    return b, None

for n in (1, 2, 4):
    torch._dynamo.reset()
    b, fwd = make_block(n)
    if fwd is None:
        b.compile(backend="neuron", dynamic=False)
        f = lambda x: b(x, cos, sin)
    else:
        cf = torch.compile(fwd, backend="neuron", dynamic=False)
        f = lambda x: cf(x, cos, sin)
    try:
        ms = timeit(f, 2)
        print(f"ffn token-chunks={n}: Block mb=2 {ms:7.2f} ms -> {ms/2:.2f} ms/2048tok", flush=True)
    except Exception as e:
        print(f"ffn token-chunks={n}: FAILED {repr(e)[:140]}", flush=True)

# numerics check
torch._dynamo.reset()
b1 = Block(cfg).to(dev)
b2, fwd2 = make_block(2); b2.load_state_dict(b1.state_dict())
xx = torch.randn(2, T, D, device=dev)
with torch.no_grad():
    d = (b1(xx, cos, sin) - fwd2(xx, cos, sin)).abs().max().item()
print(f"chunked vs whole max diff: {d:.3e}", flush=True)
print("done", flush=True)
