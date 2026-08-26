"""Does the tuned recipe's MFU generalize across model shapes?

Times ONE compiled Block (the unit that dominates the step: 16 of them are
~100% of a micro) at mb=2 for several (dim, ffn, heads) geometries, and
reports achieved TF/s and % of the 166.75 TF/s per-core bf16 peak.

FLOPs/token/layer, fwd+bwd = 3 * (8*dim^2 + 6*dim*ffn + 2*T*dim)
  (qkvo projections; SwiGLU's three matmuls; causal attention halved)
"""
import sys, time, torch, torch_neuronx
sys.path.insert(0, "/bench"); sys.path.insert(0, "/repo/src")
import bench_train as bt          # rope fix + contiguous attention patches
from xscript.model import ModelConfig, Block, _rope_cache

dev = torch.device("neuron")
T = 2048
PEAK = 166.75          # TF/s per logical core, dense bf16
def sync(): torch_neuronx.synchronize()

SHAPES = [
    # name,                dim,  ffn,  heads, layers_for_1_7B
    ("1B  current   d2048", 2048, 5632, 16, 16),
    ("1.7B deep     d2048", 2048, 5632, 16, 28),
    ("1.7B wide     d2560", 2560, 6912, 20, 18),
    ("2.5B          d3072", 3072, 8192, 24, 20),
    ("7B-ish        d4096", 4096, 11008, 32, 32),
]

print(f"{'shape':22s} {'ms/blk':>8s} {'GF/tok/layer':>13s} {'TF/s':>8s} {'%peak':>7s}  {'1.7B params':>12s}", flush=True)
for name, dim, ffn, heads, nl in SHAPES:
    cfg = ModelConfig(vocab_size=65536, dim=dim, n_layers=1, n_heads=heads,
                      n_kv_heads=heads, ffn_dim=ffn, max_seq_len=T,
                      rope_theta=10000.0, norm_eps=1e-5)
    torch._dynamo.reset()
    try:
        b = Block(cfg).to(dev)
        b.compile(backend="neuron", dynamic=False)
        cos, sin = _rope_cache(T, cfg.head_dim, cfg.rope_theta, dev, torch.float32)
        mb = 2
        x = torch.randn(mb, T, dim, device=dev, requires_grad=True)
        def run():
            with torch.autocast("neuron", dtype=torch.bfloat16):
                out = b(x, cos, sin)
            out.float().sum().backward(); x.grad = None
        for _ in range(3): run()
        sync(); t0 = time.time()
        for _ in range(6): run()
        sync(); ms = (time.time() - t0) / 6 * 1e3
        gf_tok = 3 * (8 * dim * dim + 6 * dim * ffn + 2 * T * dim) / 1e9
        tfs = gf_tok * (mb * T) / (ms / 1e3) / 1e3
        # params of a full model of this shape at nl layers (+ tied-free embeddings)
        p = nl * (4 * dim * dim + 3 * dim * ffn) + 2 * 65536 * dim
        print(f"{name:22s} {ms:8.2f} {gf_tok:13.3f} {tfs:8.1f} {tfs/PEAK*100:6.1f}%  {p/1e9:11.2f}B", flush=True)
        del b, x
    except Exception as e:
        print(f"{name:22s} FAILED {repr(e)[:110]}", flush=True)
print("done", flush=True)
