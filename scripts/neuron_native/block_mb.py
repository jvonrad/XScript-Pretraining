"""Per-token Block fwd+bwd time at mb=1 vs mb=2 (does a taller matmul pay?)."""
import os, sys, time, torch, torch_neuronx
import os as _os_, sys as _sys_
_HERE = _os_.path.dirname(_os_.path.abspath(__file__))
_REPO_SRC = _os_.path.join(_os_.path.dirname(_os_.path.dirname(_HERE)), "src")
_sys_.path.insert(0, _REPO_SRC)
_sys_.path.insert(0, _HERE)
from bench_train import MODEL   # loads contiguous-attn + view-rope patches
from xscript.model import ModelConfig, Block, _rope_cache
dev = torch.device("neuron"); cfg = ModelConfig(**MODEL)
T, D = cfg.max_seq_len, cfg.dim
def sync(): torch_neuronx.synchronize()
cos, sin = _rope_cache(T, cfg.head_dim, cfg.rope_theta, dev, torch.float32)
mbs = [int(a) for a in sys.argv[1:]] or [1, 2]
for mb in mbs:
    blk = Block(cfg).to(dev); blk.compile(backend="neuron", dynamic=False)
    h = torch.randn(mb, T, D, device=dev, requires_grad=True)
    def run():
        with torch.autocast("neuron", dtype=torch.bfloat16):
            out = blk(h, cos, sin)
        out.float().sum().backward(); h.grad = None
    for _ in range(3): run()
    sync(); t0 = time.time()
    for _ in range(10): run()
    sync(); ms = (time.time() - t0) / 10 * 1e3
    print(f"mb={mb}: Block fwd+bwd {ms:.2f} ms -> {ms/mb:.2f} ms per 2048 tokens", flush=True)
print("done", flush=True)
