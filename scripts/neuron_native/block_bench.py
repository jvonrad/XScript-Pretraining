"""Break the 22ms Block fwd+bwd into components at exact shapes (mb=1, T=2048)."""
import sys, time, torch, torch_neuronx
import torch.nn.functional as F
import os as _os_, sys as _sys_
_HERE = _os_.path.dirname(_os_.path.abspath(__file__))
_REPO_SRC = _os_.path.join(_os_.path.dirname(_os_.path.dirname(_HERE)), "src")
_sys_.path.insert(0, _REPO_SRC)
_sys_.path.insert(0, _HERE)
from bench_train import MODEL  # noqa (loads contiguous-rope patch)
from xscript.model import ModelConfig, Attention, SwiGLU, RMSNorm, _rope_cache
import xscript.model as _xm

dev = torch.device("neuron")
cfg = ModelConfig(**MODEL)
T, D, H, HD = cfg.max_seq_len, cfg.dim, cfg.n_heads, cfg.head_dim

def sync(): torch_neuronx.synchronize()

def timeit(fn, iters=10, warm=3):
    for _ in range(warm): fn()
    sync(); t0 = time.time()
    for _ in range(iters): fn()
    sync()
    return (time.time() - t0) / iters * 1e3

def fb(name, mod_or_fn, *inputs, grad_input_idx=0):
    c = torch.compile(mod_or_fn, backend="neuron", dynamic=False)
    def run():
        with torch.autocast("neuron", dtype=torch.bfloat16):
            out = c(*inputs)
        out.float().sum().backward()
        for t in inputs:
            if isinstance(t, torch.Tensor) and t.grad is not None:
                t.grad = None
    ms = timeit(run)
    print(f"{name:32s} {ms:7.2f} ms fwd+bwd", flush=True)
    return ms

x = torch.randn(1, T, D, device=dev, requires_grad=True)
cos, sin = _rope_cache(T, HD, cfg.rope_theta, dev, torch.float32)

attn = Attention(cfg).to(dev)
t_attn = fb("Attention module", attn, x, cos, sin)

ffn = SwiGLU(cfg).to(dev)
t_ffn = fb("SwiGLU module", ffn, x)

norm = RMSNorm(D, cfg.norm_eps).to(dev)
t_norm = fb("RMSNorm (x1)", norm, x)

q = torch.randn(1, H, T, HD, device=dev, requires_grad=True)
k = torch.randn(1, H, T, HD, device=dev, requires_grad=True)
v = torch.randn(1, H, T, HD, device=dev, requires_grad=True)
t_sdpa = fb("SDPA alone (causal)", lambda a,b,c_: F.scaled_dot_product_attention(a,b,c_,is_causal=True), q, k, v)

t_rope = fb("rope alone (1 tensor)", lambda a: _xm._apply_rope(a, cos, sin), q)

qkvo = torch.nn.Linear(D, D, bias=False).to(dev)
t_proj = fb("one proj matmul", qkvo, x)

est = t_attn + t_ffn + 2*t_norm
print(f"\nattn {t_attn:.1f} + ffn {t_ffn:.1f} + 2xnorm {2*t_norm:.1f} = {est:.1f} ms (Block measured ~22.1)", flush=True)
print(f"inside attention: sdpa {t_sdpa:.1f}, rope x2 ~{2*t_rope:.1f}, projs x4 ~{4*t_proj:.1f}", flush=True)
print("done", flush=True)
