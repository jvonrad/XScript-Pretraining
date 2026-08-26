"""Standalone compiled fwd+bwd timing of glue formulations (mb=1, T=2048)."""
import sys, time, torch, torch_neuronx
import torch.nn.functional as F
import os as _os_, sys as _sys_
_HERE = _os_.path.dirname(_os_.path.abspath(__file__))
_REPO_SRC = _os_.path.join(_os_.path.dirname(_os_.path.dirname(_HERE)), "src")
_sys_.path.insert(0, _REPO_SRC)
_sys_.path.insert(0, _HERE)
from bench_train import MODEL
from xscript.model import ModelConfig
dev = torch.device("neuron"); cfg = ModelConfig(**MODEL)
T, D, FF = cfg.max_seq_len, cfg.dim, cfg.ffn_dim
def sync(): torch_neuronx.synchronize()
def fb(name, fn, *inputs, iters=10):
    c = torch.compile(fn, backend="neuron", dynamic=False)
    def run():
        with torch.autocast("neuron", dtype=torch.bfloat16):
            out = c(*inputs)
        out.float().sum().backward()
        for t in inputs:
            if isinstance(t, torch.Tensor) and t.grad is not None: t.grad = None
    for _ in range(3): run()
    sync(); t0 = time.time()
    for _ in range(iters): run()
    sync(); print(f"{name:40s} {(time.time()-t0)/iters*1e3:7.2f} ms", flush=True)

x = torch.randn(1, T, D, device=dev, requires_grad=True)
w1 = torch.nn.Linear(D, FF, bias=False).to(dev); w3 = torch.nn.Linear(D, FF, bias=False).to(dev); w2 = torch.nn.Linear(FF, D, bias=False).to(dev)
fb("swiglu: w2(silu(w1x)*w3x)  [baseline]", lambda a: w2(F.silu(w1(a)) * w3(a)), x)
fb("swiglu: g*sigmoid(g)*u", lambda a: w2((lambda g, u: g * torch.sigmoid(g) * u)(w1(a), w3(a))), x)
fb("swiglu: silu in fp32", lambda a: w2((F.silu(w1(a).float()) * w3(a).float()).to(torch.bfloat16)), x)
fb("swiglu: matmuls only (w2(w1x*w3x))", lambda a: w2(w1(a) * w3(a)), x)
fb("swiglu: w2(w1x) (no gate at all)", lambda a: w2(w1(a)), x)

wn = torch.nn.Parameter(torch.ones(D, device=dev))
eps = cfg.norm_eps
def rms_ref(a):
    dt = a.dtype; f = a.float(); f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps); return (f * wn.float()).to(dt)
def rms_square(a):
    dt = a.dtype; f = a.float(); f = f * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps); return (f * wn.float()).to(dt)
ones = torch.full((D, 1), 1.0 / D, device=dev)
def rms_mm(a):
    dt = a.dtype; f = a.float(); ms = f.square() @ ones; f = f * torch.rsqrt(ms + eps); return (f * wn.float()).to(dt)
def rms_bf16stat(a):
    # statistic in fp32 but normalize in bf16 (NOT identical numerics) -- ceiling probe
    f = a.float(); r = torch.rsqrt(f.square().mean(-1, keepdim=True) + eps); return (a * r.to(a.dtype)) * wn.to(a.dtype)
xb = torch.randn(1, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
fb("rmsnorm: reference (pow.mean)", rms_ref, xb)
fb("rmsnorm: square().mean", rms_square, xb)
fb("rmsnorm: stat via matmul", rms_mm, xb)
fb("rmsnorm: bf16 normalize (probe)", rms_bf16stat, xb)
print("done", flush=True)
