import sys, time, torch, torch_neuronx
import torch.nn.functional as F
import os as _os_, sys as _sys_
_HERE = _os_.path.dirname(_os_.path.abspath(__file__))
_REPO_SRC = _os_.path.join(_os_.path.dirname(_os_.path.dirname(_HERE)), "src")
_sys_.path.insert(0, _REPO_SRC)
_sys_.path.insert(0, _HERE)
from bench_train import MODEL
dev = torch.device("neuron")
N, D, V = MODEL["max_seq_len"], MODEL["dim"], MODEL["vocab_size"]
def sync(): torch_neuronx.synchronize()
w = torch.randn(V, D, device=dev, requires_grad=True)
hx = torch.randn(N, D, device=dev, requires_grad=True)
tgt = torch.randint(0, V, (N,), dtype=torch.int32, device=dev)
def fb(name, fn, iters=10):
    c = torch.compile(fn, backend="neuron", dynamic=False)
    def run():
        with torch.autocast("neuron", dtype=torch.bfloat16):
            loss = c(hx, w, tgt)
        loss.backward(); hx.grad = None; w.grad = None
    for _ in range(3): run()
    sync(); t0 = time.time()
    for _ in range(iters): run()
    sync(); ms = (time.time()-t0)/iters*1e3
    with torch.autocast("neuron", dtype=torch.bfloat16):
        v = float(c(hx, w, tgt).detach())
    print(f"{name:44s} {ms:7.2f} ms   loss={v:.6f}", flush=True)
def lse_gather(a, W, t):
    lg = (a @ W.t()).float()
    return (torch.logsumexp(lg, -1) - lg.gather(1, t.long().unsqueeze(1)).squeeze(1)).mean()
def lse_bf16keep(a, W, t):
    lgb = a @ W.t()                                   # bf16
    lse = torch.logsumexp(lgb.float(), -1)            # upcast inside the reduction
    tl = lgb.gather(1, t.long().unsqueeze(1)).squeeze(1).float()
    return (lse - tl).mean()
def lse_bf16keep_ns(a, W, t):
    lgb = a @ W.t()
    lse = torch.logsumexp(lgb.float(), -1)
    tl = (lgb * F.one_hot(t.long(), V).to(lgb.dtype)).sum(-1).float()
    return (lse - tl).mean()
fb("lse-gather (current)", lse_gather)
fb("lse(float inside) + bf16 gather", lse_bf16keep)
fb("lse(float inside) + bf16 onehot", lse_bf16keep_ns)
print("done", flush=True)
