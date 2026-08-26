"""lm_head + CE formulations, compiled fwd+bwd, mb=1 (N=2048 tokens)."""
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

fb("F.cross_entropy(logits.float()) [baseline]", lambda a, W, t: F.cross_entropy((a @ W.t()).float(), t))
def lse_gather(a, W, t):
    lg = (a @ W.t()).float()
    lse = torch.logsumexp(lg, dim=-1)
    tl = lg.gather(1, t.long().unsqueeze(1)).squeeze(1)
    return (lse - tl).mean()
fb("logsumexp - gather (fp32)", lse_gather)
def lse_onehot(a, W, t):
    lg = (a @ W.t()).float()
    lse = torch.logsumexp(lg, dim=-1)
    tl = (lg * F.one_hot(t.long(), V).to(lg.dtype)).sum(-1)
    return (lse - tl).mean()
fb("logsumexp - onehot-sum (fp32)", lse_onehot)
def manual_lsm(a, W, t):
    lg = (a @ W.t()).float()
    m = lg.max(-1, keepdim=True).values
    lse = (lg - m).exp().sum(-1).log() + m.squeeze(1)
    tl = lg.gather(1, t.long().unsqueeze(1)).squeeze(1)
    return (lse - tl).mean()
fb("manual max/exp/sum/log (fp32)", manual_lsm)
fb("bf16 logits into CE (PROBE, numerics differ)", lambda a, W, t: F.cross_entropy((a @ W.t()), t))
print("done", flush=True)
