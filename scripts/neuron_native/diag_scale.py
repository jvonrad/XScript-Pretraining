#!/usr/bin/env python
"""Gradient accuracy vs the scalar loss scale on the compiled training path: backward of (loss*s) for several s,
compared (after dividing by s) with the CPU fp32 reference gradient of the same batch."""
import copy, sys, os, time, numpy as np, re
from pathlib import Path
_HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parents[1] / "src"))
import torch, torch.distributed as dist, torch.nn.functional as F, torch_neuronx
import train_native as tn
import xscript.model as M
ckpt = sys.argv[1]
dist.init_process_group(backend="neuron"); dev = torch.device(f"neuron:{torch_neuronx.current_device()}")
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
master = M.Transformer(M.ModelConfig(**ck["cfg"]["model"])); master.load_state_dict(ck["model"]); master = master.to(dev)
shadow = copy.deepcopy(master); lin = {id(m.weight) for m in shadow.modules() if isinstance(m, torch.nn.Linear)}; pairs = []
for ps, pm in zip(shadow.parameters(), master.parameters()):
    ps.data = pm.data.to(torch.bfloat16) if id(ps) in lin else pm.data; pairs.append((ps, pm))
for blk in shadow.layers: blk.compile(backend="neuron", dynamic=False)
emb = torch.compile(lambda w, idx: F.embedding(idx, w), backend="neuron", dynamic=False)
tail = torch.compile(tn._tail, backend="neuron", dynamic=False)
def forward_loss(idx, tgt):
    x = emb(shadow.tok_emb.weight, idx); cos, sin = shadow._rope_for(idx.shape[1], idx.device, x.dtype)
    for blk in shadow.layers: x = blk(x, cos, sin)
    return tail(x, shadow.norm.weight, shadow.norm.eps, shadow.lm_head.weight, tgt.reshape(-1))
batch = torch.load("/mnt/scratch/xscript/diag/batch.pt")
x = batch[:, :-1].contiguous().to(torch.int32).to(dev); y = batch[:, 1:].contiguous().to(torch.int32).to(dev)
ref = torch.load("/mnt/scratch/xscript/diag/ref_fp32.pt")["grads"]
NAMES = [k for k, _ in master.named_parameters()]
fam = lambda k: re.sub(r"layers\.\d+\.", "layers.*.", k)
def grad_at(scale, mode):
    for ps, pm in pairs: ps.grad = None
    with torch.autocast("neuron", dtype=torch.bfloat16): loss = forward_loss(x, y)
    if mode == "div": (loss / (1.0 / scale)).backward()
    elif mode == "mul": (loss * scale).backward()
    elif mode == "gradarg": loss.backward(gradient=torch.tensor(scale, device=dev, dtype=loss.dtype))
    return {k: (ps.grad.detach().double().cpu() / scale) for k, (ps, pm) in zip(NAMES, pairs)}
for scale, mode in ((1.0, "div"), (1/3, "div"), (1/15, "div"), (1/15, "mul"), (1/15, "gradarg"), (1/16, "div"), (1/3, "gradarg")):
    g = grad_at(scale, mode); rows = {}
    for k in g:
        a, b = g[k].flatten(), ref[k].double().flatten()
        rows.setdefault(fam(k), []).append(((a @ b / (a.norm() * b.norm() + 1e-300)).item(), (a.norm() / (b.norm() + 1e-300)).item()))
    print(f"[scale {scale:.4f} {mode:7s}] worst cos={min(min(c for c,_ in v) for v in rows.values()):.5f}; " +
          " ".join(f"{f.split('.')[-2] if 'layers' in f else f.split('.')[0]}={min(c for c,_ in v):.4f}/{min(r for _,r in v):.3f}/{max(r for _,r in v):.3f}" for f, v in rows.items()), flush=True)
print("DIAG_SCALE_DONE", flush=True)
