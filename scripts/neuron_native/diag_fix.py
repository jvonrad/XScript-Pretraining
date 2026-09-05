#!/usr/bin/env python
"""Verify the accumulation fix: unscaled backward x N, fp32 master grad / N == single unscaled grad; and vs CPU fp32."""
import copy, sys, os, numpy as np, re
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
def _hook(pm):
    def h(ps):
        g = ps.grad; pm.grad = (g.float() if g.dtype != torch.float32 else g.clone()) if pm.grad is None else pm.grad.add_(g); ps.grad = None
    return h
for ps, pm in pairs: ps.register_post_accumulate_grad_hook(_hook(pm))
for blk in shadow.layers: blk.compile(backend="neuron", dynamic=False)
emb = torch.compile(lambda w, idx: F.embedding(idx, w), backend="neuron", dynamic=False)
tail = torch.compile(tn._tail, backend="neuron", dynamic=False)
def forward_loss(idx, tgt):
    x = emb(shadow.tok_emb.weight, idx); cos, sin = shadow._rope_for(idx.shape[1], idx.device, x.dtype)
    for blk in shadow.layers: x = blk(x, cos, sin)
    return tail(x, shadow.norm.weight, shadow.norm.eps, shadow.lm_head.weight, tgt.reshape(-1))
fr = np.fromfile("/mnt/scratch/xscript/shards/fr__unigram_20lang/pool_00000.bin", dtype=np.uint16, count=32768)
en = np.fromfile("/mnt/scratch/xscript/shards/en__unigram_20lang/pool_00000.bin", dtype=np.uint16, count=32768)
def mk(off):
    b = torch.from_numpy(np.stack([fr[off:off+2049], en[off:off+2049]]).astype(np.int32)); return b[:, :-1].contiguous().to(dev), b[:, 1:].contiguous().to(dev)
A, B, C = mk(4096), mk(12288), mk(20480)
fam = lambda k: re.sub(r"layers\.\d+\.", "layers.*.", k)
def run(batches):
    for pm in master.parameters(): pm.grad = None
    for x, y in batches:
        with torch.autocast("neuron", dtype=torch.bfloat16): loss = forward_loss(x, y)
        loss.backward()                      # unscaled (the fix)
    for pm in master.parameters(): pm.grad.div_(len(batches))
    return {k: p.grad.detach().double().cpu().clone() for k, p in master.named_parameters()}
def cmp(name, got, ref):
    rows = {}
    for k in got:
        a, b = got[k].flatten(), ref[k].double().flatten()
        rows.setdefault(fam(k), []).append(((a @ b / (a.norm() * b.norm() + 1e-300)).item(), (a.norm() / (b.norm() + 1e-300)).item()))
    print(f"[fix] {name}: worst cos={min(min(c for c,_ in v) for v in rows.values()):.5f}; " +
          " ".join(f"{f.split('.')[-2] if 'layers' in f else f.split('.')[0]}={min(c for c,_ in v):.4f}/{min(r for _,r in v):.3f}/{max(r for _,r in v):.3f}" for f, v in rows.items()), flush=True)
gA, gB, gC = run([A]), run([B]), run([C])
cmp("A vs CPU fp32", gA, torch.load("/mnt/scratch/xscript/diag/ref_fp32.pt")["grads"])
cmp("15xA /15 vs A", run([A] * 15), gA)
cmp("[A,B,C] /3 vs mean", run([A, B, C]), {k: (gA[k] + gB[k] + gC[k]) / 3 for k in gA})
print("DIAG_FIX_DONE", flush=True)
