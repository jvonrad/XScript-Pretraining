#!/usr/bin/env python
"""Does grad accumulation across micro-batches work through the compiled blocks + post-accumulate hooks?
Checks master.grad(A then B) == grad(A) + grad(B) per parameter, on the training graph path (world=1)."""
import copy, sys, os, time, numpy as np
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
HOOKS = {"on": True}
def _hook(pm):
    def h(ps):
        if not HOOKS["on"]: return
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
def zero():
    for pm in master.parameters(): pm.grad = None
def run(batches, manual=False):
    zero(); HOOKS["on"] = not manual
    for ps, pm in pairs: ps.grad = None
    for x, y in batches:
        with torch.autocast("neuron", dtype=torch.bfloat16): loss = forward_loss(x, y)
        (loss / len(batches)).backward()
        if manual:
            for ps, pm in pairs:
                g = ps.grad; pm.grad = g.float().clone() if pm.grad is None else pm.grad.add_(g); ps.grad = None
    HOOKS["on"] = True
    return {k: p.grad.detach().cpu().clone() for k, p in master.named_parameters()}
gA, gB, gC = run([A]), run([B]), run([C])
gABC = run([A, B, C]); gABC_manual = run([A, B, C], manual=True)
torch.save({"gA": gA, "gB": gB, "gC": gC, "gABC": gABC, "gABC_manual": gABC_manual}, "/mnt/scratch/xscript/diag/accum.pt")
worst = 1.0; worst_k = None; ratios = []
for k in gA:
    ref = (gA[k] + gB[k] + gC[k]) / 3; got = gABC[k]
    cos = torch.dot(ref.flatten(), got.flatten()) / (ref.norm() * got.norm() + 1e-30); r = (got.norm() / (ref.norm() + 1e-30)).item()
    ratios.append(r)
    if cos < worst: worst, worst_k = cos.item(), k
print(f"[accum] worst cosine(acc(A,B,C), mean(gA,gB,gC)) = {worst:.5f} at {worst_k}; norm ratio mean={np.mean(ratios):.4f} min={min(ratios):.4f} max={max(ratios):.4f}", flush=True)
import re
fam = lambda k: re.sub(r"layers\.\d+\.", "layers.*.", k)
for name, got_all in (("hooks", gABC), ("manual", gABC_manual)):
    rows = {}
    for k in gA:
        ref = (gA[k] + gB[k] + gC[k]) / 3; got = got_all[k]
        cos = (torch.dot(ref.flatten(), got.flatten()) / (ref.norm() * got.norm() + 1e-30)).item()
        cA = (torch.dot((gA[k]/3).flatten(), got.flatten()) / ((gA[k]/3).norm() * got.norm() + 1e-30)).item()
        cC = (torch.dot((gC[k]/3).flatten(), got.flatten()) / ((gC[k]/3).norm() * got.norm() + 1e-30)).item()
        rows.setdefault(fam(k), []).append((cos, (got.norm()/(ref.norm()+1e-30)).item(), cA, cC))
    print(f"[accum:{name}] family: cos(vs mean) min | ratio min/max | cos(vs gA/3) mean | cos(vs gC/3) mean")
    for f, v in rows.items():
        print(f"   {f:30s} {min(x[0] for x in v):.4f} | {min(x[1] for x in v):.3f}/{max(x[1] for x in v):.3f} | {np.mean([x[2] for x in v]):.3f} | {np.mean([x[3] for x in v]):.3f}", flush=True)
# also: is a second single-batch pass deterministic?
gA2 = run([A]); print(f"[accum] repeat A: max rel diff = {max(((gA[k]-gA2[k]).norm()/(gA[k].norm()+1e-30)).item() for k in gA):.2e}", flush=True)
print("DIAG_ACCUM_DONE", flush=True)
