#!/usr/bin/env python
"""Isolate the accumulation defect: [A,A,A] vs A; [A,B] vs [B,A]; compiled vs eager. Saves tensors; float64 metrics on CPU."""
import copy, sys, os, time, numpy as np
from pathlib import Path
_HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parents[1] / "src"))
import torch, torch.distributed as dist, torch.nn.functional as F, torch_neuronx
import train_native as tn
import xscript.model as M
ckpt = sys.argv[1]; compile_ = "--eager" not in sys.argv; tag = "eager" if not compile_ else "compiled"
dist.init_process_group(backend="neuron"); dev = torch.device(f"neuron:{torch_neuronx.current_device()}")
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
master = M.Transformer(M.ModelConfig(**ck["cfg"]["model"])); master.load_state_dict(ck["model"]); master = master.to(dev)
shadow = copy.deepcopy(master); lin = {id(m.weight) for m in shadow.modules() if isinstance(m, torch.nn.Linear)}; pairs = []
for ps, pm in zip(shadow.parameters(), master.parameters()):
    ps.data = pm.data.to(torch.bfloat16) if id(ps) in lin else pm.data; pairs.append((ps, pm))
if compile_:
    for blk in shadow.layers: blk.compile(backend="neuron", dynamic=False)
    emb = torch.compile(lambda w, idx: F.embedding(idx, w), backend="neuron", dynamic=False)
    tail = torch.compile(tn._tail, backend="neuron", dynamic=False)
else:
    emb = lambda w, idx: F.embedding(idx, w); tail = tn._tail
def forward_loss(idx, tgt):
    x = emb(shadow.tok_emb.weight, idx); cos, sin = shadow._rope_for(idx.shape[1], idx.device, x.dtype)
    for blk in shadow.layers: x = blk(x, cos, sin)
    return tail(x, shadow.norm.weight, shadow.norm.eps, shadow.lm_head.weight, tgt.reshape(-1))
fr = np.fromfile("/mnt/scratch/xscript/shards/fr__unigram_20lang/pool_00000.bin", dtype=np.uint16, count=32768)
en = np.fromfile("/mnt/scratch/xscript/shards/en__unigram_20lang/pool_00000.bin", dtype=np.uint16, count=32768)
def mk(off):
    b = torch.from_numpy(np.stack([fr[off:off+2049], en[off:off+2049]]).astype(np.int32)); return b[:, :-1].contiguous().to(dev), b[:, 1:].contiguous().to(dev)
A, B = mk(4096), mk(12288)
NAMES = [k for k, _ in master.named_parameters()]
def run(batches, per_mb=False):
    for ps, pm in pairs: ps.grad = None; pm.grad = None
    losses = []; parts = []
    for x, y in batches:
        with torch.autocast("neuron", dtype=torch.bfloat16): loss = forward_loss(x, y)
        (loss / len(batches)).backward(); losses.append(loss.item())
        if per_mb: parts.append({k: ps.grad.detach().float().cpu().clone() for k, (ps, pm) in zip(NAMES, pairs)})
        for ps, pm in pairs:
            g = ps.grad; pm.grad = g.float().clone() if pm.grad is None else pm.grad.add_(g); ps.grad = None
    print(f"[{tag}] losses {['%.4f' % l for l in losses]}", flush=True)
    out = {k: p.grad.detach().cpu().clone() for k, p in master.named_parameters()}
    return (out, parts) if per_mb else out
t0 = time.time()
res = {"A": run([A]), "AA": run([A, A]), "AAA": run([A, A, A]), "AAAA": run([A, A, A, A])}
res["AAA_parts"] = run([A, A, A], per_mb=True)[1]
print(f"[{tag}] done in {time.time()-t0:.0f}s", flush=True)
torch.save(res, f"/mnt/scratch/xscript/diag/accum_{tag}.pt")
print("DIAG_ACCUM2_DONE", flush=True)
