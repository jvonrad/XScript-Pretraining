#!/usr/bin/env python
"""Numerics A/B for the native trainer: forward loss + master grads of a checkpoint on ONE fixed
micro-batch through exactly the training graph path (compiled blocks / emb / tail, bf16 shadow,
fp32 masters, grad hooks). Compare against the CPU fp32 / bf16-autocast reference
(/home/ubuntu/logs/diag_cpu_ref.py) with diag_compare.py.

    torchrun --nproc_per_node 1 ... scripts/neuron_native/diag_native.py <ckpt.pt> <out.pt> [--no-compile]
"""
import copy, sys, os, time
from pathlib import Path
_HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parents[1] / "src"))
import torch, torch.distributed as dist, torch.nn.functional as F, torch_neuronx
import train_native as tn          # applies the rope/attention monkeypatches at import
import xscript.model as M

ckpt, out = sys.argv[1], sys.argv[2]; compile_ = "--no-compile" not in sys.argv
dist.init_process_group(backend="neuron")
dev = torch.device(f"neuron:{torch_neuronx.current_device()}")
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
late_load = "--late-load" in sys.argv     # compile on the seed-0 init, load the checkpoint AFTER compiling (as a resumed/started run does)
torch.manual_seed(0)
master = M.Transformer(M.ModelConfig(**ck["cfg"]["model"]))
if not late_load: master.load_state_dict(ck["model"])
master = master.to(dev)
shadow = copy.deepcopy(master)
lin = {id(m.weight) for m in shadow.modules() if isinstance(m, torch.nn.Linear)}
pairs = []
for ps, pm in zip(shadow.parameters(), master.parameters()):
    ps.data = pm.data.to(torch.bfloat16) if id(ps) in lin else pm.data; pairs.append((ps, pm))
def _hook(pm):
    def h(ps):
        g = ps.grad; pm.grad = (g.float() if g.dtype != torch.float32 else g.clone()) if pm.grad is None else pm.grad.add_(g); ps.grad = None
    return h
for ps, pm in pairs: ps.register_post_accumulate_grad_hook(_hook(pm))
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
def refresh_shadow():
    with torch.no_grad():
        for ps, pm in pairs:
            if id(ps) in lin: ps.data.copy_(pm.data)
batch = torch.load("/mnt/scratch/xscript/diag/batch.pt")
if late_load:
    x0 = batch[:, :-1].contiguous().to(torch.int32).to(dev); y0 = batch[:, 1:].contiguous().to(torch.int32).to(dev)
    with torch.autocast("neuron", dtype=torch.bfloat16):
        l0 = forward_loss(x0, y0)
    l0.backward(); print(f"[diag] init-weights loss={l0.item():.4f} (compiled on init)", flush=True)
    for pm in master.parameters(): pm.grad = None
    with torch.no_grad(): master.load_state_dict(ck["model"])
    refresh_shadow()
x = batch[:, :-1].contiguous().to(torch.int32).to(dev); y = batch[:, 1:].contiguous().to(torch.int32).to(dev)
t0 = time.time()
with torch.autocast("neuron", dtype=torch.bfloat16):
    loss = forward_loss(x, y)
loss.backward()
lv = loss.item(); print(f"[diag] loss={lv:.5f} ({time.time()-t0:.0f}s incl. compile)", flush=True)
# second forward on the same batch (checks determinism / graph reuse)
with torch.no_grad(), torch.autocast("neuron", dtype=torch.bfloat16):
    lv2 = forward_loss(x, y).item()
print(f"[diag] loss again={lv2:.5f}", flush=True)
grads = {k: p.grad.detach().cpu() for k, p in master.named_parameters()}
gn = torch.sqrt(sum((g.float()**2).sum() for g in grads.values())).item()
print(f"[diag] grad_norm={gn:.4f}", flush=True)
torch.save({"loss": lv, "loss2": lv2, "grads": grads}, out)
print("DIAG_NATIVE_DONE", flush=True)

# ---- optimizer-side checks: clip norm on device, one AdamW(ZeRO) step from fresh state ----
from torch.distributed.optim import ZeroRedundancyOptimizer
before = {k: p.detach().cpu().clone() for k, p in master.named_parameters()}
tn_ = torch.nn.utils.clip_grad_norm_(master.parameters(), 1.0)
print(f"[diag] device clip_grad_norm_ total_norm={float(tn_):.5f}", flush=True)
clipped = {k: p.grad.detach().cpu() for k, p in master.named_parameters()}
gn2 = torch.sqrt(sum((g.float()**2).sum() for g in clipped.values())).item()
print(f"[diag] grad norm after clip (cpu-computed)={gn2:.5f}", flush=True)
optim = ZeroRedundancyOptimizer(master.parameters(), optimizer_class=torch.optim.AdamW, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
for g in optim.param_groups: g["lr"] = 3e-3
for g in optim.optim.param_groups: g["lr"] = 3e-3
optim.step()
delta = {k: (p.detach().cpu() - before[k]) for k, p in master.named_parameters()}
torch.save({"clipped": clipped, "delta": delta, "total_norm": float(tn_)}, out.replace(".pt", "_optstep.pt"))
print("DIAG_OPT_DONE", flush=True)
