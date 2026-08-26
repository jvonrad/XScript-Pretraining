"""Runs NEURON.md §10e's reference implementation verbatim, with bench_train.py's
seeding, and checks it reproduces the canonical loss trace and throughput."""
import os, sys, time, json
sys.path.insert(0, "/repo/src"); sys.path.insert(0, "/bench")
import torch
rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1"))
torch.manual_seed(1 + rank)
from xscript.model import ModelConfig
cfg = ModelConfig(vocab_size=65536, dim=2048, n_layers=16, n_heads=16, n_kv_heads=16,
                  ffn_dim=5632, max_seq_len=2048, rope_theta=10000.0, norm_eps=1e-5)
# --- bench_train.py's attention patch (recipe item 3) ---
import xscript.model as _xm, torch.nn.functional as F
def _attn(self, x, cos, sin):
    B, T, _ = x.shape
    q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
    k = self.wk(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()
    v = self.wv(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()
    q = _xm._apply_rope(q, cos, sin); k = _xm._apply_rope(k, cos, sin)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))
_xm.Attention.forward = _attn
# ---- one-time setup (torchrun sets RANK/WORLD_SIZE; NEURON_RT_NUM_CORES=4) ----
import copy, torch, torch.distributed as dist, torch.nn.functional as F, torch_neuronx
from torch.distributed.optim import ZeroRedundancyOptimizer
import xscript.model as M
from rope_fix import apply_rope_viewbased
M._apply_rope = apply_rope_viewbased          # item 2 (item 3 = .contiguous() in Attention.forward)
dist.init_process_group(backend="neuron")
dev = torch.device(f"neuron:{torch_neuronx.current_device()}")
master = M.Transformer(cfg).to(dev)            # fp32; owns the optimizer state
optim = ZeroRedundancyOptimizer(master.parameters(), optimizer_class=torch.optim.AdamW,
                                lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
# bf16 shadow model: Linear weights bf16, everything else SHARES the fp32 storage
shadow = copy.deepcopy(master)
lin = {id(m.weight) for m in shadow.modules() if isinstance(m, torch.nn.Linear)}
pairs = []
for ps, pm in zip(shadow.parameters(), master.parameters()):
    ps.data = pm.data.to(torch.bfloat16) if id(ps) in lin else pm.data
    pairs.append((ps, pm))
def _hook(pm):                                  # bf16 grad -> fp32 master .grad, freed immediately
    def h(ps):
        g = ps.grad
        pm.grad = (g.float() if g.dtype != torch.float32 else g.clone()) if pm.grad is None else pm.grad.add_(g)
        ps.grad = None
    return h
for ps, pm in pairs: ps.register_post_accumulate_grad_hook(_hook(pm))
for blk in shadow.layers: blk.compile(backend="neuron", dynamic=False)      # item 1: per-Block graphs
emb = torch.compile(lambda w, idx: F.embedding(idx, w), backend="neuron", dynamic=False)
def _tail(x, nw, eps, w, t):                    # final RMSNorm + lm_head + lse-gather CE, one graph
    f = x.float(); f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    lg = ((f * nw.float()).to(x.dtype).reshape(-1, x.size(-1)) @ w.t()).float()
    lse = torch.logsumexp(lg, -1); tl = lg.gather(1, t.clamp(min=0).long().unsqueeze(1)).squeeze(1)
    valid = (t != -100).to(lg.dtype); return ((lse - tl) * valid).sum() / valid.sum().clamp(min=1.0)
tail = torch.compile(_tail, backend="neuron", dynamic=False)
def forward_loss(idx, tgt):                     # idx, tgt: int32 [mb, T]
    x = emb(shadow.tok_emb.weight, idx)
    cos, sin = shadow._rope_for(idx.shape[1], idx.device, x.dtype)
    for blk in shadow.layers: x = blk(x, cos, sin)
    return tail(x, shadow.norm.weight, shadow.norm.eps, shadow.lm_head.weight, tgt.reshape(-1))

# --- bench_train.py's synthetic data + bookkeeping ---
mb, T = 2, 2048
per_step = max(1, round(1.0e6 / T)); unit = mb * world
grad_accum = (max(unit, (per_step // unit) * unit)) // unit
g = torch.Generator().manual_seed(5678 + rank)
pool = []
for _ in range(4):
    w = torch.randint(0, cfg.vocab_size, (mb, T + 1), generator=g, dtype=torch.int32)
    pool.append((w[:, :-1].contiguous().to(dev), w[:, 1:].contiguous().to(dev)))
def run_step(i):
    global micro_batches
    micro_batches = [pool[(i * grad_accum + j) % len(pool)] for j in range(grad_accum)]
    loss_acc = None
    # ---- one optimizer step (mb=2, grad_accum=61 at world=4 -> 999,424 tokens) ----
    optim.zero_grad(set_to_none=True)
    for x, y in micro_batches:
        with torch.autocast("neuron", dtype=torch.bfloat16):
            loss = forward_loss(x, y)
        (loss / grad_accum).backward()              # hooks accumulate into master .grad
    for pm in master.parameters(): dist.all_reduce(pm.grad, op=dist.ReduceOp.AVG)
    torch.nn.utils.clip_grad_norm_(master.parameters(), 1.0)
    optim.step()
    with torch.no_grad():                           # refresh the bf16 shadows ONCE per step
        for ps, pm in pairs:
            if id(ps) in lin: ps.data.copy_(pm.data)
    return None
# step() above discards the loss; re-run its loop with accounting for the check
def one_step(i):
    optim.zero_grad(set_to_none=True); acc = None
    for j in range(grad_accum):
        x, y = pool[(i * grad_accum + j) % len(pool)]
        with torch.autocast("neuron", dtype=torch.bfloat16):
            loss = forward_loss(x, y)
        (loss / grad_accum).backward()
        d = loss.detach() / grad_accum; acc = d if acc is None else acc + d
    for pm in master.parameters(): dist.all_reduce(pm.grad, op=dist.ReduceOp.AVG)
    torch.nn.utils.clip_grad_norm_(master.parameters(), 1.0)
    optim.step()
    with torch.no_grad():
        for ps, pm in pairs:
            if id(ps) in lin: ps.data.copy_(pm.data)
    return float(acc)
tokens_per_step = grad_accum * unit * T
l0 = one_step(0); torch_neuronx.synchronize()
if rank == 0: print(f"[ref] warmup loss {l0:.4f}", flush=True)
t0 = time.time(); losses = []
for i in range(1, 4):
    losses.append(one_step(i)); torch_neuronx.synchronize()
dt = time.time() - t0
if rank == 0:
    tps = tokens_per_step * 3 / dt
    print(f"[ref] step losses {[round(l, 4) for l in losses]}  (canonical: 11.4956, 11.3108, 12.3931)", flush=True)
    print(f"[ref] tok/s {tps:.0f}  MFU {tps * 6.54e9 / 667e12:.3f}  (canonical: ~47100 / 0.462)", flush=True)
