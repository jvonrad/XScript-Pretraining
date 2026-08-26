"""Device+runtime profile of ONE micro (fwd+bwd) of the best config on a single
rank: per-Block compiled graphs, compiled emb/tail, bf16 shadow weights with
hook accumulation. Reports device-busy vs wall to expose idle gaps."""
import os, sys, time, json, torch, torch_neuronx
sys.path.insert(0, "/bench"); sys.path.insert(0, "/repo/src")
from torch.profiler import ProfilerActivity, profile
from torch_neuronx.profiling import NeuronConfig, ProfileMode
import bench_train as bt
from xscript.model import ModelConfig, Transformer
import copy

dev = torch.device("neuron")
cfg = ModelConfig(**bt.MODEL)
torch.manual_seed(1)
raw = Transformer(cfg).to(dev)
shadow = copy.deepcopy(raw)
lin_s, lin_m = [], []
for (n_s, m_s), (n_m, m_m) in zip(shadow.named_modules(), raw.named_modules()):
    if isinstance(m_s, torch.nn.Linear):
        m_s.weight.data = m_m.weight.data.to(torch.bfloat16); lin_s.append(m_s.weight); lin_m.append(m_m.weight)
sl = set(id(p) for p in lin_s)
for p_s, p_m in zip(shadow.parameters(), raw.parameters()):
    if id(p_s) not in sl: p_s.data = p_m.data
for layer in shadow.layers: layer.compile(backend="neuron", dynamic=False)
def hook(param):
    param.grad = None
for p in shadow.parameters(): p.register_post_accumulate_grad_hook(hook)   # drop grads (accumulation cost measured separately)
mb, T = 2, cfg.max_seq_len
g = torch.Generator().manual_seed(5678)
w = torch.randint(0, cfg.vocab_size, (mb, T + 1), generator=g, dtype=torch.int32)
x, y = w[:, :-1].contiguous().to(dev), w[:, 1:].contiguous().to(dev)
def micro():
    with torch.autocast("neuron", dtype=torch.bfloat16):
        loss = bt.forward_loss(shadow, x, y, False, 0, True, None, 0, ce_lse=True, tail_compile=True)
    loss.backward()
for _ in range(3): micro()
torch_neuronx.synchronize()
t0 = time.time(); 
for _ in range(5): micro()
torch_neuronx.synchronize(); wall = (time.time() - t0) / 5
print(f"micro wall (no accumulation): {wall*1e3:.1f} ms", flush=True)
out = "/bench/prof_step"; os.makedirs(out, exist_ok=True)
cfgp = NeuronConfig(modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME], profile_output_dir=out, max_events_per_nc=500000, correlate_device_time=True)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1], experimental_config=cfgp) as prof:
    micro(); torch_neuronx.synchronize()
prof.export_chrome_trace(os.path.join(out, "trace.json"))
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25), flush=True)
# device busy from the chrome trace: sum of device-track event durations vs span
try:
    tr = json.load(open(os.path.join(out, "trace.json")))
    ev = [e for e in tr.get("traceEvents", []) if e.get("ph") == "X" and "dur" in e]
    devs = [e for e in ev if any(k in str(e.get("cat", "")).lower() + str(e.get("name", "")).lower() for k in ("neuron", "device", "nc", "kernel", "neff"))]
    if devs:
        t_start = min(e["ts"] for e in devs); t_end = max(e["ts"] + e["dur"] for e in devs)
        busy = sum(e["dur"] for e in devs)
        cats = {}
        for e in devs: cats[e.get("cat", "?")] = cats.get(e.get("cat", "?"), 0) + e["dur"]
        print(f"device-ish events: {len(devs)}, span {(t_end-t_start)/1e3:.1f} ms, summed dur {busy/1e3:.1f} ms", flush=True)
        for k, v in sorted(cats.items(), key=lambda kv: -kv[1])[:10]: print(f"  cat {k}: {v/1e3:.1f} ms", flush=True)
    print("trace events total:", len(ev), flush=True)
except Exception as e:
    print("trace parse failed:", repr(e)[:200], flush=True)
print("done", flush=True)
