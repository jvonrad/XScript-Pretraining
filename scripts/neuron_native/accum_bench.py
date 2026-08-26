"""Cost of gradient accumulation on this backend: eager per-param add_ (what
AccumulateGrad does) vs one _foreach_add_ vs a single big add, over tensors
shaped like the model's 219 params (1.09B fp32)."""
import sys, time, torch, torch_neuronx
sys.path.insert(0, "/bench"); sys.path.insert(0, "/repo/src")
from bench_train import MODEL
from xscript.model import ModelConfig, Transformer
dev = torch.device("neuron")
m = Transformer(ModelConfig(**MODEL))
shapes = [p.shape for p in m.parameters()]
del m
acc = [torch.zeros(s, device=dev) for s in shapes]
new = [torch.ones(s, device=dev) for s in shapes]
def sync(): torch_neuronx.synchronize()
def timeit(fn, iters=5):
    for _ in range(2): fn()
    sync(); t0 = time.time()
    for _ in range(iters): fn()
    sync(); return (time.time()-t0)/iters*1e3
n = sum(s.numel() for s in shapes)
print(f"params: {len(shapes)} tensors, {n/1e9:.2f}B fp32 = {n*4/2**30:.2f} GiB", flush=True)
print(f"eager per-tensor add_ x{len(shapes)}:   {timeit(lambda: [a.add_(b) for a, b in zip(acc, new)]):.1f} ms", flush=True)
try:
    print(f"torch._foreach_add_ (1 call):        {timeit(lambda: torch._foreach_add_(acc, new)):.1f} ms", flush=True)
except Exception as e:
    print("foreach_add_ FAILED:", repr(e)[:120], flush=True)
big_a = torch.zeros(n, device=dev); big_b = torch.ones(n, device=dev)
print(f"single flat add_ (4.4GB):            {timeit(lambda: big_a.add_(big_b)):.1f} ms", flush=True)
c = torch.compile(lambda a, b: a + b, backend="neuron", dynamic=False)
print(f"compiled flat add (4.4GB):           {timeit(lambda: c(big_a, big_b)):.1f} ms", flush=True)
print("done", flush=True)
