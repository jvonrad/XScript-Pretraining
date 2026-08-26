"""Cost of the (B,H,T,d) <-> (B,H,d,T) transposes the SDPA NKI path does (8 per attention bwd)."""
import time, torch, torch_neuronx
dev = torch.device("neuron")
def sync(): torch_neuronx.synchronize()
x = torch.randn(1, 16, 2048, 128, device=dev, dtype=torch.bfloat16, requires_grad=True)
def t1(a): return a.transpose(-2, -1).contiguous()
def t8(a):
    y = a
    for _ in range(4):
        y = y.transpose(-2, -1).contiguous() * 1.0001
        y = y.transpose(-2, -1).contiguous() * 1.0001
    return y
for name, fn in (("1 transpose fwd+bwd", t1), ("8 transposes fwd+bwd (chained)", t8)):
    c = torch.compile(fn, backend="neuron", dynamic=False)
    def run():
        c(x).float().sum().backward(); x.grad = None
    for _ in range(3): run()
    sync(); t0 = time.time()
    for _ in range(10): run()
    sync(); print(f"{name:34s} {(time.time()-t0)/10*1e3:.3f} ms", flush=True)
print("done", flush=True)
