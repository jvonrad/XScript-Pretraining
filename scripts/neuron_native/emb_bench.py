"""Embedding fwd+bwd (2048 tokens, 65536x2048 table), compiled, both bwd paths."""
import os, sys, time, torch, torch_neuronx
dev = torch.device("neuron")
V, D, T = 65536, 2048, 2048
emb = torch.nn.Embedding(V, D).to(dev)
idx = torch.randint(0, V, (1, T), dtype=torch.int32, device=dev)
c = torch.compile(lambda i: emb(i), backend="neuron", dynamic=False)
def sync(): torch_neuronx.synchronize()
def run():
    with torch.autocast("neuron", dtype=torch.bfloat16):
        out = c(idx)
    out.float().sum().backward(); emb.weight.grad = None
for _ in range(3): run()
sync(); t0 = time.time()
for _ in range(10): run()
sync()
print(f"embedding fwd+bwd ({os.environ.get('TORCH_NEURONX_EMBEDDING_BWD_NKI_THRESHOLD','default')}): {(time.time()-t0)/10*1e3:.2f} ms", flush=True)
