"""View-based interleaved rope: numerically identical to model.py's
_apply_rope, but no strided slicing / strided assignment — x[..., ::2] is
replaced by x.view(..., hd/2, 2) unbind, and the interleaved write-back by
stack+view. Same elements, same multiplies, same interleaved (GPT-NeoX)
convention."""
import torch


def apply_rope_viewbased(x, cos, sin):
    # x: (B, H, T, D)
    B, H, T, D = x.shape
    xp = x.view(B, H, T, D // 2, 2)
    x1 = xp[..., 0]
    x2 = xp[..., 1]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).view(B, H, T, D)


if __name__ == "__main__":
    import os, sys, time
    import torch_neuronx
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
    from xscript.model import _apply_rope, _rope_cache

    # exactness on CPU (fp32): must be bitwise identical
    torch.manual_seed(0)
    x = torch.randn(2, 16, 128, 128)
    cos, sin = _rope_cache(128, 128, 10000.0, torch.device("cpu"), torch.float32)
    ref = _apply_rope(x, cos, sin)
    new = apply_rope_viewbased(x, cos, sin)
    assert torch.equal(ref, new), (ref - new).abs().max()
    print("CPU bitwise identical: OK", flush=True)

    # exactness + speed on neuron, compiled, fwd+bwd
    dev = torch.device("neuron")
    T, HD = 2048, 128
    cos, sin = _rope_cache(T, HD, 10000.0, dev, torch.float32)
    q = torch.randn(1, 16, T, HD, device=dev)
    ref = _apply_rope(q, cos, sin).cpu()
    new = apply_rope_viewbased(q, cos, sin).cpu()
    print("neuron eager max diff:", (ref - new).abs().max().item(), flush=True)

    qg = q.detach().requires_grad_(True)
    c = torch.compile(apply_rope_viewbased, backend="neuron", dynamic=False)

    def sync(): torch_neuronx.synchronize()
    def run():
        out = c(qg, cos, sin)
        out.float().sum().backward()
        qg.grad = None
    for _ in range(3): run()
    sync(); t0 = time.time()
    for _ in range(10): run()
    sync()
    print(f"view-based rope compiled fwd+bwd: {(time.time()-t0)/10*1e3:.2f} ms (old: 6.74)", flush=True)


def apply_rope_rotate_half(x, cosf, sinf):
    """Rotate-half rope on a HALVES layout (evens in first 64 dims, odds in
    last 64). ONLY valid when wq/wk output rows are permuted per head to that
    layout — then q.k dot products equal the interleaved reference exactly in
    math (accumulation order differs, so not bitwise). cosf/sinf: (T, D) =
    cat([cos, cos], -1) / cat([sin, sin], -1)."""
    D = x.shape[-1]
    h = D // 2
    x1 = x[..., :h]
    x2 = x[..., h:]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cosf[None, None, :, :] + rot * sinf[None, None, :, :]
