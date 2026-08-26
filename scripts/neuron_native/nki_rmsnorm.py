"""NKI RMSNorm fwd+bwd for (N, D) rows, D on the free axis.

Forward matches model.py RMSNorm op-for-op: r = rsqrt(mean(x^2) + eps) in
fp32, out = (x*r) * w cast back to x.dtype. Backward is the closed form
  dx = g*w*r - x * r^3 * dot/D,   dot = sum_j g_j w_j x_j
  dw_j = sum_rows g_ij x_ij r_i
mathematically identical to autograd's decomposition but fp-reordered (NOT
bitwise). dw is emitted as per-tile partials; the wrapper sums them.
"""
import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch_neuronx import nki_op, wrap_nki

TILE = 128


@wrap_nki
@nki.jit
def _rmsnorm_fwd_kernel(x, w, eps):
    # x: (N, D); w: (1, D). Returns out (N, D) in x.dtype and r (N, 1) fp32.
    N, D = x.shape
    out = nl.ndarray((N, D), dtype=x.dtype, buffer=nl.shared_hbm)
    r_out = nl.ndarray((N, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    wt = nl.load(w[0:1, :], dtype=nl.float32)
    wtb = nl.broadcast_to(wt, (TILE, D))
    for i in nl.affine_range(N // TILE):
        xt = nl.load(x[nl.ds(i * TILE, TILE), :], dtype=nl.float32)
        sq = nl.ndarray((TILE, D), dtype=nl.float32, buffer=nl.sbuf)
        ssum = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=xt,
                        reduce_op=nl.add, reduce_res=ssum,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        r = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=r, op=nl.rsqrt, data=ssum, scale=1.0 / D, bias=eps)
        xn = nl.ndarray((TILE, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=xn, data=xt, op0=nl.multiply, operand0=r)
        res = nl.ndarray((TILE, D), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=res, data1=xn, data2=wtb, op=nl.multiply)
        nl.store(out[nl.ds(i * TILE, TILE), :], value=res)
        nl.store(r_out[nl.ds(i * TILE, TILE), :], value=r)
    return out, r_out


@wrap_nki
@nki.jit
def _rmsnorm_bwd_kernel(g, x, w, r):
    # g, x: (N, D); w: (1, D); r: (N, 1) fp32 from forward.
    N, D = x.shape
    dx = nl.ndarray((N, D), dtype=g.dtype, buffer=nl.shared_hbm)
    dw_part = nl.ndarray((N // TILE, D), dtype=nl.float32, buffer=nl.shared_hbm)
    wt = nl.load(w[0:1, :], dtype=nl.float32)
    wtb = nl.broadcast_to(wt, (TILE, D))
    for i in nl.affine_range(N // TILE):
        gt = nl.load(g[nl.ds(i * TILE, TILE), :], dtype=nl.float32)
        xt = nl.load(x[nl.ds(i * TILE, TILE), :], dtype=nl.float32)
        rt = nl.load(r[nl.ds(i * TILE, TILE), :], dtype=nl.float32)
        gw = nl.ndarray((TILE, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gw, data1=gt, data2=wtb, op=nl.multiply)
        gwx = nl.ndarray((TILE, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gwx, data1=gw, data2=xt, op=nl.multiply)
        dot = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=dot, op=nl.add, data=gwx, axis=1)
        r2 = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r2, data1=rt, data2=rt, op=nl.multiply)
        r3 = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r3, data1=r2, data2=rt, op=nl.multiply)
        s2 = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=s2, data1=dot, data2=r3, op=nl.multiply)
        s2d = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=s2d, data=s2, op0=nl.multiply, operand0=1.0 / D)
        t1 = nl.ndarray((TILE, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=t1, data=gw, op0=nl.multiply, operand0=rt)
        t2 = nl.ndarray((TILE, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=t2, data=xt, op0=nl.multiply, operand0=s2d)
        res = nl.ndarray((TILE, D), dtype=g.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=res, data1=t1, data2=t2, op=nl.subtract)
        nl.store(dx[nl.ds(i * TILE, TILE), :], value=res)
        gx = nl.ndarray((TILE, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gx, data1=gt, data2=xt, op=nl.multiply)
        gxr = nl.ndarray((TILE, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gxr, data=gx, op0=nl.multiply, operand0=rt)
        dwp = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_partition_reduce(dst=dwp, data=gxr, op=nl.add)
        nl.store(dw_part[nl.ds(i, 1), :], value=dwp)
    return dx, dw_part


@nki_op("xscript::rmsnorm_fwd", mutates_args={})
def rmsnorm_fwd(x: torch.Tensor, w: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    return _rmsnorm_fwd_kernel(x, w, eps)


@nki_op("xscript::rmsnorm_bwd", mutates_args={})
def rmsnorm_bwd(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _rmsnorm_bwd_kernel(g, x, w, r)


class _NKIRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        out, r = rmsnorm_fwd(x2, w.reshape(1, -1), eps)
        ctx.save_for_backward(x2, w, r)
        ctx.shp = shp
        return out.view(shp)

    @staticmethod
    def backward(ctx, g):
        x2, w, r = ctx.saved_tensors
        dx, dw_part = rmsnorm_bwd(g.contiguous().reshape(x2.shape), x2,
                                  w.reshape(1, -1), r)
        return dx.view(ctx.shp), dw_part.sum(dim=0).to(w.dtype), None


def rmsnorm_nki_forward(self, x):
    """Drop-in replacement for model.py RMSNorm.forward."""
    return _NKIRMSNorm.apply(x, self.weight, self.eps)


if __name__ == "__main__":
    import os, sys, time
    import torch_neuronx
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
    from xscript.model import RMSNorm

    dev = torch.device("neuron")
    D, N = 2048, 2048
    ref_mod = RMSNorm(D, 1e-5).to(dev)
    with torch.no_grad():
        ref_mod.weight.mul_(0)
        ref_mod.weight.add_(torch.randn(D, device=dev))

    for dt in (torch.float32, torch.bfloat16):
        x = torch.randn(1, N, D, device=dev, dtype=dt, requires_grad=True)
        go = torch.randn(1, N, D, device=dev, dtype=dt)
        ref = ref_mod(x)
        ref.backward(go)
        gref_x = x.grad.detach().float().cpu(); x.grad = None
        gref_w = ref_mod.weight.grad.detach().float().cpu(); ref_mod.weight.grad = None
        new = _NKIRMSNorm.apply(x, ref_mod.weight, ref_mod.eps)
        new.backward(go)
        gnew_x = x.grad.detach().float().cpu(); x.grad = None
        gnew_w = ref_mod.weight.grad.detach().float().cpu(); ref_mod.weight.grad = None
        fd = (ref.detach().float().cpu() - new.detach().float().cpu()).abs().max()
        print(f"{dt}: fwd maxdiff {fd:.3e}  dx maxdiff {(gref_x-gnew_x).abs().max():.3e}  "
              f"dw reldiff {((gref_w-gnew_w).abs().max()/gref_w.abs().max()):.3e}", flush=True)

    x = torch.randn(1, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    def sync(): torch_neuronx.synchronize()
    def run_ref():
        ref_mod(x).float().sum().backward(); x.grad = None; ref_mod.weight.grad = None
    def run_new():
        _NKIRMSNorm.apply(x, ref_mod.weight, ref_mod.eps).float().sum().backward()
        x.grad = None; ref_mod.weight.grad = None
    for f, n in ((run_ref, "torch"), (run_new, "nki")):
        for _ in range(3): f()
        sync(); t0 = time.time()
        for _ in range(10): f()
        sync()
        print(f"{n} rmsnorm fwd+bwd: {(time.time()-t0)/10*1e3:.2f} ms", flush=True)
    print("done", flush=True)
