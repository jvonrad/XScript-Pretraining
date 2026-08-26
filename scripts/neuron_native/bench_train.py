"""Throughput replica of the de-starved 1B-param training run on TorchNeuron Native.

Replicates src/xscript/train.py's step mechanics (same model.py import, AdamW
hyperparams, bf16 autocast + fp32 master weights, grad-accum bookkeeping, grad
clipping, tok/s accounting) with the memory levers train_neuron.py ships for
the 24GB-per-core budget: per-Block activation checkpointing (--ckpt), chunked
lm_head+CE (--ce-chunk), ZeRO-1 optimizer sharding (--zero). Synthetic random
token windows (packed German shards live on Isambard; throughput is
data-independent).

Launch (inside the neuron-native container):
  single core:      python3 bench_train.py --steps 3 --ckpt --ce-chunk 2048
  full chip (=4):   NEURON_RT_NUM_CORES=4 torchrun --nproc_per_node 4 \
      --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
      bench_train.py --steps 5 --ckpt --ce-chunk 2048 --zero
"""
import argparse, json, os, sys, time

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
import torch_neuronx

import os as _os_, sys as _sys_
_HERE = _os_.path.dirname(_os_.path.abspath(__file__))
_REPO_SRC = _os_.path.join(_os_.path.dirname(_os_.path.dirname(_HERE)), "src")
_sys_.path.insert(0, _REPO_SRC)
_sys_.path.insert(0, _HERE)
from xscript.model import ModelConfig, Transformer  # noqa: E402
import xscript.model as _xm  # noqa: E402
from rope_fix import apply_rope_viewbased  # noqa: E402
from nki_rope import apply_rope_nki  # noqa: E402

# model.py's _apply_rope (strided even/odd slice-assign) lowers to kernels
# ~60x slower than the op's bandwidth bound on this backend (6.74ms fwd+bwd
# for an 8.4MB tensor). Two verified-BITWISE-identical replacements:
# view-based torch (1.62ms) and the NKI kernel (1.08ms, nki_rope.py).
import os as _os
if _os.environ.get("SDPA_PATCH"):
    import sdpa_patch as _sp
    print("[bench] SDPA_PATCH =", _sp.apply(), flush=True)
_xm._apply_rope = apply_rope_nki if _os.environ.get("ROPE_IMPL", "view") == "nki" else apply_rope_viewbased


def _swiglu_forward_fused(self, x):
    """SwiGLU with w1/w3 as ONE [2*ffn, dim] matmul. Per-element identical:
    each output row's dot product is unchanged; cat/split are views. Wider
    matmul shapes reach higher utilization on this backend (43% @2048^3 vs
    71% @2048x2048x5632, comp_bench.py)."""
    w13 = torch.cat([self.w1.weight, self.w3.weight], dim=0)
    gu = F.linear(x, w13)
    g, u = gu.split([self.w1.weight.shape[0], self.w3.weight.shape[0]], dim=-1)
    return self.w2(F.silu(g) * u)


def _attn_forward_contig(self, x, cos, sin):
    """model.py Attention.forward with .contiguous() after each transpose.
    Numerically identical (pure layout copy). Needed because the neuron
    torch.compile backend fails Torch-MLIR lowering when _apply_rope's
    strided slice-assign runs on a non-contiguous transposed view
    ('failed to legalize torch.constant.int'); contiguous inputs compile."""
    B, T, _ = x.shape
    q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
    k = self.wk(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()
    v = self.wv(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()
    q = _xm._apply_rope(q, cos, sin)
    k = _xm._apply_rope(k, cos, sin)
    if self.kv_heads != self.n_heads:
        rep = self.n_heads // self.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = out.transpose(1, 2).contiguous().view(B, T, -1)
    return self.wo(out)


_xm.Attention.forward = _attn_forward_contig

# configs/base_de_starved_retrain.yaml (== base_main.yaml for model/optim/schedule)
MODEL = dict(vocab_size=65536, dim=2048, n_layers=16, n_heads=16, n_kv_heads=16,
             ffn_dim=5632, max_seq_len=2048, rope_theta=10000.0, norm_eps=1e-5)
OPTIM = dict(betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8, grad_clip=1.0)
PEAK_LR = 3.0e-3          # throughput is LR-independent; run at the stable-window peak
GLOBAL_BATCH_TOKENS = 1.0e6

# NEURON.md §9 accounting: model fwd+bwd FLOPs/token (body+head+attention).
MODEL_FLOPS_PER_TOKEN = 6.54e9
CKPT_RECOMPUTE_FLOPS = 1.91e9     # per-Block recompute in backward (§9)
CE_RECOMPUTE_FLOPS = 0.27e9       # lm_head fwd recompute per token under chunked CE
TRN2_CHIP_PEAK_TFLOPS = 667.0     # dense bf16 per Trainium2 chip (= 4 of these cores)


class _ChunkedLMHeadCE(torch.autograd.Function):
    """Port of train_neuron.py's memory-efficient lm_head+CE: chunks the
    flattened token dim so only (chunk, vocab) logits exist at a time, and
    recomputes each chunk's logits in backward. Same loss/grads as the
    non-chunked F.cross_entropy (token sum is associative)."""

    @staticmethod
    def forward(ctx, x, weight, targets, chunk):
        ctx.save_for_backward(x, weight, targets)
        ctx.chunk = chunk
        n_valid = (targets != -100).to(torch.float32).sum().clamp(min=1.0)
        ctx.n_valid = n_valid
        N = x.shape[0]
        total = x.new_zeros((), dtype=torch.float32)
        for i in range(0, N, chunk):
            logits_c = (x[i:i + chunk] @ weight.t()).float()
            total = total + F.cross_entropy(
                logits_c, targets[i:i + chunk], ignore_index=-100, reduction="sum")
        return total / n_valid

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, targets = ctx.saved_tensors
        chunk, n_valid = ctx.chunk, ctx.n_valid
        N = x.shape[0]
        grad_x_parts = []
        grad_w = torch.zeros_like(weight, dtype=torch.float32)
        with torch.enable_grad():
            for i in range(0, N, chunk):
                xc = x[i:i + chunk].detach().requires_grad_(True)
                wc = weight.detach().requires_grad_(True)
                logits_c = (xc @ wc.t()).float()
                loss_c = F.cross_entropy(
                    logits_c, targets[i:i + chunk], ignore_index=-100,
                    reduction="sum") / n_valid
                gx, gw = torch.autograd.grad(loss_c, (xc, wc))
                grad_x_parts.append(gx)
                grad_w = grad_w + gw.float()
        grad_x = torch.cat(grad_x_parts, dim=0)
        return (grad_x * grad_out).to(x.dtype), (grad_w * grad_out).to(weight.dtype), None, None


_compiled_head_ce = None
_USE_NKI_CE = False
_compiled_tail = None
_compiled_emb = None


def _tail(x, nw, eps, w, t):
    # final RMSNorm (model.py formula) + lm_head + lse-gather CE, one graph
    dt = x.dtype
    f = x.float()
    f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    xn = (f * nw.float()).to(dt)
    lg = (xn.reshape(-1, xn.size(-1)) @ w.t()).float()
    lse = torch.logsumexp(lg, dim=-1)
    tl = lg.gather(1, t.clamp(min=0).long().unsqueeze(1)).squeeze(1)
    valid = (t != -100).to(lg.dtype)
    return ((lse - tl) * valid).sum() / valid.sum().clamp(min=1.0)


def _emb(w, idx):
    return F.embedding(idx, w)


def _head_ce_nki(x2, w, t):
    # matmul in-graph (bf16), AWS nkilib fused CE (online lse, in-place grad)
    from nki_ce import nki_cross_entropy_mean
    return nki_cross_entropy_mean(x2 @ w.t(), t)


def _head_ce(x2, w, t):
    # logsumexp-gather form (value-identical to F.cross_entropy, faster to compile)
    lg = (x2 @ w.t()).float()
    lse = torch.logsumexp(lg, dim=-1)
    tl = lg.gather(1, t.clamp(min=0).long().unsqueeze(1)).squeeze(1)
    valid = (t != -100).to(lg.dtype)
    return ((lse - tl) * valid).sum() / valid.sum().clamp(min=1.0)


class _Group(torch.nn.Module):
    """N consecutive Blocks as one compilable unit (shares the Block modules)."""
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, cos, sin):
        for l in self.layers:
            x = l(x, cos, sin)
        return x


def _chunk_ce_sum(xc, w, tc):
    return F.cross_entropy((xc @ w.t()).float(), tc, ignore_index=-100, reduction="sum")


def forward_loss(model, idx, targets, use_ckpt, ce_chunk, ce_compile=False, groups=None,
                 ce_ckpt_chunks=0, ce_lse=True, ckpt_blocks=0, tail_compile=False):
    """train_neuron.py's _checkpointed_forward, generalized: same calls into
    model.py, optional per-Block checkpointing, optional chunked CE, optional
    single-graph compiled lm_head+CE."""
    B, T = idx.shape
    global _compiled_emb, _compiled_tail
    if tail_compile:
        if _compiled_emb is None:
            _compiled_emb = torch.compile(_emb, backend="neuron", dynamic=False)
        x = _compiled_emb(model.tok_emb.weight, idx)
    else:
        x = model.tok_emb(idx)
    cos, sin = model._rope_for(T, idx.device, x.dtype)
    units = groups if groups else model.layers
    for li, unit in enumerate(units):
        if use_ckpt or li < ckpt_blocks:
            x = ckpt.checkpoint(unit, x, cos, sin, use_reentrant=False,
                                preserve_rng_state=False)
        else:
            x = unit(x, cos, sin)
    if tail_compile:
        if _compiled_tail is None:
            _compiled_tail = torch.compile(_tail, backend="neuron", dynamic=False)
        return _compiled_tail(x, model.norm.weight, model.norm.eps, model.lm_head.weight,
                              targets.reshape(-1))
    x = model.norm(x)
    tflat = targets.reshape(-1)
    if ce_ckpt_chunks and ce_ckpt_chunks > 0:
        # In-graph chunked CE with per-chunk activation checkpointing: the
        # (N, vocab) fp32 logits are never all live, and each chunk's head
        # matmul is recomputed in backward (+0.27 GF/token). Same loss as
        # mean CE (sum over chunks / n_valid).
        x2 = x.reshape(-1, x.size(-1))
        N = x2.shape[0]
        ch = N // ce_ckpt_chunks
        total = None
        for i in range(ce_ckpt_chunks):
            part = ckpt.checkpoint(_chunk_ce_sum, x2[i * ch:(i + 1) * ch], model.lm_head.weight,
                                   tflat[i * ch:(i + 1) * ch], use_reentrant=False,
                                   preserve_rng_state=False)
            total = part if total is None else total + part
        n_valid = (tflat != -100).to(torch.float32).sum().clamp(min=1.0)
        return total / n_valid
    if ce_compile:
        global _compiled_head_ce
        if _compiled_head_ce is None:
            _compiled_head_ce = torch.compile(_head_ce_nki if _USE_NKI_CE else _head_ce,
                                              backend="neuron", dynamic=False)
        return _compiled_head_ce(x.reshape(-1, x.size(-1)), model.lm_head.weight, tflat)
    if ce_chunk and ce_chunk > 0:
        return _ChunkedLMHeadCE.apply(
            x.reshape(-1, x.size(-1)), model.lm_head.weight, tflat, ce_chunk)
    logits = model.lm_head(x).float().view(-1, model.lm_head.weight.shape[0])
    if ce_lse:
        # logsumexp - gather: same value as F.cross_entropy (verified to 1e-6),
        # same gradient (softmax - onehot), but the compiler avoids
        # materializing the full (N, vocab) fp32 log_softmax: 22.8 vs 30.5 ms
        # fwd+bwd at N=2048 (head_bench.py). ignore_index handled via mask.
        lse = torch.logsumexp(logits, dim=-1)
        tl = logits.gather(1, tflat.clamp(min=0).long().unsqueeze(1)).squeeze(1)
        valid = (tflat != -100).to(logits.dtype)
        return ((lse - tl) * valid).sum() / valid.sum().clamp(min=1.0)
    return F.cross_entropy(logits, tflat, ignore_index=-100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--micro-bsz", type=int, default=8)
    ap.add_argument("--accum", type=int, default=0, help="override grad_accum (0 = trainer bookkeeping)")
    ap.add_argument("--ckpt", action="store_true", help="per-Block activation checkpointing")
    ap.add_argument("--ce-chunk", type=int, default=0, help="chunked lm_head+CE chunk size (0 = full)")
    ap.add_argument("--ce-compile", action="store_true", help="compile lm_head+CE as one graph")
    ap.add_argument("--ce-ckpt-chunks", type=int, default=0, help="in-graph chunked+checkpointed CE (N chunks); frees the fp32 logits")
    ap.add_argument("--full-compile", action="store_true", help="compile the entire forward+loss as one graph (implies no ckpt)")
    ap.add_argument("--group", type=int, default=0, help="compile groups of N blocks instead of single blocks")
    ap.add_argument("--tensorizer", action="store_true", help="use experimental tensorizer compile backend")
    ap.add_argument("--bf16", action="store_true", help="bf16 params (PROBE ONLY: deviates from fp32-master replica)")
    ap.add_argument("--zero", action="store_true", help="ZeRO-1 via ZeroRedundancyOptimizer")
    ap.add_argument("--flatzero", action="store_true", help="lean flat-buffer ZeRO-1 (flat_zero.py): 1 reduce_scatter + 1 all_gather per step")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--dim", type=int, default=0, help="override model dim")
    ap.add_argument("--layers", type=int, default=0, help="override n_layers")
    ap.add_argument("--ffn", type=int, default=0, help="override ffn_dim")
    ap.add_argument("--heads", type=int, default=0, help="override n_heads/n_kv_heads")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--memprobe", action="store_true", help="print per-phase device memory on step 0")
    ap.add_argument("--ce-fce", action="store_true", help="use F.cross_entropy instead of the faster logsumexp-gather head")
    ap.add_argument("--foreach-clip", action="store_true")
    ap.add_argument("--coalesced", action="store_true", help="one coalesced grad all-reduce + foreach clip")
    ap.add_argument("--ckpt-blocks", type=int, default=0, help="activation-checkpoint only the first N blocks")
    ap.add_argument("--ce-nki", action="store_true", help="with --ce-compile: use AWS nkilib fused cross-entropy kernel")
    ap.add_argument("--opt-foreach", action="store_true", help="AdamW(foreach=True) inside ZeRO")
    ap.add_argument("--fuse-qkv", action="store_true", help="with --bf16-shadow-hooks: fused [3D,D] qkv and [2FF,D] gate|up shadow weights, materialized once per step")
    ap.add_argument("--hook-batch", type=int, default=1, help="with --bf16-shadow-hooks: batch N params' grad adds into one _foreach_add_")
    ap.add_argument("--bf16-shadow-hooks", action="store_true", help="bf16 shadow Linear weights + per-param hook accumulation into fp32 master (one fresh grad alive at a time)")
    ap.add_argument("--bf16-shadow", action="store_true", help="bf16 copies of Linear weights refreshed once per step (fp32 master in optimizer); embedding/norm stay fp32; grads accumulated fp32 via foreach")
    ap.add_argument("--compiled-autograd", action="store_true", help="torch._dynamo.compiled_autograd: compile the backward incl. grad accumulation")
    ap.add_argument("--manual-accum", action="store_true", help="accumulate grads via autograd.grad + one _foreach_add_ instead of AccumulateGrad's 219 eager adds")
    ap.add_argument("--overlap-ar", action="store_true", help="launch each param's grad all-reduce async as it finalizes on the last micro (overlaps with remaining backward)")
    ap.add_argument("--tail-compile", action="store_true", help="per-Block mode: also compile embedding and (final norm + head + CE) as graphs")
    args = ap.parse_args()

    global _USE_NKI_CE
    _USE_NKI_CE = bool(args.ce_nki)
    if args.dim:    MODEL["dim"] = args.dim
    if args.layers: MODEL["n_layers"] = args.layers
    if args.ffn:    MODEL["ffn_dim"] = args.ffn
    if args.heads:  MODEL["n_heads"] = MODEL["n_kv_heads"] = args.heads
    # model FLOPs/token for MFU, recomputed for the actual shape:
    # 3 * (qkvo + swiglu + causal attn) per layer, + lm_head
    global MODEL_FLOPS_PER_TOKEN
    _d, _f, _L = MODEL["dim"], MODEL["ffn_dim"], MODEL["n_layers"]
    _T, _V = MODEL["max_seq_len"], MODEL["vocab_size"]
    MODEL_FLOPS_PER_TOKEN = 3.0 * (_L * (8 * _d * _d + 6 * _d * _f + 2 * _T * _d) + 2 * _d * _V)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    dist = world > 1
    if dist:
        import torch.distributed as td
        td.init_process_group(backend="neuron", world_size=world, rank=rank)

    dev = torch.device(f"neuron:{torch_neuronx.current_device()}")
    torch.manual_seed(1 + rank)   # seed: 1 (retrain config)

    raw_model = Transformer(ModelConfig(**MODEL))
    if args.bf16:
        raw_model = raw_model.to(torch.bfloat16)
    raw_model = raw_model.to(dev)
    if rank == 0:
        print(f"[bench] params: {raw_model.num_params(False)/1e6:.1f}M non-emb, "
              f"{raw_model.num_params(True)/1e6:.1f}M total", flush=True)

    # No DDP wrapper: mirroring train_neuron.py's data-parallel mechanics --
    # grads accumulate locally over grad_accum micros, then ONE explicit
    # all-reduce per optimizer step (same comm volume as DDP's no_sync
    # pattern in train.py, without wrapper/hook interplay).
    shadow_model, shadow_lin, master_lin, shadow_other, master_other = None, [], [], [], []
    fuse_specs = []
    if args.bf16_shadow or args.bf16_shadow_hooks:
        import copy
        shadow_model = copy.deepcopy(raw_model)
        _lin_ids = set(id(m.weight) for m in shadow_model.modules() if isinstance(m, torch.nn.Linear))
        for p_s, p_m in zip(shadow_model.parameters(), raw_model.parameters()):
            if id(p_s) not in _lin_ids:
                p_s.data = p_m.data           # share fp32 storage (embedding, norms)
                shadow_other.append(p_s); master_other.append(p_m)
        for m_s, m_m in zip(shadow_model.modules(), raw_model.modules()):
            if isinstance(m_s, torch.nn.Linear):
                m_s.weight.data = m_m.weight.data.to(torch.bfloat16)
                shadow_lin.append(m_s.weight); master_lin.append(m_m.weight)
        if args.fuse_qkv:
            import fused_shadow
            fuse_specs = fused_shadow.build(shadow_model, raw_model, ModelConfig(**MODEL), dev)
            _blk_ids = set()
            for _b in raw_model.layers:
                for _l in (_b.attn.wq, _b.attn.wk, _b.attn.wv, _b.attn.wo,
                           _b.ffn.w1, _b.ffn.w3, _b.ffn.w2):
                    _blk_ids.add(id(_l.weight))
            _keep = [(a, b) for a, b in zip(shadow_lin, master_lin) if id(b) not in _blk_ids]
            shadow_lin = [a for a, _ in _keep]; master_lin = [b for _, b in _keep]
            fused_shadow.refresh(fuse_specs)          # wqkv/w13 start empty
            if rank == 0:
                print(f"[bench] fused shadow: {len(fuse_specs)} fused params, "
                      f"{len(shadow_lin)} plain shadow Linears", flush=True)
        fwd_model = shadow_model
    else:
        fwd_model = raw_model
    if args.bf16_shadow_hooks:
        _pend = {"m": [], "g": []}
        def _flush_pending():
            if not _pend["m"]:
                return
            torch._foreach_add_(_pend["m"], _pend["g"])
            _pend["m"].clear(); _pend["g"].clear()
        def _mk_acc_hook(sp, mp):
            def _h(param):
                g = param.grad
                if g is None:
                    return
                if mp.grad is None:
                    mp.grad = g.float() if g.dtype != torch.float32 else g.clone()
                elif args.hook_batch > 1:
                    _pend["m"].append(mp.grad); _pend["g"].append(g)
                    if len(_pend["m"]) >= args.hook_batch:
                        _flush_pending()
                else:
                    mp.grad.add_(g)
                param.grad = None
            return _h
        for sp_, mp_ in zip(shadow_lin, master_lin):
            sp_.register_post_accumulate_grad_hook(_mk_acc_hook(sp_, mp_))
        for sp_, mp_ in zip(shadow_other, master_other):
            sp_.register_post_accumulate_grad_hook(_mk_acc_hook(sp_, mp_))
        if fuse_specs:
            import fused_shadow
            for _fp, _masters in fuse_specs:
                _fp.register_post_accumulate_grad_hook(fused_shadow.make_hook(_fp, _masters))
    groups = None
    if args.group:
        L = list(fwd_model.layers)
        groups = [_Group(L[i:i + args.group]) for i in range(0, len(L), args.group)]
        if args.compile:
            for gm_ in groups:
                gm_.compile(backend="neuron", dynamic=False)
    elif args.compile:
        for layer in fwd_model.layers:
            layer.compile(backend="neuron", dynamic=False)

    if args.flatzero:
        from flat_zero import FlatZeRO1
        optim = FlatZeRO1(raw_model.parameters(), lr=PEAK_LR, betas=OPTIM["betas"],
                          weight_decay=OPTIM["weight_decay"], eps=OPTIM["eps"],
                          grad_clip=OPTIM["grad_clip"], world=world, rank=rank, device=dev)
    elif args.zero:
        if not dist:
            raise SystemExit("--zero needs WORLD_SIZE>1")
        from torch.distributed.optim import ZeroRedundancyOptimizer
        _okw = {"foreach": True} if args.opt_foreach else {}
        optim = ZeroRedundancyOptimizer(
            raw_model.parameters(), optimizer_class=torch.optim.AdamW, lr=PEAK_LR,
            betas=OPTIM["betas"], weight_decay=OPTIM["weight_decay"], eps=OPTIM["eps"], **_okw)
    else:
        optim = torch.optim.AdamW(raw_model.parameters(), lr=PEAK_LR,
                                  betas=OPTIM["betas"], weight_decay=OPTIM["weight_decay"],
                                  eps=OPTIM["eps"])

    if rank == 0 and args.memprobe:
        torch_neuronx.synchronize()
        try:
            ms = torch.neuron.memory_stats(dev)
            print(f"[bench] MEMPHASE after model+optim init: cur={ms['allocated_bytes.all.current']/2**30:.2f}GiB", flush=True)
        except Exception as e:
            print("[bench] MEMPHASE init n/a", flush=True)

    # --- global batch bookkeeping, verbatim from train.py ---
    seq_len = MODEL["max_seq_len"]
    mb = args.micro_bsz
    per_step_windows = max(1, round(GLOBAL_BATCH_TOKENS / seq_len))
    unit = mb * world
    global_windows = max(unit, (per_step_windows // unit) * unit)
    grad_accum = args.accum if args.accum else global_windows // unit
    tokens_per_step = grad_accum * unit * seq_len
    if rank == 0:
        print(f"[bench] world={world} mb={mb} grad_accum={grad_accum} "
              f"tokens/step={tokens_per_step/1e6:.3f}M ckpt={args.ckpt} "
              f"ce_chunk={args.ce_chunk} ce_compile={args.ce_compile} full_compile={args.full_compile} "
              f"zero={args.zero} compile={args.compile}", flush=True)

    g = torch.Generator().manual_seed(5678 + rank)   # data_seed: 5678 (retrain config)
    pool = []
    for _ in range(4):
        w = torch.randint(0, MODEL["vocab_size"], (mb, seq_len + 1), generator=g, dtype=torch.int32)
        pool.append((w[:, :-1].contiguous().to(dev), w[:, 1:].contiguous().to(dev)))

    if args.full_compile:
        def _full(x, y):
            return forward_loss(raw_model, x, y, args.ckpt, 0, False, None, args.ce_ckpt_chunks,
                                ce_lse=not args.ce_fce, ckpt_blocks=args.ckpt_blocks)
        _opts = {"use_tensorizer_backend": True} if args.tensorizer else None
        _fc = torch.compile(_full, backend="neuron", dynamic=False, options=_opts)
        fwd_fn = _fc
    else:
        def fwd_fn(x, y):
            return forward_loss(fwd_model, x, y, args.ckpt, args.ce_chunk, args.ce_compile, groups,
                                args.ce_ckpt_chunks, ce_lse=not args.ce_fce, ckpt_blocks=args.ckpt_blocks,
                                tail_compile=args.tail_compile)

    def _memnow(label):
        if rank != 0 or not args.memprobe:
            return
        torch_neuronx.synchronize()
        try:
            ms = torch.neuron.memory_stats(dev)
            print(f"[bench] MEMPHASE {label}: cur={ms['allocated_bytes.all.current']/2**30:.2f}GiB "
                  f"peak={ms['allocated_bytes.all.peak']/2**30:.2f}GiB", flush=True)
        except Exception as e:
            print(f"[bench] MEMPHASE {label}: n/a {repr(e)[:60]}", flush=True)

    _ar_state = {"active": False, "handles": []}
    if args.overlap_ar and dist:
        import torch.distributed as td
        def _mk_hook(p):
            def _hook(param):
                if _ar_state["active"] and param.grad is not None:
                    _ar_state["handles"].append(td.all_reduce(param.grad, op=td.ReduceOp.AVG, async_op=True))
            return _hook
        for p in raw_model.parameters():
            p.register_post_accumulate_grad_hook(_mk_hook(p))

    _params_list = [p for p in raw_model.parameters() if p.requires_grad]
    _ca_compiler = (lambda gm: torch.compile(gm, backend="neuron", dynamic=False)) if args.compiled_autograd else None

    def one_step(i):
        optim.zero_grad(set_to_none=not args.flatzero)
        loss_acc = None
        for j in range(grad_accum):
            x, y = pool[(i * grad_accum + j) % len(pool)]
            with torch.autocast("neuron", dtype=torch.bfloat16):
                loss = fwd_fn(x, y)
            if i == 0 and j == 0: _memnow("after fwd (micro 0)")
            _ar_state["active"] = bool(args.overlap_ar and dist and j == grad_accum - 1)
            if args.bf16_shadow:
                gs = torch.autograd.grad(loss / grad_accum, shadow_lin + shadow_other)
                g_lin, g_oth = gs[:len(shadow_lin)], gs[len(shadow_lin):]
                if j == 0:
                    for p_, g_ in zip(master_lin, g_lin): p_.grad = g_.float()
                    for p_, g_ in zip(master_other, g_oth): p_.grad = g_
                else:
                    torch._foreach_add_([p_.grad for p_ in master_lin], [g_.float() for g_ in g_lin])
                    torch._foreach_add_([p_.grad for p_ in master_other], list(g_oth))
                del gs, g_lin, g_oth
            elif args.manual_accum:
                gs = torch.autograd.grad(loss / grad_accum, _params_list)
                if j == 0:
                    for p_, g_ in zip(_params_list, gs):
                        p_.grad = g_
                else:
                    torch._foreach_add_([p_.grad for p_ in _params_list], list(gs))
                del gs
            elif args.compiled_autograd:
                import torch._dynamo.compiled_autograd as _ca
                with _ca._enable(_ca_compiler):
                    (loss / grad_accum).backward()
            else:
                (loss / grad_accum).backward()
                if args.bf16_shadow_hooks and args.hook_batch > 1:
                    _flush_pending()
            _ar_state["active"] = False
            if i == 0 and j == 0: _memnow("after bwd (micro 0)")
            d = loss.detach() / grad_accum
            loss_acc = d if loss_acc is None else loss_acc + d
        if i == 0: _memnow("after all micros")
        if args.flatzero:
            optim.step()          # reduce_scatter + global clip + sharded AdamW + all_gather
            if i == 0: _memnow("after flatzero step")
            return float(loss_acc)
        if dist and args.overlap_ar:
            for h in _ar_state["handles"]:
                h.wait()
            _ar_state["handles"].clear()
        elif dist:
            import torch.distributed as td
            grads = [p.grad for p in raw_model.parameters() if p.grad is not None]
            if args.coalesced:
                td.all_reduce_coalesced(grads, op=td.ReduceOp.AVG)
            else:
                for g_ in grads:
                    td.all_reduce(g_, op=td.ReduceOp.AVG)
        if i == 0: _memnow("after all-reduce")
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), OPTIM["grad_clip"],
                                       foreach=True if (args.coalesced or args.foreach_clip) else None)
        optim.step()
        if args.bf16_shadow or args.bf16_shadow_hooks:
            with torch.no_grad():
                for s_, m_ in zip(shadow_lin, master_lin):
                    s_.data.copy_(m_.data)          # once per STEP, not per micro
            if fuse_specs:
                import fused_shadow
                fused_shadow.refresh(fuse_specs)
        if i == 0: _memnow("after optim.step")
        return float(loss_acc)

    def _memstats(label):
        if rank != 0:
            return
        try:
            ms = torch.neuron.memory_stats(dev)
            keys = [k for k in ms if k.startswith(("allocated_bytes", "reserved_bytes")) and "peak" in k]
            print(f"[bench] MEM {label}: " + ", ".join(f"{k}={ms[k]/2**30:.2f}GiB" for k in sorted(keys)[:4]), flush=True)
        except Exception as e:
            print(f"[bench] MEM {label}: unavailable ({repr(e)[:80]})", flush=True)

    for i in range(args.warmup):
        t = time.time()
        loss = one_step(i)
        torch_neuronx.synchronize()
        if rank == 0:
            print(f"[bench] warmup {i+1}/{args.warmup}: loss {loss:.4f} "
                  f"({time.time()-t:.1f}s)", flush=True)
        _memstats(f"after warmup {i+1}")

    torch_neuronx.synchronize()
    t0 = time.time()
    for i in range(args.steps):
        loss = one_step(args.warmup + i)
        torch_neuronx.synchronize()
        if rank == 0:
            el = time.time() - t0
            print(f"[bench] step {i+1}/{args.steps}: loss {loss:.4f} | "
                  f"cum {tokens_per_step*(i+1)/el/1e3:.1f}k tok/s", flush=True)
    torch_neuronx.synchronize()
    dt = time.time() - t0

    if rank == 0:
        tps = tokens_per_step * args.steps / dt      # global tokens/s (all ranks)
        hw_flops = MODEL_FLOPS_PER_TOKEN \
            + (CKPT_RECOMPUTE_FLOPS if args.ckpt else CKPT_RECOMPUTE_FLOPS * args.ckpt_blocks / MODEL["n_layers"]) \
            + (CE_RECOMPUTE_FLOPS if (args.ce_chunk or args.ce_ckpt_chunks) else 0.0)
        rec = dict(tag=args.tag, world=world, micro_bsz=mb, grad_accum=grad_accum,
                   ckpt=args.ckpt, ce_chunk=args.ce_chunk, ce_compile=args.ce_compile, group=args.group, tensorizer=args.tensorizer, bf16=args.bf16, ce_ckpt_chunks=args.ce_ckpt_chunks, flatzero=args.flatzero, ce_lse=not args.ce_fce, coalesced=args.coalesced, ckpt_blocks=args.ckpt_blocks, ce_nki=args.ce_nki, tail_compile=args.tail_compile,
                   model=dict(MODEL), gf_per_token=round(MODEL_FLOPS_PER_TOKEN / 1e9, 3), opt_foreach=args.opt_foreach, overlap_ar=args.overlap_ar, manual_accum=args.manual_accum, compiled_autograd=args.compiled_autograd, bf16_shadow=args.bf16_shadow, bf16_shadow_hooks=args.bf16_shadow_hooks, hook_batch=args.hook_batch, fuse_qkv=args.fuse_qkv,
                   full_compile=args.full_compile, zero=args.zero,
                   compile=args.compile, steps=args.steps,
                   tokens_per_step=tokens_per_step, seconds=round(dt, 2),
                   tok_per_s=round(tps),
                   model_tflops=round(tps * MODEL_FLOPS_PER_TOKEN / 1e12, 1),
                   hw_tflops=round(tps * hw_flops / 1e12, 1),
                   mfu_vs_chip=round(tps * MODEL_FLOPS_PER_TOKEN / (TRN2_CHIP_PEAK_TFLOPS * 1e12), 4),
                   hfu_vs_chip=round(tps * hw_flops / (TRN2_CHIP_PEAK_TFLOPS * 1e12), 4),
                   note="per-chip MFU valid when all ranks are on one chip (world<=4); "
                        "world=1 uses 1/4 of the chip")
        print("[bench] RESULT " + json.dumps(rec), flush=True)
        os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
        with open(os.path.join(_HERE, "results", f"result_{args.tag}.json"), "w") as f:
            json.dump(rec, f, indent=2)

    if dist:
        import torch.distributed as td
        td.destroy_process_group()


if __name__ == "__main__":
    main()
