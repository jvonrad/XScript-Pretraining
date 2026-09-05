#!/usr/bin/env python
"""Real pretraining on the TorchNeuron Native beta: NEURON.md 10e's recipe
(per-Block torch.compile, compiled embedding + norm/head/CE tail, view-based
RoPE, bf16 shadow Linear weights with fp32 masters, ZeRO-1, mb=2) wired to
the repo's deterministic MixedStream loader, WSD schedule, log-spaced
checkpoints and W&B -- the port `bench_train.py` was the replica of.

Same experiment as the CUDA / XLA trainers: same model.py (the RoPE rewrite
and the .contiguous() inserts are numerically identical), same run configs
(runmatrix), same checkpoint keys (`model` is the plain Transformer
state_dict, so eval tooling reads these unchanged). Execution differences
only: bf16 compute with fp32 master (autocast-equivalent), grads accumulated
in fp32 from bf16 per-parameter hooks.

Launch (inside the `neuron-native` container, repo at /repo, scratch mounted):
    NEURON_RT_NUM_CORES=16 NEURON_RT_VISIBLE_CORES=0-15 torchrun --nproc_per_node 16 \
        --rdzv_backend c10d --rdzv_endpoint localhost:<port> \
        scripts/neuron_native/train_native.py <run_name> --only-30b
Resume: re-launch the same run name with the same world size (ZeRO shards).
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_neuronx
from torch.distributed.optim import ZeroRedundancyOptimizer

import xscript.model as M
from xscript import runmatrix
from xscript.data.loader import MixedStream
from xscript.paths import run_dir, ensure
from xscript.schedule import lr_at, ckpt_interval, total_tokens
from rope_fix import apply_rope_viewbased

# ---- recipe items 2 + 3: view-based rope, contiguous transposes -------------
M._apply_rope = apply_rope_viewbased


def _attn(self, x, cos, sin):
    B, T, _ = x.shape
    q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
    k = self.wk(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()
    v = self.wv(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()
    q = M._apply_rope(q, cos, sin)
    k = M._apply_rope(k, cos, sin)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))


M.Attention.forward = _attn


def _tail(x, nw, eps, w, t):
    f = x.float()
    f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    lg = ((f * nw.float()).to(x.dtype).reshape(-1, x.size(-1)) @ w.t()).float()
    lse = torch.logsumexp(lg, -1)
    tl = lg.gather(1, t.clamp(min=0).long().unsqueeze(1)).squeeze(1)
    valid = (t != -100).to(lg.dtype)
    return ((lse - tl) * valid).sum() / valid.sum().clamp(min=1.0)


def _log(rank, path, rec):
    if rank == 0:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--base", default="configs/base_main.yaml")
    ap.add_argument("--flavor", default="unigram")
    ap.add_argument("--only-30b", action="store_true")
    ap.add_argument("--micro-bsz", type=int, default=2)
    ap.add_argument("--wandb-id", default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after N steps")
    ap.add_argument("--ckpt-interval-tokens", type=float, default=0,
                    help="debug: override the checkpoint schedule with a fixed interval (tokens)")
    ap.add_argument("--empty-cache-every", type=int, default=0,
                    help="call torch_neuronx.empty_cache() every N steps (fragmentation mitigation probe)")
    args = ap.parse_args()

    dist.init_process_group(backend="neuron")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device(f"neuron:{torch_neuronx.current_device()}")

    cfg = runmatrix.get_run(str(_ROOT / args.base), args.flavor, args.name, args.only_30b)
    cfg["train"] = {**cfg["train"], "micro_batch_size": args.micro_bsz,
                    "ckpt_schedule": cfg["train"].get("ckpt_schedule",
                                                      [[2e9, 250e6], [5e9, 500e6], [1e15, 1e9]])}
    cfg["backend"] = "neuron-native"
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    mc = M.ModelConfig(**cfg["model"])
    T = mc.max_seq_len

    # ---- model: fp32 master owns the optimizer; bf16 shadow runs the graphs --
    master = M.Transformer(mc).to(dev)
    opt = cfg.get("optim", {})
    sched = dict(cfg["schedule"])
    optim = ZeroRedundancyOptimizer(master.parameters(), optimizer_class=torch.optim.AdamW,
                                    lr=sched["peak_lr"], betas=tuple(opt.get("betas", (0.9, 0.95))),
                                    weight_decay=opt.get("weight_decay", 0.1), eps=opt.get("eps", 1e-8))
    grad_clip = opt.get("grad_clip", 1.0)
    shadow = copy.deepcopy(master)
    lin = {id(m.weight) for m in shadow.modules() if isinstance(m, torch.nn.Linear)}
    pairs = []
    for ps, pm in zip(shadow.parameters(), master.parameters()):
        ps.data = pm.data.to(torch.bfloat16) if id(ps) in lin else pm.data
        pairs.append((ps, pm))

    def _hook(pm):
        def h(ps):
            g = ps.grad
            pm.grad = (g.float() if g.dtype != torch.float32 else g.clone()) if pm.grad is None else pm.grad.add_(g)
            ps.grad = None
        return h
    for ps, pm in pairs:
        ps.register_post_accumulate_grad_hook(_hook(pm))

    def refresh_shadow():
        with torch.no_grad():
            for ps, pm in pairs:
                if id(ps) in lin:
                    ps.data.copy_(pm.data)

    for blk in shadow.layers:
        blk.compile(backend="neuron", dynamic=False)
    emb = torch.compile(lambda w, idx: F.embedding(idx, w), backend="neuron", dynamic=False)
    tail = torch.compile(_tail, backend="neuron", dynamic=False)

    def forward_loss(idx, tgt):
        x = emb(shadow.tok_emb.weight, idx)
        cos, sin = shadow._rope_for(idx.shape[1], idx.device, x.dtype)
        for blk in shadow.layers:
            x = blk(x, cos, sin)
        return tail(x, shadow.norm.weight, shadow.norm.eps, shadow.lm_head.weight, tgt.reshape(-1))

    # ---- batch bookkeeping (identical to train.py / train_neuron.py) --------
    mb = args.micro_bsz
    gbt = cfg["train"]["global_batch_tokens"]
    per_step = max(1, round(gbt / T))
    unit = mb * world
    global_windows = max(unit, (per_step // unit) * unit)
    grad_accum = global_windows // unit
    tokens_per_step = global_windows * T
    target = total_tokens(sched)
    ckpt_table = [[1e15, args.ckpt_interval_tokens]] if args.ckpt_interval_tokens else cfg["train"]["ckpt_schedule"]
    mixer = MixedStream(cfg["langs"], cfg["tok_name"], T, seed=cfg.get("data_seed", 1234),
                        probs=cfg.get("probs"))
    rdir = ensure(run_dir(cfg["name"]))
    ckdir = ensure(rdir / "checkpoints")
    log_path = rdir / "train.jsonl"
    step, tokens, last_ckpt_tokens = 0, 0, 0

    # ---- resume -------------------------------------------------------------
    last = ckdir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck.get("world") != world:
            raise RuntimeError(f"resume needs world={ck.get('world')} (ZeRO shards); got {world}")
        master.load_state_dict(ck["model"])
        # optimizer: per-rank ZeRO shard sidecar (never consolidated -- consolidating
        # onto rank 0's device needs ~8.7 GB it does not have: NRT alloc error at the
        # first save). ZeRO's partition is deterministic for the same params/world.
        shard = torch.load(ckdir / f"last.optim.rank{rank}.pt", map_location="cpu", weights_only=False)
        optim.optim.load_state_dict(shard)
        mixer.load_state_dict(ck["mixer"])
        step, tokens, last_ckpt_tokens = ck["step"], ck["tokens"], ck["last_ckpt_tokens"]
        torch.set_rng_state(ck["torch_rng"])
        del ck
        if rank == 0:
            print(f"[native] resumed @ step {step}, {tokens/1e9:.3f}B tokens", flush=True)
    refresh_shadow()

    wandb = None
    if rank == 0:
        print(f"[native] {cfg['name']}: world={world} mb={mb} grad_accum={grad_accum} "
              f"tokens/step={tokens_per_step/1e6:.3f}M target={target/1e9:.1f}B "
              f"params={master.num_params(True)/1e6:.0f}M", flush=True)
        try:
            import wandb as _wb
            wandb = _wb.init(project="XScript-Pretraining", name=cfg["name"],
                             id=args.wandb_id or (cfg["name"] + "__native2"), resume="allow", config=cfg)
        except Exception as exc:
            print(f"[native] wandb disabled ({exc})", flush=True)

    def _to_cpu(o):
        if torch.is_tensor(o):
            return o.detach().cpu()
        if isinstance(o, dict):
            return {k: _to_cpu(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return type(o)(_to_cpu(v) for v in o)
        return o

    def save(tag, resumable):
        # only rank 0 materialises the 4.4 GB host copy of the model (every rank
        # doing it grew each rank's RSS by ~4 GB after the first save)
        payload = {"model": {k: v.detach().cpu() for k, v in master.state_dict().items()} if rank == 0 else None,
                   "step": step, "tokens": tokens, "cfg": cfg}
        if resumable:
            # each rank persists its OWN ZeRO shard (host-side); no device-side consolidation
            shard = _to_cpu(optim.optim.state_dict())
            tmp = ckdir / f"{tag}.optim.rank{rank}.pt.tmp"
            torch.save(shard, tmp)
            os.replace(tmp, ckdir / f"{tag}.optim.rank{rank}.pt")
            if rank == 0:
                payload.update({"mixer": mixer.state_dict(), "last_ckpt_tokens": last_ckpt_tokens,
                                "world": world, "torch_rng": torch.get_rng_state()})
        if rank == 0:
            tmp = ckdir / f"{tag}.pt.tmp"
            torch.save(payload, tmp)
            os.replace(tmp, ckdir / f"{tag}.pt")
            print(f"[native] saved {tag} ({'full' if resumable else 'model-only'}) @ {tokens/1e9:.3f}B", flush=True)
        del payload
        import gc, ctypes
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)   # hand freed host memory back to the OS
        except Exception:
            pass
        dist.barrier()

    # ---- in-loop BPB eval (mirrors eval/bpb.py: BOS+text+EOS, non-overlapping
    # seq_len windows, sum NLL / (ln2 * UTF-8 bytes)); fixed shapes so it compiles
    # once; texts sharded over ranks and the sums all-reduced ---------------------
    import math
    from xscript import flores as _flores
    from xscript.eval.bpb import load_holdout
    from xscript.tok.wrapper import Tok
    from xscript.paths import tokenizer_dir
    FL_W, FL_B = 128, 16           # FLORES sentences: <= 102 tokens under every tokenizer here
    eval_srcs, eval_ok = {}, True
    try:
        tok = Tok(tokenizer_dir(cfg["tok_name"]))
        par = _flores.load_parallel(cfg["langs"], "dev")
        for l in cfg["langs"]:
            eval_srcs[f"flores_{l}"] = (par[l], FL_W, FL_B)
            h = load_holdout(l, cfg["train"].get("eval_docs", 500))
            if h:
                eval_srcs[f"holdout_{l}"] = (h, T, mb)   # 2048-wide windows -> reuse the training graphs
    except Exception as exc:
        eval_ok = False
        if rank == 0:
            print(f"[native] eval disabled: {exc}", flush=True)

    def _windows(texts, width):
        """(x, y) int32 rows of `width` for the texts owned by this rank, plus byte total."""
        xs, ys, nbytes = [], [], 0
        for i, text in enumerate(texts):
            if i % world != rank:
                continue
            b = len(text.encode("utf-8"))
            if b == 0:
                continue
            nbytes += b
            ids = tok.encode(text, bos=True, eos=True)
            for st in range(0, len(ids) - 1, width):
                chunk = ids[st:st + width + 1]
                if len(chunk) < 2:
                    continue
                x = np.full(width, 3, np.int32); y = np.full(width, -100, np.int32)   # pad id 3, ignored target
                x[:len(chunk) - 1] = chunk[:-1]; y[:len(chunk) - 1] = chunk[1:]
                xs.append(x); ys.append(y)
        return xs, ys, nbytes

    @torch.no_grad()
    def evaluate():
        nonlocal eval_ok
        if not eval_ok:
            return {}
        try:
            out = {}
            for name, (texts, width, bs) in eval_srcs.items():
                xs, ys, nbytes = _windows(texts, width)
                nll = torch.zeros((), device=dev); ntok = torch.zeros((), device=dev)
                for i in range(0, len(xs), bs):
                    cx, cy = xs[i:i + bs], ys[i:i + bs]
                    while len(cx) < bs:                       # keep the batch shape fixed
                        cx.append(np.full(width, 3, np.int32)); cy.append(np.full(width, -100, np.int32))
                    x = torch.from_numpy(np.stack(cx)).to(dev); y = torch.from_numpy(np.stack(cy)).to(dev)
                    with torch.autocast("neuron", dtype=torch.bfloat16):
                        mean_loss = forward_loss(x, y)
                    valid = (y != -100).sum()
                    nll = nll + mean_loss.float() * valid.float(); ntok = ntok + valid.float()
                stats_t = torch.stack([nll, ntok, torch.tensor(float(nbytes), device=dev)])
                dist.all_reduce(stats_t, op=dist.ReduceOp.SUM)
                s_nll, s_tok, s_bytes = (float(v) for v in stats_t.cpu())
                out[name] = {"bpb": s_nll / (math.log(2) * max(s_bytes, 1)), "ppl_token": math.exp(s_nll / max(s_tok, 1)),
                             "bytes": int(s_bytes), "tokens": int(s_tok)}
            return out
        except Exception as exc:
            eval_ok = False
            if rank == 0:
                print(f"[native] eval failed, disabling: {exc}", flush=True)
            return {}

    def log_eval(res, final=False):
        if not res or rank != 0:
            return
        tag = "eval_final" if final else "eval"
        _log(rank, log_path, {"step": step, "tokens": tokens, tag: res})
        print(f"[{tag}] {tokens/1e9:.3f}B: " + ", ".join(f"{k}={v['bpb']:.4f}" for k, v in res.items()), flush=True)
        if wandb:
            try:
                wandb.log({f"{tag}/{k}_bpb": v["bpb"] for k, v in res.items()} |
                          {f"{tag}/{k}_ppl": v["ppl_token"] for k, v in res.items()} | {"tokens_b": tokens / 1e9}, step=step)
            except Exception:
                pass

    t_ev = time.time(); res = evaluate(); log_eval(res)   # one point at (re)start: validates the path, seeds the curve
    if rank == 0 and res:
        print(f"[native] startup eval took {time.time()-t_ev:.0f}s (includes the one-time eval-shape compile)", flush=True)

    # ---- loop ----------------------------------------------------------------
    log_every = cfg["train"].get("log_every", 20)
    t0 = time.time()
    n_since = 0
    loss_hist = []
    while tokens < target:
        lr = lr_at(tokens, sched)
        for g in optim.param_groups:
            g["lr"] = lr
        for g in getattr(optim, "optim", optim).param_groups:
            g["lr"] = lr
        arr, counts = mixer.rank_batch(global_windows, rank, world)
        t = torch.from_numpy(arr.astype(np.int32))
        optim.zero_grad(set_to_none=True)
        acc = None
        for i in range(0, t.size(0), mb):
            x = t[i:i + mb, :-1].contiguous().to(dev)
            y = t[i:i + mb, 1:].contiguous().to(dev)
            with torch.autocast("neuron", dtype=torch.bfloat16):
                loss = forward_loss(x, y)
            # NEVER scale the loss before backward on this stack: a non-power-of-two
            # grad_output scale (1/15 here) corrupts the compiled backward (gradients
            # vs CPU fp32: cos 0.90, RMSNorm-gain grads off by 0.5-1.3x; 1/2, 1/4, 1/16
            # are exact). Backprop unscaled and divide the fp32 master grads instead.
            loss.backward()
            d = loss.detach() / grad_accum
            acc = d if acc is None else acc + d
        for pm in master.parameters():
            pm.grad.div_(grad_accum)
            dist.all_reduce(pm.grad, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_(master.parameters(), grad_clip)
        optim.step()
        refresh_shadow()
        tokens += tokens_per_step
        step += 1
        n_since += 1
        loss_hist.append(acc)
        if step % log_every == 0:
            torch_neuronx.synchronize()
            lv = float(torch.stack(loss_hist).mean())
            loss_hist = []
            dt = time.time() - t0
            tps = tokens_per_step * n_since / dt
            rec = {"step": step, "tokens": tokens, "lr": lr, "loss": lv, "tok_per_s": round(tps),
                   "mix": mixer.stats(), "mfu": round(tps * 6.54e9 / (world / 4 * 667e12), 4)}
            try:   # allocator health: retries/OOM events are the fragmentation tell (NEURON.md 10b row 8)
                ms = torch_neuronx.memory_stats()
                rec["mem"] = {k: ms[k] for k in ("num_alloc_retries", "num_ooms") if k in ms}
                rec["mem"]["reserved_gib"] = round(ms.get("reserved_bytes.all.current", 0) / 2**30, 2)
                rec["mem"]["allocated_gib"] = round(ms.get("allocated_bytes.all.current", 0) / 2**30, 2)
            except Exception:
                pass
            _log(rank, log_path, rec)
            if rank == 0:
                print(f"[native] step {step} | {tokens/1e9:.3f}B | loss {lv:.4f} | lr {lr:.2e} | "
                      f"{tps/1e3:.0f}k tok/s | MFU {rec['mfu']*100:.1f}% | mem {rec.get('mem', {})}", flush=True)
                if wandb:
                    try:
                        wandb.log({**rec, "tokens_b": tokens / 1e9}, step=step)
                    except Exception:
                        pass
            t0 = time.time()
            n_since = 0
        if tokens - last_ckpt_tokens >= ckpt_interval(tokens, ckpt_table):
            last_ckpt_tokens = tokens
            save("last", True)
            save(f"step{step}_{int(tokens/1e6)}M", False)
            log_eval(evaluate())
            t0 = time.time(); n_since = 0      # don't count save+eval time in the next tok/s window
        if args.empty_cache_every and step % args.empty_cache_every == 0:
            torch_neuronx.empty_cache()
        if args.max_steps and step >= args.max_steps:
            break
    if not args.max_steps:
        save("last", True)
        save("final", False)
        log_eval(evaluate(), final=True)
        if rank == 0:
            print(f"[native] DONE {cfg['name']} @ {tokens/1e9:.2f}B tokens", flush=True)
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
