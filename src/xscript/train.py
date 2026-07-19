"""Pretraining loop: DDP + bf16, WSD schedule, deterministic resume.

One run = one (mixture, tokenizer) cell of the design. The mixture is driven
entirely by the run config's `langs`/`probs`; the tokenizer by `tok_name`. The
loader is globally deterministic and world-size-independent, so a run resumed on
a different node count sees the exact same token stream.

Cooldown branch: set `branch.from` to a `stable` checkpoint; the schedule then
has warmup=stable=0 and only decays, giving a cheap final model at a larger
token budget without retraining the trunk.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .model import ModelConfig, Transformer
from .data.loader import MixedStream
from .schedule import lr_at, ckpt_interval, stable_end_tokens, total_tokens
from .paths import run_dir, ensure


def _ddp():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        return dist.is_initialized(), rank, world, local
    return False, 0, 1, 0


def _log(rank, path, rec):
    if rank == 0:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dist, self.rank, self.world, self.local = _ddp()
        self.device = torch.device(f"cuda:{self.local}" if torch.cuda.is_available() else "cpu")
        self.is_cuda = self.device.type == "cuda"
        torch.manual_seed(cfg.get("seed", 0))
        np.random.seed(cfg.get("seed", 0))

        mc = cfg["model"]
        self.mcfg = ModelConfig(**mc)
        self.seq_len = self.mcfg.max_seq_len
        raw_model = Transformer(self.mcfg).to(self.device)
        if self.rank == 0:
            print(f"[train] params: {raw_model.num_params(False)/1e6:.1f}M non-embedding, "
                  f"{raw_model.num_params(True)/1e6:.1f}M total")
        # Keep the canonical module unwrapped for portable state_dict keys.
        # Compilation is only the forward/backward execution path.
        model = raw_model
        if cfg.get("compile", False) and self.is_cuda:
            model = torch.compile(model)
        self.raw_model = raw_model
        if self.dist:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[self.local] if self.is_cuda else None)
        self.model = model

        opt = cfg.get("optim", {})
        self.optim = torch.optim.AdamW(
            self.raw_model.parameters(),
            lr=cfg["schedule"]["peak_lr"],
            betas=tuple(opt.get("betas", (0.9, 0.95))),
            weight_decay=opt.get("weight_decay", 0.1),
            eps=opt.get("eps", 1e-8),
        )
        self.grad_clip = opt.get("grad_clip", 1.0)

        # global batch bookkeeping
        self.micro_bsz = cfg["train"]["micro_batch_size"]
        gbt = cfg["train"]["global_batch_tokens"]
        per_step_windows = max(1, round(gbt / self.seq_len))
        # round up to a multiple of micro_bsz*world so each rank gets equal work
        unit = self.micro_bsz * self.world
        self.global_windows = max(unit, (per_step_windows // unit) * unit)
        self.grad_accum = self.global_windows // unit
        self.tokens_per_step = self.global_windows * self.seq_len
        if self.rank == 0:
            print(f"[train] global batch: {self.global_windows} windows "
                  f"({self.tokens_per_step/1e6:.3f}M tokens), grad_accum={self.grad_accum}")

        # schedule (branch collapses warmup/stable)
        self.sched = dict(cfg["schedule"])
        self.branch = cfg.get("branch")
        if self.branch:
            self.sched["warmup_tokens"] = 0
            self.sched["stable_tokens"] = 0
        self.target_tokens = total_tokens(self.sched)
        self.ckpt_table = cfg["train"].get("ckpt_schedule",
                                           [[2e9, 250e6], [5e9, 500e6],
                                            [15e9, 1e9], [1e15, 2e9]])
        # token budgets at which a stable trunk saves a named branch point
        self.stable_marks = sorted(cfg["train"].get("stable_marks", []))
        self.marks_done = set()

        # data mixer
        self.mixer = MixedStream(cfg["langs"], cfg["tok_name"], self.seq_len,
                                 seed=cfg.get("data_seed", 1234),
                                 probs=cfg.get("probs"))

        self.rdir = ensure(run_dir(cfg["name"]))
        self.log_path = self.rdir / "train.jsonl"
        self.tokens = 0
        self.step = 0
        self.last_ckpt_tokens = 0
        self.saved_stable = False

        self.wandb = None
        if self.rank == 0:
            try:
                import wandb
                self.wandb = wandb.init(
                    project="XScript-Pretraining", name=cfg["name"],
                    id=cfg.get("wandb_id", cfg["name"]),
                    resume="allow", config=cfg,
                )
                self.wandb.summary["params_total_M"] = self.raw_model.num_params(True) / 1e6
                self.wandb.summary["params_non_embed_M"] = self.raw_model.num_params(False) / 1e6
            except Exception as exc:
                print(f"[train] wandb disabled ({exc})")

    # ---- checkpoint io ----
    def _ckpt_path(self, tag):
        return ensure(self.rdir / "checkpoints") / f"{tag}.pt"

    def save(self, tag, resumable=True):
        if self.rank != 0:
            return
        payload = {
            "model": self.raw_model.state_dict(),
            "step": self.step, "tokens": self.tokens,
            "cfg": self.cfg,
        }
        if resumable:
            payload.update({
                "optim": self.optim.state_dict(),
                "mixer": self.mixer.state_dict(),
                "last_ckpt_tokens": self.last_ckpt_tokens,
                "saved_stable": self.saved_stable,
                "torch_rng": torch.get_rng_state(),
            })
        torch.save(payload, self._ckpt_path(tag))
        kind = "full" if resumable else "model-only"
        print(f"[train] saved {tag} ({kind}) @ {self.tokens/1e9:.3f}B tokens")

    def maybe_resume(self):
        last = self._ckpt_path("last")
        if self.branch and not last.exists():
            ck = torch.load(self.branch["from"], map_location="cpu", weights_only=False)
            self.raw_model.load_state_dict(ck["model"])
            if self.branch.get("load_optim", True):
                self.optim.load_state_dict(ck["optim"])
            if self.rank == 0:
                print(f"[train] branched from {self.branch['from']} "
                      f"@ {ck['tokens']/1e9:.3f}B (cooldown {self.target_tokens/1e9:.1f}B)")
            return
        if last.exists():
            ck = torch.load(last, map_location="cpu", weights_only=False)
            self.raw_model.load_state_dict(ck["model"])
            self.optim.load_state_dict(ck["optim"])
            self.mixer.load_state_dict(ck["mixer"])
            self.step = ck["step"]; self.tokens = ck["tokens"]
            self.last_ckpt_tokens = ck["last_ckpt_tokens"]
            self.saved_stable = ck.get("saved_stable", False)
            self.marks_done = {m for m in self.stable_marks if m <= self.tokens}
            torch.set_rng_state(ck["torch_rng"])
            if self.rank == 0:
                print(f"[train] resumed @ step {self.step}, {self.tokens/1e9:.3f}B tokens")

    # ---- data ----
    def _next_micro_batches(self):
        """Return grad_accum micro-batches of (x, y) on device for this rank."""
        arr, counts = self.mixer.rank_batch(self.global_windows, self.rank, self.world)
        # arr: (global_windows/world, seq_len+1)
        t = torch.from_numpy(arr.astype(np.int64))
        x = t[:, :-1].to(self.device, non_blocking=True)
        y = t[:, 1:].to(self.device, non_blocking=True)
        micros = [(x[i:i + self.micro_bsz], y[i:i + self.micro_bsz])
                  for i in range(0, x.size(0), self.micro_bsz)]
        return micros, counts

    # ---- eval ----
    def _eval_sources(self):
        from .langs import LANGS
        srcs = {}
        for l in self.cfg["langs"]:
            try:
                from .eval.bpb import load_holdout
                h = load_holdout(l, self.cfg["train"].get("eval_docs", 500))
                if h:
                    srcs[f"holdout_{l}"] = h
            except Exception:
                pass
        try:
            from . import flores
            par = flores.load_parallel(list(self.cfg["langs"]), "dev")
            for l, sents in par.items():
                srcs[f"flores_{l}"] = sents
        except Exception as e:
            if self.rank == 0:
                print(f"[train] flores eval skipped: {e}")
        return srcs

    def evaluate(self):
        if self.rank != 0:
            return {}
        from .eval.bpb import eval_sources
        from .tok.wrapper import Tok
        from .paths import tokenizer_dir
        tok = Tok(tokenizer_dir(self.cfg["tok_name"]))
        res = eval_sources(self.raw_model, tok, self._eval_sources(),
                           self.device, self.seq_len)
        self.model.train()
        return res

    # ---- loop ----
    def train(self):
        self.maybe_resume()
        self.model.train()
        t0 = time.time()
        log_every = self.cfg["train"].get("log_every", 20)
        while self.tokens < self.target_tokens:
            lr = lr_at(self.tokens, self.sched)
            for g in self.optim.param_groups:
                g["lr"] = lr

            micros, counts = self._next_micro_batches()
            self.optim.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for j, (x, y) in enumerate(micros):
                sync = (not self.dist) or (j == len(micros) - 1)
                ctx = self.model.no_sync() if (self.dist and not sync) else _null()
                with ctx:
                    with torch.autocast("cuda", dtype=torch.bfloat16) if self.is_cuda else _null():
                        _, loss = self.model(x, y)
                    (loss / len(micros)).backward()
                loss_acc += loss.detach().item() / len(micros)
            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.grad_clip)
            self.optim.step()

            self.tokens += self.tokens_per_step
            self.step += 1

            if self.step % log_every == 0:
                dt = time.time() - t0
                tps = self.tokens_per_step * log_every / dt if dt > 0 else 0
                rec = {
                    "step": self.step, "tokens": self.tokens, "lr": lr,
                    "loss": loss_acc, "tok_per_s": round(tps),
                    "mix": self.mixer.stats(),
                }
                _log(self.rank, self.log_path, rec)
                if self.wandb:
                    self.wandb.log({**rec, "tokens_b": self.tokens / 1e9}, step=self.step)
                if self.rank == 0:
                    print(f"[train] step {self.step} | {self.tokens/1e9:.2f}B | "
                          f"loss {loss_acc:.4f} | lr {lr:.2e} | {tps/1e3:.0f}k tok/s")
                t0 = time.time()

            # stable checkpoint exactly once, at the trunk's decay boundary
            if (not self.branch and not self.saved_stable
                    and self.tokens >= stable_end_tokens(self.sched)):
                self.save("stable")
                self.saved_stable = True

            # named branch points for cooldown extensions (100B runs)
            for mark in self.stable_marks:
                if mark not in self.marks_done and self.tokens >= mark:
                    self.save(f"stable_{int(mark/1e6)}M")
                    self.marks_done.add(mark)

            # log-spaced checkpoint + eval
            if self.tokens - self.last_ckpt_tokens >= ckpt_interval(self.tokens, self.ckpt_table):
                self.last_ckpt_tokens = self.tokens
                self.save("last")
                self.save(f"step{self.step}_{int(self.tokens/1e6)}M", resumable=False)
                res = self.evaluate()
                _log(self.rank, self.log_path,
                     {"step": self.step, "tokens": self.tokens, "eval": res})
                if self.wandb and res:
                    self.wandb.log({f"eval/{k}_bpb": v["bpb"] for k, v in res.items()} |
                                   {f"eval/{k}_ppl": v["ppl_token"] for k, v in res.items()},
                                   step=self.step)
                if self.rank == 0 and res:
                    brief = {k: round(v["bpb"], 4) for k, v in res.items()}
                    print(f"[eval] {self.tokens/1e9:.2f}B: {brief}")

        self.save("last")
        self.save("final", resumable=False)
        res = self.evaluate()
        _log(self.rank, self.log_path,
             {"step": self.step, "tokens": self.tokens, "eval_final": res})
        if self.wandb:
            if res:
                self.wandb.log({f"eval_final/{k}_bpb": v["bpb"] for k, v in res.items()} |
                               {f"eval_final/{k}_ppl": v["ppl_token"] for k, v in res.items()},
                               step=self.step)
            self.wandb.finish()
        if self.rank == 0:
            print(f"[train] DONE {self.cfg['name']} @ {self.tokens/1e9:.2f}B tokens")
        if self.dist:
            import torch.distributed as dist
            dist.destroy_process_group()


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def run_from_config(cfg: dict):
    Trainer(cfg).train()
