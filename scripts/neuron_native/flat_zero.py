"""Lean ZeRO-1 on flat buffers (memory + collective count), math-identical to
train.py's AdamW + clip_grad_norm_.

* all fp32 params are re-pointed to views of ONE contiguous buffer `flat`;
  their .grad to views of ONE `flat_grad` (autograd accumulates in place:
  AccumulateGrad does `grad += new` when it holds the only ref to the view).
* step(): reduce_scatter(flat_grad) -> this rank's grad shard (AVG over
  ranks, like DDP); global grad norm via one tiny all_reduce; clip exactly as
  torch.nn.utils.clip_grad_norm_; torch.optim.AdamW on the shard (identical
  per-element update, uniform weight decay as in train.py); all_gather the
  updated shard back into `flat`.
* memory per rank: flat (N) + flat_grad (N) + shard grad (N/W) + Adam m,v
  (2N/W) + shard scratch (N/W). Nothing else.
"""
import torch
import torch.distributed as dist


class FlatZeRO1:
    def __init__(self, params, lr, betas, weight_decay, eps, grad_clip,
                 world, rank, device):
        self.params = [p for p in params if p.requires_grad]
        self.world, self.rank, self.grad_clip = world, rank, grad_clip
        total = sum(p.numel() for p in self.params)
        self.pad = (-total) % world
        self.N = total + self.pad
        self.S = self.N // world
        self.flat = torch.zeros(self.N, dtype=torch.float32, device=device)
        self.flat_grad = torch.zeros(self.N, dtype=torch.float32, device=device)
        off = 0
        with torch.no_grad():
            for p in self.params:
                n = p.numel()
                self.flat[off:off + n].copy_(p.data.reshape(-1))
                p.data = self.flat[off:off + n].view_as(p)
                off += n
        self._attach_grads()
        s0 = rank * self.S
        self.shard_param = torch.nn.Parameter(self.flat[s0:s0 + self.S])   # view
        self.shard_grad = torch.zeros(self.S, dtype=torch.float32, device=device)
        self.shard_param.grad = self.shard_grad
        self.opt = torch.optim.AdamW([self.shard_param], lr=lr, betas=betas,
                                     weight_decay=weight_decay, eps=eps)
        self.param_groups = self.opt.param_groups
        self.shard_scratch = torch.empty(self.S, dtype=torch.float32, device=device)

    def _attach_grads(self):
        off = 0
        for p in self.params:
            n = p.numel()
            p.grad = self.flat_grad[off:off + n].view_as(p)   # no other ref kept
            off += n

    def zero_grad(self, set_to_none=False):
        self.flat_grad.zero_()
        # re-attach in case autograd replaced any .grad out of place
        for p in self.params:
            if p.grad is None or p.grad.data_ptr() != p.grad.data_ptr():
                pass
        self._attach_grads()

    @torch.no_grad()
    def step(self):
        if self.world > 1:
            dist.reduce_scatter_tensor(self.shard_grad, self.flat_grad, op=dist.ReduceOp.AVG)
        else:
            self.shard_grad.copy_(self.flat_grad)
        # global grad norm (== clip_grad_norm_'s total_norm over all params)
        sq = self.shard_grad.pow(2).sum()
        if self.world > 1:
            dist.all_reduce(sq, op=dist.ReduceOp.SUM)
        total_norm = sq.sqrt()
        coef = (self.grad_clip / (total_norm + 1e-6)).clamp(max=1.0)
        self.shard_grad.mul_(coef)
        self.opt.step()                       # updates self.flat[s0:s0+S] in place
        if self.world > 1:
            # all_gather_into_tensor(4GB) fails on this beta ("NRT model
            # scheduling failed"); broadcast each rank's 1GB shard instead
            # (the collective ZeroRedundancyOptimizer relies on, known good).
            for r in range(self.world):
                dist.broadcast(self.flat[r * self.S:(r + 1) * self.S], src=r)
        return total_norm

    def state_dict(self):
        return {"opt": self.opt.state_dict(), "rank": self.rank, "world": self.world}
