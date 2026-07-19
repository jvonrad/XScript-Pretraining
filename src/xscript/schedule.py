"""Learning-rate and checkpoint schedules (token-indexed).

WSD (warmup-stable-decay): a long constant-LR trunk lets us checkpoint a
`stable` point and later branch a cheap cooldown to any token budget without
retraining the trunk -- the mechanism the plan uses to extend 4 runs to 100B.

All schedules are pure functions of tokens-seen-in-this-trajectory, so a
cooldown branch is just a run whose schedule has warmup=stable=0.
"""
import math


def lr_at(t: float, sched: dict) -> float:
    peak, mn = sched["peak_lr"], sched.get("min_lr", 0.0)
    w = sched.get("warmup_tokens", 0)
    s = sched.get("stable_tokens", 0)
    d = sched.get("decay_tokens", 0)
    if t < w:
        return peak * t / max(w, 1)
    if t < w + s:
        return peak
    if t < w + s + d:
        prog = (t - w - s) / max(d, 1)
        shape = sched.get("decay_shape", "1-sqrt")
        if shape == "linear":
            f = 1.0 - prog
        elif shape == "cosine":
            f = 0.5 * (1.0 + math.cos(math.pi * prog))
        else:  # "1-sqrt" (MiniCPM WSD; empirically strong)
            f = 1.0 - math.sqrt(prog)
        return mn + (peak - mn) * f
    return mn


def total_tokens(sched: dict) -> float:
    return (sched.get("warmup_tokens", 0) + sched.get("stable_tokens", 0)
            + sched.get("decay_tokens", 0))


def stable_end_tokens(sched: dict) -> float:
    """Token count at which the trunk's decay begins (branch point)."""
    return sched.get("warmup_tokens", 0) + sched.get("stable_tokens", 0)


def ckpt_interval(tokens: float, table: list) -> float:
    """Log-spaced checkpoint interval: dense early, sparse late."""
    for up_to, interval in table:
        if tokens < up_to:
            return interval
    return table[-1][1]
