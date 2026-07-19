"""Generate per-run training configs from a base yaml.

The design covers 9 mixtures (5 monolingual + 4 EN-anchored bilingual
mixtures) x 2 tokenizer conditions (starved / destarved) for one chosen BPE
flavor = 18 runs at 30B tokens each. Four of them (EN-DE and EN-AR under both
conditions) are additionally extended to 100B via a WSD cooldown branch off a
long stable trunk -- the cheap-extension mechanism.

A GPU-hour budget shortfall (2026-07-16: ~1200h available vs ~2615h needed
for the full 30-run all-pairs matrix) forced cutting the 12 non-EN-anchored
pairwise mixtures (de-fr, de-ar, de-zh, fr-ar, fr-zh, ar-zh) that a prior
expansion had added here; reverted to the original EN-anchored-only design.
Their partial checkpoints (0-12.4B tokens) are left on disk, unused.

A run's config is this base block plus: name, langs, probs, tok_name, schedule
overrides, and (for cooldowns) a `branch` pointing at a trunk checkpoint.

Run-name convention (kept in sync with eval.bts):
  "<mix>__<tok>"          e.g. en-ar__bl_destarved        (30B deliverable)
  "<mix>__<tok>__trunk"   long stable trunk for extension
  "<mix>__<tok>__100b"    100B cooldown deliverable
where <mix> is "en" or "en-ar" etc., <tok> is "<flavor>_<condition>".
"""
import copy
from pathlib import Path

from .langs import LANGS, ANCHOR, PARTNERS, tok_conditions, tok_name
from .paths import run_dir

BILINGUAL_MIXTURES = [f"{ANCHOR}-{p}" for p in PARTNERS]
MIXTURES = [*LANGS, *BILINGUAL_MIXTURES]
# Cells extended to 100B via cooldown branch (plan's recommendation): one
# same-script (EN-DE) and one cross-script (EN-AR) partner, both tokenizer
# conditions = 4 extended runs. The same-vs-cross contrast at 100B is the point.
EXTENDED_MIXTURES = [f"{ANCHOR}-de", f"{ANCHOR}-ar"]
TRUNK_TOKENS = 80e9
# cooldown budgets: {final_total_tokens: (branch_at_trunk_tokens, decay_tokens)}
COOLDOWNS = {30e9: (24e9, 6e9), 100e9: (80e9, 20e9)}


def load_base(path) -> dict:
    from . import _yaml
    return _yaml.load(path)


def mix_langs(mix: str) -> list[str]:
    return mix.split("-")


def _stamp(base: dict, name: str, mix: str, tok: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg["name"] = name
    cfg["langs"] = mix_langs(mix)
    cfg["tok_name"] = tok
    cfg.setdefault("probs", None)   # None -> uniform (token-level 50/50 bilingual)
    return cfg


def wsd_run(base, mix, tok) -> dict:
    """Self-contained warmup+stable+decay to 30B (base schedule as-is)."""
    return _stamp(base, f"{mix}__{tok}", mix, tok)


def trunk_run(base, mix, tok) -> dict:
    cfg = _stamp(base, f"{mix}__{tok}__trunk", mix, tok)
    warm = cfg["schedule"]["warmup_tokens"]
    cfg["schedule"] = {**cfg["schedule"],
                       "stable_tokens": TRUNK_TOKENS - warm, "decay_tokens": 0}
    cfg["train"] = {**cfg["train"],
                    "stable_marks": sorted({b for b, _ in COOLDOWNS.values()})}
    return cfg


def cooldown_run(base, mix, tok, total_tokens) -> dict:
    branch_at, decay = COOLDOWNS[total_tokens]
    suffix = "" if total_tokens == 30e9 else f"__{int(total_tokens/1e9)}b"
    cfg = _stamp(base, f"{mix}__{tok}{suffix}", mix, tok)
    cfg["schedule"] = {**cfg["schedule"], "warmup_tokens": 0,
                       "stable_tokens": 0, "decay_tokens": decay}
    trunk_ckpt = (run_dir(f"{mix}__{tok}__trunk") / "checkpoints"
                  / f"stable_{int(branch_at/1e6)}M.pt")
    cfg["branch"] = {"from": str(trunk_ckpt), "load_optim": True}
    return cfg


def all_runs(base: dict, flavor: str) -> dict[str, dict]:
    """Every run needed for the full experiment, keyed by run name."""
    runs: dict[str, dict] = {}
    for mix in MIXTURES:
        for cond in tok_conditions(flavor):
            tok = tok_name(flavor, cond)
            if mix in EXTENDED_MIXTURES:
                runs[f"{mix}__{tok}__trunk"] = trunk_run(base, mix, tok)
                for total in COOLDOWNS:
                    r = cooldown_run(base, mix, tok, total)
                    runs[r["name"]] = r
            else:
                r = wsd_run(base, mix, tok)
                runs[r["name"]] = r
    return runs


def all_30b_runs(base: dict, flavor: str) -> dict[str, dict]:
    """The 30 independent 30B cells, without 100B-extension trunks.

    This mode is useful when the immediate experiment scope is only 30B.  In
    the full matrix, four of these names are cooldown branches from shared
    trunks; here every name instead gets the base self-contained WSD schedule.
    """
    runs: dict[str, dict] = {}
    for mix in MIXTURES:
        for cond in tok_conditions(flavor):
            tok = tok_name(flavor, cond)
            run = wsd_run(base, mix, tok)
            runs[run["name"]] = run
    return runs


def get_run(base_path, flavor: str, name: str, only_30b: bool = False) -> dict:
    base = load_base(base_path)
    runs = all_30b_runs(base, flavor) if only_30b else all_runs(base, flavor)
    if name not in runs:
        raise KeyError(f"unknown run '{name}'. Available:\n  " +
                       "\n  ".join(sorted(runs)))
    return runs[name]


def list_runs(base_path, flavor: str, only_30b: bool = False) -> list[str]:
    base = load_base(base_path)
    runs = all_30b_runs(base, flavor) if only_30b else all_runs(base, flavor)
    return sorted(runs)
