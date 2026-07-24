"""Language-specific neurons via LAPE (Tang et al. 2024, arXiv:2402.16438).

Port of https://github.com/RUCAIBox/Language-Specific-Neurons to our
``model.Transformer``. Two halves:

1. **Recording** (their ``activation.py``, vLLM-hooked): for every SwiGLU FFN
   neuron count how often its post-activation value is positive. For SiLU,
   ``silu(z) > 0  <=>  z > 0``, so we test the *gate pre-activation*
   ``w1(ffn_norm(x)) > 0`` -- numerically identical to the paper's
   ``SiLU(gate) > 0`` but cheaper. Counts are accumulated per language over
   FLORES+ text into ``over_zero[layer, neuron]`` plus a token count ``n``.

2. **Identification** (their ``identify.py``, kept hyperparameter-for-
   hyperparameter): activation probability per (layer, neuron, language),
   normalise across languages, entropy -> LAPE. Keep the bottom ``top_rate``
   entropy neurons among those whose max probability clears the
   ``filter_rate`` percentile; a kept neuron belongs to every language whose
   probability clears the ``activation_bar_ratio`` percentile bar.

Neuron/XLA constraints honoured here (NEURON.md section 4):
  - one fixed ``[batch, width]`` graph shape for the whole run (inputs padded
    on the host, moved once);
  - no bool ``.sum()`` on device (silently returns -1 on this build) -- the
    comparison is cast to float32 before reducing;
  - no ``torch.gather``, no on-device scatter.

The counting mask excludes PAD positions and the leading BOS (its activation
is shared boilerplate, not language evidence).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..langs import LANGS
from ..paths import RUNS, ensure
from ..tok.wrapper import PAD_ID, Tok


# --------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------

@torch.no_grad()
def _over_zero_batch(model, idx, mask, cos, sin):
    """One fixed-shape forward; returns float32 ``[n_layers, ffn_dim]`` counts.

    Re-implements ``Transformer.forward``'s loop with the SwiGLU inlined so the
    gate pre-activation is observed exactly once per layer (no double compute,
    no hooks -- XLA-friendly).
    """
    x = model.tok_emb(idx)
    m = mask[:, :, None]                                   # [B, T, 1]
    counts = []
    for layer in model.layers:
        x = x + layer.attn(layer.attn_norm(x), cos, sin)
        h = layer.ffn_norm(x)
        g = layer.ffn.w1(h)                                # gate pre-activation
        counts.append(((g > 0).to(torch.float32) * m).sum(dim=(0, 1)))
        x = x + layer.ffn.w2(F.silu(g) * layer.ffn.w3(h))
    return torch.stack(counts, dim=0)                      # [n_layers, ffn_dim]


@torch.no_grad()
def record(model, seqs, device, batch: int = 16, fixed_width: int | None = None):
    """Accumulate over-zero counts for one language.

    ``seqs``: list of token-id lists (BOS-prefixed). Returns
    ``(over_zero float64 [n_layers, ffn_dim], n_tokens int)``. Positions
    counted are real non-BOS tokens only.
    """
    model.eval()
    n_layers = len(model.layers)
    ffn_dim = model.cfg.ffn_dim
    total = np.zeros((n_layers, ffn_dim), dtype=np.float64)
    n_tokens = 0
    for s0 in range(0, len(seqs), batch):
        chunk = seqs[s0:s0 + batch]
        if fixed_width is None:
            width, nrow = max(len(s) for s in chunk), len(chunk)
        else:
            width, nrow = fixed_width, batch      # constant rows: one graph
        arr = np.full((nrow, width), PAD_ID, dtype=np.int64)
        msk = np.zeros((nrow, width), dtype=np.float32)
        for i, s in enumerate(chunk):
            arr[i, :len(s)] = s
            msk[i, 1:len(s)] = 1.0                # 1: skip BOS at position 0
        idx = torch.from_numpy(arr).to(device)
        mask = torch.from_numpy(msk).to(device)
        cos, sin = model._rope_for(width, idx.device, model.tok_emb.weight.dtype)
        counts = _over_zero_batch(model, idx, mask, cos, sin)
        if device.type == "xla":
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        total += counts.cpu().numpy().astype(np.float64)
        n_tokens += int(msk.sum())
    return total, n_tokens


def fixed_width_for(encoded: dict[str, list[list[int]]]) -> int:
    """One even sequence width covering every language (NCC-5266: odd widths
    fail to compile on Neuron)."""
    w = max(len(s) for seqs in encoded.values() for s in seqs)
    return max(w + (w % 2), 2)


# --------------------------------------------------------------------------
# identification (faithful port of identify.py)
# --------------------------------------------------------------------------

def lape(over_zero: np.ndarray, n: np.ndarray,
         top_rate: float = 0.01, filter_rate: float = 0.95,
         activation_bar_ratio: float = 0.95) -> dict:
    """LAPE selection. ``over_zero``: [n_layers, ffn_dim, n_langs] counts;
    ``n``: [n_langs] token totals.

    Returns per-language neuron indices plus the intermediate statistics so
    analysis can vary thresholds without re-recording.
    """
    over_zero = torch.from_numpy(np.asarray(over_zero, dtype=np.float64))
    n = torch.from_numpy(np.asarray(n, dtype=np.float64))
    num_layers, ffn_dim, lang_num = over_zero.size()

    activation_probs = over_zero / n                       # broadcast over langs
    normed = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed[torch.isnan(normed)] = 0
    log_probs = torch.where(normed > 0, normed.log(), torch.zeros_like(normed))
    entropy = -torch.sum(normed * log_probs, dim=-1)       # [layers, ffn]
    if torch.isnan(entropy).sum():
        raise ValueError("NaN entropy")

    flattened_probs = activation_probs.flatten()
    top_prob_value = flattened_probs.kthvalue(
        round(len(flattened_probs) * filter_rate)).values.item()
    # dismiss neurons where no language's activation clears the percentile
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy = entropy.clone()
    entropy[top_position == 0] = torch.inf

    flattened_entropy = entropy.flatten()
    k = round(len(flattened_entropy) * top_rate)
    _, index = flattened_entropy.topk(k, largest=False)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index]      # [k, langs]
    activation_bar = flattened_probs.kthvalue(
        round(len(flattened_probs) * activation_bar_ratio)).values.item()

    sel_t = selected_probs.transpose(0, 1)                       # [langs, k]
    lang_idx, indice = torch.where(sel_t > activation_bar)
    merged_index = torch.stack((row_index, col_index), dim=-1)
    final_indice = []
    counts = torch.bincount(lang_idx, minlength=lang_num).tolist()
    for chunk in indice.split(counts):
        pairs = sorted(tuple(r.tolist()) for r in merged_index[chunk])
        layer_index = [[] for _ in range(num_layers)]
        for l, h in pairs:
            layer_index[l].append(h)
        final_indice.append([torch.tensor(h).long() for h in layer_index])

    return {
        "neurons": final_indice,           # [lang][layer] -> LongTensor of ids
        "entropy": entropy,                # [layers, ffn] (inf = filtered out)
        "activation_probs": activation_probs,
        "selected_rowcol": merged_index,   # [k, 2] (layer, neuron) of low-entropy set
        "top_prob_value": top_prob_value,
        "activation_bar": activation_bar,
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def compute(run_name: str, tok_name: str, langs: list[str], split: str = "both",
            model=None, device=None, batch: int = 16, max_tokens: int = 256,
            limit: int | None = None) -> dict:
    """Record over-zero counts for ``run_name`` on every language in ``langs``.

    Returns {langs, n, over_zero [L, ffn, lang], width, meta...}; identification
    is left to the analysis side so thresholds stay tunable.
    """
    from .. import flores
    from ..paths import tokenizer_dir
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_langs = None
    if model is None:
        model, train_langs = _load_model(run_name, dev)
    langs = [l for l in LANGS if l in set(langs)]

    if split == "both":
        halves = [flores.load_parallel(langs, s) for s in ("dev", "devtest")]
        par = {l: halves[0][l] + halves[1][l] for l in langs}
    else:
        par = flores.load_parallel(langs, split)
    if limit:
        par = {l: s[:limit] for l, s in par.items()}
    tok = Tok(tokenizer_dir(tok_name))
    encoded = {l: [tok.encode(t, bos=True)[:max_tokens] for t in par[l]]
               for l in langs}
    width = fixed_width_for(encoded) if dev.type == "xla" else None

    over, ns = [], []
    for l in langs:
        oz, n = record(model, encoded[l], dev, batch=batch, fixed_width=width)
        print(f"[lape] {run_name} {l}: n={n} mean_p={oz.sum() / (n * oz.size):.4f}",
              flush=True)
        over.append(oz)
        ns.append(n)
    return {
        "run": run_name, "tok": tok_name, "langs": langs,
        "train_langs": train_langs, "split": split,
        "n_sentences": len(par[langs[0]]), "max_tokens": max_tokens,
        "fixed_width": width,
        "n": np.array(ns, dtype=np.int64),
        "over_zero": np.stack(over, axis=-1),   # [layers, ffn, langs]
    }


def save(out_dir, res: dict) -> Path:
    out_dir = ensure(Path(out_dir))
    out = out_dir / f"{res['run']}.npz"
    tmp = out.with_suffix(".npz.tmp")
    meta = {k: v for k, v in res.items() if k not in ("n", "over_zero")}
    with open(tmp, "wb") as fh:
        np.savez(fh, __meta__=np.array(json.dumps(meta)),
                 n=res["n"], over_zero=res["over_zero"])
    tmp.rename(out)
    return out


def load(out_dir, run_name: str) -> dict:
    z = np.load(Path(out_dir) / f"{run_name}.npz")
    meta = json.loads(str(z["__meta__"]))
    return {**meta, "n": z["n"], "over_zero": z["over_zero"]}


def _load_model(run_name: str, device, tag: str = "final"):
    from ..model import ModelConfig, Transformer
    ck = torch.load(RUNS / run_name / "checkpoints" / f"{tag}.pt",
                    map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**ck["cfg"]["model"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), list(ck["cfg"]["langs"])
