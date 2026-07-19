"""MEXA-style cross-lingual representation alignment on FLORES+ dev.

For every layer, embed each language's parallel sentences by mean-pooling that
layer's hidden states, then measure how well EN sentences retrieve their
translations (and vice versa). High alignment on cross-script pairs would say
the model builds a shared multilingual space despite the script gap -- the
representation-side counterpart to the BPB/BTS story.

Reported per (EN, partner) pair, at the best-aligned layer:
  - top-1 EN->L and L->EN retrieval accuracy
  - mutual nearest-neighbour rate
  - mean cosine similarity of translations and its margin over non-pairs
Cross-script (AR/ZH) vs same-script (DE/FR), and starved vs destarved, is the
comparison of interest.

Only languages in the checkpoint's training mixture are embedded. Monolingual
runs therefore have no cross-lingual pair; an EN-partner bilingual run reports
exactly that pair.
"""
import json
from pathlib import Path

import numpy as np
import torch

from ..langs import ANCHOR, LANGS
from ..paths import RUNS, RESULTS, tokenizer_dir, ensure
from ..tok.wrapper import Tok


@torch.no_grad()
def _embed(model, tok, sentences, device, seq_len, batch=32) -> np.ndarray:
    """(n_layers+1, N, dim) L2-normalised mean-pooled embeddings."""
    model.eval()
    out = None
    N = len(sentences)
    for s0 in range(0, N, batch):
        chunk = sentences[s0:s0 + batch]
        seqs = [tok.encode(t, bos=True)[:seq_len] for t in chunk]
        lens = [len(s) for s in seqs]
        maxlen = max(lens)
        arr = np.zeros((len(seqs), maxlen), dtype=np.int64)
        for i, s in enumerate(seqs):
            arr[i, :len(s)] = s
        idx = torch.from_numpy(arr).to(device)
        reps = model.layer_reps(idx).float()             # (Lr, b, T, d)
        mask = torch.zeros(len(seqs), maxlen, device=device)
        for i, ln in enumerate(lens):
            mask[i, :ln] = 1.0
        m = mask[None, :, :, None]
        pooled = (reps * m).sum(2) / m.sum(2).clamp(min=1)   # (Lr, b, d)
        pooled = torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()
        if out is None:
            out = [np.zeros((N, pooled.shape[-1]), dtype=np.float32)
                   for _ in range(pooled.shape[0])]
        for lyr in range(pooled.shape[0]):
            out[lyr][s0:s0 + len(seqs)] = pooled[lyr]
    return np.stack(out, axis=0)


def _retrieval(E: np.ndarray, F: np.ndarray) -> dict:
    sim = E @ F.T
    n = sim.shape[0]
    en_to = sim.argmax(1)
    to_en = sim.argmax(0)
    diag = np.arange(n)
    top1_ef = float((en_to == diag).mean())
    top1_fe = float((to_en == diag).mean())
    mutual = float(((en_to == diag) & (to_en[en_to] == diag)).mean())
    matched = float(np.diag(sim).mean())
    if n > 1:
        nonmatched = float((sim.sum() - np.trace(sim)) / (n * (n - 1)))
    else:
        nonmatched = 0.0
    return {"top1_en2l": top1_ef, "top1_l2en": top1_fe, "mutual_nn": mutual,
            "cosine_matched": matched, "cosine_nonmatched": nonmatched,
            "cosine_margin": matched - nonmatched}


def compute(run_name: str, tok_name: str, split: str = "dev",
            model=None, device=None, seq_len: int = 2048,
            langs: list[str] | None = None) -> dict:
    from .. import flores
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model, ck_langs = _load_model(run_name, dev)
        langs = langs or ck_langs
    elif langs is None:
        raise ValueError("langs is required when passing an in-memory model")
    partners = [lang for lang in langs if lang != ANCHOR] if ANCHOR in langs else []
    if not partners:
        return {"run": run_name, "split": split, "langs": langs, "pairs": {}}
    eval_langs = [ANCHOR] + partners
    par = flores.load_parallel(eval_langs, split)
    tok = Tok(tokenizer_dir(tok_name))
    emb = {l: _embed(model, tok, par[l], dev, seq_len) for l in par}
    n_layers = emb[ANCHOR].shape[0]
    pairs = {}
    for p in partners:
        per_layer = [_retrieval(emb[ANCHOR][ly], emb[p][ly]) for ly in range(n_layers)]
        best = max(range(n_layers), key=lambda ly: per_layer[ly]["mutual_nn"])
        pairs[p] = {"same_script": LANGS[p].same_script_as_en,
                    "best_layer": best, "best": per_layer[best],
                    "per_layer": per_layer}
    return {"run": run_name, "split": split, "langs": langs, "pairs": pairs}


def _load_model(run_name: str, device, tag: str = "final"):
    from ..model import ModelConfig, Transformer
    ck = torch.load(RUNS / run_name / "checkpoints" / f"{tag}.pt",
                    map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**ck["cfg"]["model"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), list(ck["cfg"]["langs"])


def run(run_name: str, tok_name: str, split: str = "dev",
        out_dir: Path | None = None) -> dict:
    out_dir = ensure(Path(out_dir) if out_dir else RESULTS / "alignment")
    res = compute(run_name, tok_name, split)
    (out_dir / f"{run_name}.json").write_text(json.dumps(res, indent=2))
    md = [f"# Alignment: {run_name} (FLORES+ {split})", "",
          "| partner | script | best layer | top1 EN->L | top1 L->EN | mutual-NN | cosine pair | cosine margin |",
          "|---|---|---|---|---|---|---|---|"]
    for p, v in res["pairs"].items():
        b = v["best"]
        md.append(f"| {p} | {'same' if v['same_script'] else 'cross'} | "
                  f"{v['best_layer']} | {b['top1_en2l']:.3f} | "
                  f"{b['top1_l2en']:.3f} | {b['mutual_nn']:.3f} | "
                  f"{b['cosine_matched']:.3f} | {b['cosine_margin']:.3f} |")
    if not res["pairs"]:
        md.extend(["", "No EN-partner bilingual pair exists in this run."])
    (out_dir / f"{run_name}.md").write_text("\n".join(md) + "\n")
    print(f"[align] wrote {out_dir}/{run_name}.md")
    return res
