"""MEXA-style cross-lingual representation alignment on FLORES+ dev.

For every layer, embed each language's parallel sentences by mean-pooling that
layer's hidden states, then measure how well one language's sentences retrieve
their translations in another. High alignment on cross-script pairs would say
the model builds a shared multilingual space despite the script gap -- the
representation-side counterpart to the BPB/BTS story.

Reported per unordered language pair, per layer:
  - top-1 A->B and B->A retrieval accuracy, mutual nearest-neighbour rate
  - mean cosine similarity of translations and its margin over non-pairs
  - linear CKA between the two languages' representation matrices

...each in two variants:
  - ``raw``: embeddings as-is.
  - ``centered``: each language's sentence-mean subtracted before comparing.
    Cross-lingual spaces are dominated by a per-language centroid ("language
    identity") direction; removing it is the representation-space analogue of
    the PMI debiasing that XNLI needed (CLAUDE.md section 6). Raw numbers can
    look flat purely because that centroid swamps the translation signal.

**Every model is embedded on every language by default**, not only the ones in
its training mixture. That is deliberate: the tokenizers (``unigram_starved`` /
``unigram_destarved``) cover all five languages, so an EN-only model *can* be
fed Arabic, and its EN-AR alignment is the control that says how much of a
bilingual model's EN-AR alignment is real rather than surface cues (shared
digits, Latin-script named entities, punctuation, sentence length). Without
that baseline a raw retrieval number is uninterpretable. Pass ``langs`` to
restrict. ``compute(langs=None)`` keeps the old behaviour (training mixture
only) so the ``xscript eval-align`` CLI and the smoke test are unaffected.

That is not a theoretical worry -- it is what the data does. Measured here:
``ar-fair`` is an **Arabic-only** model, yet it retrieves EN<->AR translations
at 0.96 top-1, and ``zh-fair-12b`` (**Chinese-only**) hits 0.97 on EN<->ZH.
Neither ever trained on English. Restricted to the trained pairs you would
read a bilingual model's ~0.96 EN-AR as "cross-script alignment emerges from
bilingual pretraining"; the monolingual controls show that number is
approximately free. Two corollaries the harness now handles explicitly:

  - ``lexical_baseline()`` records a **model-free** TF-IDF token-overlap floor
    per pair. Monolingual models score *below* that floor on pairs they don't
    know, so those cells carry no representational signal at all.
  - Top-1 over one FLORES split (997) **saturates** -- hence ``split="both"``
    (dev+devtest, 2009) as the default pool, and why ``cka`` / ``cosine_margin``
    are reported alongside: they do not ceiling. A near-1.0 control means a
    bilingual-minus-monolingual delta measures headroom, not transfer.

Per-example retrieval hit lists are recorded at two layers -- ``ref`` (a fixed
depth fraction, the same index for every model, so cross-model deltas carry no
layer-selection bias) and ``best`` (that model's own argmax layer, descriptive
only) -- so `scripts/external_bench/analyze_alignment.py` can put paired
bootstrap CIs on bilingual-minus-monolingual deltas exactly the way
`bootstrap_transfer.py` does for the downstream benchmarks.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from ..langs import LANGS
from ..paths import RUNS, RESULTS, tokenizer_dir, ensure
from ..tok.wrapper import PAD_ID, Tok

# Depth fraction used for the bias-free "reference" layer. Alignment in decoder
# LMs typically peaks in the upper-middle of the stack, not at the top (the last
# blocks specialise back toward next-token prediction).
REF_LAYER_FRAC = 0.75

VARIANTS = ("raw", "centered")


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------

def _encode(tok, sentences, seq_len):
    return [tok.encode(t, bos=True)[:seq_len] for t in sentences]


@torch.no_grad()
def _embed(model, seqs, device, batch=16, fixed_width=None):
    """(n_layers+1, N, dim) L2-normalised mean-pooled embeddings.

    With ``fixed_width`` set (the XLA/Neuron path) every forward uses one
    constant ``[batch, fixed_width]`` shape so the whole run compiles a single
    graph -- dynamic per-batch shapes are catastrophic on Neuron (CLAUDE.md
    section 4). Input and mask tensors are built on the HOST and moved once;
    per-row in-place scatter on an XLA tensor trips ``NRT_EXEC_OOB``.
    """
    model.eval()
    N = len(seqs)
    out = None
    for s0 in range(0, N, batch):
        chunk = seqs[s0:s0 + batch]
        rows = len(chunk)
        if fixed_width is None:
            width, nrow = max(len(s) for s in chunk), rows
        else:
            # constant row count too: a short final batch would otherwise be a
            # second graph shape.
            width, nrow = fixed_width, batch
        arr = np.full((nrow, width), PAD_ID, dtype=np.int64)
        msk = np.zeros((nrow, width), dtype=np.float32)
        for i, s in enumerate(chunk):
            arr[i, :len(s)] = s
            msk[i, :len(s)] = 1.0
        idx = torch.from_numpy(arr).to(device)
        mask = torch.from_numpy(msk).to(device)[None, :, :, None]
        reps = model.layer_reps(idx).float()                  # (Lr, nrow, W, d)
        pooled = (reps * mask).sum(2) / mask.sum(2).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        if device.type == "xla":
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        pooled = pooled.cpu().numpy()                          # (Lr, nrow, d)
        if out is None:
            out = np.zeros((pooled.shape[0], N, pooled.shape[-1]), dtype=np.float32)
        out[:, s0:s0 + rows] = pooled[:, :rows]
    return out


def _fixed_width(encoded):
    """One constant sequence width covering every language in this run."""
    w = max(len(s) for seqs in encoded.values() for s in seqs)
    # NCC-5266: neuronx-cc's matmul transpose lowering requires an even step for
    # non-FP32 dst dims, so an odd width reliably fails to compile (CLAUDE.md
    # section 4.3). The extra column is inert padding, masked out of the pooling.
    return max(w + (w % 2), 2)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def lexical_baseline(encoded: dict[str, list[list[int]]]) -> dict[str, dict]:
    """Model-free TF-IDF retrieval over shared token ids -- the floor any
    alignment number has to beat.

    The tokenizers are shared across all five languages, and FLORES sentences
    leak a lot across scripts: digits, dates, and Latin-script named entities
    survive translation into Arabic and Chinese verbatim. Matching sentences on
    IDF-weighted token overlap alone therefore already retrieves translations
    far above chance -- measured here at ~0.46 EN-DE / ~0.60 EN-FR / ~0.22
    EN-ZH on FLORES+ dev. A model scoring *below* this floor on a pair has
    contributed nothing on that pair, which is easy to mistake for a real
    same-script alignment result. Depends only on the tokenizer, so it is
    identical for every model sharing one; recorded per run anyway so results
    are self-contained.
    """
    from scipy import sparse
    langs = list(encoded)
    vocab = sorted({t for seqs in encoded.values() for s in seqs for t in s})
    col = {t: i for i, t in enumerate(vocab)}
    n_docs = sum(len(s) for s in encoded.values())
    df = np.zeros(len(vocab))
    for seqs in encoded.values():
        for s in seqs:
            df[[col[t] for t in set(s)]] += 1
    idf = np.log((n_docs + 1) / (df + 1))
    # Sparse: the dense [n_docs, |vocab|] form is ~10 GB over the dev+devtest
    # pool and minutes of numpy; each row has <=max_tokens nonzeros.
    mats = {}
    for l, seqs in encoded.items():
        rows = [i for i, s in enumerate(seqs) for _ in s]
        cols = [col[t] for s in seqs for t in s]
        M = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                              shape=(len(seqs), len(vocab)))
        M = M.multiply(idf[None, :]).tocsr()
        norm = np.sqrt(M.multiply(M).sum(1)).A.ravel()
        M = sparse.diags(1.0 / np.maximum(norm, 1e-9)) @ M
        mats[l] = M
    out = {}
    for a, b in combinations(langs, 2):
        r = _retrieval_sim(np.asarray((mats[a] @ mats[b].T).todense()))
        out[f"{a}-{b}"] = {k: r[k] for k in
                           ("top1_a2b", "top1_b2a", "mutual_nn", "cosine_margin")}
    return out


def _center(E):
    """Remove the language centroid, then re-normalise."""
    C = E - E.mean(0, keepdims=True)
    n = np.linalg.norm(C, axis=1, keepdims=True)
    return C / np.maximum(n, 1e-12)


def _cka(X, Y):
    """Linear centered kernel alignment -- retrieval-free similarity of two
    representation matrices over the same (parallel) sentences."""
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    xy = float(np.linalg.norm(Xc.T @ Yc) ** 2)
    xx = float(np.linalg.norm(Xc.T @ Xc))
    yy = float(np.linalg.norm(Yc.T @ Yc))
    return xy / (xx * yy) if xx > 0 and yy > 0 else 0.0


def _retrieval(E, F, with_hits=False):
    """Metrics for retrieving F from E (and back). Rows of E and F are parallel."""
    r = _retrieval_sim(E @ F.T, with_hits=with_hits)
    r["cka"] = _cka(E, F)
    return r


def _discriminability(sim):
    """Per-query margin and d' from a [n, n] similarity matrix.

    ``d' = (matched - mean_nonmatched) / std_nonmatched`` per query. Unlike
    top-1 accuracy this is continuous and cannot ceiling -- top-1 saturated
    above 0.95 even for monolingual CONTROLS on this pool, which is why it is
    the preferred statistic (see the module docstring). Dividing by the
    non-match spread also makes it scale-free: models differ in overall cosine
    scale, so a raw margin is not comparable across them.

    Computed from row sums rather than materialising the off-diagonal (which
    would be an n x (n-1) copy).
    """
    n = sim.shape[0]
    matched = np.diag(sim).astype(np.float64)
    s1 = sim.sum(1).astype(np.float64) - matched
    s2 = (sim.astype(np.float64) ** 2).sum(1) - matched ** 2
    m = n - 1
    mean_off = s1 / m
    var_off = np.maximum(s2 / m - mean_off ** 2, 0.0)
    margin = matched - mean_off
    dprime = margin / np.maximum(np.sqrt(var_off), 1e-9)
    return margin, dprime


def _retrieval_sim(sim, with_hits=False):
    """Retrieval metrics from a precomputed [n, n] similarity matrix."""
    n = sim.shape[0]
    diag = np.arange(n)
    a2b = sim.argmax(1)
    b2a = sim.argmax(0)
    hit_ab = (a2b == diag)
    hit_ba = (b2a == diag)
    matched = float(np.diag(sim).mean())
    nonmatched = float((sim.sum() - np.trace(sim)) / (n * (n - 1))) if n > 1 else 0.0
    res = {"top1_a2b": float(hit_ab.mean()),
           "top1_b2a": float(hit_ba.mean()),
           "mutual_nn": float((hit_ab & (b2a[a2b] == diag)).mean()),
           "cosine_matched": matched,
           "cosine_nonmatched": nonmatched,
           "cosine_margin": matched - nonmatched,
           "n": n}
    margin_q, dprime_q = _discriminability(sim)
    res["dprime"] = float(dprime_q.mean())
    if with_hits:
        res["hits"] = {"a2b": hit_ab.astype(np.uint8).tolist(),
                       "b2a": hit_ba.astype(np.uint8).tolist()}
        # per-query d' so analyze_alignment.py can paired-bootstrap it exactly
        # the way it does the 0/1 retrieval hits.
        res["dprime_q"] = [round(float(x), 4) for x in dprime_q]
    return res


def _pair_layers(EA, EB, ref_layer):
    """Per-layer metrics for one language pair, both variants.

    ``EA``/``EB`` are (n_layers+1, N, dim). Hit lists are attached only at the
    fixed ``ref`` layer and at each variant's own argmax layer.
    """
    n_layers = EA.shape[0]
    out = {}
    for variant in VARIANTS:
        A = [EA[ly] if variant == "raw" else _center(EA[ly]) for ly in range(n_layers)]
        B = [EB[ly] if variant == "raw" else _center(EB[ly]) for ly in range(n_layers)]
        # The per-layer scan skips CKA (a d x d gram product per layer, the
        # dominant cost); it is only needed at the two reported layers.
        per_layer = [_retrieval_sim(A[ly] @ B[ly].T) for ly in range(n_layers)]
        best = max(range(n_layers), key=lambda ly: per_layer[ly]["mutual_nn"])
        out[variant] = {
            "per_layer": per_layer,
            "ref_layer": ref_layer,
            "best_layer": best,
            "ref": _retrieval(A[ref_layer], B[ref_layer], with_hits=True),
            "best": _retrieval(A[best], B[best], with_hits=True),
        }
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def compute(run_name: str, tok_name: str, split: str = "dev",
            model=None, device=None, seq_len: int = 2048,
            langs: list[str] | None = None, batch: int = 16,
            max_tokens: int = 256, limit: int | None = None,
            emb_dir=None) -> dict:
    """Alignment for every unordered pair among ``langs``.

    ``langs=None`` falls back to the checkpoint's own training mixture (the
    original behaviour, kept for the ``xscript eval-align`` CLI). Callers that
    want the baseline cells -- a model scored on a language it never saw --
    must pass ``langs`` explicitly; see the module docstring for why that is
    the interpretable version.
    """
    from .. import flores
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_langs = None
    if model is None:
        model, train_langs = _load_model(run_name, dev)
    eval_langs = list(langs) if langs else train_langs
    if eval_langs is None:
        raise ValueError("langs is required when passing an in-memory model")
    eval_langs = [l for l in LANGS if l in set(eval_langs)]      # canonical order
    meta = {"run": run_name, "tok": tok_name, "split": split,
            "train_langs": train_langs, "eval_langs": eval_langs}
    if len(eval_langs) < 2:
        return {**meta, "pairs": {}}

    if split == "both":
        # 997 dev + 1012 devtest. A bigger candidate pool makes retrieval
        # harder, which matters because top-1 over dev alone saturates (>0.95)
        # even for monolingual control models -- see the module docstring.
        halves = [flores.load_parallel(eval_langs, s) for s in ("dev", "devtest")]
        par = {l: halves[0][l] + halves[1][l] for l in eval_langs}
    else:
        par = flores.load_parallel(eval_langs, split)
    if limit:
        par = {l: s[:limit] for l, s in par.items()}
    tok = Tok(tokenizer_dir(tok_name))
    encoded = {l: _encode(tok, par[l], min(seq_len, max_tokens)) for l in eval_langs}
    width = _fixed_width(encoded) if dev.type == "xla" else None
    emb = {l: _embed(model, encoded[l], dev, batch=batch, fixed_width=width)
           for l in eval_langs}

    n_layers = emb[eval_langs[0]].shape[0]
    ref_layer = min(n_layers - 1, max(0, round(REF_LAYER_FRAC * (n_layers - 1))))
    if emb_dir:
        p = save_embeddings(emb_dir, run_name,
                            {**meta, "n_layers": n_layers, "ref_layer": ref_layer,
                             "n_sentences": len(par[eval_langs[0]]),
                             "max_tokens": max_tokens, "dtype": "float32"}, emb)
        print(f"[align] cached embeddings -> {p}")
    pairs = {}
    for a, b in combinations(eval_langs, 2):
        same_script = LANGS[a].script == LANGS[b].script
        pairs[f"{a}-{b}"] = {
            "a": a, "b": b, "same_script": same_script,
            "trained_pair": bool(train_langs and a in train_langs and b in train_langs),
            **_pair_layers(emb[a], emb[b], ref_layer),
        }
    return {**meta, "n_layers": n_layers, "ref_layer": ref_layer,
            "n_sentences": len(par[eval_langs[0]]), "fixed_width": width,
            "lexical_baseline": lexical_baseline(encoded), "pairs": pairs}


def save_embeddings(emb_dir, run_name, meta: dict, emb: dict) -> Path:
    """Persist the pooled per-layer embeddings for later metric work.

    ``(n_layers+1, N, dim)`` fp32 per language -- ~280 MB/language/model, so
    ~1.4 GB per model and ~36 GB for the full 26-model roster. That is cheap
    against the alternative: every new statistic otherwise costs a full sweep
    (~100 GB of checkpoint re-download plus 26 forward passes), because the
    forward pass is 84% of runtime. With these cached, a new metric is a
    pure-CPU pass over local arrays and needs neither Neuron nor the network.

    fp32 (not fp16) so recomputed metrics reproduce the in-run numbers exactly.
    Sentence order is `flores.load_parallel(langs, split)`'s -- the sorted
    intersection of FLORES ids -- and is deterministic, so it can be
    regenerated from the recorded `split`/`eval_langs`.
    """
    emb_dir = ensure(Path(emb_dir))
    out = emb_dir / f"{run_name}.npz"
    tmp = out.with_suffix(".npz.tmp")
    # Write via an open handle: np.savez APPENDS ".npz" to a path-like target
    # that lacks it, which would silently produce "<name>.npz.tmp.npz".
    with open(tmp, "wb") as fh:
        np.savez(fh, __meta__=np.array(json.dumps(meta)),
                 **{l: v.astype(np.float32) for l, v in emb.items()})
    tmp.rename(out)                                   # atomic: no partial file
    return out


def load_embeddings(emb_dir, run_name) -> tuple[dict, dict]:
    """Inverse of `save_embeddings` -> (meta, {lang: (n_layers+1, N, dim)})."""
    z = np.load(Path(emb_dir) / f"{run_name}.npz")
    meta = json.loads(str(z["__meta__"]))
    return meta, {k: z[k] for k in z.files if k != "__meta__"}


def _load_model(run_name: str, device, tag: str = "final"):
    from ..model import ModelConfig, Transformer
    ck = torch.load(RUNS / run_name / "checkpoints" / f"{tag}.pt",
                    map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**ck["cfg"]["model"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), list(ck["cfg"]["langs"])


def _table(res: dict, variant: str, layer_key: str) -> list[str]:
    lex = res.get("lexical_baseline", {})
    md = [f"### {variant}, {layer_key} layer", "",
          "`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 "
          "means the model adds nothing on that pair.", "",
          "| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN "
          "| cos margin | CKA | lex | vs lex |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in res["pairs"].items():
        m, ly = v[variant][layer_key], v[variant][f"{layer_key}_layer"]
        acc = (m["top1_a2b"] + m["top1_b2a"]) / 2
        b = lex.get(name)
        bacc = (b["top1_a2b"] + b["top1_b2a"]) / 2 if b else None
        md.append(f"| {name} | {'same' if v['same_script'] else 'cross'} | "
                  f"{'yes' if v['trained_pair'] else 'no'} | {ly} | "
                  f"{m['top1_a2b']:.3f} | {m['top1_b2a']:.3f} | {m['mutual_nn']:.3f} | "
                  f"{m['cosine_margin']:.3f} | {m['cka']:.3f} | "
                  f"{f'{bacc:.3f}' if bacc is not None else '-'} | "
                  f"{f'{acc - bacc:+.3f}' if bacc is not None else '-'} |")
    return md + [""]


def run(run_name: str, tok_name: str, split: str = "dev",
        out_dir: Path | None = None, **kw) -> dict:
    out_dir = ensure(Path(out_dir) if out_dir else RESULTS / "alignment")
    res = compute(run_name, tok_name, split, **kw)
    (out_dir / f"{run_name}.json").write_text(json.dumps(res))
    md = [f"# Alignment: {run_name} (FLORES+ {split}, "
          f"n={res.get('n_sentences', 0)})", ""]
    if res["pairs"]:
        md += [f"Languages embedded: {', '.join(res['eval_langs'])}; "
               f"trained on: {', '.join(res['train_langs'] or ['?'])}.", ""]
        for variant in VARIANTS:
            md += _table(res, variant, "ref")
            md += _table(res, variant, "best")
    else:
        md += ["Fewer than two languages to pair."]
    (out_dir / f"{run_name}.md").write_text("\n".join(md) + "\n")
    print(f"[align] wrote {out_dir}/{run_name}.md")
    return res
