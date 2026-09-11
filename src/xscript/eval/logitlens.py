"""Logit lens over the bilingual checkpoints: where in the network does a
bilingual switch to its output language, is there a latent English pivot, and
does the fair tokenizer move either? (CLAUDE.md 6l.)

Method lineage, and where this deviates:

* Wendler et al. 2024 (arXiv:2402.10588, "Do Llamas work in English?"):
  translation / repetition / cloze prompts over ~140 single-token nouns, the
  logit lens read at the answer position, P(output-language word) vs
  P(English word) per layer.
* Wang et al. 2025 (arXiv:2504.04264, "Lost in Multilinguality"): factual
  recall prompts, rank of the correct object in the query language and in
  English per layer, language of the top-10 decoded tokens per layer, and the
  finding that errors concentrate in the final language-transition layers.
* Schut et al. 2025 (arXiv:2502.15603, "Do Multilingual LLMs Think In
  English?"): open-ended generation, per-token lens, whether the intermediate
  decoded token is the ENGLISH TRANSLATION of the token eventually emitted,
  split by lexical vs function words.

Three deviations, all forced by the fact that the whole contrast in this
project is a TOKENIZER (CLAUDE.md 6g's acc_tokennorm theorem, 6i's acc_norm
script-dependence):

1. **Full-string log-probabilities, not first-token probabilities.** All three
   papers read P(token) for a word that is a single token in Llama's
   vocabulary. A word that is one token under `unigram_destarved` is often
   three under `unigram_starved` (Arabic: 1.48x fertility), and the first
   fragment of a fragmented word is a high-prior prefix shared by many words,
   so first-token probability is NOT comparable across the two tokenizer
   conditions. The tokenizer-invariant quantity is the teacher-forced
   log-probability of the WHOLE string under the layer-l lens,
   sum_i log p_l(t_i | prefix, t_<i), the same reason 6 measures BPB rather
   than per-token loss. First-token numbers are stored alongside for
   replication of the papers, restricted to words that are single tokens
   under BOTH tokenizers.
2. **A per-token language map built from each tokenizer's own holdout text**
   replaces fastText / GPT-4o judgements of "what language is this token".
   P(lang | token) is proportional to the token's per-language rate on the
   FineWeb2-HQ holdout shards, equal prior over languages; tokens with no
   letter (punctuation, digits, bare space) and tokens too rare to estimate
   are routed to an explicit `other` class instead of being guessed. Every
   layer's full next-token distribution is then summarised as a probability
   mass per language -- a soft, script-agnostic, tokenizer-agnostic version
   of "the language of the top-10 tokens".
3. **Every latent-language quantity has a matched control.** P_l(English
   translation of the answer) is only evidence of a latent English pivot if
   it exceeds P_l(English translation of a DIFFERENT word) by more than
   chance; the same for the parallel-sentence token sets in the generation
   task (a random other English sentence as control). Raw levels are not
   comparable across tokenizers (vocabulary-dependent priors); the
   same-minus-mismatched contrast is.

Nothing here touches the accelerator: the forward pass is bf16 on the host
CPU (AMX), the lens matmul fp32.
"""
from __future__ import annotations

import io
import json
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..tok.wrapper import BOS_ID, PAD_ID, UNK_ID, EOS_ID

LANGS = ("en", "de", "fr", "ar", "zh")
LANG_NAMES = {"en": "English", "de": "Deutsch", "fr": "Français",
              "ar": "العربية", "zh": "中文"}
OTHER = len(LANGS)          # index of the `other` class in mass vectors

# Localized answer cue, identical to c5_tasks/polyfact/utils.py so a lens number
# and a PolyFact number differ by the analysis and not by the scaffolding.
CUE = {"en": "Answer:", "de": "Antwort:", "fr": "Réponse :",
       "ar": "الإجابة:", "zh": "答案："}


# ---------------------------------------------------------------------------
# token -> language map
# ---------------------------------------------------------------------------

def read_holdout(path: Path, max_chars: int) -> list[str]:
    """Documents from one FineWeb2-HQ holdout shard (jsonl.zst), up to max_chars."""
    import zstandard
    out, n = [], 0
    with open(path, "rb") as f:
        r = zstandard.ZstdDecompressor().stream_reader(f)
        for line in io.TextIOWrapper(r, encoding="utf-8"):
            t = json.loads(line)["text"]
            out.append(t)
            n += len(t)
            if n >= max_chars:
                break
    return out


def _piece_is_lexical(tok, i: int) -> bool:
    """True if a vocab entry can carry language identity: has a letter, or is
    a non-ASCII byte-fallback piece (part of a multi-byte character)."""
    if i <= PAD_ID:
        return False
    if tok.is_byte_piece(i):
        return tok.piece_bytes(i)[0] >= 0x80
    s = tok.piece(i).replace("▁", "")
    return any(unicodedata.category(ch)[0] == "L" for ch in s)


def build_token_lang_map(tok, holdout_dir: Path, langs=LANGS,
                         max_chars: int = 6_000_000, min_count: int = 3,
                         chunk_chars: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """P(lang | token) as an [V, len(langs)] float32 array plus the eligibility
    mask. Rows of ineligible tokens are all-zero: their mass goes to `other`.

    Rate-based, equal prior over languages: P(l|t) = r_l(t) / sum_l' r_l'(t)
    with r_l(t) = count_l(t) / N_l. Documents are cut into ~chunk_chars pieces
    before encoding so SentencePiece never sees a 100 kB string.
    """
    V = tok.vocab_size
    counts = np.zeros((len(langs), V), np.int64)
    for li, lang in enumerate(langs):
        texts = read_holdout(holdout_dir / f"{lang}_00000.jsonl.zst", max_chars)
        pieces = []
        for t in texts:
            for j in range(0, len(t), chunk_chars):
                pieces.append(t[j:j + chunk_chars])
        for b in range(0, len(pieces), 512):
            ids = np.concatenate([np.asarray(x, np.int64)
                                  for x in tok.encode_batch(pieces[b:b + 512])])
            counts[li] += np.bincount(ids, minlength=V)
    rates = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    tot = rates.sum(0)
    P = np.where(tot > 0, rates / np.maximum(tot, 1e-30), 0.0).T.astype(np.float32)  # [V, L]
    elig = counts.sum(0) >= min_count
    lex = np.fromiter((_piece_is_lexical(tok, i) for i in range(V)), bool, V)
    elig &= lex
    P[~elig] = 0.0
    return P, elig


def load_or_build_tokmap(tok, cache: Path, holdout_dir: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)["P"]
    P, elig = build_token_lang_map(tok, holdout_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, P=P, elig=elig, langs=np.array(LANGS))
    return P


# ---------------------------------------------------------------------------
# tokenisation helpers
# ---------------------------------------------------------------------------

def encode_pair(tok, ctx: str, cont: str) -> tuple[list[int], int]:
    """Whole-string tokenisation of ctx+cont with BOS; returns (ids, k) where
    ids[k:] is the continuation. k is the common-prefix length with the
    ctx-only tokenisation, so a boundary re-tokenisation (rare with Unigram
    at a space / quote boundary) is absorbed into the continuation rather
    than silently mis-scored. The probability read is then exactly
    P(ctx+cont | ctx) at the string level."""
    a = tok.encode(ctx, bos=True)
    b = tok.encode(ctx + cont, bos=True)
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    k = max(k, 1)
    return b, k


def single_token(tok, s: str) -> bool:
    """Is the word exactly one vocabulary piece (in its natural, space-prefixed
    `▁word` form)? SentencePiece prepends its own dummy `▁`, so the string is
    encoded WITHOUT a leading space and a bare leading `▁` piece (what CJK
    words get) is ignored."""
    ids = tok.encode(s.strip())
    if ids and tok.piece(ids[0]) == "▁":
        ids = ids[1:]
    return len(ids) == 1


def token_byte_spans(tok, ids: list[int]) -> list[tuple[int, int]]:
    """Byte offsets of each piece into (b' ' + text) -- SentencePiece's dummy
    prefix is a leading space. Specials get an empty span."""
    spans, pos = [], 0
    for i in ids:
        n = len(tok.piece_bytes(i))
        spans.append((pos, pos + n))
        pos += n
    return spans


# ---------------------------------------------------------------------------
# the lens
# ---------------------------------------------------------------------------

@dataclass
class Request:
    ids: list[int]                       # BOS-prefixed token ids
    k: int                               # first position whose PREDICTION we read is k-1
    sets: list[np.ndarray] = field(default_factory=list)   # optional token-id sets for set-mass
    meta: dict = field(default_factory=dict)


@dataclass
class Response:
    lp: np.ndarray        # [L+1, n] log p_l(ids[k+j] | ids[:k+j])   (n = len(ids)-k)
    top1: np.ndarray      # [L+1, n] argmax token per layer at each read position
    rank: np.ndarray      # [L+1, n] rank (0 = best) of the true next token
    mass: np.ndarray      # [L+1, n, len(LANGS)+1] language mass, last col = other
    setmass: np.ndarray   # [S, L+1, n] mass on each request set (S may be 0)

    def full_lp(self) -> np.ndarray:      # [L+1]
        return self.lp.sum(1)


class Lens:
    """Logit lens for xscript.model.Transformer. bf16 forward, fp32 unembedding."""

    def __init__(self, model, tokmap: np.ndarray, dtype=torch.bfloat16,
                 row_chunk: int = 96):
        self.dtype = dtype
        self.model = model.to(dtype).eval()
        self.eps = model.norm.eps
        self.norm_w = model.norm.weight.detach().float().clone()
        self.W = model.lm_head.weight.detach().float().clone()           # [V, D]
        self.P = torch.from_numpy(tokmap)                                 # [V, L]
        self.n_layers = model.cfg.n_layers
        self.row_chunk = row_chunk

    @torch.no_grad()
    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model.layer_reps(idx)                                 # [L+1, B, T, D]

    @torch.no_grad()
    def unembed(self, h: torch.Tensor) -> torch.Tensor:
        """h [L+1, R, D] (any dtype) -> fp32 logits [L+1, R, V] via final norm."""
        h = h.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm_w
        return h @ self.W.T

    @torch.no_grad()
    def run(self, reqs: list[Request], batch_size: int = 32) -> list[Response]:
        out: list[Response | None] = [None] * len(reqs)
        order = sorted(range(len(reqs)), key=lambda i: len(reqs[i].ids))
        for b0 in range(0, len(order), batch_size):
            bi = order[b0:b0 + batch_size]
            T = max(len(reqs[i].ids) for i in bi)
            idx = torch.full((len(bi), T), PAD_ID, dtype=torch.long)
            for r, i in enumerate(bi):
                idx[r, :len(reqs[i].ids)] = torch.tensor(reqs[i].ids)
            reps = self.hidden(idx)                                       # [L+1, B, T, D]
            rows_b, rows_t, tgt, owner = [], [], [], []
            for r, i in enumerate(bi):
                rq = reqs[i]
                for p in range(rq.k - 1, len(rq.ids) - 1):
                    rows_b.append(r); rows_t.append(p); tgt.append(rq.ids[p + 1]); owner.append(r)
            L1 = self.n_layers + 1
            R = len(rows_b)
            lp = np.zeros((L1, R), np.float32)
            top1 = np.zeros((L1, R), np.int32)
            rank = np.zeros((L1, R), np.int32)
            mass = np.zeros((L1, R, len(LANGS) + 1), np.float32)
            n_sets = max((len(reqs[i].sets) for i in bi), default=0)
            setmass = np.zeros((n_sets, L1, R), np.float32)
            rb, rt, tg = (torch.tensor(x) for x in (rows_b, rows_t, tgt))
            for c0 in range(0, R, self.row_chunk):
                sl = slice(c0, c0 + self.row_chunk)
                h = reps[:, rb[sl], rt[sl], :]                             # [L+1, r, D]
                logits = self.unembed(h)                                   # [L+1, r, V]
                logp = torch.log_softmax(logits, -1)
                t = tg[sl]
                lp[:, sl] = logp.gather(-1, t.view(1, -1, 1).expand(L1, -1, 1)).squeeze(-1).numpy()
                top1[:, sl] = logits.argmax(-1).numpy()
                tl = logits.gather(-1, t.view(1, -1, 1).expand(L1, -1, 1))
                rank[:, sl] = (logits > tl).sum(-1).numpy()
                probs = logp.exp()
                m = probs @ self.P                                         # [L+1, r, L]
                mass[:, sl, :len(LANGS)] = m.numpy()
                mass[:, sl, OTHER] = (1.0 - m.sum(-1)).clamp(min=0).numpy()
                if n_sets:
                    for j in range(c0, min(c0 + self.row_chunk, R)):
                        rq = reqs[bi[owner[j]]]
                        for s, ids in enumerate(rq.sets):
                            if len(ids):
                                setmass[s, :, j] = probs[:, j - c0, torch.from_numpy(ids)].sum(-1).numpy()
            # scatter back
            pos = 0
            for r, i in enumerate(bi):
                n = len(reqs[i].ids) - reqs[i].k
                sl = slice(pos, pos + n)
                out[i] = Response(lp[:, sl].copy(), top1[:, sl].copy(), rank[:, sl].copy(),
                                  mass[:, sl].copy(), setmass[:len(reqs[i].sets), :, sl].copy())
                pos += n
        return out  # type: ignore[return-value]

    @torch.no_grad()
    def greedy(self, prompts: list[list[int]], n_new: int, batch_size: int = 64) -> list[list[int]]:
        """Greedy continuation of each prompt by n_new tokens (no KV cache --
        the model has none; n_new full forwards over a right-padded batch).
        Returns only the generated ids."""
        gens = [[] for _ in prompts]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for b0 in range(0, len(order), batch_size):
            bi = order[b0:b0 + batch_size]
            cur = [list(prompts[i]) for i in bi]
            for _ in range(n_new):
                T = max(len(c) for c in cur)
                idx = torch.full((len(bi), T), PAD_ID, dtype=torch.long)
                for r, c in enumerate(cur):
                    idx[r, :len(c)] = torch.tensor(c)
                x = self.model.tok_emb(idx)
                cos, sin = self.model._rope_for(T, idx.device, x.dtype)
                for layer in self.model.layers:
                    x = layer(x, cos, sin)
                x = self.model.norm(x)
                last = torch.tensor([len(c) - 1 for c in cur])
                logits = self.model.lm_head(x[torch.arange(len(bi)), last]).float()
                logits[:, :EOS_ID + 1] = -1e9        # never emit <unk>/<bos>/<eos>; PAD is 3 -> also blocked below
                logits[:, PAD_ID] = -1e9
                nxt = logits.argmax(-1).tolist()
                for r, t in enumerate(nxt):
                    cur[r].append(int(t))
            for r, i in enumerate(bi):
                gens[i] = cur[r][len(prompts[i]):]
        return gens


def load_model(ckpt_path: Path):
    from ..model import ModelConfig, Transformer
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**ck["cfg"]["model"]))
    model.load_state_dict(ck["model"])
    return model.eval()


def lens_self_check(lens: Lens, tok) -> float:
    """Layer-16 lens must equal the model's own output distribution."""
    ids = tok.encode("The capital of France is", bos=True)
    idx = torch.tensor([ids])
    ref = lens.model(idx).float()[0, -1]                       # last-position logits
    reps = lens.hidden(idx)
    lens_logits = lens.unembed(reps[:, 0:1, -1, :])[-1, 0]
    return float((F.log_softmax(ref, -1) - F.log_softmax(lens_logits, -1)).abs().max())
