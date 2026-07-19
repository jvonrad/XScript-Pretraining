"""Bits-per-byte evaluation.

BPB = (sum of next-token NLL in nats) / (ln2 * total UTF-8 bytes of the text).

The byte denominator makes BPB comparable across tokenizers with different
fertilities -- the whole reason the plan reports BPB rather than per-token loss.
Each document is scored with a leading <bos>; the <bos> itself is never a target
and <eos> is included as a target (it is real signal the model must predict).
Long documents are scored in sliding windows of the model's context length.
"""
import json
from pathlib import Path

import torch

from ..tok.wrapper import BOS_ID, EOS_ID


@torch.no_grad()
def score_texts(model, tok, texts, device, seq_len: int, batch_tokens: int = 8192):
    """Return (nll_nats, n_bytes, n_target_tokens) summed over `texts`."""
    model.eval()
    total_nll = 0.0
    total_bytes = 0
    total_tokens = 0
    for text in texts:
        b = len(text.encode("utf-8"))
        if b == 0:
            continue
        ids = tok.encode(text, bos=True, eos=True)
        total_bytes += b
        # sliding, non-overlapping windows of seq_len+1 (predict positions 1..)
        for st in range(0, len(ids) - 1, seq_len):
            chunk = ids[st:st + seq_len + 1]
            if len(chunk) < 2:
                continue
            x = torch.tensor(chunk[:-1], device=device).unsqueeze(0)
            y = torch.tensor(chunk[1:], device=device).unsqueeze(0)
            logits, _ = model(x, y)
            nll = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
            total_nll += float(nll)
            total_tokens += y.numel()
    return total_nll, total_bytes, total_tokens


def bpb(nll_nats: float, n_bytes: int) -> float:
    import math
    return nll_nats / (math.log(2) * max(n_bytes, 1))


def eval_sources(model, tok, sources: dict[str, list[str]], device, seq_len: int) -> dict:
    """sources: name -> list[str]. Returns {name: {bpb, ppl_tok, bytes, tokens}}."""
    import math
    out = {}
    for name, texts in sources.items():
        nll, nbytes, ntok = score_texts(model, tok, texts, device, seq_len)
        out[name] = {
            "bpb": bpb(nll, nbytes),
            "ppl_token": math.exp(nll / max(ntok, 1)),
            "bytes": nbytes,
            "tokens": ntok,
        }
    return out


def run(run_name: str, tok_name: str, tag: str = "final",
        langs=None, out_dir=None) -> dict:
    """Re-evaluate a saved checkpoint's BPB on FLORES+ dev and holdout."""
    from pathlib import Path
    from ..model import ModelConfig, Transformer
    from ..tok.wrapper import Tok
    from ..paths import RUNS, RESULTS, tokenizer_dir, ensure
    from .. import flores
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(RUNS / run_name / "checkpoints" / f"{tag}.pt",
                    map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**ck["cfg"]["model"]))
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    tok = Tok(tokenizer_dir(tok_name))
    langs = langs or list(ck["cfg"]["langs"])
    srcs = {}
    for l in langs:
        h = load_holdout(l)
        if h:
            srcs[f"holdout_{l}"] = h
    for l, sents in flores.load_parallel(langs, "dev").items():
        srcs[f"flores_{l}"] = sents
    res = eval_sources(model, tok, srcs, device, model.cfg.max_seq_len)
    out_dir = ensure(Path(out_dir) if out_dir else RESULTS / "bpb")
    (out_dir / f"{run_name}_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"[bpb] {run_name} ({tag}): " +
          ", ".join(f"{k}={v['bpb']:.4f}" for k, v in res.items()))
    return res


def load_holdout(lang: str, max_docs: int = 2000) -> list[str]:
    """In-domain eval text from the reserved FineWeb holdout shard."""
    from ..paths import HOLDOUT
    import zstandard, io
    texts = []
    for p in sorted(HOLDOUT.glob(f"{lang}_*.jsonl.zst")):
        with open(p, "rb") as raw:
            r = zstandard.ZstdDecompressor().stream_reader(raw)
            for line in io.TextIOWrapper(r, encoding="utf-8"):
                if line.strip():
                    texts.append(json.loads(line)["text"])
                    if len(texts) >= max_docs:
                        return texts
    return texts
