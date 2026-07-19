"""Tokenizer analysis gate (thesis-plan next-action #2).

For every trained tokenizer x study language, measured on FLORES+ (parallel,
so 'tokens per sentence relative to English' is content-normalized fertility):

  - bytes/token, tokens/char, tokens/word, tokens/sentence
  - parity = tokens-per-sentence relative to English on the same sentences
  - %% of emitted tokens that are raw-byte atoms (the literal byte tax)
  - %% single-character tokens (allocation starvation for ZH shows up here)
  - unique vocab entries used
  - full 64k vocabulary allocation by script

Plus segmentation samples for eyeballing subword meaningfulness (the
user-facing fidelity check that decides which flavor trains models).

Gate (plan): proceed to model training only if the AR/ZH fertility gap
between starved and destarved conditions is large.
"""
import json
import unicodedata
from pathlib import Path

from .. import flores
from ..langs import LANGS, TOK_FLAVORS, tok_name, all_tok_names
from ..paths import RESULTS, tokenizer_dir, ensure
from .wrapper import Tok

# codepoint-range -> script bucket (coarse; enough for allocation accounting)
_RANGES = [
    (0x0041, 0x024F, "Latin"), (0x1E00, 0x1EFF, "Latin"), (0x2C60, 0x2C7F, "Latin"),
    (0x0370, 0x03FF, "Greek"),
    (0x0400, 0x052F, "Cyrillic"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"), (0x0750, 0x077F, "Arabic"), (0x08A0, 0x08FF, "Arabic"),
    (0xFB50, 0xFDFF, "Arabic"), (0xFE70, 0xFEFF, "Arabic"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x0DFF, "OtherIndic"), (0x0E00, 0x0E7F, "Thai"),
    (0x1100, 0x11FF, "Hangul"), (0xAC00, 0xD7AF, "Hangul"),
    (0x3040, 0x30FF, "Kana"),
    (0x3400, 0x4DBF, "Han"), (0x4E00, 0x9FFF, "Han"), (0xF900, 0xFAFF, "Han"),
    (0x0E80, 0x0FFF, "OtherSEA"), (0x1000, 0x109F, "OtherSEA"),
    (0x10A0, 0x10FF, "Georgian"), (0x0530, 0x058F, "Armenian"),
    (0x1200, 0x139F, "Ethiopic"),
]


def _char_bucket(ch: str) -> str:
    cp = ord(ch)
    if cp < 0x41:
        return "ascii_sym" if not ch.isspace() else "space"
    for lo, hi, name in _RANGES:
        if lo <= cp <= hi:
            return name
    cat = unicodedata.category(ch)
    if cat.startswith("L"):
        return "OtherScript"
    return "sym"


def classify_piece(raw: bytes) -> str:
    if not raw:
        return "special"
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "byte_atom" if len(raw) == 1 else "partial_utf8"
    letters = [c for c in s if unicodedata.category(c).startswith("L")]
    if not letters:
        return "sym_num_space"
    counts = {}
    for c in letters:
        b = _char_bucket(c)
        counts[b] = counts.get(b, 0) + 1
    top, n = max(counts.items(), key=lambda kv: kv[1])
    return top if n == len(letters) else "mixed"


def vocab_allocation(tok: Tok) -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in range(tok.vocab_size):
        b = "byte_atom" if tok.is_byte_piece(i) else classify_piece(tok.piece_bytes(i))
        counts[b] = counts.get(b, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _lang_metrics(tok: Tok, texts: list[str]) -> dict:
    n_tok = n_byte = n_char = n_word = n_bytepieces = n_singlechar = 0
    used = set()
    for ids, text in zip(tok.encode_batch(texts), texts):
        n_tok += len(ids)
        n_byte += len(text.encode("utf-8"))
        n_char += len(text)
        n_word += len(text.split())
        used.update(ids)
        for i in ids:
            if tok.is_byte_piece(i):
                n_bytepieces += 1
            else:
                try:
                    if len(tok.piece_bytes(i).decode("utf-8").strip()) == 1:
                        n_singlechar += 1
                except UnicodeDecodeError:
                    pass
    n_sent = len(texts)
    return {
        "n_sentences": n_sent,
        "tokens": n_tok,
        "bytes_per_token": n_byte / n_tok,
        "tokens_per_char": n_tok / n_char,
        "tokens_per_word": n_tok / n_word,
        "tokens_per_sentence": n_tok / n_sent,
        "pct_byte_tokens": 100.0 * n_bytepieces / n_tok,
        "pct_single_char_tokens": 100.0 * n_singlechar / n_tok,
        "unique_tokens_used": len(used),
    }


def _segment(tok: Tok, text: str) -> str:
    ids = tok.encode(text)
    parts = []
    for i in ids:
        try:
            parts.append(tok.piece_bytes(i).decode("utf-8"))
        except UnicodeDecodeError:
            parts.append(f"<{tok.piece_bytes(i).hex()}>")
    return "|".join(parts)


def run(tok_names=None, out_dir: Path | None = None, n_samples: int = 3) -> dict:
    out_dir = ensure(Path(out_dir) if out_dir else RESULTS / "tok_analysis")
    tok_names = tok_names or all_tok_names()
    toks = [Tok(tokenizer_dir(n)) for n in tok_names]

    par = flores.load_parallel(list(LANGS), "dev")
    par_test = flores.load_parallel(list(LANGS), "devtest")
    texts = {l: par[l] + par_test[l] for l in LANGS}

    metrics, alloc = {}, {}
    for tok in toks:
        m = {l: _lang_metrics(tok, texts[l]) for l in LANGS}
        en_tps = m["en"]["tokens_per_sentence"]
        for l in LANGS:
            m[l]["parity_vs_en"] = m[l]["tokens_per_sentence"] / en_tps
        metrics[tok.name] = m
        alloc[tok.name] = vocab_allocation(tok)
        print(f"[analyze] {tok.name} done")

    # ---- gate summary: starved-vs-destarved fertility ratio per flavor ----
    gate = {}
    for f in TOK_FLAVORS:
        s, d = f"{f}_starved", f"{f}_destarved"
        if s in metrics and d in metrics:
            gate[f] = {l: metrics[s][l]["tokens_per_sentence"] /
                          metrics[d][l]["tokens_per_sentence"] for l in LANGS}

    result = {"metrics": metrics, "vocab_allocation": alloc,
              "starved_over_destarved_tokens": gate}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    # ---- markdown tables ----
    cols = ["bytes_per_token", "tokens_per_char", "tokens_per_word",
            "tokens_per_sentence", "parity_vs_en", "pct_byte_tokens",
            "pct_single_char_tokens", "unique_tokens_used"]
    md = ["# Tokenizer fertility on FLORES+ (dev+devtest)", ""]
    for name, m in metrics.items():
        md += [f"## {name}", "", "| lang | " + " | ".join(cols) + " |",
               "|" + "---|" * (len(cols) + 1)]
        for l in LANGS:
            md.append("| " + l + " | " +
                      " | ".join(f"{m[l][c]:.3f}" if isinstance(m[l][c], float)
                                 else str(m[l][c]) for c in cols) + " |")
        md.append("")
    md += ["# Gate: starved/destarved token-count ratio (per flavor)", ""]
    for f, g in gate.items():
        md.append(f"- **{f}**: " + ", ".join(f"{l}={v:.3f}" for l, v in g.items()))
    md += ["", "# Vocab allocation (64k pieces by script)", ""]
    buckets = sorted({b for a in alloc.values() for b in a})
    md += ["| tokenizer | " + " | ".join(buckets) + " |",
           "|" + "---|" * (len(buckets) + 1)]
    for name, a in alloc.items():
        md.append("| " + name + " | " + " | ".join(str(a.get(b, 0)) for b in buckets) + " |")
    (out_dir / "report.md").write_text("\n".join(md) + "\n")

    # ---- segmentation samples for the fidelity eyeball check ----
    smp = ["# Segmentation samples (FLORES+ dev)", ""]
    for l in LANGS:
        smp.append(f"## {l}")
        for k in range(n_samples):
            smp += ["", f"> {par[l][k]}", ""]
            for tok in toks:
                smp.append(f"- **{tok.name}**: `{_segment(tok, par[l][k])}`")
        smp.append("")
    (out_dir / "samples.md").write_text("\n".join(smp) + "\n")

    print(f"[analyze] wrote {out_dir}/report.md, samples.md, metrics.json")
    return result
