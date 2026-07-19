#!/usr/bin/env python
"""Compare how two tokenizers segment a string.

Usage (activate the venv first: `source $XSCRIPT_SCRATCH/venv/bin/activate`):

  # one-shot:
  python scripts/tok_compare.py "عندما كنت في معهد برقيبة انغست في الثقاقة العربية واصبحت مغروما بها"

  # interactive REPL (reads lines until Ctrl-D):
  python scripts/tok_compare.py

  # pick which tokenizers (default: unigram_starved vs unigram_destarved):
  python scripts/tok_compare.py --toks unigram_starved,bpe_starved "some text"

Resolves tokenizers from $XSCRIPT_SCRATCH/tokenizers/<name>/ -- sp.model for the
SentencePiece (unigram) flavors, tokenizer.json for the bpe/pa flavors. Reports
piece list, token count, tokens/char, and how many pieces fell back to raw
UTF-8 bytes (<0xXX>, SentencePiece byte fallback).
"""
import argparse
import os
import re
import sys
from pathlib import Path

BYTE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def _tok_root() -> Path:
    """$XSCRIPT_SCRATCH/tokenizers, falling back to the package's paths module."""
    scratch = os.environ.get("XSCRIPT_SCRATCH")
    if scratch:
        return Path(scratch) / "tokenizers"
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from xscript.paths import TOKENIZERS
        return TOKENIZERS
    except Exception:
        return Path(f"/scratch/{os.environ.get('USER', '')}/xscript") / "tokenizers"


class Tok:
    """Wrap either a SentencePiece sp.model or a HuggingFace tokenizer.json."""

    def __init__(self, name: str, root: Path):
        self.name = name
        d = root / name
        if (d / "sp.model").exists():
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor(model_file=str(d / "sp.model"))
            self.kind = "spm"
        elif (d / "tokenizer.json").exists():
            from tokenizers import Tokenizer
            self.hf = Tokenizer.from_file(str(d / "tokenizer.json"))
            self.kind = "hf"
        else:
            raise FileNotFoundError(f"no sp.model or tokenizer.json under {d}")

    def pieces(self, text: str) -> list[str]:
        if self.kind == "spm":
            return self.sp.encode(text, out_type=str)
        return self.hf.encode(text).tokens


def show(text: str, toks: list[Tok]) -> None:
    chars = len(text)
    print(f"\ninput ({chars} chars): {text!r}")
    counts = []
    for t in toks:
        ps = t.pieces(text)
        counts.append(len(ps))
        fb = sum(1 for p in ps if BYTE.match(p))
        f = len(ps) / chars if chars else 0.0
        extra = f", {fb} byte-fallback" if fb else ""
        print(f"  [{t.name}]  {len(ps)} tokens  ({f:.3f} tok/char{extra})")
        print(f"     {ps}")
    if len(toks) == 2 and counts[0] != counts[1]:
        lo, hi = min(counts), max(counts)
        fewer = toks[counts.index(lo)].name
        print(f"  -> {fewer} uses {hi - lo} fewer tokens ({lo} vs {hi})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Compare tokenizer segmentations.")
    ap.add_argument("text", nargs="*", help="text to tokenize (omit for interactive)")
    ap.add_argument("--toks", default="unigram_starved,unigram_destarved",
                    help="comma-separated tokenizer names under "
                         "$XSCRIPT_SCRATCH/tokenizers (default: %(default)s)")
    a = ap.parse_args(argv)

    root = _tok_root()
    names = [n.strip() for n in a.toks.split(",") if n.strip()]
    toks = [Tok(n, root) for n in names]
    print(f"tokenizers: {', '.join(names)}   (from {root})")

    if a.text:
        show(" ".join(a.text), toks)
        return
    print("enter text, one line at a time (Ctrl-D to quit):")
    try:
        for line in sys.stdin:
            line = line.rstrip("\n")
            if line:
                show(line, toks)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
