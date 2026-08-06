#!/usr/bin/env python
"""Compare how two tokenizers segment a string.

Usage:

  # interactive REPL -- type a word/sentence, press enter, get both segmentations:
  python scripts/tok_compare.py

  # one-shot:
  ~/.tokvenv/bin/python scripts/tok_compare.py "عندما كنت في معهد برقيبة انغست في الثقاقة العربية"

  # pipe a file through it:
  cat sentences.txt | python scripts/tok_compare.py

  # pick which tokenizers (default: unigram_starved vs unigram_destarved):
  python scripts/tok_compare.py --toks unigram_starved,bpe_starved "some text

Tokenizers resolve in this order, first hit wins:

  1. $XSCRIPT_SCRATCH/tokenizers/<name>/   (the training cluster's layout)
  2. ~/.cache/xscript-tokenizers/<name>/   (this script's download cache)
  3. hf_hub_download from --repo           (jvonrad/xscript-eval), cached into 2.

sp.model for the SentencePiece (unigram) flavors, tokenizer.json for the
bpe/pa flavors. Reports the piece list, token count, tokens/char, bytes/token
(the fertility figure CLAUDE.md quotes), and how many pieces fell back to raw
UTF-8 bytes (<0xXX>, SentencePiece byte fallback) -- see CLAUDE.md 6g for why
byte- and char-normalisation are the tokenizer-invariant units and token
normalisation is not.

REPL commands: `:ids` toggles token ids, `:q` quits.
"""
import argparse
import os
import re
import sys
from pathlib import Path

BYTE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
DEFAULT_REPO = "jvonrad/xscript-eval"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "xscript-tokenizers"
# Files a tokenizer dir may hold; meta.json is optional (only the repo's own
# dirs carry it) but cheap to mirror, and Tok in src/xscript/tok/wrapper.py
# needs it if you later point that at the same cache.
TOK_FILES = ("sp.model", "tokenizer.json", "meta.json")

# Alternating colors so piece boundaries are visible without separators
# eating into the text -- the point of the tool is seeing *where* it splits.
PALETTE = ("\033[48;5;24;97m", "\033[48;5;53;97m")
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"


def _scratch_root() -> Path | None:
    """$XSCRIPT_SCRATCH/tokenizers, or the package's paths module if importable."""
    scratch = os.environ.get("XSCRIPT_SCRATCH")
    if scratch:
        return Path(scratch) / "tokenizers"
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from xscript.paths import TOKENIZERS
        return TOKENIZERS
    except Exception:
        return None


def _has_model(d: Path) -> bool:
    return (d / "sp.model").exists() or (d / "tokenizer.json").exists()


def resolve(name: str, repo: str) -> Path:
    """Directory holding `name`'s model file, downloading from HF if needed."""
    root = _scratch_root()
    if root is not None and _has_model(root / name):
        return root / name
    if _has_model(CACHE / name):
        return CACHE / name

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    dest = CACHE / name
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[tok_compare] fetching {name} from {repo} -> {dest}", file=sys.stderr)
    for fname in TOK_FILES:
        try:
            local = hf_hub_download(repo_id=repo, filename=f"tokenizers/{name}/{fname}")
        except EntryNotFoundError:
            continue        # flavor doesn't ship this file (spm vs bpe)
        (dest / fname).write_bytes(Path(local).read_bytes())
    if not _has_model(dest):
        raise FileNotFoundError(
            f"no sp.model or tokenizer.json for {name!r} in {repo} "
            f"(checked {root}, {CACHE})")
    return dest


class Tok:
    """Wrap either a SentencePiece sp.model or a HuggingFace tokenizer.json."""

    def __init__(self, name: str, repo: str = DEFAULT_REPO):
        self.name = name
        self.dir = resolve(name, repo)
        if (self.dir / "sp.model").exists():
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor(model_file=str(self.dir / "sp.model"))
            self.kind = "spm"
            self.vocab_size = self.sp.get_piece_size()
        else:
            from tokenizers import Tokenizer
            self.hf = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
            self.kind = "hf"
            self.vocab_size = self.hf.get_vocab_size()

    def encode(self, text: str) -> tuple[list[str], list[int]]:
        if self.kind == "spm":
            return self.sp.encode(text, out_type=str), self.sp.encode(text)
        enc = self.hf.encode(text)
        return enc.tokens, enc.ids


def render(pieces: list[str], color: bool) -> str:
    """One line of pieces, '▁' shown as a visible space marker."""
    if not color:
        return " | ".join(pieces)
    out = []
    for i, p in enumerate(pieces):
        out.append(f"{PALETTE[i % 2]}{p}{RESET}")
    return "".join(out)


def show(text: str, toks: list["Tok"], color: bool, ids: bool) -> None:
    chars = len(text)
    nbytes = len(text.encode("utf-8"))
    head = f"{BOLD}{text}{RESET}" if color else repr(text)
    print(f"\n{head}")
    print(f"{DIM if color else ''}  {chars} chars / {nbytes} bytes{RESET if color else ''}")

    width = max(len(t.name) for t in toks)
    counts = []
    for t in toks:
        pieces, tok_ids = t.encode(text)
        n = len(pieces)
        counts.append(n)
        fb = sum(1 for p in pieces if BYTE.match(p))
        unk = sum(1 for i in tok_ids if i == 0)
        stats = f"{n:>4} tok  {n / chars if chars else 0:.3f} tok/char  " \
                f"{nbytes / n if n else 0:.2f} B/tok"
        if fb:
            stats += f"  {fb} byte-fallback"
        if unk:
            stats += f"  {unk} <unk>"
        print(f"\n  {t.name:<{width}}  {stats}")
        print(f"    {render(pieces, color)}")
        if ids:
            print(f"    {DIM if color else ''}{tok_ids}{RESET if color else ''}")

    if len(toks) == 2 and counts[0] != counts[1]:
        lo, hi = min(counts), max(counts)
        more = toks[counts.index(hi)].name
        print(f"\n  -> {more} uses {hi - lo} MORE tokens "
              f"({hi} vs {lo}, x{hi / lo:.3f} fertility)")
    elif len(toks) == 2:
        print(f"\n  -> same token count ({counts[0]})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Compare tokenizer segmentations.")
    ap.add_argument("text", nargs="*", help="text to tokenize (omit for interactive)")
    ap.add_argument("--toks", default="unigram_starved,unigram_destarved",
                    help="comma-separated tokenizer names (default: %(default)s)")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="HF repo to fetch tokenizers from (default: %(default)s)")
    ap.add_argument("--ids", action="store_true", help="also print token ids")
    ap.add_argument("--no-color", action="store_true", help="plain output")
    a = ap.parse_args(argv)

    color = not a.no_color and sys.stdout.isatty()
    names = [n.strip() for n in a.toks.split(",") if n.strip()]
    toks = [Tok(n, a.repo) for n in names]
    show_ids = a.ids

    if a.text:
        show(" ".join(a.text), toks, color, show_ids)
        return

    for t in toks:
        print(f"{t.name:<20} {t.kind}  vocab {t.vocab_size}  ({t.dir})")
    interactive = sys.stdin.isatty()
    if interactive:
        try:                       # arrow keys + history in the REPL
            import readline  # noqa: F401
        except ImportError:
            pass
        print("\ntype text and press enter  ·  `:ids` toggles ids  ·  `:q` or Ctrl-D quits")

    while True:
        try:
            line = input("\n> " if interactive else "")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        line = line.strip()
        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            return
        if line == ":ids":
            show_ids = not show_ids
            print(f"ids {'on' if show_ids else 'off'}")
            continue
        show(line, toks, color, show_ids)


if __name__ == "__main__":
    main()
