#!/usr/bin/env python
"""Where does an N-language tokenizer sit between starved and fair?

Prints, for each tokenizer, the 64k allocation by script, the own-script
multi-character piece counts (CLAUDE.md results/tok_analysis/FINDINGS.md's
mechanism), and per-language emitted-token stats on FLORES+, each with a
"position" = (x - starved)/(fair - starved) so 0 = starved, 1 = fair.

    XSCRIPT_SCRATCH=/mnt/scratch/xscript python scripts/tok_allocation_compare.py \
        unigram_10lang unigram_20lang unigram_50lang
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xscript import flores
from xscript.langs import LANGS
from xscript.paths import tokenizer_dir
from xscript.tok.wrapper import Tok
from xscript.tok.analyze import classify_piece, vocab_allocation, _lang_metrics

mids = sys.argv[1:] or ["unigram_50lang"]
names = ["unigram_starved", *mids, "unigram_destarved"]
toks = {n: Tok(tokenizer_dir(n)) for n in names}
par = flores.load_parallel(list(LANGS), "dev"); pt = flores.load_parallel(list(LANGS), "devtest")
texts = {l: par[l] + pt[l] for l in LANGS}

def multi(tok):
    out = {}
    for i in range(tok.vocab_size):
        if tok.is_byte_piece(i):
            continue
        b = classify_piece(tok.piece_bytes(i))
        if b in ("Latin", "Arabic", "Han", "Cyrillic"):
            n = len(tok.piece_bytes(i).decode("utf-8").strip())
            d = out.setdefault(b, [0, 0]); d[0] += 1; d[1] += n > 1
    return out

alloc = {n: vocab_allocation(t) for n, t in toks.items()}
mc = {n: multi(t) for n, t in toks.items()}
met = {n: {l: _lang_metrics(t, texts[l]) for l in LANGS} for n, t in toks.items()}
S, F = names[0], names[-1]
def pos(x, s, f):
    return f"{(x - s) / (f - s):5.2f}" if f != s else "  n/a"
short = lambda n: n.replace("unigram_", "").replace("destarved", "fair")
hdr = "".join(f"{short(n):>10s}" for n in names)

print("== 64k vocab pieces by script ==            " + hdr + "   position of " + ", ".join(short(m) for m in mids))
for b in ["Latin", "Arabic", "Han", "Cyrillic", "Kana", "Hangul", "Greek", "Hebrew", "Devanagari", "Thai", "OtherIndic", "OtherScript", "mixed", "sym_num_space"]:
    v = [alloc[n].get(b, 0) for n in names]
    print(f"{b:14s}{'':30s}" + "".join(f"{x:10d}" for x in v) + "   " + " ".join(pos(alloc[m].get(b, 0), v[0], v[-1]) for m in mids))
print("\n== own-script MULTI-character pieces (the mechanism) ==")
for b in ["Latin", "Arabic", "Han", "Cyrillic"]:
    v = [mc[n].get(b, [0, 0])[1] for n in names]
    print(f"{b:14s}{'':30s}" + "".join(f"{x:10d}" for x in v) + "   " + " ".join(pos(mc[m].get(b, [0, 0])[1], v[0], v[-1]) for m in mids))
print("\n== emitted tokens on FLORES+ (dev+devtest), per language ==")
for key, label in [("tokens_per_sentence", "tokens/sentence"), ("unique_tokens_used", "unique pieces used"), ("pct_single_char_tokens", "% single-char tokens")]:
    print(f"-- {label}")
    for l in LANGS:
        v = [met[n][l][key] for n in names]
        fmt = (lambda x: f"{x:10.1f}") if isinstance(v[0], float) else (lambda x: f"{x:10d}")
        print(f"  {l:12s}{'':30s}" + "".join(fmt(x) for x in v) + "   " + " ".join(pos(met[m][l][key], v[0], v[-1]) for m in mids))
