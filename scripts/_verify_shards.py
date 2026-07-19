#!/usr/bin/env python3
"""Verify packed uint16 token shards before training: structural integrity
(index.json vs on-disk .bin sizes, no orphans/missing files, meta.json
consistency) plus a real round-trip through PackedStream + Tok.decode to
sanity-check content, exercising the exact code path the training loop uses.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from xscript.paths import shard_dir, tokenizer_dir, POOLS
from xscript.tok.wrapper import Tok, BOS_ID, EOS_ID, UNK_ID, PAD_ID

LANGS = ["en", "de", "fr", "ar", "zh"]
TOKS = ["unigram_starved", "unigram_destarved"]


def verify_combo(lang: str, tok: str) -> list[str]:
    problems = []
    d = shard_dir(lang, tok)
    idx_path = d / "index.json"
    meta_path = d / "meta.json"
    if not idx_path.exists():
        return [f"MISSING index.json"]
    if not meta_path.exists():
        return [f"MISSING meta.json"]

    index = json.loads(idx_path.read_text())
    meta = json.loads(meta_path.read_text())

    on_disk = {p.name for p in d.glob("*.bin")}
    listed = set(index)
    missing = listed - on_disk
    orphans = on_disk - listed
    if missing:
        problems.append(f"index.json lists {len(missing)} .bin files not on disk: {sorted(missing)[:5]}")
    if orphans:
        problems.append(f"{len(orphans)} .bin files on disk not in index.json: {sorted(orphans)[:5]}")

    size_mismatches = []
    n_zero = 0
    for name, n_tokens in index.items():
        p = d / name
        if not p.exists():
            continue
        actual_bytes = p.stat().st_size
        expected_bytes = n_tokens * 2  # uint16
        # n_tokens==0 with actual_bytes==0 is the known legacy case (pool
        # checkpoint boundaries can produce a valid-but-empty source shard):
        # loader.py's PackedStream explicitly filters `index[name] > 0` before
        # ever mmap'ing, so these are harmless, not a live bug -- only an
        # inconsistency between the two (one zero, one not) is a real problem.
        if n_tokens == 0 and actual_bytes == 0:
            n_zero += 1
        elif actual_bytes != expected_bytes:
            size_mismatches.append((name, expected_bytes, actual_bytes))
    if size_mismatches:
        problems.append(f"{len(size_mismatches)} size mismatches (expected 2*n_tokens): {size_mismatches[:3]}")
    if n_zero:
        print(f"    (info) {lang}/{tok}: {n_zero} known-empty legacy shards (harmless, filtered by loader.py)")

    sum_tokens = sum(index.values())
    if sum_tokens != meta["total_tokens"]:
        problems.append(f"sum(index.json)={sum_tokens} != meta.json total_tokens={meta['total_tokens']}")
    if len(index) != meta["n_shards"]:
        problems.append(f"len(index.json)={len(index)} != meta.json n_shards={meta['n_shards']}")

    # cross-check against the source pool's byte budget: bytes/token should be
    # in a sane range (not exactly right since pool text_bytes includes docs
    # not yet packed if pool grew after packing, but a huge deviation = bug)
    pool_stats = POOLS / lang / "stats.json"
    if pool_stats.exists():
        st = json.loads(pool_stats.read_text())
        bpt = st["text_bytes"] / meta["total_tokens"]
        if not (0.5 < bpt < 20):
            problems.append(f"suspicious bytes/token={bpt:.2f} (pool text_bytes={st['text_bytes']}, tokens={meta['total_tokens']})")

    return problems


def roundtrip_sample(lang: str, tok_name: str, n_samples: int = 2, seq_len: int = 256):
    from xscript.data.loader import PackedStream
    stream = PackedStream(lang, tok_name, seq_len=seq_len, seed=1234)
    tok = Tok(tokenizer_dir(tok_name))
    assert tok.vocab_size == 65536, f"expected vocab 65536, got {tok.vocab_size}"
    out = []
    for k in range(n_samples):
        w = stream.get(k * 1000)
        assert w.dtype == np.uint16
        assert w.shape == (seq_len + 1,)
        assert w.max() < tok.vocab_size, f"token id {w.max()} >= vocab_size {tok.vocab_size}"
        n_bos = int((w == BOS_ID).sum())
        n_eos = int((w == EOS_ID).sum())
        n_unk = int((w == UNK_ID).sum())
        text = tok.decode(w.tolist())
        out.append({
            "window": int(k * 1000), "n_bos": n_bos, "n_eos": n_eos, "n_unk": n_unk,
            "preview": text[:200].replace("\n", " "),
        })
    return out


def main():
    all_ok = True
    for lang in LANGS:
        for tok in TOKS:
            problems = verify_combo(lang, tok)
            if problems:
                all_ok = False
                print(f"[FAIL] {lang}/{tok}:")
                for p in problems:
                    print(f"    - {p}")
            else:
                print(f"[OK] {lang}/{tok}: structural checks passed")

    print("\n--- round-trip decode samples ---")
    for lang in LANGS:
        for tok in TOKS:
            try:
                samples = roundtrip_sample(lang, tok)
                for s in samples:
                    print(f"[{lang}/{tok}] window={s['window']} bos={s['n_bos']} eos={s['n_eos']} unk={s['n_unk']}")
                    print(f"    {s['preview']!r}")
            except Exception as exc:
                all_ok = False
                print(f"[FAIL] {lang}/{tok} round-trip: {exc}")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
