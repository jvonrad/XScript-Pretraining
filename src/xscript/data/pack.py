"""Tokenize text pools into uint16 token shards (one .bin per pool shard).

Documents are shuffled within each pool shard (seeded) and packed as
<bos> doc <eos> <bos> doc <eos> ... with no padding; the training loader
reads contiguous (seq_len+1)-token windows. Vocab is 65536 so uint16 is
exact.
"""
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from ..paths import pool_dir, shard_dir, tokenizer_dir, ensure

ENCODE_BATCH = 512


def _iter_pool_docs(path: Path):
    import zstandard
    with open(path, "rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        import io
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)["text"]


def _pack_one(args) -> tuple[str, int]:
    pool_shard, out_path, tok_dir, seed = args
    from ..tok.wrapper import Tok
    import random
    tok = Tok(tok_dir)
    docs = list(_iter_pool_docs(Path(pool_shard)))
    random.Random(seed).shuffle(docs)
    n = 0
    with open(out_path, "wb") as out:
        for i in range(0, len(docs), ENCODE_BATCH):
            batch = tok.encode_batch(docs[i:i + ENCODE_BATCH], bos=True, eos=True)
            flat = np.concatenate([np.asarray(ids, dtype=np.uint16) for ids in batch])
            flat.tofile(out)
            n += len(flat)
    return str(out_path), n


def pack(lang: str, tok_name: str, workers: int = 8, seed: int = 1234,
         max_tokens: float | None = None) -> dict:
    src = pool_dir(lang)
    all_shards = sorted(src.glob("pool_*.jsonl.zst"))
    if not all_shards:
        raise FileNotFoundError(f"no pool shards in {src} - run `xscript pool --lang {lang}`")
    # While `xscript pool` is still running, its writer may have the highest-
    # indexed shard open (only closed on the next roll/checkpoint) -- hold it
    # back so we never read a truncated file mid-download. Safe to include
    # once the pool has reached its byte budget; picked up on the next
    # incremental `pack` call otherwise.
    pool_done = False
    stats_path = src / "stats.json"
    if stats_path.exists():
        pst = json.loads(stats_path.read_text())
        pool_done = pst["text_bytes"] >= pst["budget_bytes"] * 0.99
    shards = all_shards if pool_done else all_shards[:-1]
    out = ensure(shard_dir(lang, tok_name))
    tok_dir = str(tokenizer_dir(tok_name))

    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    import zlib
    jobs = []
    for s in shards:
        dst = out / (s.name.replace(".jsonl.zst", ".bin"))
        if dst.name not in index:
            jobs.append((str(s), str(dst), tok_dir, seed ^ zlib.crc32(s.name.encode())))

    total = sum(index.values())
    if jobs:
        with mp.Pool(workers) as pool:
            for dst, n in pool.imap_unordered(_pack_one, jobs):
                if n == 0:
                    # Empty pool checkpoint shards contain no documents. Do
                    # not expose a zero-byte file to np.memmap in PackedStream.
                    Path(dst).unlink(missing_ok=True)
                    continue
                index[Path(dst).name] = n
                total += n
                index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
                print(f"[pack] {lang}/{tok_name}: {Path(dst).name} = {n/1e6:.1f}M tokens "
                      f"(total {total/1e9:.2f}B)")
                if max_tokens and total >= max_tokens:
                    print("[pack] reached max_tokens; stopping")
                    pool.terminate()
                    break
    meta = {"lang": lang, "tokenizer": tok_name, "total_tokens": total,
            "n_shards": len(index)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[pack] {lang}/{tok_name}: {total/1e9:.3f}B tokens in {len(index)} shards")
    return meta
