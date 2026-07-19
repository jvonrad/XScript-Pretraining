import zstandard, io, json, sys
from pathlib import Path
src = Path(sys.argv[1])
shards = sorted(src.glob("pool_*.jsonl.zst"))
print("total shards:", len(shards), flush=True)
bad = []
for i, s in enumerate(shards):
    try:
        with open(s, "rb") as raw:
            reader = zstandard.ZstdDecompressor().stream_reader(raw)
            n = 0
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                if line.strip():
                    json.loads(line)
                    n += 1
    except Exception as e:
        bad.append((s.name, str(e)))
        print("BAD:", s.name, e, flush=True)
    if i % 20 == 0:
        print("checked", i, flush=True)
print("DONE, bad shards:", bad, flush=True)
