"""Model-training text pools from FineWeb-HQ (EN) / FineWeb2-HQ (DE/FR/AR/ZH).

Both datasets come from the same lab and the same XLM-R-embedding quality
classifier (top-10%% selection), keeping filtering methodology constant across
all five languages -- with two documented exceptions: FineWeb2-HQ's arb_Arab
split is only ~29GB total (85 parquet files, confirmed exhausted 2026-07-13),
and fra_Latn -HQ fully exhausted at 124.1GB (2026-07-14), both far short of
what our token budgets need. `FALLBACK_SOURCES` lets a language keep pulling
from a second, less-strict source (FineWeb2's own dedup+quality-filtered
"train" split, one filter step short of the top-decile HQ cut -- still same
lab/pipeline) once its primary source is exhausted, rather than epoch heavily
over a small pool. This is a real, deliberate deviation from strict
filter-parity for these two languages; note it in the thesis.

We stream parquet with column pruning (only `text`), so the large `embeddings`
column in FineWeb2-HQ is never downloaded. Output pools are zstd jsonl shards
of ~1GB uncompressed text. The first parquet file of the primary source is
reserved exclusively for the in-domain eval holdout (never enters the pool).
"""
import json
from pathlib import Path

from ..langs import LANGS
from ..paths import MANIFEST_CACHE, POOLS, HOLDOUT, pool_dir, ensure

POOL_SHARD_BYTES = 1 << 30      # uncompressed text per pool shard
HOLDOUT_BYTES = 30 * (1 << 20)  # 30MB per language

FALLBACK_SOURCES: dict[str, tuple[str, str]] = {
    "ar": ("HuggingFaceFW/fineweb-2", "data/arb_Arab/train"),
    # fra_Latn -HQ (epfml/FineWeb2-HQ) fully exhausted 2026-07-14 at 124.1GB,
    # short of the 152.2GB budget -- same fallback pattern as ar.
    "fr": ("HuggingFaceFW/fineweb-2", "data/fra_Latn/train"),
}


def _sources_for(lang: str) -> list[tuple[str, str]]:
    """[(repo, subdir), ...] in priority order: primary (quality-filtered -HQ)
    first, then any FALLBACK_SOURCES entry once the primary is exhausted."""
    L = LANGS[lang]
    srcs = [(L.fineweb_repo, L.fineweb_subdir)]
    if lang in FALLBACK_SOURCES:
        srcs.append(FALLBACK_SOURCES[lang])
    return srcs


def _list_parquets(repo: str, subdir: str) -> list[str]:
    """Manifest of parquet files under repo/subdir, round-robin across CC dumps
    for raw FineWeb-HQ (EN)."""
    from huggingface_hub import HfApi
    cache = ensure(MANIFEST_CACHE / "fineweb_hq_pools") / \
        f"{repo.replace('/', '__')}__{subdir.replace('/', '_')}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    api = HfApi()
    files = [e.path for e in api.list_repo_tree(repo, subdir,
                                                repo_type="dataset", recursive=True)
             if e.__class__.__name__ == "RepoFile" and e.path.endswith(".parquet")]
    if repo == "epfml/FineWeb-HQ":
        # data/CC-MAIN-YYYY-WW/000_xxxxx.parquet: interleave dumps so the pool
        # spans the full 2013-2024 crawl range like FineWeb2-HQ does.
        by_dump: dict[str, list[str]] = {}
        for f in sorted(files):
            by_dump.setdefault(f.split("/")[1], []).append(f)
        out = []
        for i in range(max(len(v) for v in by_dump.values())):
            for d in sorted(by_dump):
                if i < len(by_dump[d]):
                    out.append(by_dump[d][i])
        files = out
    else:
        files = sorted(files)
    cache.write_text(json.dumps(files))
    return files


def _iter_texts(repo: str, path_in_repo: str):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    with fs.open(f"datasets/{repo}/{path_in_repo}", "rb") as f:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["text"])
            for t in tbl.column("text").to_pylist():
                if t:
                    yield t


class _PoolWriter:
    def __init__(self, out_dir: Path, prefix: str = "pool", start_idx: int = -1,
                 total_bytes: int = 0, total_docs: int = 0):
        import zstandard
        self.dir = ensure(out_dir)
        self.prefix = prefix
        self.zstd = zstandard
        self.idx = start_idx
        self.cur = None
        self.cur_bytes = 0
        self.total_bytes = total_bytes
        self.total_docs = total_docs
        self._roll()

    def _roll(self):
        if self.cur:
            self.cur.close()
        self.idx += 1
        self.cur_bytes = 0
        raw = open(self.dir / f"{self.prefix}_{self.idx:05d}.jsonl.zst", "wb")
        self.cur = self.zstd.ZstdCompressor(level=3).stream_writer(raw, closefd=True)

    def write(self, text: str):
        line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
        b = line.encode("utf-8")
        self.cur.write(b)
        n = len(text.encode("utf-8"))
        self.cur_bytes += n
        self.total_bytes += n
        self.total_docs += 1
        if self.cur_bytes >= POOL_SHARD_BYTES:
            self._roll()

    def close(self):
        if self.cur:
            self.cur.close()
            self.cur = None


CHECKPOINT_EVERY_N_FILES = 3   # bound on re-downloaded work if the process dies


def _next_shard_idx(out_dir: Path, prefix: str = "pool") -> int:
    existing = sorted(out_dir.glob(f"{prefix}_*.jsonl.zst"))
    if not existing:
        return -1
    stem = existing[-1].name[len(prefix) + 1:]        # "NNNNN.jsonl.zst"
    return int(stem.split(".", 1)[0])


def build_pool(lang: str, budget_bytes: float, holdout_bytes: int = HOLDOUT_BYTES) -> dict:
    """Build (or resume) a language's text pool.

    Crash-resumable: every CHECKPOINT_EVERY_N_FILES source files, the current
    shard is closed (a truncated zstd frame from a mid-write kill can't be
    decoded, so a checkpoint never leaves a shard half-written) and
    `stats.json` records which manifest files are already consumed plus the
    index of that last cleanly-closed shard (`shard_idx`). A crash *between*
    checkpoints (e.g. mid-way through one large parquet file) can still leave
    a higher-numbered shard on disk with an unterminated zstd frame -- since
    none of its bytes were counted into `text_bytes`/`docs` at the last
    checkpoint, resuming deletes any shard past `shard_idx` before writing
    starts again, so a stray corrupt shard never lingers for `pack()` to trip
    on. Re-running `xscript pool` after an interruption (SSH drop, node
    hiccup, ^C, session teardown) picks up from there instead of
    re-downloading from scratch. `files_consumed` entries are tagged
    "repo::path" so a language with a FALLBACK_SOURCES entry can track
    consumption across sources without collision; old untagged checkpoints
    (single-source) are migrated on load by assuming the primary repo.
    """
    sources = _sources_for(lang)
    primary_repo, primary_subdir = sources[0]
    first_files = _list_parquets(primary_repo, primary_subdir)
    if not first_files:
        raise RuntimeError(f"no parquet files found for {lang}")
    out = pool_dir(lang)
    stats_path = out / "stats.json"
    resume = None
    if stats_path.exists():
        st = json.loads(stats_path.read_text())
        if st["text_bytes"] >= budget_bytes * 0.99:
            print(f"[pool] {lang}: cached ({st['text_bytes']/1e9:.1f}GB)")
            return st
        if st.get("files_consumed"):
            resume = st
            print(f"[pool] {lang}: resuming from checkpoint "
                  f"({st['text_bytes']/1e9:.1f}/{budget_bytes/1e9:.1f}GB, "
                  f"{len(st['files_consumed'])} files already consumed)")

    if resume is None:
        # holdout from the primary source's first file only; pool starts at the second
        hw = _PoolWriter(HOLDOUT, prefix=lang)
        got = 0
        for t in _iter_texts(primary_repo, first_files[0]):
            hw.write(t)
            got += len(t.encode("utf-8"))
            if got >= holdout_bytes:
                break
        hw.close()
        used: list[str] = []
        pw = _PoolWriter(out)
    else:
        got = resume["holdout_bytes"]
        used = [u if "::" in u else f"{primary_repo}::{u}" for u in resume["files_consumed"]]
        # shard_idx is missing on checkpoints written before this field existed;
        # best-effort fall back to whatever's on disk (pre-existing behaviour).
        last_good_idx = resume.get("shard_idx")
        if last_good_idx is None:
            last_good_idx = _next_shard_idx(out)
        else:
            for stray in out.glob("pool_*.jsonl.zst"):
                idx = int(stray.name[len("pool_"):].split(".", 1)[0])
                if idx > last_good_idx:
                    stray.unlink()   # never checkpointed -- may be a truncated zstd frame
            # The file *at* shard_idx is the one _checkpoint()'s _roll() had just
            # opened (empty) when that checkpoint was written -- everything written
            # into it since then, up to a crash, was never counted in text_bytes/
            # docs. It's fine (already fully closed) if the run reached this point
            # via the final `pw.close()` on graceful completion, but indistinguishable
            # from a crash-mid-write from stats.json alone -- so verify by decoding.
            at_idx = out / f"pool_{last_good_idx:05d}.jsonl.zst"
            if at_idx.exists():
                import io
                import zstandard
                try:
                    with open(at_idx, "rb") as raw:
                        reader = zstandard.ZstdDecompressor().stream_reader(raw)
                        for jline in io.TextIOWrapper(reader, encoding="utf-8"):
                            if jline.strip():
                                json.loads(jline)
                except Exception as exc:
                    print(f"[pool] {lang}: {at_idx.name} failed validation ({exc}) "
                          f"-- discarding (never counted in checkpointed totals)")
                    at_idx.unlink()
        pw = _PoolWriter(out, start_idx=last_good_idx,
                         total_bytes=resume["text_bytes"], total_docs=resume["docs"])

    def _checkpoint():
        pw._roll()   # close the current shard so it's a complete, valid zstd frame
        st = {"lang": lang, "budget_bytes": budget_bytes, "text_bytes": pw.total_bytes,
              "docs": pw.total_docs, "holdout_bytes": got, "holdout_file": first_files[0],
              "files_consumed": used, "shard_idx": pw.idx, "exhausted": False}
        stats_path.write_text(json.dumps(st, indent=2))

    done = False
    for i, (repo, subdir) in enumerate(sources):
        files = first_files if i == 0 else _list_parquets(repo, subdir)
        pool_files = files[1:] if i == 0 else files   # only the primary source reserves a holdout file
        for f in pool_files:
            tag = f"{repo}::{f}"
            if tag in used:
                continue
            used.append(tag)
            try:
                for t in _iter_texts(repo, f):
                    pw.write(t)
                    if pw.total_bytes >= budget_bytes:
                        break
            except Exception as exc:
                print(f"[pool] WARN {tag}: {exc}")
                used.pop()   # not actually consumed -- retry it on the next run
            if pw.total_bytes >= budget_bytes:
                done = True
                _checkpoint()
                break
            if len(used) % CHECKPOINT_EVERY_N_FILES == 0:
                _checkpoint()
            if len(used) % 20 == 0:
                print(f"[pool] {lang}: {pw.total_bytes/1e9:.1f}/{budget_bytes/1e9:.1f}GB "
                      f"({len(used)} files, source {i+1}/{len(sources)}: {repo})")
        if done:
            break
        if i + 1 < len(sources):
            print(f"[pool] {lang}: source {i+1}/{len(sources)} ({repo}) exhausted at "
                  f"{pw.total_bytes/1e9:.1f}GB -> falling back to {sources[i+1][0]}")
    pw.close()
    st = {"lang": lang, "budget_bytes": budget_bytes, "text_bytes": pw.total_bytes,
          "docs": pw.total_docs, "holdout_bytes": got, "holdout_file": first_files[0],
          "files_consumed": used, "shard_idx": pw.idx,
          "exhausted": pw.total_bytes < budget_bytes * 0.99}
    stats_path.write_text(json.dumps(st, indent=2))
    if st["exhausted"]:
        print(f"[pool] WARNING {lang}: corpus exhausted at {pw.total_bytes/1e9:.1f}GB "
              f"< budget {budget_bytes/1e9:.1f}GB -> training will epoch over this pool")
    print(f"[pool] {lang}: {pw.total_bytes/1e9:.2f}GB text, {pw.total_docs} docs")
    return st


def _measured_bytes_per_token(flavor: str = "unigram", condition: str = "destarved") -> dict[str, float]:
    """Real bytes/token per study language, measured on FLORES+ dev with the
    tokenizer that will actually pack the pool text. Destarved is the more
    byte-hungry of the two conditions (better fertility -> more input bytes
    needed per token), so it's the binding case for sizing. Scripts vary a
    lot (Arabic ~6.8 bytes/token vs Chinese ~4.0) -- returns {} (caller falls
    back to a flat estimate) if the tokenizer or FLORES+ aren't ready yet.
    """
    from .. import flores
    from ..langs import tok_name
    from ..paths import tokenizer_dir
    from ..tok.wrapper import Tok

    tdir = tokenizer_dir(tok_name(flavor, condition))
    if not (tdir / "meta.json").exists():
        return {}
    try:
        tok = Tok(tdir)
        par = flores.load_parallel(list(LANGS), "dev")
    except Exception as exc:
        print(f"[pool] WARN: couldn't measure real bytes/token ({exc}); "
              f"falling back to the flat estimate")
        return {}
    out = {}
    for l, sents in par.items():
        b = sum(len(s.encode("utf-8")) for s in sents)
        t = sum(len(tok.encode(s)) for s in sents)
        out[l] = b / t
    return out


def plan_budgets(tokens_per_run: float = 30e9, est_bytes_per_token: float = 4.5,
                 safety: float = 1.15) -> dict[str, float]:
    """Per-language pool byte budgets.

    Monolingual runs need the full token budget in one language; bilingual
    runs need half. The pool must cover the *max* need across planned runs
    under the worst-case (most byte-hungry, i.e. destarved) tokenizer.

    Sized from each language's REAL measured bytes/token (destarved
    tokenizer on FLORES+), not a flat guess -- a single constant badly
    under/over-shoots per language given how much bytes/token varies by
    script. `est_bytes_per_token` is only a fallback for languages the
    measurement can't cover yet (e.g. before the tokenizer gate).
    """
    need_tokens = {l: tokens_per_run for l in LANGS}  # monolingual dominates
    measured = _measured_bytes_per_token()
    return {l: need_tokens[l] * measured.get(l, est_bytes_per_token) * safety
            for l in LANGS}
