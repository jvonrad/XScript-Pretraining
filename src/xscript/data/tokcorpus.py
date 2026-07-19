"""Tokenizer-training corpora from raw (unfiltered) FineWeb / FineWeb2.

Both tokenizer conditions are sampled from the SAME corpus family as the
model-training pools (data/fineweb.py's FineWeb-HQ / FineWeb2-HQ), just the
unfiltered releases -- HuggingFaceFW/fineweb (English) and
HuggingFaceFW/fineweb-2 (everyone else). This is a deliberate deviation from
ATLAS's literal MADLAD-400-trained tokenizer: it removes a tokenizer-corpus-
vs-model-corpus domain-mismatch confound that could otherwise hit AR/ZH
differently than DE/FR (MADLAD's non-Latin-script cleaning/LangID is less
consistent than FineWeb's), at the cost of no longer being a byte-for-byte
ATLAS replication for the starved tokenizer's source text.

  starved   -- ATLAS-style: T=100 temperature sampling (p_l ~ n_l^(1/100),
               i.e. near-uniform) over ~419 languages -- English (raw FineWeb)
               plus the ~418 largest FineWeb2 language-script configs by
               volume. "Largest by volume" is the FineWeb2 analogue of how
               MADLAD-400's own ~419-language "clean" set was itself
               determined (languages with enough clean text to clear a
               volume floor), so it preserves the same selection logic, just
               applied to a different corpus.
  destarved -- our 5 study languages only; per-language byte budgets scaled
               by the FLORES+ byte premium so *content* (not bytes) is
               uniform across languages.

Both stream parquet with column pruning (reusing data.fineweb's `_iter_texts`),
so only the `text` column is ever pulled.
"""
import json
import random
import urllib.request
from pathlib import Path

from ..langs import LANGS
from ..paths import MANIFEST_CACHE, TOK_CORPORA, ensure
from .fineweb import _iter_texts

FINEWEB_EN_REPO = "HuggingFaceFW/fineweb"
FINEWEB2_REPO = "HuggingFaceFW/fineweb-2"
N_STARVED_LANGS = 419  # matches MADLAD-400/ATLAS's ~419-language scale

# SentencePiece skips sentences longer than max_sentence_length (default 4192
# bytes); we pre-split long documents so no text is silently dropped.
MAX_LINE_BYTES = 4000


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def _fineweb2_size_manifest(refresh: bool = False) -> dict[str, int]:
    """{config_name: num_bytes_original_files} for every FineWeb2 language-script config."""
    cache = ensure(MANIFEST_CACHE) / "fineweb2_sizes.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    d = _get_json(f"https://datasets-server.huggingface.co/size?dataset={FINEWEB2_REPO}")
    sizes = {c["config"]: c["num_bytes_original_files"] for c in d["size"]["configs"]}
    cache.write_text(json.dumps(sizes))
    print(f"[tokcorpus] fetched FineWeb2 sizes for {len(sizes)} configs")
    return sizes


def _fineweb_en_size(refresh: bool = False) -> int:
    cache = ensure(MANIFEST_CACHE) / "fineweb_en_size.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())["bytes"]
    d = _get_json(f"https://datasets-server.huggingface.co/size?dataset={FINEWEB_EN_REPO}&config=default")
    n = d["size"]["config"]["num_bytes_original_files"]
    cache.write_text(json.dumps({"bytes": n}))
    return n


def select_starved_languages(n_langs: int = N_STARVED_LANGS) -> dict[str, int]:
    """{code: available_bytes} for the starved condition's language universe.

    "en" (raw FineWeb) plus the (n_langs-1) largest FineWeb2 configs by volume.
    Our other 4 study languages (de/fr/ar/zh) are always near the top of that
    ranking on volume alone (verified: ranks 2-13 of 1314), so no forced
    inclusion is needed.
    """
    sizes = _fineweb2_size_manifest()
    top = sorted(sizes.items(), key=lambda kv: -kv[1])[: n_langs - 1]
    universe = {"en": _fineweb_en_size(), **dict(top)}
    return universe


def _source_for_code(code: str) -> tuple[str, str]:
    """(repo, subdir) of raw FineWeb-family text for a language code.

    `code` is either "en", one of our other 4 study codes, or a raw FineWeb2
    language-script config name (a starved-only competing language).
    """
    if code == "en":
        return FINEWEB_EN_REPO, "data"
    if code in LANGS:
        code = LANGS[code].fineweb_subdir  # study code -> FineWeb2 config name
    return FINEWEB2_REPO, f"data/{code}/train"


def _list_parquet_files(repo: str, subdir: str) -> list[str]:
    cache = ensure(MANIFEST_CACHE / "parquet_files") / f"{repo.replace('/', '__')}__{subdir.replace('/', '_')}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from huggingface_hub import HfApi
    api = HfApi()
    files = [e.path for e in api.list_repo_tree(repo, subdir, repo_type="dataset", recursive=True)
             if e.__class__.__name__ == "RepoFile" and e.path.endswith(".parquet")]
    # dump-stratified sources (raw FineWeb's CC-MAIN-*/) interleave across dumps
    # for a temporally representative sample; FineWeb2's per-language train/
    # files have no such structure, so plain sort is enough.
    by_dump: dict[str, list[str]] = {}
    for f in sorted(files):
        parts = f[len(subdir):].strip("/").split("/")
        key = parts[0] if len(parts) > 1 and "CC-MAIN" in parts[0] else ""
        by_dump.setdefault(key, []).append(f)
    if len(by_dump) > 1:
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


def _split_line(text: str):
    """Yield non-empty lines, splitting any line over MAX_LINE_BYTES at whitespace."""
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        enc = ln.encode("utf-8")
        while len(enc) > MAX_LINE_BYTES:
            cut = enc[:MAX_LINE_BYTES].rfind(b" ")
            if cut < MAX_LINE_BYTES // 2:
                cut = MAX_LINE_BYTES
                while cut > 0 and (enc[cut] & 0xC0) == 0x80:  # don't split mid UTF-8 char
                    cut -= 1
            yield enc[:cut].decode("utf-8", errors="replace").strip()
            enc = enc[cut:].lstrip()
        if enc:
            yield enc.decode("utf-8", errors="replace")


def _collect(code: str, budget_bytes: int, out_path: Path, seed: int = 0) -> int:
    """Stream parquet files (shuffled order) for one language until budget is met."""
    repo, subdir = _source_for_code(code)
    files = list(_list_parquet_files(repo, subdir))
    random.Random(seed).shuffle(files)
    got = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for f in files:
            if got >= budget_bytes:
                break
            try:
                for doc in _iter_texts(repo, f):
                    for ln in _split_line(doc):
                        out.write(ln + "\n")
                        got += len(ln.encode("utf-8")) + 1
                    if got >= budget_bytes:
                        break
            except Exception as exc:
                print(f"[tokcorpus] WARN {repo}/{f} failed mid-stream: {exc}")
    return got


def build_starved(total_bytes: float = 4e9, T: float = 100.0, seed: int = 0,
                  n_langs: int = N_STARVED_LANGS) -> Path:
    """T-temperature sample over ~419 FineWeb/FineWeb2 languages (ATLAS-scale replication)."""
    universe = select_starved_languages(n_langs)
    out_dir = ensure(TOK_CORPORA / "starved")
    weights = {l: b ** (1.0 / T) for l, b in universe.items()}
    z = sum(weights.values())
    stats = {}
    for i, code in enumerate(sorted(universe)):
        budget = int(total_bytes * weights[code] / z)
        out_path = out_dir / f"{code}.txt"
        if out_path.exists() and out_path.stat().st_size >= 0.9 * budget:
            stats[code] = {"budget": budget, "bytes": out_path.stat().st_size, "cached": True}
            continue
        got = _collect(code, budget, out_path, seed=seed + i)
        stats[code] = {"budget": budget, "bytes": got}
        print(f"[starved] {code}: {got/1e6:.1f}MB / budget {budget/1e6:.1f}MB "
              f"({i+1}/{len(universe)})")
    (out_dir / "stats.json").write_text(json.dumps(
        {"total_bytes": total_bytes, "T": T, "n_langs": len(universe), "per_lang": stats},
        indent=2))
    return out_dir


def build_destarved(total_bytes: float = 4e9, seed: int = 0) -> Path:
    """5 study languages; byte budgets scaled by FLORES+ byte premium (content-uniform)."""
    from ..byte_premium import load_premiums
    premiums = load_premiums()
    out_dir = ensure(TOK_CORPORA / "destarved")
    z = sum(premiums[l] for l in LANGS)
    stats = {}
    for i, code in enumerate(LANGS):
        budget = int(total_bytes * premiums[code] / z)
        out_path = out_dir / f"{code}.txt"
        if out_path.exists() and out_path.stat().st_size >= 0.9 * budget:
            stats[code] = {"budget": budget, "bytes": out_path.stat().st_size, "cached": True}
            continue
        got = _collect(code, budget, out_path, seed=seed + i)
        stats[code] = {"budget": budget, "bytes": got}
        print(f"[destarved] {code}: {got/1e6:.1f}MB / budget {budget/1e6:.1f}MB")
    (out_dir / "stats.json").write_text(json.dumps(
        {"total_bytes": total_bytes, "premiums": premiums, "per_lang": stats}, indent=2))
    return out_dir


def corpus_files(condition: str) -> list[Path]:
    d = TOK_CORPORA / condition
    files = sorted(d.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no corpus at {d} - run `xscript tok-corpus --condition {condition}`")
    return files
