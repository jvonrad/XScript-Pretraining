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

from ..langs import (LANGS, ANCHOR, PARTNERS, EN_SHARE_50LANG,
                     BILINGUAL_TOK_CONDITIONS, TOK_CONDITION_50LANG,
                     NLANG_CONDITIONS, nlang_of)
from ..paths import MANIFEST_CACHE, TOK_CORPORA, ensure
from .fineweb import _iter_texts

FINEWEB_EN_REPO = "HuggingFaceFW/fineweb"
FINEWEB2_REPO = "HuggingFaceFW/fineweb-2"
N_STARVED_LANGS = 419  # matches MADLAD-400/ATLAS's ~419-language scale
N_50LANG = 50          # corroboration condition: our 5 + the 45 largest others

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


# --------------------------------------------------------------------------- #
# Corroboration conditions (2026-09-04): 50lang and bi_<X>
# --------------------------------------------------------------------------- #
# Both reuse `_collect` unchanged. The five study languages are collected ONCE
# into TOK_CORPORA/cache/<code>.txt at the largest budget any condition needs
# and every condition takes a byte-prefix of that file (cut at a line
# boundary). Because `_collect` streams the seeded-shuffled file list in
# order until the budget is met, a prefix of a larger collection is exactly
# what a smaller collection with the same seed would have produced -- so the
# cache is a pure download saving, not a change in what the corpus contains.

_CACHE_SEED = {l: 100 + i for i, l in enumerate(LANGS)}


def select_50lang_languages(n_langs: int = N_50LANG) -> list[str]:
    """["en"] + our four partners + the largest remaining FineWeb2 configs by
    volume up to n_langs, as raw config names (deu_Latn, ...), in volume
    order. The partners are ranks 2/3/5/13 of 1314, so for n_langs >= 14
    this is exactly the top-(n_langs-1); below that (10lang) Arabic would
    otherwise be dropped, so the study languages are forced in."""
    sizes = _fineweb2_size_manifest()
    ranked = [c for c, _ in sorted(sizes.items(), key=lambda kv: -kv[1])]
    study = [LANGS[l].fineweb_subdir for l in LANGS if l != "en"]
    if n_langs - 1 < len(study):
        raise ValueError(f"n_langs={n_langs} cannot hold the 5 study languages")
    fill = [c for c in ranked if c not in study][: n_langs - 1 - len(study)]
    chosen = set(study) | set(fill)
    return ["en", *[c for c in ranked if c in chosen]]


def _study_code(code: str) -> str:
    """FineWeb2 config name -> study short code where one exists, else itself."""
    for l, L in LANGS.items():
        if code == l or code == L.fineweb_subdir:
            return l
    return code


def study_cache_budgets(total_bytes: float = 4e9) -> dict[str, int]:
    """Largest byte budget each study language needs across 50lang + bi_<X>."""
    from ..byte_premium import load_premiums
    prem = load_premiums()
    n_min = min(nlang_of(c) for c in NLANG_CONDITIONS)
    other = total_bytes * (1.0 - EN_SHARE_50LANG) / (n_min - 1)
    need = {"en": total_bytes * EN_SHARE_50LANG}
    for l in LANGS:
        if l != "en":
            need[l] = other
    for cond, (a, x) in BILINGUAL_TOK_CONDITIONS.items():
        en_b = total_bytes / (1.0 + prem[x])
        need["en"] = max(need["en"], en_b)
        need[x] = max(need[x], total_bytes - en_b)
    return {l: int(v) for l, v in need.items()}


def _cache_path(code: str) -> Path:
    return ensure(TOK_CORPORA / "cache") / f"{code}.txt"


def _collect_job(args) -> tuple[str, int]:
    code, budget, out_path, seed = args
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size >= 0.9 * budget:
        return code, out_path.stat().st_size
    got = _collect(code, budget, out_path, seed=seed)
    print(f"[tokcorpus] {code}: {got/1e6:.1f}MB / budget {budget/1e6:.1f}MB", flush=True)
    return code, got


def _run_jobs(jobs, workers: int) -> dict[str, int]:
    import multiprocessing as mp
    out = {}
    if workers <= 1 or len(jobs) <= 1:
        for j in jobs:
            code, got = _collect_job(j)
            out[code] = got
        return out
    with mp.Pool(workers) as pool:
        for code, got in pool.imap_unordered(_collect_job, jobs):
            out[code] = got
    return out


def build_study_cache(total_bytes: float = 4e9, workers: int = 4) -> dict[str, int]:
    budgets = study_cache_budgets(total_bytes)
    jobs = [(l, b, str(_cache_path(l)), _CACHE_SEED[l]) for l, b in budgets.items()]
    return _run_jobs(jobs, workers)


def _materialize(code: str, budget: int, out_path: Path) -> int:
    """Byte-prefix of the cached collection, cut at a newline."""
    src = _cache_path(code)
    if not src.exists() or src.stat().st_size < 0.9 * budget:
        raise FileNotFoundError(f"cache for {code} missing/short ({src}); run build_study_cache")
    if out_path.exists() and out_path.stat().st_size >= 0.9 * budget:
        return out_path.stat().st_size
    got = 0
    with open(src, "rb") as f, open(out_path, "wb") as o:
        while got < budget:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            if got + len(chunk) > budget:
                cut = chunk.rfind(b"\n", 0, budget - got)
                chunk = chunk[: cut + 1] if cut >= 0 else chunk
                o.write(chunk)
                got += len(chunk)
                break
            o.write(chunk)
            got += len(chunk)
    return got


def build_nlang(n_langs: int, total_bytes: float = 4e9, seed: int = 0,
                workers: int = 4, en_share: float = EN_SHARE_50LANG) -> Path:
    """N languages: English fixed at `en_share` of the bytes, the other N-1
    uniform in bytes (NOT byte-premium adjusted -- deliberately mirrors the
    near-uniform-in-bytes T=100 starved mixture, just over N languages
    instead of 419). Condition name is f"{n_langs}lang"."""
    langs = select_50lang_languages(n_langs)
    out_dir = ensure(TOK_CORPORA / f"{n_langs}lang")
    other = total_bytes * (1.0 - en_share) / (len(langs) - 1)
    budgets = {c: int(total_bytes * en_share) if c == "en" else int(other) for c in langs}
    build_study_cache(total_bytes, workers)
    stats, jobs = {}, []
    for i, code in enumerate(langs):
        sc = _study_code(code)
        out_path = out_dir / f"{sc}.txt"
        if sc in LANGS:
            got = _materialize(sc, budgets[code], out_path)
            stats[sc] = {"budget": budgets[code], "bytes": got, "from_cache": True}
        else:
            jobs.append((code, budgets[code], str(out_path), seed + 1000 + i))
    for code, got in _run_jobs(jobs, workers).items():
        stats[code] = {"budget": budgets[code], "bytes": got}
    (out_dir / "stats.json").write_text(json.dumps(
        {"total_bytes": total_bytes, "en_share": en_share, "n_langs": len(langs),
         "languages": langs, "per_lang": stats}, indent=2))
    print(f"[{n_langs}lang] {len(langs)} languages, {sum(v['bytes'] for v in stats.values())/1e9:.2f}GB")
    return out_dir


def build_50lang(total_bytes: float = 4e9, seed: int = 0, workers: int = 4,
                 en_share: float = EN_SHARE_50LANG) -> Path:
    return build_nlang(50, total_bytes, seed, workers, en_share)


def build_bilingual(condition: str, total_bytes: float = 4e9, workers: int = 4) -> Path:
    """en + one partner, byte budgets scaled by the FLORES+ byte premium so
    content (not bytes) is matched -- the destarved recipe restricted to the
    pair a bilingual model actually trains on."""
    from ..byte_premium import load_premiums
    a, x = BILINGUAL_TOK_CONDITIONS[condition]
    prem = load_premiums()
    out_dir = ensure(TOK_CORPORA / condition)
    z = prem[a] + prem[x]
    budgets = {a: int(total_bytes * prem[a] / z), x: int(total_bytes * prem[x] / z)}
    build_study_cache(total_bytes, workers)
    stats = {}
    for code, b in budgets.items():
        got = _materialize(code, b, out_dir / f"{code}.txt")
        stats[code] = {"budget": b, "bytes": got, "from_cache": True}
    (out_dir / "stats.json").write_text(json.dumps(
        {"total_bytes": total_bytes, "premiums": {a: prem[a], x: prem[x]},
         "per_lang": stats}, indent=2))
    print(f"[{condition}] " + ", ".join(f"{c}={v['bytes']/1e6:.0f}MB" for c, v in stats.items()))
    return out_dir
