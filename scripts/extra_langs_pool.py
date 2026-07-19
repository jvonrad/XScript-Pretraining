#!/usr/bin/env python3
"""Standalone: byte-premium + FineWeb2-HQ pool download for Russian/Indonesian.

NOT part of the 5-language thesis pipeline (EN/DE/FR/AR/ZH, see langs.py) --
deliberately kept separate so it can't affect LANGS, the run matrix, tokenizer
training, or the official results/byte_premium/byte_premiums.json. Reuses the
lang-agnostic building blocks from xscript.data.fineweb (which already take an
explicit repo/subdir, not a `lang` key) and xscript.flores's private jsonl
parsing helpers, so this has zero import-time dependency on LANGS.

Usage:
  python scripts/extra_langs_pool.py premium          # byte premium only
  python scripts/extra_langs_pool.py pool --langs ru id --budget-from-premium
  python scripts/extra_langs_pool.py both
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xscript.paths import FLORES_DIR, RESULTS, ensure, pool_dir, HOLDOUT
from xscript.data.fineweb import (
    _list_parquets, _iter_texts, _PoolWriter, CHECKPOINT_EVERY_N_FILES,
    HOLDOUT_BYTES,
)

# Code doubles as the FLORES+ file stem and the FineWeb2-HQ subdir -- both
# datasets use the same <iso639-3>_<script> convention for these languages.
EXTRA_LANGS = {"ru": "rus_Cyrl", "id": "ind_Latn"}
EN_FLORES_CODE = "eng_Latn"
FINEWEB2_HQ_REPO = "epfml/FineWeb2-HQ"
FINEWEB2_REPO = "HuggingFaceFW/fineweb-2"

# Same defaults as fineweb.plan_budgets()'s fallback (no destarved-tokenizer
# measurement exists for ru/id, since they're not in the tokenizer's corpus).
DEFAULT_TOKENS = 30e9
DEFAULT_BYTES_PER_TOKEN = 4.5
DEFAULT_SAFETY = 1.15
DEFAULT_BUDGET_BYTES = DEFAULT_TOKENS * DEFAULT_BYTES_PER_TOKEN * DEFAULT_SAFETY  # ~155.25GB
EN_30B_BUDGET_BYTES = 151.4e9  # measured destarved EN pool target for the 30B tier


def _download_flores(code: str, token=None) -> None:
    from huggingface_hub import hf_hub_download
    ensure(FLORES_DIR)
    for split in ("dev", "devtest"):
        hf_hub_download(
            repo_id="openlanguagedata/flores_plus", repo_type="dataset",
            filename=f"{split}/{code}.jsonl", local_dir=FLORES_DIR, token=token,
        )


def _load_flores(code: str, split: str) -> dict[int, str]:
    from xscript.flores import _pick, _TEXT_KEYS, _ID_KEYS
    path = FLORES_DIR / split / f"{code}.jsonl"
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[int(_pick(rec, _ID_KEYS))] = str(_pick(rec, _TEXT_KEYS))
    return out


def compute_premiums() -> dict:
    """Same methodology as byte_premium.compute(): bytes(L)/bytes(EN) over the
    id-intersection of parallel FLORES+ sentences, per split and combined."""
    for code in [EN_FLORES_CODE, *EXTRA_LANGS.values()]:
        _download_flores(code)

    per_split = {}
    totals = {"en": 0, **{l: 0 for l in EXTRA_LANGS}}
    for split in ("dev", "devtest"):
        en = _load_flores(EN_FLORES_CODE, split)
        per_lang_text = {"en": en}
        for l, code in EXTRA_LANGS.items():
            per_lang_text[l] = _load_flores(code, split)
        common = sorted(set.intersection(*(set(v) for v in per_lang_text.values())))
        b = {l: sum(len(t[i].encode("utf-8")) for i in common) for l, t in per_lang_text.items()}
        per_split[split] = {
            "n_sentences": len(common),
            "bytes": b,
            "premium": {l: b[l] / b["en"] for l in EXTRA_LANGS},
        }
        for l in totals:
            totals[l] += b[l]

    result = {
        "per_split": per_split,
        "bytes_total": totals,
        "premium": {l: totals[l] / totals["en"] for l in EXTRA_LANGS},
    }
    out_dir = ensure(RESULTS / "byte_premium")
    (out_dir / "byte_premiums_extra.json").write_text(json.dumps(result, indent=2))

    lines = ["| lang | premium (dev) | premium (devtest) | premium (combined) |",
             "|---|---|---|---|"]
    for l in EXTRA_LANGS:
        lines.append(
            f"| {l} | {per_split['dev']['premium'][l]:.4f} "
            f"| {per_split['devtest']['premium'][l]:.4f} "
            f"| {result['premium'][l]:.4f} |")
    table = "\n".join(lines)
    (out_dir / "byte_premiums_extra.md").write_text(table + "\n")
    print(table)
    return result


def build_pool_standalone(lang: str, subdir: str, budget_bytes: float,
                          holdout_bytes: int = HOLDOUT_BYTES) -> dict:
    """Mirrors fineweb.build_pool()'s crash-safe resume logic (checkpoint
    every CHECKPOINT_EVERY_N_FILES files, shard_idx-based orphan cleanup on
    resume). FineWeb2-HQ is consumed first, then the standard FineWeb2 train
    split supplies any remaining bytes; this matches the AR/FR pool policy."""
    sources = [(FINEWEB2_HQ_REPO, subdir),
               (FINEWEB2_REPO, f"data/{subdir}/train")]
    files = _list_parquets(*sources[0])
    if not files:
        raise RuntimeError(f"no parquet files found for {lang} ({subdir})")
    out = pool_dir(lang)
    stats_path = out / "stats.json"
    resume = None
    if stats_path.exists():
        st = json.loads(stats_path.read_text())
        if st["text_bytes"] >= budget_bytes * 0.99:
            print(f"[extra-pool] {lang}: cached ({st['text_bytes']/1e9:.1f}GB)")
            return st
        if st.get("files_consumed"):
            resume = st
            print(f"[extra-pool] {lang}: resuming from checkpoint "
                  f"({st['text_bytes']/1e9:.1f}/{budget_bytes/1e9:.1f}GB, "
                  f"{len(st['files_consumed'])} files already consumed)")

    if resume is None:
        hw = _PoolWriter(HOLDOUT, prefix=lang)
        got = 0
        for t in _iter_texts(FINEWEB2_HQ_REPO, files[0]):
            hw.write(t)
            got += len(t.encode("utf-8"))
            if got >= holdout_bytes:
                break
        hw.close()
        used: list[str] = []
        pw = _PoolWriter(out)
    else:
        got = resume["holdout_bytes"]
        # Migrate checkpoints made by the older HQ-only version, whose entries
        # were untagged paths, by assigning them to the primary source.
        used = [u if "::" in u else f"{FINEWEB2_HQ_REPO}::{u}"
                for u in resume["files_consumed"]]
        last_good_idx = resume.get("shard_idx")
        if last_good_idx is None:
            from xscript.data.fineweb import _next_shard_idx
            last_good_idx = _next_shard_idx(out)
        else:
            for stray in out.glob("pool_*.jsonl.zst"):
                idx = int(stray.name[len("pool_"):].split(".", 1)[0])
                if idx > last_good_idx:
                    stray.unlink()
        pw = _PoolWriter(out, start_idx=last_good_idx,
                         total_bytes=resume["text_bytes"], total_docs=resume["docs"])

    def _checkpoint():
        pw._roll()
        st = {"lang": lang, "budget_bytes": budget_bytes, "text_bytes": pw.total_bytes,
              "docs": pw.total_docs, "holdout_bytes": got, "holdout_file": files[0],
              "files_consumed": used, "shard_idx": pw.idx, "exhausted": False}
        stats_path.write_text(json.dumps(st, indent=2))

    done = False
    for source_idx, (repo, source_subdir) in enumerate(sources):
        source_files = files if source_idx == 0 else _list_parquets(repo, source_subdir)
        pool_files = source_files[1:] if source_idx == 0 else source_files
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
                print(f"[extra-pool] WARN {tag}: {exc}")
                used.pop()
            if pw.total_bytes >= budget_bytes:
                done = True
                _checkpoint()
                break
            if len(used) % CHECKPOINT_EVERY_N_FILES == 0:
                _checkpoint()
            if len(used) % 20 == 0:
                print(f"[extra-pool] {lang}: {pw.total_bytes/1e9:.1f}/"
                      f"{budget_bytes/1e9:.1f}GB ({len(used)} files, {repo})")
        if done:
            break
        if source_idx + 1 < len(sources):
            print(f"[extra-pool] {lang}: {repo} exhausted at "
                  f"{pw.total_bytes/1e9:.1f}GB; falling back to {sources[source_idx+1][0]}")
    pw.close()
    st = {"lang": lang, "budget_bytes": budget_bytes, "text_bytes": pw.total_bytes,
          "docs": pw.total_docs, "holdout_bytes": got, "holdout_file": files[0],
          "files_consumed": used, "shard_idx": pw.idx,
          "exhausted": pw.total_bytes < budget_bytes * 0.99}
    stats_path.write_text(json.dumps(st, indent=2))
    if st["exhausted"]:
        print(f"[extra-pool] WARNING {lang}: corpus exhausted at {pw.total_bytes/1e9:.1f}GB "
              f"< budget {budget_bytes/1e9:.1f}GB")
    print(f"[extra-pool] {lang}: {pw.total_bytes/1e9:.2f}GB text, {pw.total_docs} docs")
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["premium", "pool", "both"])
    ap.add_argument("--langs", nargs="*", default=list(EXTRA_LANGS), choices=list(EXTRA_LANGS))
    ap.add_argument("--budget-gb", type=float)
    ap.add_argument("--budget-from-premium", action="store_true",
                    help="scale the measured 30B English byte budget by each "
                         "language's FLORES+ byte-premium ratio")
    args = ap.parse_args()

    if args.cmd in ("premium", "both"):
        compute_premiums()
    if args.cmd in ("pool", "both"):
        if args.budget_from_premium:
            premium_path = RESULTS / "byte_premium" / "byte_premiums_extra.json"
            if not premium_path.exists():
                compute_premiums()
            premiums = json.loads(premium_path.read_text())["premium"]
        for l in args.langs:
            if args.budget_from_premium:
                budget = EN_30B_BUDGET_BYTES * premiums[l]
            else:
                budget = (args.budget_gb * 1e9 if args.budget_gb is not None
                          else DEFAULT_BUDGET_BYTES)
            print(f"[extra-pool] {l}: target {budget/1e9:.2f}GB")
            build_pool_standalone(l, EXTRA_LANGS[l], budget)


if __name__ == "__main__":
    main()
