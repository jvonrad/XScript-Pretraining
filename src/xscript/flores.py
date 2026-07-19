"""FLORES+ (openlanguagedata/flores_plus) loading.

FLORES+ is gated ("auto"): accept the terms on Hugging Face once and export
HF_TOKEN. Files are per-language jsonl: dev/<code>.jsonl, devtest/<code>.jsonl.

We key sentences by their FLORES `id` and align across languages on the
intersection of ids, so byte premiums and retrieval eval always compare the
same parallel content.
"""
import json
from pathlib import Path

from .langs import LANGS
from .paths import FLORES_DIR, ensure

REPO_ID = "openlanguagedata/flores_plus"
SPLITS = ("dev", "devtest")

_TEXT_KEYS = ("text", "sentence")
_ID_KEYS = ("id", "sentence_id")


def download(langs=None, splits=SPLITS, token=None) -> None:
    from huggingface_hub import hf_hub_download
    langs = langs or list(LANGS)
    ensure(FLORES_DIR)
    for lc in langs:
        code = LANGS[lc].flores_code
        for split in splits:
            hf_hub_download(
                repo_id=REPO_ID, repo_type="dataset",
                filename=f"{split}/{code}.jsonl",
                local_dir=FLORES_DIR, token=token,
            )


def _pick(d: dict, keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"none of {keys} in FLORES+ record with keys {sorted(d)}")


def load(lang: str, split: str = "dev") -> dict[int, str]:
    """Return {sentence_id: text} for one language/split."""
    path = FLORES_DIR / split / f"{LANGS[lang].flores_code}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run `xscript flores-download` (requires HF_TOKEN "
            f"with accepted terms for {REPO_ID})")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[int(_pick(rec, _ID_KEYS))] = str(_pick(rec, _TEXT_KEYS))
    return out


def load_parallel(langs, split: str = "dev") -> dict[str, list[str]]:
    """Aligned parallel sentences: same order, intersection of ids."""
    per_lang = {l: load(l, split) for l in langs}
    common = sorted(set.intersection(*(set(v) for v in per_lang.values())))
    if not common:
        raise RuntimeError(f"no common sentence ids across {langs} ({split})")
    return {l: [per_lang[l][i] for i in common] for l in langs}
