"""Taxi-1500 (Ma et al. 2023, arXiv:2305.08487) as a 0-shot cloze task.

Six topics over Parallel Bible Corpus verses -- Recommendation, Faith,
Description, Sin, Grace, Violence -- so chance is 1/6 = 0.167. It is the
backup to SIB-200 in this repo's five-language suite: it covers all five
languages, but its domain (scripture) is a much worse match for
FineWeb2-trained checkpoints than SIB-200's FLORES newswire, and its register
is uneven across our languages (see EDITIONS below). Read a Taxi-1500 number
as a corroboration of SIB-200, not as an independent measurement of the same
thing.

**Assembly.** The labelled dataset is not distributable: only the ENGLISH
labels are public (`eng_data/eng_{train,dev,test}.tsv` in the project's GitHub
repo, 1077 verses keyed by PBC verse id), and the per-language text has to be
joined onto them from a Bible corpus. The authors' own `data_preprocess.py`
does exactly this join and expects Parallel Bible Corpus files, which need an
access request. What IS freely downloadable is **Taxi1500-c v3.0**
(<https://cis.lmu.de/~yehao/data/Taxi1500-c_v3.0.zip>, 786 MB, 1384 Bible
editions), the openly-licensed subset of the same corpus in the same
`verse_id<TAB>text` format -- and it contains editions covering all 1077
labelled verses in every one of our five languages. `stage()` below downloads
it once, keeps only the five editions and the label files (a few MB), and
throws the archive away.

**Edition choice is a real methodological knob**, so it is pinned here rather
than picked at runtime. Requirements: full coverage of all 1077 labelled
verses, and the most modern register available (these checkpoints saw web
text, not early-modern prose). Note Arabic has exactly ONE full-coverage
edition in the open corpus and it is from 1865, so the Arabic column is in a
noticeably more archaic register than the others -- an asymmetry that
disfavours Arabic for reasons unrelated to the model. This is not fixable
without PBC access, and is the single strongest reason to treat SIB-200 as
primary.

**Splits.** Nothing is finetuned on Taxi-1500 anywhere in this repo, so the
train/dev/test distinction carries no meaning for these checkpoints; all 1077
labelled verses are merged into one `test` split and sorted by verse id, which
also makes the doc order identical across the five languages.
"""
import csv
import io
import json
import os
import zipfile
from pathlib import Path

import datasets

CORPUS_URL = "https://cis.lmu.de/~yehao/data/Taxi1500-c_v3.0.zip"
LABELS_URL = ("https://raw.githubusercontent.com/cisnlp/Taxi1500/main/"
              "eng_data/eng_{split}.tsv")
LABEL_SPLITS = ("train", "dev", "test")

# Topic order is the paper's (Table 1 of arXiv:2305.08487).
CATEGORIES = ["Recommendation", "Faith", "Description", "Sin", "Grace", "Violence"]

# One pinned edition per language: full coverage of all 1077 labelled verses,
# most modern register available. See the module docstring on Arabic.
EDITIONS = {
    "eng_Latn": "eng_engbsb.ebible.txt",       # Berean Study Bible, 2023
    "deu_Latn": "deu_deu1951.ebible.txt",      # Schlachter 1951
    "fra_Latn": "fra_francl.ebible.txt",       # néo-Crampon Libre (modernized)
    "arb_Arab": "arb-x-bible.txt",             # 1865 -- the ONLY full-coverage Arabic
    "zho_Hans": "cmn_cmn-cu89s.ebible.txt",    # Chinese Union Version, simplified
}

LABEL_WORDS = {
    "eng_Latn": ["Recommendation", "Faith", "Description", "Sin", "Grace", "Violence"],
    "deu_Latn": ["Empfehlung", "Glaube", "Beschreibung", "Sünde", "Gnade", "Gewalt"],
    "fra_Latn": ["recommandation", "foi", "description", "péché", "grâce", "violence"],
    "arb_Arab": ["توصية", "إيمان", "وصف", "خطيئة", "نعمة", "عنف"],
    "zho_Hans": ["劝勉", "信仰", "描述", "罪", "恩典", "暴力"],
}


def cache_dir() -> Path:
    """Where the staged few-MB extract lives.

    Under `XSCRIPT_SCRATCH` when the eval runners set it, so a fan-out shares
    one copy instead of every process pulling 786 MB.
    """
    root = os.environ.get("XSCRIPT_SCRATCH")
    base = Path(root) if root else Path.home() / ".cache" / "xscript"
    return base / "taxi1500"


def stage(force: bool = False) -> Path:
    """Download + extract the five editions and the English labels. Idempotent.

    Safe to call from several processes at once: each writes to a private tmp
    path and `os.replace`s into place, so a concurrent reader sees either the
    old file or the complete new one, never a partial.
    """
    out = cache_dir()
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(EDITIONS.values())
    have = {f.name for f in out.glob("*")}
    need_labels = [s for s in LABEL_SPLITS if force or f"eng_{s}.tsv" not in have]
    need_corpus = force or not wanted <= have

    import urllib.request

    for split in need_labels:
        url = LABELS_URL.format(split=split)
        with urllib.request.urlopen(url) as r:
            data = r.read()
        tmp = out / f"eng_{split}.tsv.tmp.{os.getpid()}"
        tmp.write_bytes(data)
        os.replace(tmp, out / f"eng_{split}.tsv")

    if need_corpus:
        print(f"[taxi1500] downloading {CORPUS_URL} (786 MB, once) ...", flush=True)
        with urllib.request.urlopen(CORPUS_URL) as r:
            blob = r.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            members = {Path(n).name: n for n in z.namelist()}
            missing = wanted - set(members)
            if missing:
                raise RuntimeError(f"editions absent from the corpus: {sorted(missing)}")
            for name in wanted:
                tmp = out / f"{name}.tmp.{os.getpid()}"
                tmp.write_bytes(z.read(members[name]))
                os.replace(tmp, out / name)
        del blob
        print(f"[taxi1500] staged {len(wanted)} editions in {out}", flush=True)
    return out


def _read_labels(root: Path) -> dict[str, str]:
    """verse id -> topic, over all three label files (see docstring on splits)."""
    labels = {}
    for split in LABEL_SPLITS:
        with open(root / f"eng_{split}.tsv", encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) < 3:
                    continue
                vid, topic = row[0].strip(), row[1].strip()
                if topic not in CATEGORIES:
                    raise ValueError(f"unknown topic {topic!r} for verse {vid}")
                labels[vid] = topic
    return labels


def _read_edition(path: Path) -> dict[str, str]:
    """verse id -> verse text. PBC/eBible format is `id<TAB>text` after a
    variable-length `#`-comment header (the authors' script hardcodes 11 header
    lines; the open corpus's headers are not all 11 lines long, so filter on the
    comment marker instead)."""
    verses = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2 and parts[1].strip():
                verses[parts[0].strip()] = parts[1].strip()
    return verses


def build_dataset(lang: str, **kwargs) -> datasets.DatasetDict:
    root = stage()
    labels = _read_labels(root)
    verses = _read_edition(root / EDITIONS[lang])
    missing = sorted(set(labels) - set(verses))
    if missing:
        raise RuntimeError(
            f"{lang}: edition {EDITIONS[lang]} is missing {len(missing)} of "
            f"{len(labels)} labelled verses (first: {missing[:3]}). EDITIONS "
            "must only list full-coverage editions -- see the module docstring.")
    rows = [{"verse_id": vid, "topic": labels[vid], "text": verses[vid],
             "label": CATEGORIES.index(labels[vid])}
            for vid in sorted(labels)]
    return datasets.DatasetDict({"test": datasets.Dataset.from_list(rows)})


def _choices(lang: str):
    def fn(doc):
        return LABEL_WORDS[lang]
    return fn


choices_eng_Latn = _choices("eng_Latn")
choices_deu_Latn = _choices("deu_Latn")
choices_fra_Latn = _choices("fra_Latn")
choices_arb_Arab = _choices("arb_Arab")
choices_zho_Hans = _choices("zho_Hans")


if __name__ == "__main__":
    # `python -m xscript.eval.c5_tasks.taxi1500.utils` pre-warms the cache, so a
    # fan-out does not have N processes racing on the same 786 MB download.
    root = stage()
    labels = _read_labels(root)
    print(json.dumps({"cache": str(root), "labelled_verses": len(labels),
                      "editions": EDITIONS}, indent=2))
