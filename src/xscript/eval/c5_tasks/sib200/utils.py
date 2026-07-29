"""SIB-200 topic classification (Adelani et al. 2024, arXiv:2309.07445) as a
0-shot cloze task for base LMs.

SIB-200 annotates FLORES-200 sentences with one of 7 topics
(science/technology, travel, politics, sports, health, entertainment,
geography), so chance is 1/7 = 0.143. Three properties make it the right
addition to this repo's suite:

* it covers all five of our languages (it is FLORES-derived, same language set
  as `belebele_cloze_*` and the alignment/BPB pools);
* it is *lexical* topic classification, not multi-step reasoning -- the kind of
  task a 1B/30B model can actually do, unlike Global-MMLU/ARC (CLAUDE.md 6);
* it is scored on the same FLORES sentences 6b's alignment and 6's BPB use,
  so downstream capability and representation alignment are measured on
  identical text.

Loading: `Davlan/sib200` ships one directory of TSVs per language
(`data/<code>/{train,dev,test}.tsv`, columns `index_id`, `category`, `text`).
All three splits are merged into a single `test` split here -- 1004 sentences
per language instead of the 204 in `test.tsv` alone. Nothing is finetuned on
SIB-200 anywhere in this repo, so every split is unseen data for these
checkpoints and the merge is purely a power gain (SE on an accuracy near
chance drops from ~2.4 to ~1.1 points). Docs are sorted by `index_id`, which
is the FLORES sentence id, so **the doc order is identical across all five
languages** -- per-example hit lists line up language-to-language, not just
model-to-model.

The label set is English in the source data for every language. That is a
confound for exactly the comparison this repo cares about (a zh-only model
being asked to rank English words), so each language gets two tasks:

    sib200_<code>        localized prompt + localized label words  (primary)
    sib200_enlab_<code>  localized text, ENGLISH prompt and labels (control)

Reporting both is the point: the gap between them separates "can the model
classify this text" from "can the model read the English label words".
"""
import csv

import datasets

# Label strings exactly as they appear in `category`, in the canonical order
# from the dataset's own labels.txt (identical for every language).
CATEGORIES = [
    "science/technology",
    "travel",
    "politics",
    "sports",
    "health",
    "entertainment",
    "geography",
]

# Localized surface forms, same order as CATEGORIES. Kept here (not in the
# yamls) so the ordering can never drift out of sync with CATEGORIES.
LABELS = {
    "eng_Latn": ["science and technology", "travel", "politics", "sports",
                 "health", "entertainment", "geography"],
    "deu_Latn": ["Wissenschaft und Technik", "Reisen", "Politik", "Sport",
                 "Gesundheit", "Unterhaltung", "Geografie"],
    "fra_Latn": ["science et technologie", "voyage", "politique", "sport",
                 "santé", "divertissement", "géographie"],
    "arb_Arab": ["العلوم والتكنولوجيا", "السفر", "السياسة", "الرياضة",
                 "الصحة", "الترفيه", "الجغرافيا"],
    "zho_Hans": ["科学技术", "旅游", "政治", "体育", "健康", "娱乐", "地理"],
}

SPLITS = ("train", "dev", "test")


def build_dataset(lang: str, **kwargs) -> datasets.DatasetDict:
    """Merge `Davlan/sib200`'s train+dev+test TSVs into one `test` split.

    Parsed with the stdlib csv reader at QUOTE_NONE: the sentences contain bare
    double quotes (they are newswire prose), which pandas' default quoting --
    what `datasets`' csv builder would use -- silently swallows, merging fields
    and dropping rows.
    """
    from huggingface_hub import hf_hub_download

    rows = []
    for split in SPLITS:
        path = hf_hub_download("Davlan/sib200", f"data/{lang}/{split}.tsv",
                               repo_type="dataset")
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if row["category"] not in CATEGORIES:
                    raise ValueError(f"{lang}/{split}: unknown category "
                                     f"{row['category']!r}")
                rows.append({
                    "index_id": int(row["index_id"]),
                    "category": row["category"],
                    "text": row["text"].strip(),
                    "label": CATEGORIES.index(row["category"]),
                })
    # FLORES sentence id order -> identical doc order in every language.
    rows.sort(key=lambda r: r["index_id"])
    ids = [r["index_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{lang}: duplicate index_id across splits")
    return datasets.DatasetDict({"test": datasets.Dataset.from_list(rows)})


def _choices(lang: str):
    def fn(doc):
        return LABELS[lang]
    return fn


# doc_to_choice hooks -- one per language, referenced from the yamls. The
# English-label control tasks all point at `choices_eng_Latn`.
choices_eng_Latn = _choices("eng_Latn")
choices_deu_Latn = _choices("deu_Latn")
choices_fra_Latn = _choices("fra_Latn")
choices_arb_Arab = _choices("arb_Arab")
choices_zho_Hans = _choices("zho_Hans")
