"""X-CSQA (multilingual CommonsenseQA) as a 0-shot CLOZE task, aligned across
our five languages.

X-CSQA is the CommonsenseQA half of **XCSR** (Lin et al. 2021,
arXiv:2106.06937, "Common Sense Beyond English"): CommonsenseQA re-partitioned
into a 1,000-item dev and a 1,074-item test set and translated into 16
languages, of which en/de/fr/ar/zh are ours. Five options per question, so
nominal chance is 0.200.

Why it is worth adding to a suite that already has six MuBench families: it is
the only **commonsense-reasoning** benchmark here whose items are questions
rather than sentence completions. HellaSwag and StoryCloze test narrative
plausibility (pick the continuation that reads naturally), which a language
model can do largely on fluency; X-CSQA asks a question whose answer is not in
the stem and whose distractors are deliberately same-category
("hospital"/"town"/"schools"/"office building"), so fluency alone does not
rank them. ARC-Easy is the nearest neighbour but is school science, i.e.
knowledge rather than commonsense.

FIVE THINGS VERIFIED BEFORE USE (scripts/external_bench/verify_xcsqa.py), per
CLAUDE.md 6e's standing rule that a benchmark can be "the same benchmark" in
name only:

1. **Item counts are equal**: 1000 (validation) and 1074 (test) in all five
   languages -- the cheapest identity check, and the one that caught
   ARC-Easy(en)-vs-ARC-Challenge(others).
2. **`id` sets are identical across all five languages** -- so this is ONE
   item set rendered five ways, like MuBench and unlike okapi.
3. **The `test` split is BLIND** (`answerKey == ""` in all 1074 rows, every
   language -- it is a leaderboard split). Only `validation` is usable, and it
   is exposed below as lm-eval's `test` split. Scoring the real test split
   would silently produce garbage, not an error.
4. **Options are PERMUTED per language** and `answerKey` tracks the
   permutation correctly (verified by spot-check: en `hospital`=C, de
   `Krankenhaus`=A, zh `医院`=D are the same item). So the gold *letter*
   disagrees across languages on ~80% of items while the gold *meaning*
   agrees. Harmless for cloze -- we score the option TEXT -- but it means a
   letter-format rendering of this dataset would have a different gold
   distribution per language, and per-example hit lists only line up after
   sorting by `id`, which `_load` does.
5. **German has six rows carrying an empty option string** (see `_drop_ids`),
   which is a silent scoring defect, not a cosmetic one -- details there.

ESTIMATOR. The candidates are per-document text ("hospital", "wisconsin"), not
a choice set shared across documents, so `acc_cal` does NOT apply and X-CSQA
is deliberately absent from `rawscores.SHARED_CHOICE_TASKS` -- calibrating it
would silently return a position-bias correction (CLAUDE.md 6e). By CLAUDE.md
6g's structural rule the candidates are *short fixed phrases* (~10 chars, the
ARC-Easy/BMLAMA regime), so **`acc` is expected to beat `acc_norm`** here --
but that is a prediction to check against the raw sidecars, not a setting.
`acc_mutual_info` is therefore in the metric list: X-CSQA distractors are bare
nouns with wildly unequal unigram priors ("michigan" vs "hospital"), so the
prior-normalized estimator is the principled correction, and issuing the
unconditional requests now is what makes `acc_pmi` derivable on CPU forever
instead of costing another accelerator pass.
"""
import json

import datasets

LANGS = ("en", "de", "fr", "ar", "zh")

# Localized answer cue. Identical table to c5_tasks/gmmlu_probe/utils.py --
# kept in sync deliberately, because CLAUDE.md 6e records that the FIRST
# version of gmmlu_probe hardcoded English "Answer:" for all five languages
# and called it this project's sixth format artifact, introduced while fixing
# the fifth. Measured effect of the cue language on gmmlu was exactly 0.000,
# but the control is shipped anyway (xcsqa_encue_*) rather than assumed.
ANSWER_CUE = {
    "en": "Answer:",
    "de": "Antwort:",
    "fr": "Réponse :",
    "ar": "الإجابة:",
    "zh": "答案：",
}

_CACHE: dict[str, list[dict]] = {}
_DROP: set[str] | None = None


def _raw(lang: str) -> list[dict]:
    """Read one language's dev split straight from the repo's parquet.

    `INK-USC/xcsr` ships plain parquet per config, so this reads the file
    directly rather than going through `load_dataset`'s builder -- the same
    choice c5_tasks/mubench/utils.py makes, and it keeps the loader immune to
    dataset-script deprecations across `datasets` versions.

    NOTE the file named `validation` is the only labelled split; see the
    module docstring, point 3.
    """
    if lang in _CACHE:
        return _CACHE[lang]
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("INK-USC/xcsr",
                           f"X-CSQA-{lang}/validation-00000-of-00001.parquet",
                           repo_type="dataset")
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    out = []
    for r in rows:
        q = r["question"]
        out.append({
            "id": str(r["id"]),
            "stem": (q["stem"] or "").strip(),
            "labels": list(q["choices"]["label"]),
            "options": [(t or "").strip() for t in q["choices"]["text"]],
            "answerKey": (r["answerKey"] or "").strip(),
        })
    _CACHE[lang] = out
    return out


def _drop_ids() -> set[str]:
    """Ids to exclude, computed over ALL FIVE languages at once.

    German carries six rows in which an option that has no idiomatic
    translation ("badarse", "getting high", "plethora", "maundering", ...) was
    left as the EMPTY STRING. This is the same class of silent data defect as
    the four malformed zh HellaSwag `endings` in CLAUDE.md 6d, and it is worse
    than it looks, for two independent reasons:

    * An empty continuation is scored over ZERO tokens, so its summed
      loglikelihood is exactly 0.0 while every real candidate is negative.
      Under `acc` the empty option therefore wins on EVERY document that has
      one -- it is not noise, it is a guaranteed wrong answer. Under
      `acc_norm` it is 0/0.
    * On one of the six (`a1a1ab3b47e42234`) the empty string IS the gold, so
      that row is unanswerable in German at all.

    Dropping them only from German would break the property this whole file
    exists to provide -- that every language renders ONE item set -- so the
    ids are dropped from ALL five languages, leaving n=994 x 5. The scan is
    general (any language, blank option, blank stem, or unusable gold) rather
    than a hardcoded id list, so a future re-release of the data cannot
    silently reintroduce the defect.
    """
    global _DROP
    if _DROP is not None:
        return _DROP
    drop: set[str] = set()
    for lang in LANGS:
        for r in _raw(lang):
            if not r["stem"] or any(not o for o in r["options"]):
                drop.add(r["id"])
            elif r["answerKey"] not in r["labels"]:
                drop.add(r["id"])
            elif len(r["options"]) != len(r["labels"]) or len(r["options"]) < 2:
                drop.add(r["id"])
    _DROP = drop
    return drop


def _rows(lang: str, en_options: bool = False,
          cue_lang: str | None = None) -> list[dict]:
    """Build one language's documents as plain dicts.

    Kept separate from `_load` so `verify_xcsqa.py` can check exactly the rows
    that will be scored without constructing a `datasets.Dataset` -- the same
    pure-stdlib-analysis split the rest of this repo uses (`analyze_*.py`,
    `verify_mubench.py`), and it keeps the gate runnable anywhere.

    Docs are sorted by `id`, which makes the doc order IDENTICAL across all
    five languages (the raw files are not in a common order -- verified), so
    per-example hit lists line up language-to-language and not merely
    model-to-model.

    `en_options=True` builds the English-option control: the localized stem
    with the ENGLISH option texts and the English gold. This is the X-CSQA
    analogue of `sib200_enlab_*`, which measured ~14 accuracy points in
    CLAUDE.md 6d -- by far the largest prompt-side effect found in this
    project, and the one that made a cross-language table wrong in both
    directions at once. Note it takes the English options in ENGLISH order
    together with the English gold, rather than trying to re-map the
    per-language permutation onto the localized option order: the two are
    self-consistent, and there is no field in the data from which the
    permutation could be recovered reliably.
    """
    drop = _drop_ids()
    cue = ANSWER_CUE[cue_lang or lang]
    src = {r["id"]: r for r in _raw(lang)}
    ref = {r["id"]: r for r in _raw("en")} if en_options else src
    out = []
    for _id in sorted(src):
        if _id in drop:
            continue
        loc, opt = src[_id], ref[_id]
        out.append({
            "_id": _id,
            # The cue is appended here rather than in `doc_to_text` or via
            # `Dataset.map`: map() would drag the whole row through datasets'
            # dill-based fingerprinting for no benefit, and putting it in the
            # yaml would split the prompt definition across two files.
            "stem": f"{loc['stem']}\n{cue}",
            "options": list(opt["options"]),
            "label": opt["labels"].index(opt["answerKey"]),
        })
    return out


def _load(lang: str, en_options: bool = False,
          cue_lang: str | None = None) -> datasets.Dataset:
    return datasets.Dataset.from_list(
        _rows(lang, en_options=en_options, cue_lang=cue_lang))


def build(lang: str, en_options: bool = False, cue_lang: str | None = None):
    def fn(**kwargs):
        return datasets.DatasetDict(
            {"test": _load(lang, en_options=en_options, cue_lang=cue_lang)})
    return fn


def text(doc):
    return doc["stem"]


def choices(doc):
    return doc["options"]


# Generated hooks.
#   build_<lang>        localized stem + localized options + localized cue
#   build_enopt_<lang>  localized stem + ENGLISH options   + localized cue
#   build_encue_<lang>  localized stem + localized options + ENGLISH cue
# There is no `enopt`/`encue` hook for English: both would be byte-identical to
# the primary task, exactly as sib200 has no `sib200_enlab_eng_Latn`.
for _l in LANGS:
    globals()[f"build_{_l}"] = build(_l)
    if _l != "en":
        globals()[f"build_enopt_{_l}"] = build(_l, en_options=True)
        globals()[f"build_encue_{_l}"] = build(_l, cue_lang="en")
