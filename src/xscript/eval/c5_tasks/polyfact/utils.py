"""PolyFact (jvonrad/PolyFact) as a 0-shot CLOZE task for base LMs.

PolyFact is a parallel multilingual multiple-choice QA set over Wikidata
triples: 12 languages, 2039 test facts, four candidate entities per fact. It
covers all five of this project's languages, which makes it the second
all-five-language *knowledge* instrument after the native MMLU exams -- and
unlike translated MMLU it is not Anglocentric by construction, because every
item is a Wikidata triple rendered natively in each language rather than an
English exam question pushed through MT.

Verified before use (CLAUDE.md 6e's standing rule -- this is the check that
caught arc_easy(en)-vs-ARC-Challenge(others)):

  * 2039 rows in every one of en/de/fr/ar/zh;
  * `fact_id` sequence IDENTICAL across all five, so the doc order lines up
    language-to-language and per-example hit lists are directly comparable;
  * the GOLD ENTITY (Wikidata QID) is the same item in every language --
    0 mismatches;
  * option ID *sets* align across languages (0 mismatches) even though the
    option *positions* are independently shuffled per language, which the
    dataset card warns about explicitly;
  * 0 items with a duplicate option, 0 out-of-range gold indices;
  * gold position is ~uniform (.24-.28 per slot in every language), so there
    is no position prior for a letter-format task to exploit.

`jvonrad/PolyFact` and `jvonrad/PolyFact-Clean` were compared and are
BYTE-IDENTICAL on the test split for all five languages; this uses the former.

CLOZE, not A/B/C/D. CLAUDE.md 6e established that these checkpoints hold
measurable world knowledge but "cannot do the A/B/C/D indirection at all"
(Global-MMLU letter format sits at 0.2499 against a 0.248 null and STAYS there
after calibration, while the cloze format of the same items clears its null by
15 sigma). Scoring the entity's own name as the continuation is also the
Appendix-D methodology the rest of this suite uses.

Two tasks per language, per the practice CLAUDE.md 6e mandates after the sixth
format artifact (the first version of `gmmlu_probe` hardcoded an English
"Answer:" for all five languages):

    polyfact_<lang>        localized question + localized answer cue  (primary)
    polyfact_encue_<lang>  localized question + ENGLISH "Answer:"     (control)

6e measured the cue effect at exactly 0.000 on Arabic Global-MMLU, so the
control is expected to be a no-op -- it exists so that expectation is
falsifiable from stored data rather than assumed.

NOTE ON ESTIMATORS. `acc_cal` must NOT be used here and this task is
deliberately absent from `rawscores.SHARED_CHOICE_TASKS`: the four candidates
are document-specific entity names, so choice index `c` does not denote the
same thing in any two documents. `_calibrate` would still return a number, but
it would be a position-bias correction misread as a label-prior fix -- exactly
the error 6e documents. Candidates are short entity names, so 6g's structural
rule predicts plain `acc` over `acc_norm`; both are emitted, plus PMI, and the
choice is made on over-null / entropy as 6g does rather than assumed.
"""
import datasets

REPO = "jvonrad/PolyFact"
LANGS = ("en", "de", "fr", "ar", "zh")

# Localized answer cue. Same table as gmmlu_probe/utils.py -- kept identical on
# purpose so a PolyFact number and a Global-MMLU number differ by the benchmark
# and not by the scaffolding around it.
CUE = {
    "en": "Answer:",
    "de": "Antwort:",
    "fr": "Réponse :",
    "ar": "الإجابة:",
    "zh": "答案：",
}


def build_dataset(lang: str = "en", **kwargs) -> datasets.DatasetDict:
    """Test split for one language, in the dataset's own fact_id order.

    Order is NOT re-sorted: the fact_id sequence was verified identical across
    all five languages as shipped, so leaving it alone is what keeps the doc
    order aligned language-to-language.
    """
    ds = datasets.load_dataset(REPO, lang, split="test")

    def _prep(doc):
        return {
            "question": doc["question"],
            "choices": [doc["option_a"], doc["option_b"],
                        doc["option_c"], doc["option_d"]],
            "label": int(doc["answer_index"]),
        }

    ds = ds.map(_prep, remove_columns=[c for c in ds.column_names
                                       if c not in ("question",)])
    return datasets.DatasetDict({"test": ds})


def _mk(lang):
    def build(**kwargs):
        return build_dataset(lang=lang, **kwargs)
    return build


build_en = _mk("en")
build_de = _mk("de")
build_fr = _mk("fr")
build_ar = _mk("ar")
build_zh = _mk("zh")


def _doc_to_text(cue):
    def f(doc):
        return f"{doc['question']}\n{cue}"
    return f


doc_to_text_en = _doc_to_text(CUE["en"])
doc_to_text_de = _doc_to_text(CUE["de"])
doc_to_text_fr = _doc_to_text(CUE["fr"])
doc_to_text_ar = _doc_to_text(CUE["ar"])
doc_to_text_zh = _doc_to_text(CUE["zh"])
doc_to_text_encue = _doc_to_text(CUE["en"])   # control: English cue everywhere


def doc_to_choice(doc):
    return doc["choices"]
