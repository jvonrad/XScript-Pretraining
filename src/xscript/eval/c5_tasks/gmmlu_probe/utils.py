"""Global-MMLU as a single full-size task, in BOTH answer formats.

CLAUDE.md 6 concludes Global-MMLU is genuinely at chance ("letter, cloze, and
cloze+PMI scoring all stay ~0.21-0.23 for en"), but that was measured with
`--limit 200`. At n=200 the smallest effect detectable at 2 sigma is +6.1
accuracy points, so the experiment could only ever have ruled out a LARGE
effect; a 1B model at 30B tokens would plausibly sit +2..+4 above chance and
be invisible. `CohereForAI/Global-MMLU` has n=14042 for en (the `-Lite` build
lm-eval defaults to has 400), which takes the 2-sigma threshold to +0.7
points.

Two tasks per language, from the same documents in the same order:

  gmmlu_letter_{lang}  lm-eval's registered format: options are listed in the
                       prompt and the four candidates are "A"/"B"/"C"/"D".
                       The candidate set is SHARED across documents, so
                       `acc_cal` applies -- but note what it corrects. The
                       letter is a POINTER, not content: "A" denotes a
                       different answer in every question. So calibration here
                       removes the model's selection bias (a fixed preference
                       for emitting one option id), NOT a label-prior on
                       meaning as it does for SIB-200. That is legitimate only
                       because the gold letter is near-uniform (A/B/C/D =
                       94/108/97/101 in Lite), so a letter preference is pure
                       noise. It cannot manufacture knowledge: with no signal,
                       the per-document deviation from the letter mean is
                       noise and calibrated accuracy stays at 0.25.

  gmmlu_cloze_{lang}   the answer TEXT is the scored continuation. This is the
                       format that actually tests world knowledge, since the
                       model never has to bind "the third option" to a letter
                       token. Choices are per-document, so calibration does
                       NOT apply (`has_shared_choices` returns False) and
                       `acc_norm` is the correct length correction.

Read the pair together: letter-vs-cloze separates "cannot do the MCQ format"
from "does not know the answer", and `acc - null` in the degeneracy report
separates both from "always says B".
"""
import datasets

LETTERS = ["A", "B", "C", "D"]


def _load(lang: str) -> datasets.Dataset:
    ds = datasets.load_dataset("CohereForAI/Global-MMLU", lang, split="test")
    rows = []
    for d in ds:
        ans = str(d["answer"]).strip().upper()
        if ans not in LETTERS:
            continue
        opts = [d["option_a"], d["option_b"], d["option_c"], d["option_d"]]
        if any(o is None or not str(o).strip() for o in opts):
            continue
        rows.append({
            "sample_id": str(d["sample_id"]),
            "question": str(d["question"]).strip(),
            "options": [str(o).strip() for o in opts],
            "label": LETTERS.index(ans),
        })
    # Stable order across languages and across runs -- but ROUND-ROBIN over
    # subjects, not plain sample_id order. `sample_id` is "<subject>/test/<n>",
    # so sorting by it groups all 57 subjects into contiguous blocks and a
    # `--limit N` prefix would silently cover only the alphabetically-early
    # ones (25 of 57 at N=4000). Interleaving makes any prefix a stratified
    # sample, so `--limit` trades power for time without biasing the subject
    # mix. Order is still deterministic and identical across languages
    # (Global-MMLU is parallel: same sample_ids in every language).
    rows.sort(key=lambda r: r["sample_id"])
    by_subject = {}
    for r in rows:
        by_subject.setdefault(r["sample_id"].split("/")[0], []).append(r)
    order, subjects = [], sorted(by_subject)
    for i in range(max(len(v) for v in by_subject.values())):
        for s in subjects:
            if i < len(by_subject[s]):
                order.append(by_subject[s][i])
    return datasets.Dataset.from_list(order)


def build(lang: str):
    def fn(**kwargs) -> datasets.DatasetDict:
        return datasets.DatasetDict({"test": _load(lang)})
    return fn


build_en, build_de = build("en"), build("de")
build_fr, build_ar = build("fr"), build("ar")
build_zh = build("zh")


def letter_text(doc) -> str:
    o = doc["options"]
    return (f"{doc['question']}\n"
            f"A. {o[0]}\nB. {o[1]}\nC. {o[2]}\nD. {o[3]}\nAnswer:")


def letter_choice(doc):
    return LETTERS


# Localized answer cue, one per language. The cue MUST be in the document's
# language: CLAUDE.md 6d measured the label-language effect on SIB-200 at ~14
# accuracy points, and an English "Answer:" after an Arabic question is exactly
# that confound. The first version of this file hardcoded "Answer:" for all
# five languages, which put every non-English cell at chance -- Arabic sat at
# +0.002..+0.016 over its own null at every token budget under both
# tokenizers, which reads as "the model knows nothing" but was measuring the
# prompt. `cloze_text` is the localized (primary) form; `cloze_text_en` keeps
# the English cue as an explicit control, the same primary/control split
# `sib200_*` vs `sib200_enlab_*` already uses.
ANSWER_CUE = {
    "en": "Answer:",
    "de": "Antwort:",
    "fr": "Réponse :",
    "ar": "الإجابة:",
    "zh": "答案：",
}


def _cloze_text(lang: str):
    cue = ANSWER_CUE[lang]

    def fn(doc) -> str:
        return f"{doc['question']}\n{cue}"
    return fn


cloze_text_en_cue = _cloze_text("en")
cloze_text_de = _cloze_text("de")
cloze_text_fr = _cloze_text("fr")
cloze_text_ar = _cloze_text("ar")
cloze_text_zh = _cloze_text("zh")
cloze_text_en = _cloze_text("en")


def cloze_text(doc) -> str:
    """Back-compat English-cue form; the per-language hooks above are what the
    yamls reference."""
    return f"{doc['question']}\nAnswer:"


def cloze_choice(doc):
    return doc["options"]
