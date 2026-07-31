"""MMMLU (openai/MMMLU) as a 0-shot cloze task, parallel with English MMLU.

Why this exists alongside `gmmlu_probe/`: Messmer et al. 2025 (arXiv:2502.10361),
the paper CLAUDE.md 6d replicates, evaluates non-English MMLU on **MMMLU**,
which is translated by *professional translators*. `CohereForAI/Global-MMLU`
is largely machine-translated. Both are renderings of the same 14,042 MMLU
test items, so they are directly comparable, and the difference isolates
translation quality -- which is the one prompt-side explanation for Arabic
sitting at chance that has NOT been ruled out (the English-cue confound was
tested and came back at exactly 0.000, see CLAUDE.md 6e).

Format follows `gmmlu_probe`: CLOZE (the answer text is the scored
continuation), not the A/B/C/D letter format lm-eval's registered
`mmmlu_*` tasks use -- our checkpoints are at chance on letters and stay
there after calibration, so letters measure nothing here. The answer cue is
localized, which is correct even though it was measured not to matter.

English has no MMMLU config (MMMLU *is* the translation of English MMLU), so
`mmmlu_cloze_en` reads `cais/mmlu` and the five tasks are renderings of one
item set.

Documents are ordered round-robin over subjects so a `--limit N` prefix stays
stratified across all 57 subjects -- see gmmlu_probe/utils.py for why plain
sorting silently truncates the subject mix.
"""
import datasets

LETTERS = ["A", "B", "C", "D"]

# openai/MMMLU config per language; English falls back to cais/mmlu.
MMMLU_CONFIG = {"de": "DE_DE", "fr": "FR_FR", "ar": "AR_XY", "zh": "ZH_CN"}

ANSWER_CUE = {
    "en": "Answer:", "de": "Antwort:", "fr": "Réponse :",
    "ar": "الإجابة:", "zh": "答案：",
}


def _rows_en():
    ds = datasets.load_dataset("cais/mmlu", "all", split="test")
    out = []
    for i, d in enumerate(ds):
        opts = [str(o).strip() for o in d["choices"]]
        if len(opts) != 4 or any(not o for o in opts):
            continue
        out.append({"subject": d["subject"], "idx": i, "question": d["question"].strip(),
                    "options": opts, "label": int(d["answer"])})
    return out


def _rows_mmmlu(lang):
    ds = datasets.load_dataset("openai/MMMLU", MMMLU_CONFIG[lang], split="test")
    out = []
    for i, d in enumerate(ds):
        ans = str(d["Answer"]).strip().upper()
        if ans not in LETTERS:
            continue
        opts = [str(d[k]).strip() for k in LETTERS]
        if any(not o for o in opts):
            continue
        out.append({"subject": str(d["Subject"]), "idx": i,
                    "question": str(d["Question"]).strip(),
                    "options": opts, "label": LETTERS.index(ans)})
    return out


def _load(lang: str) -> datasets.Dataset:
    rows = _rows_en() if lang == "en" else _rows_mmmlu(lang)
    rows.sort(key=lambda r: (r["subject"], r["idx"]))
    by = {}
    for r in rows:
        by.setdefault(r["subject"], []).append(r)
    order, subjects = [], sorted(by)
    for i in range(max(len(v) for v in by.values())):
        for s in subjects:
            if i < len(by[s]):
                order.append(by[s][i])
    return datasets.Dataset.from_list(order)


def build(lang: str):
    def fn(**kwargs) -> datasets.DatasetDict:
        return datasets.DatasetDict({"test": _load(lang)})
    return fn


build_en, build_de, build_fr = build("en"), build("de"), build("fr")
build_ar, build_zh = build("ar"), build("zh")


def _text(lang: str):
    cue = ANSWER_CUE[lang]

    def fn(doc) -> str:
        return f"{doc['question']}\n{cue}"
    return fn


text_en, text_de, text_fr = _text("en"), _text("de"), _text("fr")
text_ar, text_zh = _text("ar"), _text("zh")


def choices(doc):
    return doc["options"]
