"""NATIVE-language knowledge exams: ArabicMMLU (ar) and CMMLU (zh), 0-shot cloze.

Why this exists. Translated MMLU asks whether a model knows *Anglocentric*
facts rendered in another language -- US law, US medical licensing, US history.
Our Arabic checkpoints sit at chance there on BOTH the machine-translated
Global-MMLU and the professionally-translated MMMLU (CLAUDE.md 6e), which
rules out translation quality but not the possibility that the *content* is
simply absent from Arabic web text.

Messmer et al. 2025 -- the paper 6d replicates -- do not use translated MMLU
for non-English either: they evaluate zh/fr/ar with the **FineTasks** suite,
which selects NATIVE benchmarks per language. That is why their per-language
panels are not comparable with each other, and why comparing our translated-
MMLU numbers against them was invalid.

So this measures the thing translated MMLU cannot: whether the model knows
facts that Arabic and Chinese web text actually contains.

    nativemmlu_cloze_ar   MBZUAI/ArabicMMLU, n=14455, Arabic school/university
                          exams (Arabic-world curricula, not translations).
                          Option count VARIES 2-5 per item, so per-item chance
                          varies -- read `acc - null` from the degeneracy
                          report, never a fixed 1/k.
    nativemmlu_cloze_zh   haonan-li/cmmlu, native Chinese exams.

Both are cloze with a localized cue, matching gmmlu_probe/mmmlu_probe so the
three are directly comparable. lm-eval's registered `arabicmmlu` wraps the
Arabic question in an ENGLISH template ("This is a {subject} question ...
Select the correct answer!") and scores A/B/C/D letters; `cmmlu` localizes its
cue but is still letter-format. Our checkpoints are at chance on letters
regardless of calibration, so neither registered task is usable here.
"""
import datasets

CUE = {"ar": "الإجابة:", "zh": "答案："}


def _rows_ar():
    ds = datasets.load_dataset("MBZUAI/ArabicMMLU", "All", split="test")
    keys = [f"Option {i}" for i in range(1, 6)]
    out = []
    for i, d in enumerate(ds):
        opts = [str(d[k]).strip() for k in keys if d.get(k)]
        opts = [o for o in opts if o]
        ak = str(d["Answer Key"]).strip().upper()
        gold = "ABCDE".find(ak)
        if len(opts) < 2 or not (0 <= gold < len(opts)):
            continue
        q = d["Question"] if not d.get("Context") else f"{d['Context']}\n\n{d['Question']}"
        out.append({"subject": str(d.get("Subject") or ""), "idx": i,
                    "question": str(q).strip(), "options": opts, "label": gold})
    return out


def _rows_zh():
    subjects = datasets.get_dataset_config_names("haonan-li/cmmlu",
                                                 trust_remote_code=True)
    out = []
    for s in subjects:
        ds = datasets.load_dataset("haonan-li/cmmlu", s, split="test",
                                   trust_remote_code=True)
        for i, d in enumerate(ds):
            opts = [str(d[k]).strip() for k in ("A", "B", "C", "D")]
            gold = "ABCD".find(str(d["Answer"]).strip().upper())
            if any(not o for o in opts) or gold < 0:
                continue
            out.append({"subject": s, "idx": i, "question": str(d["Question"]).strip(),
                        "options": opts, "label": gold})
    return out


def _load(lang: str) -> datasets.Dataset:
    rows = _rows_ar() if lang == "ar" else _rows_zh()
    rows.sort(key=lambda r: (r["subject"], r["idx"]))
    by = {}
    for r in rows:
        by.setdefault(r["subject"], []).append(r)
    order, subs = [], sorted(by)
    for i in range(max(len(v) for v in by.values())):
        for s in subs:
            if i < len(by[s]):
                order.append(by[s][i])
    return datasets.Dataset.from_list(order)


def build(lang):
    def fn(**kwargs):
        return datasets.DatasetDict({"test": _load(lang)})
    return fn


build_ar, build_zh = build("ar"), build("zh")


def _text(lang):
    cue = CUE[lang]

    def fn(doc):
        return f"{doc['question']}\n{cue}"
    return fn


text_ar, text_zh = _text("ar"), _text("zh")


def choices(doc):
    return doc["options"]
