"""ARC-Challenge with PMI scoring, like-for-like across all five languages.

Two defects in the C.5 ARC column, both fixed here.

1. POOL MISMATCH. `arc_easy` (English, n=2376) was compared against
   `arc_{de,fr,ar,zh}`, which come from okapi's `alexandrainst/m_arc` and are
   **100% ARC-Challenge** (verified by question matching: 1169/1169 of m_arc's
   English rows appear in ARC-Challenge, 1/1169 in ARC-Easy). Easy and
   Challenge differ by ~20 points at this scale, which is the whole of
   CLAUDE.md 6d's "striking English-only pattern" (en +0.184 over chance vs
   -0.006..+0.017 elsewhere). Measured like-for-like, English ARC-Challenge is
   .268-.285, i.e. +0.02..+0.035 -- roughly two points above the others, not
   twenty-five.

2. NO PMI. lm-eval's ARC tasks report only acc and acc_norm. ARC's candidates
   are per-document answer texts, so `acc_cal` does not apply (there is no
   stable choice index to average over) and `acc_norm` only corrects length.
   What is left uncorrected is the ANSWER-STRING PRIOR: some answers are more
   likely strings regardless of the question. PMI -- log P(answer|question) -
   log P(answer) -- is the standard correction (Holtzman et al. 2021) and is
   the one lever not yet pulled on ARC.

The English source is ARC-Challenge; the others are m_arc. Prompt formatting
is kept EXACTLY as okapi/lm-eval build it ("Question: {q}\\nAnswer:", English
scaffolding and all) so that adding PMI is the only variable changed. That
scaffolding is itself non-localized, but the cue-language effect was measured
at 0.000 on Global-MMLU (CLAUDE.md 6e), so it is not the confound to chase
here.
"""
import datasets

LANG_CFG = {"de": "de", "fr": "fr", "ar": "ar", "zh": "zh"}


def _rows_en():
    ds = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    out = []
    for i, d in enumerate(ds):
        texts = list(d["choices"]["text"])
        labels = [str(x) for x in d["choices"]["label"]]
        key = str(d["answerKey"])
        if key not in labels or len(texts) < 2:
            continue
        out.append({"idx": i, "query": "Question: " + d["question"].strip() + "\nAnswer:",
                    "choices": [str(t).strip() for t in texts],
                    "label": labels.index(key)})
    return out


def _rows_ml(lang):
    ds = datasets.load_dataset("alexandrainst/m_arc", LANG_CFG[lang], split="test")
    keys = ["option_a", "option_b", "option_c", "option_d", "option_e"]
    out = []
    for i, d in enumerate(ds):
        opts = [str(d[k]).strip() for k in keys if d.get(k) not in (None, "")]
        ans = str(d["answer"]).strip()
        # okapi stores the answer as the option letter
        gold = "abcde".find(ans.lower().replace("option ", "")[:1])
        if len(opts) < 2 or not (0 <= gold < len(opts)):
            continue
        out.append({"idx": i,
                    "query": "Question: " + str(d["instruction"]).strip() + "\nAnswer:",
                    "choices": opts, "label": gold})
    return out


def _load(lang):
    rows = _rows_en() if lang == "en" else _rows_ml(lang)
    rows.sort(key=lambda r: r["idx"])
    return datasets.Dataset.from_list(rows)


def build(lang):
    def fn(**kwargs):
        return datasets.DatasetDict({"test": _load(lang)})
    return fn


build_en, build_de, build_fr = build("en"), build("de"), build("fr")
build_ar, build_zh = build("ar"), build("zh")


def text(doc):
    return doc["query"]


def choices(doc):
    return doc["choices"]
