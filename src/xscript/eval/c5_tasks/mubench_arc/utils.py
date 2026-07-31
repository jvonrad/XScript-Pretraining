"""ARC-Easy aligned across all five languages, from MuBench (aialt/MuBench).

This is the fix for the defect in CLAUDE.md 6d's ARC column: English was
scored on ARC-**Easy** (n=2376) while de/fr/ar/zh were scored on okapi's
m_arc, which is 100% ARC-**Challenge** (n=1169). Easy and Challenge differ by
~20 points at this scale, which is the entirety of the "striking English-only
pattern" (en +0.184 over chance vs -0.006..+0.017 elsewhere). Measured
like-for-like on ARC-Challenge, English is +0.02..+0.035 -- about two points
above the others, not twenty-five.

MuBench ships ARC-Easy translated into 61 languages, aligned by `_id`, and
verified here: 100.0% of its English questions appear in real ARC-Easy and
0.0% in ARC-Challenge, with the original option-count distribution
(2348 four-way, 7 three-way, 4 five-way).

Two conversions applied:

* MuBench is LETTER format -- the prompt embeds the options and `choices` is
  ["A","B","C","D"], ending "Answer with A, B, C, D, ...\\nAnswer:". Our
  checkpoints are at chance on letter formats and stay there after
  calibration (CLAUDE.md 6e), so the options are parsed back out of the
  prompt and scored as CLOZE continuations.
* `acc_mutual_info` is added. ARC's candidates are per-document answer texts,
  so `acc_cal` does not apply (no stable choice index) and `acc_norm` only
  corrects length; the uncorrected term is the answer-string prior, which is
  what PMI removes (Holtzman et al. 2021). lm-eval's own ARC tasks report
  neither.

Scaffolding is kept as "Question: {q}\\nAnswer:", matching okapi/lm-eval, so
these numbers sit alongside the existing ARC column with format held constant.

NOTE MuBench also ships HellaSwag, MMLU, MMLU-Pro, StoryCloze, WinoGrande,
MNLI, SNLI, TruthfulQA, GPQA and BMLAMA on the same 61 aligned languages. Our
HellaSwag column has the same shape of risk as ARC did (English n=10042 vs
okapi hellaswag_de n=9368 -- different pools), and MuBench would settle it.
"""
import json
import re

import datasets

LANGS = ("en", "de", "fr", "ar", "zh")
_HEAD = re.compile(r"Question:\s*(.*?)\n([A-E]:.*)", re.S)
_OPT = re.compile(r"^([A-E]):\s*(.*)$", re.M)


def _parse(prompt: str):
    m = _HEAD.match(prompt)
    if not m:
        return None, None
    tail = m.group(2)
    # Drop MuBench's letter-answer instruction line before parsing options.
    tail = tail.split("\nAnswer with")[0]
    opts = [o[1].strip() for o in _OPT.findall(tail)]
    return m.group(1).strip(), opts


def _load(lang: str) -> datasets.Dataset:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("aialt/MuBench",
                           f"ARCEasyDataset/en_template_{lang}_test.jsonl",
                           repo_type="dataset")
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            q, opts = _parse(d["prompt"])
            gold = int(d["label"])
            if not q or len(opts) < 2 or not (0 <= gold < len(opts)):
                continue
            out.append({"_id": str(d["_id"]), "question": q,
                        "options": opts, "label": gold})
    # `_id` is "test_<n>" and is shared across languages -> identical doc order.
    out.sort(key=lambda r: int(r["_id"].rsplit("_", 1)[-1]))
    return datasets.Dataset.from_list(out)


def build(lang):
    def fn(**kwargs):
        return datasets.DatasetDict({"test": _load(lang)})
    return fn


build_en, build_de, build_fr = build("en"), build("de"), build("fr")
build_ar, build_zh = build("ar"), build("zh")


def text(doc):
    return "Question: " + doc["question"] + "\nAnswer:"


def choices(doc):
    return doc["options"]
