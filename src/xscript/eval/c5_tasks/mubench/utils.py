"""MuBench (aialt/MuBench) as 0-shot CLOZE tasks, aligned across our 5 languages.

MuBench ships 12 benchmarks translated into 61 languages and aligned by `_id`.
That solves the defect ARC exposed in CLAUDE.md 6d: our C.5 columns paired
English datasets against non-English ones that were not the same pool
(`arc_easy` n=2376 vs okapi m_arc, which is 100% ARC-Challenge n=1169 -- a
~20-point difficulty gap that produced the entire "ARC is English-only"
claim). With MuBench every language is a rendering of ONE item set.

Two conversions, both forced by things established in CLAUDE.md 6e:

* LETTER -> CLOZE. MuBench is letter-format: the prompt embeds the options and
  `choices` is ["A","B",...]. Our checkpoints sit at chance on letter formats
  and stay there after calibration, so the option texts are parsed back out of
  the prompt and scored as continuations.
* The trailing instruction line ("Answer with A, B, C or D:") and the option
  block are stripped from the stem, so what remains is a natural cloze prefix.

MNLI/SNLI are different in kind and are handled as-is: they embed no options,
so their candidates ARE the label words. Those are LOCALIZED per language
(Arabic gives التضمين / محايد / تناقض, not entailment/neutral/contradiction),
so there is no SIB-200-style label-language confound here -- verified, not
assumed. The choice set is nonetheless shared across documents, so `acc_cal`
applies and these belong in rawscores.SHARED_CHOICE_TASKS.

Verified per benchmark before use (see verify_mubench.py): item counts, `_id`
alignment across languages, gold index validity, and -- where an English
original exists -- pool identity by question matching.
"""
import json
import re

import datasets

LANGS = ("en", "de", "fr", "ar", "zh")

# TEMPLATE: MuBench ships `en_template_*` (English instructions around
# translated content) and `local_template_*` (instructions translated too).
# Use local: it is the correct form, and it costs nothing. Note the parser
# below must therefore be LANGUAGE-AGNOSTIC -- it cannot match English
# instruction strings, because they are no longer English.
TEMPLATE = "local"

# folder -> (has option lines?, stem ends with an instruction line to drop?)
#
# The option MARKER is localized in the local template -- Arabic writes
# "الخيار A: ...", Chinese "选项 A: ..." or "选项A: ...", while ARC-Easy and
# BMLAMA use a bare "A: ...". So options cannot be found by matching an
# English prefix; `_OPTION_LINE` instead requires a single Latin capital
# immediately before a colon (ASCII or full-width), with any prefix allowed,
# and `_parse_options` additionally requires the recovered letters to run
# A, B, C, ... in order. That ordering check is what makes the loose prefix
# safe: a content line like "Premise: ..." cannot match (the character before
# the colon is lowercase), and a stray match cannot form a consecutive run.
SPECS = {
    "arceasy":    ("ARCEasyDataset",     True,  False),
    "hellaswag":  ("HellaswagDataset",   True,  True),
    "mmlu":       ("MMLUDataset",        True,  False),
    "storycloze": ("StoryClozeDataset",  True,  True),
    "winogrande": ("WinoGrandeDataset",  True,  True),
    "bmlama":     ("BMLAMADataset",      True,  False),
    "truthfulqa": ("TruthfulQADataset",  True,  False),
    # No embedded options: candidates are the label words themselves.
    "mnli":       ("MNLIDataset",        False, False),
    "snli":       ("SNLIDataset",        False, False),
}
_OPTION_LINE = re.compile(r"^[^:\uff1a]*?([A-Z])\s*[:\uff1a]\s*(.+)$")


def _parse_options(prompt: str):
    """Return (options, index of first option line) or (None, None)."""
    lines = prompt.split("\n")
    opts, first, letters = [], None, []
    for i, ln in enumerate(lines):
        m = _OPTION_LINE.match(ln.strip())
        if not m:
            continue
        letter, text = m.group(1), m.group(2).strip()
        want = chr(ord("A") + len(letters))
        if letter != want:            # must run A, B, C, ... consecutively
            continue
        if first is None:
            first = i
        letters.append(letter)
        opts.append(text)
    if len(opts) < 2:
        return None, None
    return opts, first


def _load(bench: str, lang: str) -> datasets.Dataset:
    from huggingface_hub import hf_hub_download
    folder, has_options, drop_instruction = SPECS[bench]
    path = hf_hub_download("aialt/MuBench",
                           f"{folder}/{TEMPLATE}_template_{lang}_test.jsonl",
                           repo_type="dataset")
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            prompt = d["prompt"]
            gold = int(d["label"])
            # The LAST non-empty line of the prompt is MuBench's own answer cue,
            # already localized ("Answer:" / "الإجابة:" / "答案："). Reuse it
            # rather than hardcoding a per-language table.
            plines = [ln for ln in prompt.split("\n") if ln.strip()]
            cue = plines[-1].strip() if plines else "Answer:"
            if not has_options:                  # MNLI / SNLI
                opts = [str(c) for c in d["choices"]]
                stem = prompt.strip()            # already ends with the cue
            else:
                opts, first = _parse_options(prompt)
                if opts is None:
                    continue
                head = [ln for ln in prompt.split("\n")[:first] if ln.strip()]
                if drop_instruction and len(head) > 1:
                    head = head[:-1]             # drop the localized instruction
                if not head:
                    continue
                stem = "\n".join(head).strip() + "\n" + cue
            if not stem or len(opts) < 2 or not (0 <= gold < len(opts)):
                continue
            if any(not o for o in opts):
                continue
            out.append({"_id": str(d["_id"]), "stem": stem,
                        "options": opts, "label": gold})
    out.sort(key=lambda r: int(re.sub(r"\D", "", r["_id"]) or 0))
    return datasets.Dataset.from_list(out)


def build(bench, lang):
    def fn(**kwargs):
        return datasets.DatasetDict({"test": _load(bench, lang)})
    return fn


def text(doc):
    return doc["stem"]


def choices(doc):
    return doc["options"]


# Generated hooks: build_<bench>_<lang>
for _b in SPECS:
    for _l in LANGS:
        globals()[f"build_{_b}_{_l}"] = build(_b, _l)
