"""Chinese HellaSwag: loader + the upstream okapi preprocessing.

`process_docs`/`preprocess` are a VERBATIM copy of
lm_eval/tasks/okapi/hellaswag_multilingual/utils.py (v0.4.12). Duplicated
rather than imported so `hellaswag_zh` preprocesses EXACTLY like the registered
`hellaswag_{de,fr,ar}` tasks it is compared against -- including the quirks
(``.capitalize()`` lowercases the rest of ctx_b; the WikiHow bracket stripping
is an English-corpus artifact that survives translation). Any drift here would
make the Chinese column non-comparable with the others, which is the only
reason this task exists.

`build_dataset` is the part that is NOT upstream, and is why the task needs a
`custom_dataset:` hook instead of a plain `dataset_path:` (see the yaml).
"""
import json
import re

import datasets


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)


def build_dataset(**kwargs) -> datasets.DatasetDict:
    """Load `alexandrainst/m_hellaswag` zh, repairing 4 malformed `endings`.

    `datasets.load_dataset("alexandrainst/m_hellaswag", "zh")` FAILS outright:

        ArrowInvalid: JSON parse error: Column(/endings/[]) changed from
        string to object in row 153

    In `data/zh/val.jsonl`, 4 of the 37,064 endings (all in doc index 5074)
    were written as `{"zh": ..., "en": ...}` dicts instead of bare strings --
    a leak from the translation pipeline. pyarrow infers the schema from the
    first chunk, hits the type change, and refuses the whole file, so the
    entire language is unloadable. That, not a missing translation, is why
    CLAUDE.md's C.5 table had an empty ZH-HellaSwag column and why lm-eval
    0.4.12 ships no `hellaswag_zh.yaml` while shipping 31 other okapi
    languages.

    Reading the jsonl with the stdlib and taking the `"zh"` member of any dict
    ending recovers all 9266 documents; every other field is already
    well-typed. The affected row is a single document (~0.01% of the split),
    and it is repaired rather than dropped so the doc order stays a fixed,
    reproducible list -- `bootstrap_transfer.py`'s paired resampling requires
    both models in a comparison to score the identical doc sequence.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("alexandrainst/m_hellaswag", "data/zh/val.jsonl",
                           repo_type="dataset")
    rows = []
    repaired = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            endings = []
            for end in row["endings"]:
                if isinstance(end, dict):
                    end = end["zh"]
                    repaired += 1
                endings.append(end)
            row["endings"] = endings
            rows.append(row)
    if repaired:
        print(f"[hellaswag_zh] repaired {repaired} dict-typed ending(s) "
              f"in {len(rows)} docs")
    return datasets.DatasetDict({"val": datasets.Dataset.from_list(rows)})
