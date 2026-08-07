#!/usr/bin/env python
"""Pool-identity gate for X-CSQA, the same check `verify_mubench.py` runs.

CLAUDE.md 6e's standing rule: *before comparing a benchmark across languages,
verify the non-English version renders the same pool as the English one.*
That rule exists because ARC did not -- our C.5 table scored English on
ARC-Easy and every other language on ARC-Challenge, a ~20-point difficulty
gap that produced an entire retracted finding. Item counts are the cheapest
form of the check and are what caught it.

Pure CPU, no accelerator and no checkpoint. Run it before any X-CSQA sweep and
again after any change to c5_tasks/xcsqa/utils.py.

    python verify_xcsqa.py                 # checks the built tasks
    python verify_xcsqa.py --csqa          # also match English against real
                                           # CommonsenseQA (needs tau/commonsense_qa)

Checks, in order of how badly a failure would corrupt results:

  1. `test` split really is blind      -- scoring it would yield garbage, not an
                                          error, since every answerKey is "".
  2. equal item counts across langs    -- the ARC check.
  3. identical `id` sets AND identical
     doc order after the build         -- per-example hit lists must line up
                                          language-to-language, not just
                                          model-to-model.
  4. no empty / malformed candidates   -- an empty continuation scores ll=0.0,
                                          which beats every real (negative)
                                          candidate under `acc`, so it is a
                                          guaranteed wrong answer rather than
                                          noise. Six German rows had this.
  5. gold validity + arity             -- ragged arity would make nominal
                                          chance meaningless (cf. BMLAMA).
  6. options are permuted per language -- expected and harmless for cloze, but
                                          asserted so a future release that
                                          silently un-permutes is noticed.
  7. pool identity vs CommonsenseQA    -- --csqa; the question-matching check.
"""
import argparse
import collections
import sys
from pathlib import Path

LANGS = ("en", "de", "fr", "ar", "zh")
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

FAILURES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csqa", action="store_true",
                    help="also verify English pool identity against the real "
                         "CommonsenseQA (downloads tau/commonsense_qa)")
    args = ap.parse_args()

    from xscript.eval.c5_tasks.xcsqa import utils as U

    print("\n1. THE `test` SPLIT IS BLIND (only `validation` is labelled)")
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    for lang in LANGS:
        p = hf_hub_download("INK-USC/xcsr",
                            f"X-CSQA-{lang}/test-00000-of-00001.parquet",
                            repo_type="dataset")
        keys = [(r["answerKey"] or "").strip() for r in pq.read_table(p).to_pylist()]
        check(all(k == "" for k in keys), f"{lang}: upstream `test` unlabelled",
              f"n={len(keys)} labelled={sum(1 for k in keys if k)}")

    print("\n2. RAW ITEM COUNTS EQUAL ACROSS LANGUAGES")
    raw = {l: U._raw(l) for l in LANGS}
    counts = {l: len(v) for l, v in raw.items()}
    check(len(set(counts.values())) == 1, "equal raw counts", str(counts))

    print("\n3. DROPPED IDS, AND WHY")
    drop = U._drop_ids()
    print(f"       dropping {len(drop)} id(s) from ALL {len(LANGS)} languages")
    for _id in sorted(drop):
        for lang in LANGS:
            r = next((x for x in raw[lang] if x["id"] == _id), None)
            if r and (not r["stem"] or any(not o for o in r["options"])):
                print(f"       {_id}  [{lang}] options={r['options']}")

    print("\n4. BUILT TASKS: COUNTS, ID SETS, DOC ORDER")
    built = {l: U._rows(l) for l in LANGS}
    n = {l: len(v) for l, v in built.items()}
    check(len(set(n.values())) == 1, "equal built counts", str(n))
    ids = {l: [r["_id"] for r in built[l]] for l in LANGS}
    base = ids["en"]
    check(all(set(ids[l]) == set(base) for l in LANGS), "identical id SETS")
    check(all(ids[l] == base for l in LANGS),
          "identical DOC ORDER (hit lists line up across languages)")
    check(all(_id not in drop for _id in base), "no dropped id survived the build")

    print("\n5. CANDIDATES AND GOLD")
    for lang in LANGS:
        rows = built[lang]
        arity = collections.Counter(len(r["options"]) for r in rows)
        empty = sum(1 for r in rows if any(not o.strip() for o in r["options"]))
        badgold = sum(1 for r in rows if not 0 <= r["label"] < len(r["options"]))
        nostem = sum(1 for r in rows if not r["stem"].strip())
        check(empty == 0 and badgold == 0 and nostem == 0 and set(arity) == {5},
              f"{lang}: 5 non-empty options, gold in range",
              f"arity={dict(arity)} empty={empty} badgold={badgold} nostem={nostem}")

    print("\n6. OPTIONS ARE PERMUTED PER LANGUAGE (expected; gold tracks it)")
    gold_en = {r["_id"]: r["label"] for r in built["en"]}
    for lang in LANGS:
        if lang == "en":
            continue
        diff = sum(1 for r in built[lang] if r["label"] != gold_en[r["_id"]])
        check(diff > 0, f"en vs {lang}: gold INDEX differs (permuted)",
              f"{diff}/{len(built[lang])} items")

    print("\n7. THE ENGLISH-OPTION CONTROL IS SELF-CONSISTENT")
    for lang in ("de", "fr", "ar", "zh"):
        ctl = U._rows(lang, en_options=True)
        same_opts = all(a["options"] == b["options"]
                        for a, b in zip(ctl, built["en"]))
        same_gold = all(a["label"] == b["label"] for a, b in zip(ctl, built["en"]))
        same_stem = all(a["stem"] == b["stem"] for a, b in zip(ctl, built[lang]))
        check(same_opts and same_gold and same_stem,
              f"xcsqa_enopt_{lang}: English options+gold, localized stem")

    if args.csqa:
        print("\n8. POOL IDENTITY: IS X-CSQA-en REALLY CommonsenseQA?")
        # Read the parquet directly rather than via `load_dataset`, for the
        # same reason c5_tasks/xcsqa/utils.py does: no builder, no dataset
        # script, no fingerprinting -- just the bytes.
        norm = lambda s: " ".join(s.lower().split())
        pool, per_split = set(), {}
        for split in ("train", "validation"):
            p = hf_hub_download("tau/commonsense_qa",
                                f"data/{split}-00000-of-00001.parquet",
                                repo_type="dataset")
            qs = {norm(q) for q in pq.read_table(p).column("question").to_pylist()}
            per_split[split] = qs
            pool |= qs
        stems = [norm(r["stem"]) for r in U._raw("en")]
        hit = sum(1 for s in stems if s in pool)
        check(hit == len(stems), "every X-CSQA-en question is a real CSQA question",
              f"{hit}/{len(stems)}")
        # Which CSQA split they came from. Recorded, not asserted: XCSR
        # re-partitioned CSQA, so X-CSQA's dev set is mostly CSQA *train*.
        # Irrelevant for contamination here (these checkpoints see only
        # FineWeb2-HQ, and any web-leakage is identical for every checkpoint,
        # so it cancels in every within-benchmark contrast), but it means
        # X-CSQA dev is NOT a held-out slice of CSQA dev.
        for split, qs in per_split.items():
            n = sum(1 for s in stems if s in qs)
            print(f"       from CSQA {split}: {n}/{len(stems)}")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
