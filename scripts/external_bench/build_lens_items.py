#!/usr/bin/env python
"""Build the fixed item sets for the logit-lens analysis (CLAUDE.md 6l).

Deterministic and model-independent, so every checkpoint -- both tokenizer
conditions, every partner -- is scored on IDENTICAL items and fair-vs-starved
contrasts can be paired item by item.

  results/logitlens/items/translation.json   Wendler et al. word pairs, en <-> {de,fr,zh}
                                              from epfl-dlab/llm-latent-language, plus an
                                              en <-> ar list written for this project and
                                              cross-checked against MUSE (below)
  results/logitlens/items/cloze.json         Wendler's masked-definition prompts (de/fr/zh/en)
  results/logitlens/items/factual.json       800 PolyFact facts (jvonrad/PolyFact), all five
                                              languages, options aligned by Wikidata QID
  results/logitlens/items/generation.json    300 FLORES+ devtest sentence ids, all five
                                              languages (parallel), plus the generation prompts

Arabic is absent from Wendler's word lists. The en->ar list below was written
by hand for the ~140 English nouns in those lists (one common-noun sense,
no article, no multi-word entries) and is checked against the MUSE en-ar
dictionary; the agreement rate and every disagreement are printed so the
list can be audited. Items whose English word is a substring of the partner
word (or vice versa) are dropped, as in Wendler's notebook, because for them
"latent English" and "output language" are not distinguishable.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

DATA = Path("/mnt/scratch/xscript_lens/data")
OUT = REPO / "results" / "logitlens" / "items"

# en -> ar, written for this project (see module docstring). Multi-word or
# ambiguous entries deliberately omitted (tie, Han, grid=net collision).
EN_AR = {
    "advance": "تقدم", "attempt": "محاولة", "bag": "حقيبة", "ball": "كرة", "beauty": "جمال",
    "benefit": "فائدة", "book": "كتاب", "bridge": "جسر", "cave": "كهف", "child": "طفل",
    "city": "مدينة", "cloth": "قماش", "cloud": "سحابة", "color": "لون", "dance": "رقص",
    "day": "يوم", "defeat": "هزيمة", "delete": "حذف", "door": "باب", "east": "شرق",
    "eight": "ثمانية", "electric": "كهربائي", "example": "مثال", "exchange": "تبادل",
    "eye": "عين", "face": "وجه", "field": "حقل", "fire": "نار", "five": "خمسة",
    "flat": "مسطح", "flower": "زهرة", "foot": "قدم", "forest": "غابة", "four": "أربعة",
    "friend": "صديق", "fruit": "فاكهة", "generation": "جيل", "gold": "ذهب",
    "ground": "أرض", "group": "مجموعة", "hair": "شعر", "hand": "يد", "head": "رأس",
    "heart": "قلب", "history": "تاريخ", "horse": "حصان", "house": "بيت",
    "household": "أسرة", "hundred": "مئة", "item": "عنصر", "judge": "قاضي", "kind": "نوع",
    "lake": "بحيرة", "language": "لغة", "law": "قانون", "left": "يسار", "light": "ضوء",
    "line": "خط", "machine": "آلة", "male": "ذكر", "meaning": "معنى", "meeting": "اجتماع",
    "middle": "وسط", "moon": "قمر", "mother": "أم", "mountain": "جبل", "mouth": "فم",
    "net": "شبكة", "nine": "تسعة", "north": "شمال", "ocean": "محيط", "office": "مكتب",
    "official": "رسمي", "one": "واحد", "painting": "لوحة", "pair": "زوج", "part": "جزء",
    "person": "شخص", "platform": "منصة", "point": "نقطة", "pond": "بركة",
    "position": "موضع", "power": "قوة", "province": "مقاطعة", "red": "أحمر",
    "return": "عودة", "rice": "أرز", "rise": "ارتفاع", "river": "نهر", "rock": "صخرة",
    "sand": "رمل", "school": "مدرسة", "sea": "بحر", "seat": "مقعد", "section": "قسم",
    "seven": "سبعة", "shop": "متجر", "site": "موقع", "six": "ستة", "snow": "ثلج",
    "soil": "تربة", "song": "أغنية", "sound": "صوت", "source": "مصدر", "south": "جنوب",
    "speech": "خطاب", "speed": "سرعة", "spring": "نابض", "square": "مربع", "star": "نجمة",
    "station": "محطة", "street": "شارع", "strip": "شريط", "structure": "هيكل",
    "summer": "صيف", "sun": "شمس", "tail": "ذيل", "talent": "موهبة", "tea": "شاي",
    "ten": "عشرة", "thousand": "ألف", "three": "ثلاثة", "time": "وقت", "tower": "برج",
    "tribe": "قبيلة", "two": "اثنان", "valley": "وادي", "version": "نسخة",
    "village": "قرية", "virtue": "فضيلة", "water": "ماء", "wave": "موجة", "white": "أبيض",
    "wood": "خشب", "word": "كلمة", "yellow": "أصفر",
}


def wendler_lists() -> dict[str, dict[str, dict]]:
    """{lang: {en_word: row}} for fr/de/zh/en (Wendler's clean.csv files)."""
    out = {}
    for lang in ("fr", "de", "zh", "en"):
        rows = {}
        for r in csv.DictReader(open(DATA / "wendler" / f"{lang}.csv", encoding="utf-8")):
            if r.get("error", "no error") != "no error":
                continue
            rows[r["word_original"].strip()] = r
        out[lang] = rows
    return out


def muse(pair: str) -> dict[str, set[str]]:
    d: dict[str, set[str]] = {}
    for line in open(DATA / "muse" / f"{pair}.txt", encoding="utf-8"):
        parts = line.rstrip("\n").split("\t") if "\t" in line else line.split()
        if len(parts) >= 2:
            d.setdefault(parts[0].strip().lower(), set()).add(parts[1].strip())
    return d


def build_translation(rng: random.Random) -> dict:
    wl = wendler_lists()
    pairs = {}
    for lang in ("de", "fr", "zh"):
        items = []
        for en, r in wl[lang].items():
            x = r["word_translation"].strip()
            if not x or " " in x or " " in en:
                continue
            if lang != "zh" and (en.lower() in x.lower() or x.lower() in en.lower()):
                continue
            items.append({"en": en, "x": x})
        pairs[lang] = items
    # Arabic: this project's list, audited against MUSE en-ar
    m = muse("en-ar")
    agree, seen, disagreements = 0, 0, []
    ar_items = []
    for en, ar in EN_AR.items():
        ar_items.append({"en": en, "x": ar})
        if en in m:
            seen += 1
            if ar in m[en]:
                agree += 1
            else:
                disagreements.append((en, ar, sorted(m[en])[:6]))
    pairs["ar"] = ar_items
    print(f"[items] en-ar list: {len(ar_items)} words; MUSE covers {seen}, agrees on {agree} "
          f"({agree / max(seen, 1):.0%})")
    for en, ar, alts in disagreements:
        print(f"    MUSE disagrees: {en} -> {ar}   (MUSE: {', '.join(alts)})")
    # de-duplicate on the partner side (mismatched-word controls must differ)
    for lang, items in pairs.items():
        seen_x, uniq = set(), []
        for it in items:
            if it["x"] in seen_x:
                continue
            seen_x.add(it["x"]); uniq.append(it)
        pairs[lang] = uniq
        # fixed demo indices (5 other items) and a fixed mismatched-control partner
        n = len(uniq)
        for i, it in enumerate(uniq):
            others = [j for j in range(n) if j != i]
            it["demos"] = rng.sample(others, 5)
            it["ctl"] = (i + 7) % n if (i + 7) % n != i else (i + 1) % n
        print(f"[items] translation en-{lang}: {n} items")
    return pairs


def _unwrap(s: str) -> str:
    """Drop one layer of stray outer quotes some csv rows carry ('"A "soil" ..."'),
    but never the quote that opens a '"_"' blank."""
    s = s.strip()
    while len(s) > 2 and s[0] in "'\"" and s[-1] == s[0] and not s.startswith(s[0] + "_"):
        s = s[1:-1].strip()
    return s


def build_cloze(rng: random.Random) -> dict:
    """Wendler's masked-definition prompts, quotes stripped so the answer is a
    plain space-prefixed word (zh: no space). Query = prompt cut before the
    final answer; the cut answer is the continuation."""
    wl = wendler_lists()
    out = {}
    for lang in ("de", "fr", "zh", "en"):
        items = []
        for en, r in wl[lang].items():
            x = r["word_translation"].strip()
            full = _unwrap(r["blank_prompt_translation"])
            masked = _unwrap(r["blank_prompt_translation_masked"])
            if not x or not masked or " " in x:
                continue
            # normalise quotes: "___" / "_" -> ___, "word" -> word
            for q in ('"___"', '"_"', '“___”'):
                masked = masked.replace(q, "___")
                full = full.replace(q, "___")
            masked = masked.replace(f'"{x}"', x).replace(f'“{x}”', x)
            full = full.replace(f'"{x}"', x).replace(f'“{x}”', x)
            cut = masked.rfind(x)
            if cut <= 0 or "___" not in masked[:cut]:
                continue
            ctx = masked[:cut].rstrip()
            cont = ("" if lang == "zh" else " ") + x
            if lang not in ("zh", "en") and (en.lower() in x.lower() or x.lower() in en.lower()):
                continue
            items.append({"en": en, "x": x, "ctx": ctx, "cont": cont, "demo": full})
        n = len(items)
        for i, it in enumerate(items):
            it["demos"] = rng.sample([j for j in range(n) if j != i], 5)
            it["ctl"] = (i + 7) % n if (i + 7) % n != i else (i + 1) % n
        out[lang] = items
        print(f"[items] cloze {lang}: {n} items")
    return out


def build_factual(rng: random.Random, n: int = 800) -> dict:
    import datasets
    rows = {l: datasets.load_dataset("jvonrad/PolyFact", l, split="test") for l in ("en", "de", "fr", "ar", "zh")}
    ids = rows["en"]["fact_id"]
    for l in rows:
        assert rows[l]["fact_id"] == ids, l
    pick = sorted(rng.sample(range(len(ids)), n))
    facts = []
    for i in pick:
        f = {"fact_id": ids[i], "relation": rows["en"][i]["relation"], "gold_qid": None, "langs": {}}
        for l, ds in rows.items():
            r = ds[i]
            opts = [r["option_a"], r["option_b"], r["option_c"], r["option_d"]]
            qids = r["option_ids"]
            gold = qids[r["answer_index"]]
            if f["gold_qid"] is None:
                f["gold_qid"] = gold
            assert f["gold_qid"] == gold, (ids[i], l)
            f["langs"][l] = {"question": r["question"], "options": dict(zip(qids, opts)), "answer": r["answer_text"]}
        # a fixed common QID order so candidate index c means the same entity in every language
        f["qids"] = sorted(f["langs"]["en"]["options"])
        f["gold_idx"] = f["qids"].index(f["gold_qid"])
        facts.append(f)
    # mismatched-fact control: the English answer of the next sampled fact
    for j, f in enumerate(facts):
        f["ctl"] = (j + 1) % len(facts)
    print(f"[items] factual: {len(facts)} PolyFact facts")
    return {"facts": facts}


def build_generation(rng: random.Random, n: int = 300) -> dict:
    from xscript.flores import load
    sents = {l: load(l, "devtest") for l in ("en", "de", "fr", "ar", "zh")}
    common = sorted(set.intersection(*(set(s) for s in sents.values())))
    pick = sorted(rng.sample(common, n))
    items = []
    for sid in pick:
        it = {"id": sid, "text": {l: sents[l][sid] for l in sents}}
        # generation prompt: first ~40% of the words (>= 3), zh: first 40% of characters
        prm = {}
        for l, t in it["text"].items():
            if l == "zh":
                prm[l] = t[:max(4, int(len(t) * 0.4))]
            else:
                w = t.split()
                prm[l] = " ".join(w[:max(3, int(len(w) * 0.4))])
        it["prompt"] = prm
        items.append(it)
    for j, it in enumerate(items):
        it["ctl"] = (j + 11) % len(items)          # control sentence for set-mass
    print(f"[items] generation: {len(items)} FLORES+ devtest sentences x 5 languages")
    return {"items": items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--n-facts", type=int, default=800)
    ap.add_argument("--n-sents", type=int, default=300)
    ap.add_argument("--all-facts", action="store_true",
                    help="write factual_all.json with every PolyFact fact (2039) and nothing else")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    if args.all_facts:
        json.dump(build_factual(rng, 2039), open(OUT / "factual_all.json", "w"), ensure_ascii=False, indent=1)
        return
    json.dump(build_translation(rng), open(OUT / "translation.json", "w"), ensure_ascii=False, indent=1)
    json.dump(build_cloze(rng), open(OUT / "cloze.json", "w"), ensure_ascii=False, indent=1)
    json.dump(build_factual(rng, args.n_facts), open(OUT / "factual.json", "w"), ensure_ascii=False, indent=1)
    json.dump(build_generation(rng, args.n_sents), open(OUT / "generation.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
