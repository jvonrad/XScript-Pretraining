"""Pairwise vocabulary overlap between the five study languages, per tokenizer.

For each tokenizer (fair = unigram_destarved, starved = unigram_starved) we
encode the SAME parallel FLORES+ sentences (dev+devtest, n=2009) in all five
languages and ask how much of the vocabulary the languages actually SHARE.

Three families of numbers, because they answer different questions:

  type-level   Jaccard  |A n B| / |A u B|   and overlap coefficient
               |A n B| / min(|A|,|B|)  over the SET of pieces each language
               emits.  "How much of the vocabulary is common property?"
  token-level  coverage_{A->B} = share of A's token OCCURRENCES whose piece is
               also emitted by B.  Asymmetric, and the one that says how much
               of running text rides on shared machinery.
  content-only the same, restricted to pieces that contain a letter
               (classify_piece != sym_num_space / byte_atom / special), because
               punctuation, digits and Latin-script named entities are shared
               across scripts for free -- the "lexical floor" of CLAUDE.md 6b.

Static (usage-free) view too: how many of the 65536 pieces are reachable from
each language at all, and how the shared pool splits by script.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xscript.tok.wrapper import Tok
from xscript.tok.analyze import classify_piece

LANGS = ["en", "de", "fr", "ar", "zh"]
FLORES_CODE = {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn",
               "ar": "arb_Arab", "zh": "cmn_Hans"}
SCRIPT = {"en": "Latin", "de": "Latin", "fr": "Latin", "ar": "Arabic", "zh": "Han"}
TOKS = {"fair": "unigram_destarved", "starved": "unigram_starved"}
SPECIALS = {0, 1, 2, 3}


def load_parallel(flores_dir: Path) -> dict[str, list[str]]:
    per = {}
    for lg in LANGS:
        d = {}
        for split in ("dev", "devtest"):
            path = flores_dir / split / f"{FLORES_CODE[lg]}.jsonl"
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        d[(split, int(r["id"]))] = r["text"]
        per[lg] = d
    common = sorted(set.intersection(*(set(v) for v in per.values())))
    return {lg: [per[lg][k] for k in common] for lg in LANGS}, len(common)


def profile(tok: Tok, texts: list[str]):
    """counts[piece_id] -> occurrences, over one language's sentences."""
    counts: dict[int, int] = {}
    for ids in tok.encode_batch(texts):
        for i in ids:
            if i in SPECIALS:
                continue
            counts[i] = counts.get(i, 0) + 1
    return counts


def is_content(tok: Tok, i: int) -> bool:
    if tok.is_byte_piece(i):
        return False
    return classify_piece(tok.piece_bytes(i)) not in ("sym_num_space", "special",
                                                      "partial_utf8")


def pair_stats(cA: dict, cB: dict):
    A, B = set(cA), set(cB)
    inter, union = A & B, A | B
    nA, nB = sum(cA.values()), sum(cB.values())
    shA = sum(cA[i] for i in inter)
    shB = sum(cB[i] for i in inter)
    return {
        "types_a": len(A), "types_b": len(B), "types_shared": len(inter),
        "jaccard": len(inter) / len(union) if union else 0.0,
        "overlap_coef": len(inter) / min(len(A), len(B)) if A and B else 0.0,
        "cov_a_to_b": shA / nA if nA else 0.0,
        "cov_b_to_a": shB / nB if nB else 0.0,
    }



def topk(counts: dict, k: int) -> set:
    return {i for i, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]}


def matched_overlap(cA: dict, cB: dict, k: int):
    """Jaccard on the top-k types of each language: set sizes are equal by
    construction, so this is immune to the fair tokenizer simply having more
    distinct pieces per language (which mechanically depresses raw Jaccard)."""
    A, B = topk(cA, k), topk(cB, k)
    inter = A & B
    nA, nB = sum(cA.values()), sum(cB.values())
    return {
        "k": k,
        "shared": len(inter),
        "jaccard": len(inter) / len(A | B) if (A or B) else 0.0,
        "frac_shared": len(inter) / min(len(A), len(B)) if A and B else 0.0,
        "mass_a_in_shared": sum(cA[i] for i in inter) / nA if nA else 0.0,
        "mass_b_in_shared": sum(cB[i] for i in inter) / nB if nB else 0.0,
    }


def piece_len(tok, i) -> int:
    if tok.is_byte_piece(i):
        return 0
    try:
        return len(tok.piece_bytes(i).decode("utf-8").replace("\u2581", ""))
    except UnicodeDecodeError:
        return 0


def _script_mix(tok, counts):
    mix = {}
    for i in counts:
        b = classify_piece(tok.piece_bytes(i))
        mix[b] = mix.get(b, 0) + 1
    return dict(sorted(mix.items(), key=lambda kv: -kv[1]))


def shared_composition(tok, cA, cB):
    """What the shared pieces actually ARE: script bucket + single vs multi
    char. Cross-script 'overlap' is mostly Latin entities/loanwords and bare
    characters, not shared subwords."""
    inter = set(cA) & set(cB)
    by_script, single, multi = {}, 0, 0
    for i in inter:
        b = "byte_atom" if tok.is_byte_piece(i) else classify_piece(tok.piece_bytes(i))
        by_script[b] = by_script.get(b, 0) + 1
        if piece_len(tok, i) <= 1:
            single += 1
        else:
            multi += 1
    return {"by_script": dict(sorted(by_script.items(), key=lambda kv: -kv[1])),
            "single_char": single, "multi_char": multi,
            "pct_multi_char": 100.0 * multi / len(inter) if inter else 0.0}


def main():
    flores_dir = Path(sys.argv[1] if len(sys.argv) > 1
                      else "/mnt/scratch/xscript/flores_plus")
    tokdir = Path(sys.argv[2] if len(sys.argv) > 2
                  else "/mnt/scratch/xscript/tokenizers")
    out = Path(sys.argv[3] if len(sys.argv) > 3
               else "results/vocab_overlap")
    out.mkdir(parents=True, exist_ok=True)

    texts, n_sent = load_parallel(flores_dir)
    print(f"parallel sentences: {n_sent}", file=sys.stderr)

    report = {"n_sentences": n_sent, "tokenizers": {}}
    for cond, name in TOKS.items():
        tok = Tok(tokdir / name)
        assert tok.vocab_size == 65536, tok.vocab_size
        full = {lg: profile(tok, texts[lg]) for lg in LANGS}
        cont = {lg: {i: c for i, c in full[lg].items() if is_content(tok, i)}
                for lg in LANGS}
        # genuine subwords only: letter-bearing AND >= 3 characters. Removes the
        # bare-character / punctuation floor, and the confound that starved
        # pieces are simply shorter (short pieces are reused across languages
        # more easily, so raw overlap favours the starved tokenizer by
        # construction).
        sub = {lg: {i: c for i, c in cont[lg].items() if piece_len(tok, i) >= 3}
               for lg in LANGS}

        # static: piece-level script buckets of the whole vocab, and how many
        # pieces are shared by >= 2 / all languages in usage
        seen_by = {}
        for lg in LANGS:
            for i in full[lg]:
                seen_by[i] = seen_by.get(i, 0) + 1
        shared_any = sum(1 for v in seen_by.values() if v >= 2)
        shared_all = sum(1 for v in seen_by.values() if v == len(LANGS))

        entry = {
            "vocab_size": tok.vocab_size,
            "per_lang": {lg: {"types": len(full[lg]),
                              "types_content": len(cont[lg]),
                              "tokens": sum(full[lg].values()),
                              "tokens_content": sum(cont[lg].values())}
                         for lg in LANGS},
            "used_by_any_lang": len(seen_by),
            "used_by_2plus_langs": shared_any,
            "used_by_all_5": shared_all,
            "pairs": {}, "pairs_content": {},
            "pairs_matched": {}, "pairs_matched_content": {},
            "pairs_subword": {}, "pairs_matched_subword": {},
            "shared_composition": {},
            "mean_piece_len": {lg: (sum(full[lg][i] * piece_len(tok, i)
                                        for i in full[lg]) /
                                    sum(full[lg].values())) for lg in LANGS},
            "subword_types": {lg: len(sub[lg]) for lg in LANGS},
            "subword_script_mix": {
                lg: _script_mix(tok, sub[lg]) for lg in LANGS},
        }
        ks = [500, 1000, 2000]
        for a, b in combinations(LANGS, 2):
            entry["pairs"][f"{a}-{b}"] = pair_stats(full[a], full[b])
            entry["pairs_content"][f"{a}-{b}"] = pair_stats(cont[a], cont[b])
            entry["pairs_matched"][f"{a}-{b}"] = {
                str(k): matched_overlap(full[a], full[b], k) for k in ks}
            entry["pairs_matched_content"][f"{a}-{b}"] = {
                str(k): matched_overlap(cont[a], cont[b], k) for k in ks}
            entry["pairs_subword"][f"{a}-{b}"] = pair_stats(sub[a], sub[b])
            entry["pairs_matched_subword"][f"{a}-{b}"] = {
                str(k): matched_overlap(sub[a], sub[b], k) for k in [500, 1000]}
            entry["shared_composition"][f"{a}-{b}"] = shared_composition(
                tok, full[a], full[b])
        report["tokenizers"][cond] = entry

    (out / "overlap.json").write_text(json.dumps(report, indent=2))
    write_md(report, out / "overlap.md")
    print(f"wrote {out}/overlap.json and overlap.md", file=sys.stderr)


def group(a, b):
    if SCRIPT[a] == SCRIPT[b]:
        return "same-script"
    return "cross-script"


def write_md(rep, path):
    L = []
    L.append("# Vocabulary overlap between languages, per tokenizer\n")
    L.append(f"FLORES+ dev+devtest, {rep['n_sentences']} parallel sentences per "
             "language (identical content in all five). Specials excluded.\n")
    for cond in ("fair", "starved"):
        e = rep["tokenizers"][cond]
        L.append(f"\n## {cond} (`{TOKS[cond]}`)\n")
        L.append("| lang | types used | content types | tokens |")
        L.append("|---|---|---|---|")
        for lg in LANGS:
            p = e["per_lang"][lg]
            L.append(f"| {lg} | {p['types']} | {p['types_content']} | {p['tokens']} |")
        L.append(f"\nPieces used by >=1 language: {e['used_by_any_lang']} of "
                 f"{e['vocab_size']}; by >=2: {e['used_by_2plus_langs']}; "
                 f"by all 5: {e['used_by_all_5']}.\n")
        for key, label in (("pairs", "all pieces"),
                           ("pairs_content", "content pieces only (letter-bearing)")):
            L.append(f"\n### {label}\n")
            L.append("| pair | group | shared types | Jaccard | overlap coef | "
                     "cov A->B | cov B->A |")
            L.append("|---|---|---|---|---|---|---|")
            for a, b in combinations(LANGS, 2):
                s = e[key][f"{a}-{b}"]
                L.append(f"| {a}-{b} | {group(a,b)} | {s['types_shared']} | "
                         f"{s['jaccard']:.3f} | {s['overlap_coef']:.3f} | "
                         f"{s['cov_a_to_b']:.3f} | {s['cov_b_to_a']:.3f} |")
        L.append("\n### matched-size overlap (top-K types per language, K equal "
                 "in both langs and both tokenizers)\n")
        L.append("| pair | group | K=500 shared | K=1000 shared | K=2000 shared | "
                 "K=2000 Jaccard | K=2000 mass A | K=2000 mass B |")
        L.append("|---|---|---|---|---|---|---|---|")
        for a, b in combinations(LANGS, 2):
            m = e["pairs_matched"][f"{a}-{b}"]
            L.append(f"| {a}-{b} | {group(a,b)} | {m['500']['shared']} | "
                     f"{m['1000']['shared']} | {m['2000']['shared']} | "
                     f"{m['2000']['jaccard']:.3f} | "
                     f"{m['2000']['mass_a_in_shared']:.3f} | "
                     f"{m['2000']['mass_b_in_shared']:.3f} |")
        L.append("\n### genuine subwords only (letter-bearing, >=3 chars)\n")
        L.append("| pair | group | A types | B types | shared | Jaccard | "
                 "overlap coef | top-500 shared | top-1000 shared |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for a, b in combinations(LANGS, 2):
            st = e["pairs_subword"][f"{a}-{b}"]
            m = e["pairs_matched_subword"][f"{a}-{b}"]
            L.append(f"| {a}-{b} | {group(a,b)} | {st['types_a']} | "
                     f"{st['types_b']} | {st['types_shared']} | "
                     f"{st['jaccard']:.3f} | {st['overlap_coef']:.3f} | "
                     f"{m['500']['shared']} | {m['1000']['shared']} |")
        L.append("\nScript of each language's own >=3-char pieces "
                 "(a language whose long pieces are mostly Latin can only "
                 "'share vocabulary' with English trivially):\n")
        L.append("| lang | >=3-char types | script mix |")
        L.append("|---|---|---|")
        for lg in LANGS:
            mix = e["subword_script_mix"][lg]
            top = ", ".join(f"{k} {v}" for k, v in list(mix.items())[:3])
            L.append(f"| {lg} | {e['subword_types'][lg]} | {top} |")
        L.append("\nMean emitted piece length (chars): " +
                 ", ".join(f"{lg} {e['mean_piece_len'][lg]:.2f}" for lg in LANGS) +
                 ".\n")
        L.append("\n### what the shared pieces ARE\n")
        L.append("| pair | group | shared types | multi-char % | top script buckets |")
        L.append("|---|---|---|---|---|")
        for a, b in combinations(LANGS, 2):
            c = e["shared_composition"][f"{a}-{b}"]
            top = ", ".join(f"{k} {v}" for k, v in list(c["by_script"].items())[:4])
            L.append(f"| {a}-{b} | {group(a,b)} | "
                     f"{c['single_char']+c['multi_char']} | "
                     f"{c['pct_multi_char']:.1f} | {top} |")
    L.append("\n## fair - starved (all pieces / content only)\n")
    L.append("| pair | group | dJaccard | dJaccard content | dOverlapCoef | "
             "dOverlapCoef content |")
    L.append("|---|---|---|---|---|---|")
    f, s = rep["tokenizers"]["fair"], rep["tokenizers"]["starved"]
    for a, b in combinations(LANGS, 2):
        k = f"{a}-{b}"
        L.append(f"| {k} | {group(a,b)} | "
                 f"{f['pairs'][k]['jaccard']-s['pairs'][k]['jaccard']:+.3f} | "
                 f"{f['pairs_content'][k]['jaccard']-s['pairs_content'][k]['jaccard']:+.3f} | "
                 f"{f['pairs'][k]['overlap_coef']-s['pairs'][k]['overlap_coef']:+.3f} | "
                 f"{f['pairs_content'][k]['overlap_coef']-s['pairs_content'][k]['overlap_coef']:+.3f} |")
    L.append("\n## fair - starved, SIZE-MATCHED (top-2000 types per language)\n")
    L.append("| pair | group | fair shared | starved shared | delta | "
             "fair mass A | starved mass A | delta |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a, b in combinations(LANGS, 2):
        k = f"{a}-{b}"
        fm, sm = f["pairs_matched"][k]["2000"], s["pairs_matched"][k]["2000"]
        L.append(f"| {k} | {group(a,b)} | {fm['shared']} | {sm['shared']} | "
                 f"{fm['shared']-sm['shared']:+d} | "
                 f"{fm['mass_a_in_shared']:.3f} | {sm['mass_a_in_shared']:.3f} | "
                 f"{fm['mass_a_in_shared']-sm['mass_a_in_shared']:+.3f} |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
