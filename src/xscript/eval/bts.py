"""Bilingual Transfer Score and the thesis's interaction estimate.

BTS(L) = (BPB_mono(L) - BPB_bi(L)) / BPB_mono(L)   (positive = bilingual helps)

Two matchings (the plan asks for both):
  - matched-total   : mono-final vs bilingual-final (both at the same TOTAL
                      tokens). This is ATLAS's framing; under a starved tokenizer
                      the cross-script partner saw fewer of its own tokens.
  - matched-lang    : bilingual-final vs the mono checkpoint at the same number
                      of *that-language* tokens the bilingual run actually saw
                      (= total x mixing-prob). Removes the token-count confound.

Headline interaction (the untested contribution):
  penalty(C) = mean BTS(same-script) - mean BTS(cross-script)   under tokenizer C
  interaction = penalty(starved) - penalty(destarved)
A large positive interaction => the cross-script "penalty" is largely a
tokenizer starvation artifact, not intrinsic script transfer.

Run naming convention (see configs/matrix.py): "<mix>__<tok_name>", where mix is
one language ("en") or "en-<partner>", and tok_name is e.g. "bl_destarved".
"""
import json
from pathlib import Path

from ..langs import ANCHOR, PARTNERS, LANGS
from ..paths import RUNS, RESULTS, ensure


def _read_evals(name: str) -> list[dict]:
    """[(tokens, {source: bpb})...] sorted by tokens, from a run's train.jsonl."""
    log = RUNS / name / "train.jsonl"
    if not log.exists():
        return []
    series = {}
    for line in log.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = rec.get("eval") or rec.get("eval_final")
        if ev:
            series[rec["tokens"]] = {s: v["bpb"] for s, v in ev.items()}
    return [{"tokens": t, "bpb": series[t]} for t in sorted(series)]


def _final_bpb(name: str, source: str) -> float | None:
    ev = _read_evals(name)
    for rec in reversed(ev):
        if source in rec["bpb"]:
            return rec["bpb"][source]
    return None


def _bpb_at_lang_tokens(name: str, source: str, lang_tokens: float) -> float | None:
    """Mono BPB at the checkpoint closest to `lang_tokens` (matched-lang)."""
    best, bd = None, None
    for rec in _read_evals(name):
        if source in rec["bpb"]:
            d = abs(rec["tokens"] - lang_tokens)
            if bd is None or d < bd:
                bd, best = d, rec["bpb"][source]
    return best


def compute(tok_name: str, source_kind: str = "flores",
            mix_prob: float = 0.5, total_tokens: float = 30e9) -> dict:
    """BTS for every partner under one tokenizer condition."""
    rows = {}
    for p in PARTNERS:
        src = f"{source_kind}_{p}"
        mono, bi = f"{p}__{tok_name}", f"{ANCHOR}-{p}__{tok_name}"
        bpb_mono = _final_bpb(mono, src)
        bpb_bi = _final_bpb(bi, src)
        entry = {"same_script": LANGS[p].same_script_as_en,
                 "bpb_mono_final": bpb_mono, "bpb_bi_final": bpb_bi}
        if bpb_mono and bpb_bi:
            entry["bts_matched_total"] = (bpb_mono - bpb_bi) / bpb_mono
            bpb_mono_lang = _bpb_at_lang_tokens(mono, src, total_tokens * mix_prob)
            if bpb_mono_lang:
                entry["bpb_mono_at_lang_tokens"] = bpb_mono_lang
                entry["bts_matched_lang"] = (bpb_mono_lang - bpb_bi) / bpb_mono_lang
        rows[p] = entry
    return rows


def _penalty(rows: dict, key: str) -> float | None:
    same = [r[key] for r in rows.values() if r.get("same_script") and key in r]
    cross = [r[key] for r in rows.values() if not r.get("same_script") and key in r]
    if not same or not cross:
        return None
    return sum(same) / len(same) - sum(cross) / len(cross)


def run(flavor: str = "unigram", source_kind: str = "flores",
        total_tokens: float = 30e9, mix_prob: float = 0.5,
        out_dir: Path | None = None) -> dict:
    out_dir = ensure(Path(out_dir) if out_dir else RESULTS / "bts")
    conds = {c: compute(f"{flavor}_{c}", source_kind, mix_prob, total_tokens)
             for c in ("starved", "destarved")}
    inter = {}
    for key in ("bts_matched_total", "bts_matched_lang"):
        ps = {c: _penalty(conds[c], key) for c in conds}
        if ps["starved"] is not None and ps["destarved"] is not None:
            inter[key] = {"penalty_starved": ps["starved"],
                          "penalty_destarved": ps["destarved"],
                          "interaction": ps["starved"] - ps["destarved"]}
    result = {"flavor": flavor, "source": source_kind, "by_condition": conds,
              "interaction": inter}
    (out_dir / f"bts_{flavor}_{source_kind}.json").write_text(json.dumps(result, indent=2))

    md = [f"# BTS ({flavor}, eval on {source_kind})", ""]
    for c, rows in conds.items():
        md += [f"## {c}", "",
               "| partner | script | BPB mono | BPB bi | BTS (total) | BTS (lang) |",
               "|---|---|---|---|---|---|"]
        for p, r in rows.items():
            md.append("| {} | {} | {} | {} | {} | {} |".format(
                p, "same" if r["same_script"] else "cross",
                _f(r.get("bpb_mono_final")), _f(r.get("bpb_bi_final")),
                _f(r.get("bts_matched_total")), _f(r.get("bts_matched_lang"))))
        md.append("")
    md += ["## Interaction (same-script penalty - cross-script penalty)", ""]
    for key, v in inter.items():
        md.append(f"- **{key}**: penalty(starved)={v['penalty_starved']:.4f}, "
                  f"penalty(destarved)={v['penalty_destarved']:.4f}, "
                  f"**interaction={v['interaction']:.4f}**")
    md += ["", "> interaction >> 0  =>  cross-script penalty is a tokenizer-"
           "starvation artifact.", "> interaction ~ 0  =>  penalty persists "
           "under a fair tokenizer (genuine script effect)."]
    (out_dir / f"bts_{flavor}_{source_kind}.md").write_text("\n".join(md) + "\n")
    print(f"[bts] wrote {out_dir}/bts_{flavor}_{source_kind}.md")
    return result


def _f(x):
    return f"{x:.4f}" if isinstance(x, float) else "-"
