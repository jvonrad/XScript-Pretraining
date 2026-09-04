#!/usr/bin/env python
"""Do culturally-loaded words drift APART across languages as training goes on?

THE HYPOTHESIS (2026-09-03, user-proposed)
==========================================
Under the FAIR tokenizer, with the language centroid removed, cross-lingual
alignment at early/mid layers DECLINES over training. One reading: the model
is not losing alignment, it is learning language-SPECIFIC lexical semantics --
`debt` and `Schuld` name overlapping but non-identical concepts (German
`Schuld` also carries "guilt"), so a better model should place them FURTHER
apart once the language-identity offset is gone.

That predicts an INTERACTION, not a main effect: culturally-loaded pairs
should separate more (or converge less) over budget than culturally-neutral
pairs. A main effect alone proves nothing -- if every pair drifts apart, the
finding is "centered alignment declines with training", which §6b already
documents and which says nothing about culture.

⛔ THREE WAYS THIS MEASURES ITSELF INSTEAD OF THE MODEL
=======================================================
1. **The centroid must not come from the probe words.** `alignment._center`
   subtracts the mean of whatever set it is handed. If the nuanced and control
   sets each contributed to their own centroid, a systematic difference
   between the sets would partly cancel into the centroid and the contrast
   would be self-referential. The default centroid (`--centroid flores`) is
   therefore the **FLORES+ language centroid**: the mean sentence embedding
   over FLORES+ dev (n=997) per language and layer, which is the same object
   §6b centres on and is fixed independently of the probe lists.

   `--centroid filler` swaps in a frequent-token vocabulary (probe words
   excluded) as a robustness check. Note the centroid choice is largely
   INERT for the headline contrast: whatever it is, it is subtracted from
   both groups within a language, so it shifts absolute cosines but mostly
   cancels from the nuanced-vs-control difference. It matters for reading
   raw alignment levels, not for the interaction.

   ⚠️ FLORES is sentence-level while the probes are bare words, so the
   FLORES centroid carries a sentence-vs-word domain offset. That offset is
   common to both groups (see above), but it does mean absolute cosines
   under `--centroid flores` are not directly comparable to §6b's
   sentence-on-sentence numbers.

2. **The groups must be confounder-matched, or the confounders reported.**
   Token count, character length and orthographic identity all move
   similarity on their own. A nuanced set that happens to be longer or more
   fragmented would separate for mechanical reasons. This script MEASURES all
   three per group and prints them; it does not silently assume balance.
   Cognates (identical strings across languages) are flagged separately --
   they are trivially aligned and would dilute any real effect.

3. **Word choice is researcher discretion.** The lists below are fixed in
   source and were written before any model was run. They are not tuned to
   the result. They are also small (~10-14 per group per language), so treat
   this as exploratory: it can motivate a pre-registered study, it cannot
   settle one on its own.

METHOD
======
For each bilingual checkpoint (EN + partner) at several mid-stable budgets:
  * embed each probe word ALONE (BOS + word), mean-pooling `layer_reps` over
    the word's own subword positions -> (n_layers+1, dim) per word;
  * per (language, layer), subtract the FLORES+ language centroid and
    re-normalise -- the same centre-then-normalise as `alignment._center`,
    but with a fixed, probe-independent centroid;
  * cosine(EN word, partner word) per layer;
  * track that against training budget, nuanced vs control.

The reported quantity is the per-layer slope of cos vs log(budget) for each
group, and the DIFFERENCE of slopes (control - nuanced). Positive difference
= nuanced pairs separate faster than neutral ones, i.e. the hypothesis.

Bare words, not sentences, on purpose: a template context would put the same
carrier text around every item and risk measuring the template. The cost is
that this is lexical rather than contextual semantics -- a real limitation,
stated rather than hidden.

    python probe_cultural_drift.py --repo jvonrad/xscript-eval \\
        --runs en-de-fair-2b en-de-fair-5b en-de-fair-10b en-de-fair-23b \\
        --workdir $WORK --device cpu
"""
import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# --- probe sets -------------------------------------------------------------
# NUANCED: the partner word's conventional sense is documented as broader,
# narrower or differently-valenced than the English gloss (`Schuld` = debt AND
# guilt; `Heimat` carries belonging/nationhood English `home` does not;
# `guanxi` is a social-obligation network, not just "relationship").
# CONTROL: concrete, low cultural load, intended to translate near-1:1.
# Both lists are per-partner because cultural loading is language-specific --
# there is no German analogue of `guanxi` to hold constant.
PROBES = {
    "de": {
        "nuanced": [("debt", "Schuld"), ("home", "Heimat"), ("spirit", "Geist"),
                    ("luck", "Glück"), ("law", "Recht"), ("power", "Macht"),
                    ("order", "Ordnung"), ("guilt", "Schuld"),
                    ("education", "Bildung"), ("shame", "Scham"),
                    ("duty", "Pflicht"), ("mind", "Geist")],
        "control": [("water", "Wasser"), ("stone", "Stein"), ("iron", "Eisen"),
                    ("salt", "Salz"), ("tree", "Baum"), ("river", "Fluss"),
                    ("hand", "Hand"), ("moon", "Mond"), ("snow", "Schnee"),
                    ("window", "Fenster"), ("milk", "Milch"), ("finger", "Finger")],
    },
    "fr": {
        "nuanced": [("freedom", "liberté"), ("secularism", "laïcité"),
                    ("spirit", "esprit"), ("law", "droit"),
                    ("homeland", "patrie"), ("shame", "honte"),
                    ("happiness", "bonheur"), ("power", "pouvoir"),
                    ("citizen", "citoyen"), ("taste", "goût"),
                    ("duty", "devoir"), ("solidarity", "solidarité")],
        "control": [("water", "eau"), ("stone", "pierre"), ("iron", "fer"),
                    ("salt", "sel"), ("tree", "arbre"), ("river", "rivière"),
                    ("hand", "main"), ("moon", "lune"), ("snow", "neige"),
                    ("window", "fenêtre"), ("milk", "lait"), ("finger", "doigt")],
    },
    "ar": {
        "nuanced": [("honor", "شرف"), ("forbidden", "حرام"), ("trust", "أمانة"),
                    ("patience", "صبر"), ("dignity", "كرامة"),
                    ("shame", "عيب"), ("fate", "قدر"), ("charity", "زكاة"),
                    ("community", "أمة"), ("modesty", "حياء"),
                    ("struggle", "جهاد"), ("lawful", "حلال")],
        "control": [("water", "ماء"), ("stone", "حجر"), ("iron", "حديد"),
                    ("salt", "ملح"), ("tree", "شجرة"), ("river", "نهر"),
                    ("hand", "يد"), ("moon", "قمر"), ("snow", "ثلج"),
                    ("window", "نافذة"), ("milk", "حليب"), ("finger", "إصبع")],
    },
    "zh": {
        "nuanced": [("relationship", "关系"), ("face", "面子"),
                    ("filial piety", "孝"), ("fate", "缘分"),
                    ("courtesy", "客气"), ("harmony", "和谐"),
                    ("effort", "辛苦"), ("hardship", "吃苦"),
                    ("propriety", "礼"), ("humaneness", "仁"),
                    ("obligation", "人情"), ("hot", "上火")],
        "control": [("water", "水"), ("stone", "石头"), ("iron", "铁"),
                    ("salt", "盐"), ("tree", "树"), ("river", "河"),
                    ("hand", "手"), ("moon", "月亮"), ("snow", "雪"),
                    ("window", "窗户"), ("milk", "牛奶"), ("finger", "手指")],
    },
}

FILLER_N = 200          # vocabulary size for the language centroid
FILLER_MIN_CHARS = 2


def filler_words(tok, sentences, exclude: set[str], n=FILLER_N):
    """Frequent decoded tokenizer pieces -- a probe-INDEPENDENT vocabulary for
    the language centroid. Built from the model's own tokenizer so it works
    for zh (no whitespace) exactly as for the Latin-script languages."""
    cnt = Counter()
    for s in sentences:
        for i in tok.encode(s):
            cnt[i] += 1
    out, seen = [], set()
    for i, _ in cnt.most_common():
        try:
            w = tok.decode([i]).strip()
        except Exception:
            continue
        # Reject punctuation/mixed pieces: the centroid is meant to be the
        # language's LEXICAL centre, and frequent tokens are otherwise
        # dominated by quotes and full stops (observed: '。”', ':“'), whose
        # geometry is not what "language identity" should mean here.
        if (len(w) < FILLER_MIN_CHARS or w in seen or w.lower() in exclude
                or any(c.isdigit() for c in w)
                or not all(c.isalpha() for c in w)):
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= n:
            break
    return out


def embed_words(model, tok, words, device, batch=16):
    """(n_words, n_layers+1, dim): mean over each word's OWN subword positions
    (BOS excluded), from one forward pass per batch."""
    import torch
    reps = []
    for s0 in range(0, len(words), batch):
        chunk = words[s0:s0 + batch]
        enc = [tok.encode(w, bos=True) for w in chunk]
        width = max(len(e) for e in enc)
        x = torch.zeros(len(enc), width, dtype=torch.long)
        for r, e in enumerate(enc):
            x[r, :len(e)] = torch.tensor(e)
        with torch.no_grad():
            L = model.layer_reps(x.to(device))        # (n_layers+1, B, T, d)
        L = L.float().cpu()
        for r, e in enumerate(enc):
            # positions 1..len(e)-1 are the word's own pieces; 0 is BOS
            sl = L[:, r, 1:len(e), :]
            reps.append(sl.mean(1).numpy())           # (n_layers+1, d)
    import numpy as np
    return np.stack(reps)


def flores_centroid(model, tok, sentences, device, batch=16, max_tok=128):
    """(n_layers+1, dim): the FLORES+ language centroid -- mean over sentence
    embeddings, each mean-pooled over its own real token positions (BOS
    excluded, matching `embed_words`). This is the object §6b's `centered`
    variant removes; estimating it from 997 parallel sentences makes it both
    probe-independent and far better sampled than any hand-built word list."""
    import numpy as np
    import torch
    acc, n = None, 0
    for s0 in range(0, len(sentences), batch):
        chunk = sentences[s0:s0 + batch]
        enc = [tok.encode(t, bos=True)[:max_tok] for t in chunk]
        enc = [e for e in enc if len(e) > 1]
        if not enc:
            continue
        width = max(len(e) for e in enc)
        x = torch.zeros(len(enc), width, dtype=torch.long)
        for r, e in enumerate(enc):
            x[r, :len(e)] = torch.tensor(e)
        with torch.no_grad():
            L = model.layer_reps(x.to(device)).float().cpu().numpy()
        for r, e in enumerate(enc):
            v = L[:, r, 1:len(e), :].mean(1)          # (n_layers+1, d)
            acc = v if acc is None else acc + v
            n += 1
    if not n:
        raise RuntimeError("no usable FLORES sentences for the centroid")
    return acc / n


def centred(E, centroid):
    """alignment._center's operation, but with an EXTERNALLY supplied centroid
    so the probe groups never define their own reference frame."""
    import numpy as np
    C = E - centroid
    n = np.linalg.norm(C, axis=-1, keepdims=True)
    return C / np.maximum(n, 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--centroid", default="flores", choices=["flores", "filler"],
                    help="what 'language identity' is estimated from. flores "
                         "(default): the FLORES+ dev language centroid, as in "
                         "§6b. filler: a frequent-token vocabulary, probe "
                         "words excluded. Robustness check, not a rerun.")
    ap.add_argument("--centroid-sents", type=int, default=500,
                    help="FLORES+ dev sentences behind the centroid. MEASURED "
                         "(2026-09-03, zh-fair-12b): two DISJOINT 150-sentence "
                         "centroids agree at cos 0.996-0.999 for layers 4-16 "
                         "but only 0.969 at LAYER 0, and swapping between them "
                         "moves a probe cosine by 0.014 -- the same order as "
                         "the effects being looked for. That is an argument "
                         "for more sentences, but mostly it is an argument for "
                         "a FIXED basis: every checkpoint here centres on the "
                         "SAME sentences, so centroid sampling error is a "
                         "constant per-language offset and cancels from the "
                         "trajectory (the quantity of interest). It survives "
                         "in absolute cosines, so do not read those too "
                         "finely, least of all at layer 0.")
    ap.add_argument("--out", type=Path,
                    default=REPO / "results" / "cultural_drift" / "raw.json")
    args = ap.parse_args()

    import numpy as np
    work = args.workdir.resolve()
    scratch = work / "xscript"
    os.environ["XSCRIPT_SCRATCH"] = str(scratch)
    sys.path.insert(0, str(REPO / "src"))
    from huggingface_hub import hf_hub_download
    import torch
    from xscript.model import ModelConfig, Transformer
    from xscript.tok.wrapper import Tok
    from xscript.paths import tokenizer_dir
    from xscript import flores

    models = json.loads((work / "_repo" / "models.json").read_text()) \
        if (work / "_repo" / "models.json").exists() else \
        json.loads(Path(hf_hub_download(repo_id=args.repo, filename="models.json")).read_text())
    device = torch.device(args.device)
    out_rows = []

    for run in args.runs:
        langs = models[run]["langs"]
        partner = [l for l in langs if l != "en"]
        if "en" not in langs or len(partner) != 1 or partner[0] not in PROBES:
            print(f"[drift] skip {run}: needs an EN-anchored bilingual, got {langs}")
            continue
        pl = partner[0]
        ck_path = scratch / f"runs/{run}/checkpoints/final.pt"
        if not ck_path.exists():
            print(f"[drift] skip {run}: checkpoint not staged at {ck_path}")
            continue
        print(f"\n===== {run} (en-{pl}, tok={models[run]['tok']}) =====", flush=True)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = Transformer(ModelConfig(**ck["cfg"]["model"])).to(device).eval()
        model.load_state_dict(ck["model"])
        tok = Tok(tokenizer_dir(models[run]["tok"]))

        pairs = {g: PROBES[pl][g] for g in ("nuanced", "control")}
        probe_en = {w for g in pairs for w, _ in pairs[g]}
        probe_pl = {w for g in pairs for _, w in pairs[g]}
        sents = flores.load_parallel(["en", pl], "dev")
        cent = {}
        if args.centroid == "flores":
            for lg in ("en", pl):
                cent[lg] = flores_centroid(model, tok, sents[lg][:args.centroid_sents],
                                           device)
            print(f"[drift] FLORES+ centroid from "
                  f"{min(args.centroid_sents, len(sents['en']))} sentences/lang")
        else:
            fill = {"en": filler_words(tok, sents["en"], {w.lower() for w in probe_en}),
                    pl: filler_words(tok, sents[pl], {w.lower() for w in probe_pl})}
            print(f"[drift] filler vocab: en={len(fill['en'])}, {pl}={len(fill[pl])}")
            for lg in ("en", pl):
                cent[lg] = embed_words(model, tok, fill[lg], device).mean(0)

        for group, plist in pairs.items():
            we = [a for a, _ in plist]
            wp = [b for _, b in plist]
            Ee = centred(embed_words(model, tok, we, device), cent["en"])
            Ep = centred(embed_words(model, tok, wp, device), cent[pl])
            n_layers = Ee.shape[1]
            for i, (a, b) in enumerate(plist):
                cos = (Ee[i] * Ep[i]).sum(-1)                  # per layer
                out_rows.append({
                    "run": run, "partner": pl, "tok": models[run]["tok"],
                    "centroid": args.centroid,
                    "group": group, "en": a, "partner_word": b,
                    "identical": a.strip().lower() == b.strip().lower(),
                    "n_tok_en": len(tok.encode(a)), "n_tok_partner": len(tok.encode(b)),
                    "chars_en": len(a), "chars_partner": len(b),
                    "cos_by_layer": [float(c) for c in cos],
                })
            print(f"[drift]   {group}: {len(plist)} pairs, "
                  f"mid-layer mean cos = "
                  f"{np.mean([r['cos_by_layer'][n_layers//2] for r in out_rows if r['run']==run and r['group']==group]):+.4f}")
        del model

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_rows, indent=1))
    print(f"\n[drift] wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
