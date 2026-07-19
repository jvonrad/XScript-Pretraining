"""Language metadata: the single source of truth for the 5 study languages.

Design (thesis-plan.txt): EN anchor; DE, FR same-script (Latin); AR, ZH
cross-script. FLORES+ uses `cmn_Hans` for Simplified-Mandarin while
FineWeb2-HQ labels its Chinese subset `cmn_Hani` (Han script umbrella,
overwhelmingly Simplified web text) -- both refer to the same language here.
"""
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class Lang:
    code: str            # our short code
    name: str
    script: str          # ISO 15924 of the training data
    same_script_as_en: bool
    flores_code: str     # file stem in openlanguagedata/flores_plus (dev/<code>.jsonl)
    fineweb_repo: str    # HF dataset repo for model-training text (quality-filtered -HQ)
    fineweb_subdir: str  # language-script config/subdir; also the raw FineWeb2 config
                         # name used for the (unfiltered) tokenizer-training corpus


LANGS: dict[str, Lang] = {
    "en": Lang("en", "English", "Latn", True, "eng_Latn",
               "epfml/FineWeb-HQ", "data"),
    "de": Lang("de", "German", "Latn", True, "deu_Latn",
               "epfml/FineWeb2-HQ", "deu_Latn"),
    "fr": Lang("fr", "French", "Latn", True, "fra_Latn",
               "epfml/FineWeb2-HQ", "fra_Latn"),
    "ar": Lang("ar", "Arabic", "Arab", False, "arb_Arab",
               "epfml/FineWeb2-HQ", "arb_Arab"),
    "zh": Lang("zh", "Chinese", "Hans", False, "cmn_Hans",
               "epfml/FineWeb2-HQ", "cmn_Hani"),
}

ANCHOR = "en"
PARTNERS = ["de", "fr", "ar", "zh"]

# Run matrix: 5 monolingual + all 10 pairwise bilingual mixtures, x 2
# tokenizer conditions. PARTNERS remains the EN-centric BTS comparison set.
MONOLINGUAL_RUNS = [(l,) for l in LANGS]
BILINGUAL_RUNS = list(combinations(LANGS, 2))

# Tokenizers (see xscript.tok.train):
#   flavor:    "unigram" = SentencePiece Unigram + byte fallback (ATLAS's actual
#                          algorithm; the faithful replication point)
#              "bpe"     = byte-level BPE, classical (swiss-ai/parity-aware-bpe
#                          learn_bpe.py) -- the baseline parity-aware modifies
#              "pa"      = parity-aware byte-level BPE (same repo), fertility-
#                          equalized over a multi-parallel FLORES+ dev set
#   condition: "starved"   = raw FineWeb/FineWeb2, T=100 sampling over ~419
#                            languages (matching MADLAD-400/ATLAS's scale)
#              "destarved" = raw FineWeb/FineWeb2, our 5 languages, byte-
#                            premium-adjusted
#
# Tokenizer corpus is FineWeb-family (not MADLAD-400) so that it is in the same
# corpus family as the model-training pools (also FineWeb-family) under both
# conditions -- avoiding a tokenizer-corpus-vs-model-corpus domain mismatch
# that could otherwise hit AR/ZH differently than DE/FR. This is a deliberate
# deviation from ATLAS's literal MADLAD-trained tokenizer; see data/tokcorpus.py.
#
# Parity-aware balances a fixed multi-parallel dev set (our 5 languages), so it
# only has a destarved form; a 420-language "starved" version is undefined. It
# is therefore an analysis-only fidelity reference, NOT a model-training flavor
# (the starved-vs-destarved contrast needs a flavor with both conditions).
TOK_FLAVORS = ["unigram", "bpe", "pa"]
TOK_CONDITIONS = ["starved", "destarved"]
MODEL_FLAVORS = ["unigram", "bpe"]  # eligible as the model-training tokenizer


def tok_name(flavor: str, condition: str) -> str:
    return f"{flavor}_{condition}"


def tok_conditions(flavor: str) -> list[str]:
    return ["destarved"] if flavor == "pa" else TOK_CONDITIONS


def all_tok_names() -> list[str]:
    return [tok_name(f, c) for f in TOK_FLAVORS for c in tok_conditions(f)]
