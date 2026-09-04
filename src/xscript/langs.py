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

# Corroboration conditions (2026-09-04), unigram only. They sit BETWEEN the
# two headline arms on the vocabulary-competition axis and are used to check
# that the alignment (CLAUDE.md 6b), consistency and parametric-sharing (6j)
# results are not specific to the 419-vs-5 contrast:
#   50lang  -- 50 languages: our 5 plus the 45 largest remaining FineWeb2
#              configs by volume; English fixed at EN_SHARE_50LANG (5%) of
#              corpus bytes, the other 49 uniform in bytes.
#   bi_<X>  -- a 2-language tokenizer for the en-X bilingual (X in PARTNERS),
#              byte-premium content-aligned between en and X like destarved.
#              Only meaningful for the mixtures {en, X, en-X}; see
#              condition_langs() and runmatrix.
TOK_CONDITION_50LANG = "50lang"
NLANG_CONDITIONS = ["50lang", "20lang", "10lang"]   # "<N>lang": N-way competition, en 5%
EN_SHARE_50LANG = 0.05                    # English share for every <N>lang
BILINGUAL_TOK_CONDITIONS: dict[str, tuple[str, str]] = {
    f"bi_{p}": (ANCHOR, p) for p in PARTNERS}
EXTRA_TOK_CONDITIONS = [*NLANG_CONDITIONS, *BILINGUAL_TOK_CONDITIONS]


def nlang_of(condition: str) -> int | None:
    """50 for "50lang", 20 for "20lang", None for anything else."""
    import re
    m = re.fullmatch(r"(\d+)lang", condition)
    return int(m.group(1)) if m else None
ALL_TOK_CONDITIONS = TOK_CONDITIONS + EXTRA_TOK_CONDITIONS


def tok_name(flavor: str, condition: str) -> str:
    return f"{flavor}_{condition}"


def tok_conditions(flavor: str, extra: bool = False) -> list[str]:
    """Conditions a flavor is trained in. `extra=True` adds the corroboration
    conditions (unigram only) -- off by default so the headline matrix,
    the gate report and every existing caller are unchanged."""
    if flavor == "pa":
        return ["destarved"]
    if extra and flavor == "unigram":
        return ALL_TOK_CONDITIONS
    return TOK_CONDITIONS


def condition_langs(condition: str) -> list[str]:
    """Study languages a tokenizer condition is meant to train models on."""
    if condition in BILINGUAL_TOK_CONDITIONS:
        return list(BILINGUAL_TOK_CONDITIONS[condition])
    return list(LANGS)


def all_tok_names(extra: bool = False) -> list[str]:
    return [tok_name(f, c) for f in TOK_FLAVORS for c in tok_conditions(f, extra)]
