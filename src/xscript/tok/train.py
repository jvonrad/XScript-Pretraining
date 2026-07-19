"""Train the study's tokenizers.

We pretrain models with TWO SentencePiece Unigram tokenizers only. The `bpe`
and `pa` flavors are trained purely as tokenizer-analysis comparators for the
fertility/allocation gate (`xscript tok-analyze`); no model run ever uses them.

MODEL-TRAINING tokenizers -- SentencePiece Unigram, character_coverage=0.999995,
byte fallback:
  unigram_starved   -- ATLAS-style replication arm: T=100 temperature mixture
                       over ~419 languages. Matches both ATLAS's ~uniform 420-
                       language mixture and the Unigram algorithm of the MADLAD-
                       400 lineage its tokenizer descends from.
  unigram_destarved -- the intervention arm: our 5 study languages only, byte-
                       premium content-aligned (equal *content*, not bytes, per
                       language; see data/tokcorpus.py). Same algorithm as the
                       starved arm, so the starved-vs-destarved contrast isolates
                       vocabulary allocation rather than confounding it with the
                       tokenizer algorithm.

On the algorithm choice: Unigram is MADLAD-400's confirmed algorithm (its
released 256k *model* tokenizer is SentencePiece Unigram). ATLAS's 64k tokenizer
is a SEPARATE artifact -- trained by the MADLAD-400 authors (Kudugunta et al.)
on the same T=100 recipe -- whose algorithm ATLAS does not state in-text, though
Unigram is the natural inference from that lineage. Do not conflate ATLAS's 64k
with MADLAD's 256k; they are different tokenizers.

ANALYSIS-ONLY comparators -- trained for the gate, never used to pretrain:
  bpe -- byte-level BPE (Whitespace + ByteLevel pre-tokenization) trained with
         HuggingFace `tokenizers`' Rust `BpeTrainer`. Quantifies how much the
         Unigram-vs-BPE algorithm choice alone moves fertility/allocation.
  pa  -- parity-aware byte-level BPE via swiss-ai/parity-aware-bpe's
         `parity_aware_learn_bpe.py` (window variant, for ZH), fertility-
         equalized over the 5-way-parallel FLORES+ dev set. Same byte-level
         alphabet as `bpe`; the merge criterion (parity-balanced vs frequency)
         is the only difference -> a clean upper bound on fertility
         equalization. Destarved only (it balances a fixed dev-language set).
         Uses the slow single-threaded reference trainer -- tolerable only
         because its corpus is 5 languages, not 419.

Every flavor exposes exactly `VOCAB_SIZE` pieces with our four specials at ids
0..3, so packed token ids stay uint16 and every downstream module stays flavor-
agnostic. VOCAB_SIZE is overridable via XSCRIPT_VOCAB for the CPU smoke test.
"""
import json
import os
import subprocess
from pathlib import Path

from ..langs import tok_name
from ..paths import TOK_CORPORA, tokenizer_dir, ensure
from ..data.tokcorpus import corpus_files

VOCAB_SIZE = int(os.environ.get("XSCRIPT_VOCAB", "65536"))
SPECIALS = ["<unk>", "<bos>", "<eos>", "<pad>"]  # ids 0..3 in every flavor
PA_REPO = "swiss-ai/parity-aware-bpe"


# --------------------------------------------------------------------------- #
# unigram (SentencePiece)
# --------------------------------------------------------------------------- #
def train_unigram(condition: str, seed: int = 42) -> Path:
    import sentencepiece as spm
    if hasattr(spm, "set_random_generator_seed"):
        spm.set_random_generator_seed(seed)   # not a TrainerSpec field in >=0.2
    files = corpus_files(condition)
    out = ensure(tokenizer_dir(tok_name("unigram", condition)))
    spm.SentencePieceTrainer.train(
        input=",".join(str(f) for f in files),
        model_prefix=str(out / "sp"),
        model_type="unigram",
        vocab_size=VOCAB_SIZE,
        character_coverage=0.999995,
        byte_fallback=True,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>", pad_piece="<pad>",
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        remove_extra_whitespaces=False,
        num_threads=max(1, (os.cpu_count() or 8) - 2),
    )
    _write_meta(out, "unigram", condition, files)
    return out


# --------------------------------------------------------------------------- #
# byte-level BPE + parity-aware BPE (swiss-ai/parity-aware-bpe)
# --------------------------------------------------------------------------- #
def _n_merges() -> int:
    # vocab = 4 specials + 256 byte-level base alphabet + merges
    return VOCAB_SIZE - len(SPECIALS) - 256


def train_bpe(condition: str) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    files = corpus_files(condition)
    out = ensure(tokenizer_dir(tok_name("bpe", condition)))

    tok = Tokenizer(models.BPE(unk_token=None, fuse_unk=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace(), pre_tokenizers.ByteLevel(use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIALS,                       # ids 0..3, in order
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # full 256 bytes
        show_progress=True,
    )
    tok.train([str(f) for f in files], trainer)
    tok.save(str(out / "tokenizer.json"))
    _write_meta(out, "bpe", condition, files,
                extra={"vocab_size_actual": tok.get_vocab_size(),
                       "source": "huggingface-tokenizers-bpe"})
    return out


def train_pa(condition: str = "destarved", variant: str = "window") -> Path:
    if condition != "destarved":
        raise ValueError("parity-aware BPE is destarved-only (see langs.tok_conditions)")
    inputs = corpus_files("destarved")            # one file per study language
    dev = _write_pa_dev(inputs)                    # aligned FLORES+ dev per lang
    out = ensure(tokenizer_dir(tok_name("pa", condition)))
    merges = out / "merges.raw.txt"
    # parity-aware's multi-worker vocab builder is broken in the released
    # version (pickle.load on a text-mode file), so force single-worker.
    pa_workers = os.environ.get("XSCRIPT_PA_WORKERS", "1")
    cmd = ["python", "-m", "parity_aware_bpe.parity_aware_learn_bpe",
           "--variant", variant, "--symbols", str(_n_merges()),
           "--num-workers", pa_workers, "--output", str(merges),
           "--input", *[str(f) for f in inputs],
           "--dev", *[str(f) for f in dev]]
    _run(cmd)
    _bytelevel_from_merges(merges, out, "pa", condition, inputs)
    return out


def _write_pa_dev(inputs) -> list[Path]:
    """FLORES+ dev text per language, in the SAME order as `inputs` (stem=code)."""
    from .. import flores
    d = ensure(TOK_CORPORA / "pa_dev")
    dev = []
    for f in inputs:
        code = f.stem
        sents = list(flores.load(code, "dev").values())
        p = d / f"{code}.dev.txt"
        p.write_text("\n".join(sents) + "\n", encoding="utf-8")
        dev.append(p)
    return dev


def _bytelevel_from_merges(merges_path: Path, out: Path, flavor: str,
                           condition: str, corpus_files_used) -> None:
    """Merge rules -> HuggingFace byte-level BPE tokenizer, exactly VOCAB_SIZE."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders

    lines = [l.strip() for l in merges_path.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    if lines and lines[0].startswith("#version"):
        lines = lines[1:]

    vocab: dict[str, int] = {s: i for i, s in enumerate(SPECIALS)}     # 0..3
    for ch in pre_tokenizers.ByteLevel.alphabet():                     # 256 bytes
        vocab.setdefault(ch, len(vocab))
    keep = max(0, VOCAB_SIZE - len(vocab))                             # merges budget
    merges: list[tuple[str, str]] = []
    for line in lines:
        if len(merges) >= keep:
            break
        a, b = line.split(" ")
        if a not in vocab or b not in vocab:      # order guarantees this won't hit
            continue
        merges.append((a, b))
        vocab.setdefault(a + b, len(vocab))

    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges,
                               unk_token=None, fuse_unk=False))
    # EXACT pre-tokenizer/decoder the repo trains and loads with (byte-level)
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace(), pre_tokenizers.ByteLevel(use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.save(str(out / "tokenizer.json"))
    _write_meta(out, flavor, condition, corpus_files_used,
                extra={"vocab_size_actual": tok.get_vocab_size(),
                       "n_merges": len(merges), "source": PA_REPO})


def _run(cmd, shell: bool = False) -> None:
    print(f"[tok] $ {cmd if shell else ' '.join(cmd)}")
    subprocess.run(cmd, shell=shell, check=True)


# --------------------------------------------------------------------------- #
def _write_meta(out: Path, flavor: str, condition: str, files, extra=None) -> None:
    meta = {
        "flavor": flavor,
        "condition": condition,
        "vocab_size": VOCAB_SIZE,
        "specials": SPECIALS,
        "corpus_files": [str(f) for f in files],
    }
    if extra:
        meta.update(extra)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[tok] trained {flavor}_{condition} -> {out}")


def train(flavor: str, condition: str) -> Path:
    if flavor == "unigram":
        return train_unigram(condition)
    if flavor == "bpe":
        return train_bpe(condition)
    if flavor == "pa":
        return train_pa(condition)
    raise ValueError(f"unknown flavor {flavor!r} (want unigram|bpe|pa)")
