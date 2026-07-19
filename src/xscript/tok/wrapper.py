"""Uniform interface over the tokenizer flavors.

`unigram` loads a SentencePiece model; `bpe`/`pa` load a HuggingFace byte-level
BPE. Ids 0..3 are <unk>/<bos>/<eos>/<pad> in every flavor, so packing, training
and eval code never branches on flavor.
"""
import json
from functools import cached_property
from pathlib import Path

UNK_ID, BOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3


class Tok:
    def __init__(self, path: str | Path):
        self.dir = Path(path)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.flavor = self.meta["flavor"]
        self.condition = self.meta["condition"]
        self.name = f"{self.flavor}_{self.condition}"
        if self.flavor == "unigram":
            import sentencepiece as spm
            self._sp = spm.SentencePieceProcessor(model_file=str(self.dir / "sp.model"))
            self.vocab_size = self._sp.get_piece_size()
        else:
            from tokenizers import Tokenizer
            self._hf = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
            self.vocab_size = self._hf.get_vocab_size()

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        if self.flavor == "unigram":
            ids = self._sp.encode(text)
        else:
            ids = self._hf.encode(text).ids
        if bos:
            ids = [BOS_ID] + ids
        if eos:
            ids = ids + [EOS_ID]
        return ids

    def encode_batch(self, texts: list[str], bos: bool = False, eos: bool = False):
        if self.flavor == "unigram":
            batch = self._sp.encode(texts)
        else:
            batch = [e.ids for e in self._hf.encode_batch(texts)]
        if bos or eos:
            batch = [([BOS_ID] if bos else []) + ids + ([EOS_ID] if eos else [])
                     for ids in batch]
        return batch

    def decode(self, ids: list[int]) -> str:
        if self.flavor == "unigram":
            return self._sp.decode([i for i in ids if i > PAD_ID])
        return self._hf.decode([i for i in ids if i > PAD_ID], skip_special_tokens=True)

    # --- introspection for the analysis gate ---

    def piece(self, idx: int) -> str:
        """Raw vocabulary piece as stored (SP: '▁'-form / '<0xNN>'; BL: bytelevel-mapped)."""
        if self.flavor == "unigram":
            return self._sp.id_to_piece(idx)
        return self._id_to_piece_bl[idx]

    @cached_property
    def _id_to_piece_bl(self) -> list[str]:
        vocab = self._hf.get_vocab()
        pieces = [""] * self.vocab_size
        for p, i in vocab.items():
            pieces[i] = p
        return pieces

    @cached_property
    def _bl_byte_map(self) -> dict[str, int]:
        # inverse of the GPT-2 bytes<->unicode table used by ByteLevel
        bs = list(range(ord("!"), ord("~") + 1)) + \
             list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {chr(c): b for b, c in zip(bs, cs)}

    def piece_bytes(self, idx: int) -> bytes:
        """The exact bytes a vocab entry emits (specials -> b'')."""
        p = self.piece(idx)
        if p in ("<unk>", "<bos>", "<eos>", "<pad>"):
            return b""
        if self.flavor == "unigram":
            if len(p) == 6 and p.startswith("<0x") and p.endswith(">"):
                return bytes([int(p[3:5], 16)])  # byte-fallback piece
            return p.replace("▁", " ").encode("utf-8")
        return bytes(self._bl_byte_map[ch] for ch in p)

    def is_byte_piece(self, idx: int) -> bool:
        """True if this vocab entry is a raw-byte atom (SP fallback / BL base byte)."""
        p = self.piece(idx)
        if self.flavor == "unigram":
            return len(p) == 6 and p.startswith("<0x") and p.endswith(">")
        return idx >= 4 and len(self.piece_bytes(idx)) == 1
