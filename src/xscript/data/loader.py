"""Deterministic streaming loader over packed uint16 shards.

- One PackedStream per language: contiguous (seq_len+1)-token windows,
  epoch-level window permutation (seeded), transparent epoch rollover with
  epoch counting (matters for AR, whose pool may be smaller than the budget).
- MixedStream: bilingual token-level mixing. For every global sample slot a
  stateless hash of (seed, slot) picks the language, so the mixture is exactly
  reproducible, independent of world size, and resumable from a slot counter.

Mixing mode (thesis decision, configurable in the run config):
  token-level p=0.5 matches ATLAS's compute-matched framing: each language
  contributes half the training *tokens*; under a starved tokenizer the
  cross-script partner therefore sees less *content* -- that is exactly the
  effect under audit. `content` mode instead sets p from the byte premiums so
  content is matched and token counts differ.
"""
import json
from pathlib import Path

import numpy as np

from ..paths import shard_dir


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


class PackedStream:
    def __init__(self, lang: str, tok_name: str, seq_len: int, seed: int):
        self.lang, self.seq_len, self.seed = lang, seq_len, seed
        d = shard_dir(lang, tok_name)
        index = json.loads((d / "index.json").read_text())
        # Pool checkpoints can create complete but empty zstd shards. Packing
        # those historically left zero-token .bin entries in index.json; mmap
        # cannot open a zero-byte file when a window crosses that boundary.
        # They carry no data, so filter them losslessly at load time.
        names = [name for name in sorted(index) if index[name] > 0]
        self.paths = [d / name for name in names]
        self.lens = np.array([index[name] for name in names], dtype=np.int64)
        self.cum = np.concatenate([[0], np.cumsum(self.lens)])
        self.total_tokens = int(self.cum[-1])
        self.n_windows = self.total_tokens // (seq_len + 1)
        if self.n_windows == 0:
            raise RuntimeError(f"shards for {lang}/{tok_name} too small")
        self._mm = [None] * len(self.paths)
        self._epoch = -1
        self._perm = None
        self.consumed = 0  # windows consumed (resume state)

    def _memmap(self, i: int):
        if self._mm[i] is None:
            self._mm[i] = np.memmap(self.paths[i], dtype=np.uint16, mode="r")
        return self._mm[i]

    def _window(self, w: int) -> np.ndarray:
        start = w * (self.seq_len + 1)
        i = int(np.searchsorted(self.cum, start, side="right") - 1)
        out = np.empty(self.seq_len + 1, dtype=np.uint16)
        filled = 0
        while filled < self.seq_len + 1:
            off = start + filled - self.cum[i]
            take = int(min(self.lens[i] - off, self.seq_len + 1 - filled))
            out[filled:filled + take] = self._memmap(i)[off:off + take]
            filled += take
            i += 1
        return out

    def get(self, k: int) -> np.ndarray:
        """k-th window in the global consumption order (deterministic)."""
        epoch, pos = divmod(k, self.n_windows)
        if epoch != self._epoch:
            rng = np.random.default_rng([self.seed, epoch, 0xDA7A])
            self._perm = rng.permutation(self.n_windows)
            self._epoch = epoch
        return self._window(int(self._perm[pos]))

    def next(self) -> np.ndarray:
        x = self.get(self.consumed)
        self.consumed += 1
        return x

    @property
    def epochs(self) -> float:
        return self.consumed / self.n_windows

    def state_dict(self):
        return {"consumed": self.consumed}

    def load_state_dict(self, s):
        self.consumed = s["consumed"]


class MixedStream:
    """Token-level language mixing across one or two PackedStreams."""

    def __init__(self, langs: list[str], tok_name: str, seq_len: int, seed: int,
                 probs: list[float] | None = None):
        self.langs = langs
        self.streams = {l: PackedStream(l, tok_name, seq_len, seed + j)
                        for j, l in enumerate(langs)}
        self.probs = probs or [1.0 / len(langs)] * len(langs)
        assert abs(sum(self.probs) - 1.0) < 1e-6
        self.seed = seed
        self.slot = 0  # global sample counter (resume state)

    def _choose(self, slot: int) -> str:
        if len(self.langs) == 1:
            return self.langs[0]
        u = _splitmix64(self.seed * 0x100000001 + slot) / 2**64
        acc = 0.0
        for l, p in zip(self.langs, self.probs):
            acc += p
            if u < acc:
                return l
        return self.langs[-1]

    def next_batch(self, n: int) -> tuple[np.ndarray, dict[str, int]]:
        """n windows -> array (n, seq_len+1); also per-language window counts."""
        rows, counts = [], {l: 0 for l in self.langs}
        for _ in range(n):
            l = self._choose(self.slot)
            rows.append(self.streams[l].next())
            counts[l] += 1
            self.slot += 1
        return np.stack(rows), counts

    def draw(self) -> tuple[str, int]:
        """Advance one global slot; return (lang, occurrence-index) WITHOUT reading.

        Occurrence-index is the k-th window of that language in global order, so
        stream.get(occ) is a pure deterministic lookup. All DDP ranks call draw()
        for every slot (keeping counters identical) but only materialise their own
        rows via streams[lang].get(occ) -- giving each rank a disjoint, reproducible
        1/world shard of the global batch.
        """
        l = self._choose(self.slot)
        occ = self.streams[l].consumed
        self.streams[l].consumed += 1
        self.slot += 1
        return l, occ

    def rank_batch(self, n_global: int, rank: int, world: int
                   ) -> tuple[np.ndarray, dict[str, int]]:
        """n_global windows drawn globally; rows n_global/world owned by `rank`."""
        rows, counts = [], {l: 0 for l in self.langs}
        for j in range(n_global):
            l, occ = self.draw()
            if j % world == rank:
                rows.append(self.streams[l].get(occ))
                counts[l] += 1
        return np.stack(rows), counts

    def skip_to(self, slot: int) -> None:
        """Fast-forward mixture state without reading data (for resume)."""
        assert slot >= self.slot
        for s in range(self.slot, slot):
            self.streams[self._choose(s)].consumed += 1
        self.slot = slot

    def state_dict(self):
        return {"slot": self.slot,
                "streams": {l: s.state_dict() for l, s in self.streams.items()}}

    def load_state_dict(self, s):
        self.slot = s["slot"]
        for l, st in s["streams"].items():
            self.streams[l].load_state_dict(st)

    def stats(self) -> dict:
        return {l: {"windows": s.consumed, "epochs": round(s.epochs, 4)}
                for l, s in self.streams.items()}
