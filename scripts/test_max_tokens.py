#!/usr/bin/env python
"""Regression test for `train.max_tokens` (the GPU-hour budget cap).

scripts/smoke.py is the usual end-to-end check, but it cannot serve here for
two reasons: it trains the `pa` (parity-aware BPE) tokenizer, whose learner is
the optional `[tok]` git dependency and is absent from the container runtime;
and it pins its own scratch dir and writes a pool `stats.json` without the
`budget_bytes` key `data.pack.pack()` requires. `pa` is an analysis-only
comparator no model is ever trained with, so rather than install it and patch
fixtures, this builds its own minimal fixture -- a packed uint16 shard of
random ids -- and exercises only the code path `max_tokens` touches.

What it proves, on the real Trainer over real packed shards:
  1. an uncapped run ends at the schedule's own total (warmup+stable+decay);
  2. a capped run ends at the cap instead;
  3. the cap does NOT perturb the LR schedule -- the capped run's final LR is
     still peak, i.e. it stopped mid-stable rather than being cooled early.

(3) is the whole point. Shortening `stable_tokens` would also stop the run
early, but would start the decay early too and de-match every checkpoint from
the runs it is meant to be compared against (CLAUDE.md 6's LR-state confound).

    python scripts/test_max_tokens.py
"""
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

VOCAB = 8192
SHARD_TOKENS = 400_000


def build_fixture(scratch: Path, tok_name: str, lang: str) -> None:
    """A packed shard of random ids + a tokenizer dir (Trainer.evaluate always
    constructs a Tok, even when no eval sources exist)."""
    sd = scratch / "shards" / f"{lang}__{tok_name}"
    sd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ids = rng.integers(4, VOCAB, size=SHARD_TOKENS, dtype=np.uint16)
    ids.tofile(sd / "shard_00000.bin")
    (sd / "index.json").write_text(json.dumps({"shard_00000.bin": int(SHARD_TOKENS)}))

    td = scratch / "tokenizers" / tok_name
    td.mkdir(parents=True, exist_ok=True)
    real = Path(os.environ.get("XS_REAL_TOK", "")) if os.environ.get("XS_REAL_TOK") else None
    if real and (real / "sp.model").exists():
        shutil.copy(real / "sp.model", td / "sp.model")
        meta = json.loads((real / "meta.json").read_text())
    else:                                    # minimal SP model trained inline
        import sentencepiece as spm
        corpus = td / "corpus.txt"
        corpus.write_text("\n".join(" ".join(
            f"w{rng.integers(0, 400)}" for _ in range(12)) for _ in range(4000)))
        spm.SentencePieceTrainer.train(
            input=str(corpus), model_prefix=str(td / "sp"), vocab_size=1000,
            model_type="unigram", character_coverage=0.9995,
            unk_id=0, bos_id=1, eos_id=2, pad_id=3)
        meta = {}
    meta.update({"flavor": "unigram", "condition": tok_name.split("_", 1)[1],
                 "vocab_size": VOCAB, "specials": ["<unk>", "<bos>", "<eos>", "<pad>"]})
    (td / "meta.json").write_text(json.dumps(meta))


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="xs_maxtok_"))
    os.environ["XSCRIPT_SCRATCH"] = str(scratch)
    os.environ["XSCRIPT_RESULTS"] = str(scratch / "results")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    print(f"[test] scratch = {scratch}")

    from xscript import _yaml
    from xscript.runmatrix import all_runs
    from xscript.schedule import lr_at, total_tokens
    from xscript.train import run_from_config

    build_fixture(scratch, "unigram_destarved", "en")
    print("[test] fixture ready")

    base = _yaml.load(str(_ROOT / "configs" / "base_smoke.yaml"))
    base["model"]["vocab_size"] = VOCAB
    sched_total = total_tokens(base["schedule"])
    cap = 12000.0
    assert cap < sched_total, "cap must be below the schedule total to be a test"

    results = {}
    for label, max_tokens in [("uncapped", None), ("capped", cap)]:
        cfg = all_runs(copy.deepcopy(base), "unigram")["en__unigram_destarved"]
        cfg["name"] = f"en__unigram_destarved__{label}"
        if max_tokens is not None:
            cfg["train"]["max_tokens"] = max_tokens
        run_from_config(cfg)
        recs = [json.loads(l) for l in
                (scratch / "runs" / cfg["name"] / "train.jsonl").read_text().splitlines()
                if l.strip()]
        results[label] = max(r["tokens"] for r in recs if "tokens" in r)
        print(f"[test] {label:9} ended at {results[label]:.0f} tokens")

    step = 2048  # base_smoke global_batch_tokens
    ok = True
    print()
    print(f"schedule total        : {sched_total:.0f}")
    print(f"uncapped final tokens : {results['uncapped']:.0f}")
    print(f"capped   final tokens : {results['capped']:.0f}   (cap {cap:.0f})")
    if results["uncapped"] < sched_total:
        print("FAIL: uncapped run did not reach the schedule total"); ok = False
    if not (cap <= results["capped"] < cap + 2 * step):
        print("FAIL: capped run did not stop at the cap"); ok = False
    if results["capped"] >= results["uncapped"]:
        print("FAIL: cap did not shorten the run"); ok = False

    lr_end = lr_at(results["capped"], base["schedule"])
    peak = float(base["schedule"]["peak_lr"])
    print(f"LR at capped stop     : {lr_end:.3e}   (peak {peak:.3e})")
    if abs(lr_end - peak) > 1e-12:
        print("FAIL: capped run not still at peak LR -- cap perturbed the schedule")
        ok = False

    shutil.rmtree(scratch, ignore_errors=True)
    print("\nPASS: max_tokens stops the run early and leaves the LR schedule intact"
          if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
