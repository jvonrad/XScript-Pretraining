"""End-to-end smoke test for the Neuron/XLA training path on REAL Trainium
hardware (not CPU emulation) -- the Neuron twin of `scripts/smoke.py`.

Reuses smoke.py's synthetic-data fabrication (tiny fake FLORES+/pools/
tokenizer corpora, no network, no HF_TOKEN) so this costs nothing but a few
seconds of one NeuronCore, then drives the REAL `train_neuron.NeuronTrainer`
loop (forward/backward/xm.reduce_gradients/xm.mark_step/xm.save/resume) on an
actual XLA device. This is the fast, cheap way to catch device-side breakage
(shape/dtype/API mismatches) before committing to any real 30B-token run.

Run (single core, from inside the neuron venv):
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; export PATH="$HOME/.local/bin:$PATH"
    source ~/neuron_venv/bin/activate
    NEURON_RT_VISIBLE_CORES=0-1 python scripts/neuron_train/smoke_neuron.py
"""
import os
import sys
from pathlib import Path

SP = os.environ.get("SMOKE_SCRATCH",
                    os.path.join(os.path.dirname(__file__), "..", "..", ".smoke_scratch_neuron"))
os.environ["XSCRIPT_SCRATCH"] = os.path.abspath(SP)
os.environ["XSCRIPT_RESULTS"] = os.path.join(os.path.abspath(SP), "results")
os.environ.setdefault("PJRT_DEVICE", "NEURON")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))  # reuse smoke.py's fabrication helpers

import smoke as cpu_smoke  # noqa: E402  (scripts/smoke.py)
from xscript import paths, runmatrix, _yaml, train_neuron  # noqa: E402


def _fake_pools():
    """Like cpu_smoke.fake_pools(), but with a stats.json pack.py accepts."""
    import json
    import zstandard
    from xscript.langs import LANGS
    zc = zstandard.ZstdCompressor(level=1)
    for lang in LANGS:
        pd = paths.ensure(paths.pool_dir(lang))
        docs = [json.dumps({"text": " ".join(cpu_smoke._sentence(lang) for _ in range(3))},
                           ensure_ascii=False) for _ in range(1500)]
        blob = ("\n".join(docs) + "\n").encode("utf-8")
        with open(pd / "pool_00000.jsonl.zst", "wb") as f:
            f.write(zc.compress(blob))
        text_bytes = sum(len(json.loads(d)["text"].encode("utf-8")) for d in docs)
        (pd / "stats.json").write_text(json.dumps({
            "lang": lang, "budget_bytes": text_bytes, "text_bytes": text_bytes,
            "docs": len(docs), "exhausted": False,
        }))
        hd = paths.ensure(paths.HOLDOUT)
        hdocs = [json.dumps({"text": cpu_smoke._sentence(lang)}, ensure_ascii=False)
                 for _ in range(40)]
        with open(hd / f"{lang}_00000.jsonl.zst", "wb") as f:
            f.write(zc.compress(("\n".join(hdocs) + "\n").encode("utf-8")))
    print("[smoke-neuron] wrote synthetic pools + holdout (pack.py-compatible stats.json)")


def main():
    root = Path(os.environ["XSCRIPT_SCRATCH"])
    print(f"[smoke-neuron] scratch = {root}")

    # ---- fabricate synthetic inputs (identical to the CPU smoke test) ----
    cpu_smoke.fake_flores()
    cpu_smoke.fake_tok_corpora()
    _fake_pools()  # NOT cpu_smoke.fake_pools(): its stats.json lacks the
    # `budget_bytes` key pack.py now requires (pre-existing mismatch between
    # scripts/smoke.py and src/xscript/data/pack.py, unrelated to this port --
    # confirmed by running the CPU smoke test directly, which fails earlier
    # anyway on a missing `parity_aware_bpe` package in this venv).

    from xscript.tok import train as tt
    tt.train_unigram("destarved")
    print("[smoke-neuron] trained synthetic unigram_destarved tokenizer")

    from xscript.data import pack
    for lang in ("en", "fr"):
        pack.pack(lang, "unigram_destarved", workers=2)
    print("[smoke-neuron] packed synthetic en/fr shards")

    # ---- drive the REAL NeuronTrainer on REAL XLA hardware ----
    base = _yaml.load(str(_ROOT / "configs" / "base_smoke.yaml"))
    mono = runmatrix._stamp(base, "en__unigram_destarved__smoke", "en", "unigram_destarved")
    train_neuron.run_from_config(mono)
    print("[smoke-neuron] monolingual XLA train step(s) OK")

    bil = runmatrix._stamp(base, "en-fr__unigram_destarved__smoke", "en-fr", "unigram_destarved")
    train_neuron.run_from_config(bil)
    print("[smoke-neuron] bilingual XLA train step(s) OK")

    # resume: re-running the same config must pick up from "last" and finish
    # immediately (tokens already >= target) -- exercises maybe_resume()'s
    # checkpoint-load path on a real device.
    train_neuron.run_from_config(dict(bil))
    print("[smoke-neuron] resume-from-checkpoint OK")

    print("\n[smoke-neuron] ALL STAGES PASSED")


if __name__ == "__main__":
    main()
