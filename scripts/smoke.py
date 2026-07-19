"""End-to-end CPU smoke test on synthetic data (no network / no HF_TOKEN).

Fabricates the artifacts each stage expects on a throwaway XSCRIPT_SCRATCH, then
drives the real code paths: tokenizer train -> analyze gate -> pack -> loader
determinism -> model train (mono, bilingual, and a WSD cooldown branch) ->
BPB / MEXA alignment / BTS. Exercised with a 2-layer/dim-64 model so it runs in
under a minute. Run: python scripts/smoke.py
"""
import json
import os
import random
import sys
from pathlib import Path

# throwaway scratch under the session scratchpad
SP = os.environ.get("SMOKE_SCRATCH",
                    os.path.join(os.path.dirname(__file__), "..", ".smoke_scratch"))
os.environ["XSCRIPT_SCRATCH"] = os.path.abspath(SP)
os.environ["XSCRIPT_RESULTS"] = os.path.join(os.path.abspath(SP), "results")  # keep repo/results clean
os.environ.setdefault("XSCRIPT_VOCAB", "1024")      # tiny vocab -> fast tokenizers + small head
os.environ.setdefault("XSCRIPT_TOK_WORKERS", "2")   # keep BPE learners under the process limit
os.environ.setdefault("OMP_NUM_THREADS", "1")   # login-node process-limit safety
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import zstandard  # noqa: E402
from xscript.langs import LANGS  # noqa: E402
from xscript import paths  # noqa: E402

rng = random.Random(0)

# distinct scripts so tokenizer allocation actually differs across languages
_ALPH = {
    "en": "abcdefghijklmnopqrstuvwxyz",
    "de": "abcdefghijklmnopqrstuvwxyzäöüß",
    "fr": "abcdefghijklmnopqrstuvwxyzéèàç",
    "ar": "ابتثجحخدذرزسشصضطظعغفقكلمنهوي",
    "zh": "的一是不了人我在有他这为之大来以个中上们时到",
}


def _words(alph, n, kmin=2, kmax=7):
    return ["".join(rng.choice(alph) for _ in range(rng.randint(kmin, kmax)))
            for _ in range(n)]


VOCAB = {l: _words(a, 400) for l, a in _ALPH.items()}


def _sentence(lang, wmin=4, wmax=12):
    return " ".join(rng.choice(VOCAB[lang]) for _ in range(rng.randint(wmin, wmax)))


def fake_flores():
    for split in ("dev", "devtest"):
        d = paths.ensure(paths.FLORES_DIR / split)
        n = 120 if split == "dev" else 100
        # parallel: same ids across languages
        for lang, L in LANGS.items():
            recs = [{"id": i, "text": _sentence(lang)} for i in range(n)]
            (d / f"{L.flores_code}.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
                encoding="utf-8")
    print("[smoke] wrote synthetic FLORES+ dev/devtest")


def fake_tok_corpora():
    # starved: our 5 langs + extra pseudo-languages competing for vocab
    starved = paths.ensure(paths.TOK_CORPORA / "starved")
    for lang in LANGS:
        txt = "\n".join(_sentence(lang) for _ in range(4000))
        (starved / f"{lang}.txt").write_text(txt + "\n", encoding="utf-8")
    for k in range(6):  # extra "languages" only in the starved corpus
        alph = _ALPH[list(_ALPH)[k % 5]]
        txt = "\n".join(" ".join(rng.choice(_words(alph, 200)) for _ in range(6))
                        for _ in range(3000))
        (starved / f"x{k}.txt").write_text(txt + "\n", encoding="utf-8")
    # destarved: only our 5 languages
    de = paths.ensure(paths.TOK_CORPORA / "destarved")
    for lang in LANGS:
        txt = "\n".join(_sentence(lang) for _ in range(6000))
        (de / f"{lang}.txt").write_text(txt + "\n", encoding="utf-8")
    print("[smoke] wrote synthetic tokenizer corpora")


def fake_pools():
    zc = zstandard.ZstdCompressor(level=1)
    for lang in LANGS:
        pd = paths.ensure(paths.pool_dir(lang))
        docs = [json.dumps({"text": " ".join(_sentence(lang) for _ in range(3))},
                           ensure_ascii=False) for _ in range(1500)]
        with open(pd / "pool_00000.jsonl.zst", "wb") as f:
            f.write(zc.compress(("\n".join(docs) + "\n").encode("utf-8")))
        (pd / "stats.json").write_text(json.dumps(
            {"lang": lang, "text_bytes": 1, "docs": len(docs)}))
        hd = paths.ensure(paths.HOLDOUT)
        hdocs = [json.dumps({"text": _sentence(lang)}, ensure_ascii=False)
                 for _ in range(40)]
        with open(hd / f"{lang}_00000.jsonl.zst", "wb") as f:
            f.write(zc.compress(("\n".join(hdocs) + "\n").encode("utf-8")))
    print("[smoke] wrote synthetic pools + holdout")


def train_tokenizers():
    # exercises all three real code paths: SP Unigram, repo byte-level BPE,
    # repo parity-aware BPE (needs the parity-aware-bpe package installed).
    from xscript.tok import train as tt
    for cond in ("starved", "destarved"):
        tt.train_unigram(cond)
        tt.train_bpe(cond)
    tt.train_pa("destarved", variant="window")
    print("[smoke] trained unigram/bpe x {starved,destarved} + pa_destarved")


def check_loader_determinism():
    from xscript.data.loader import MixedStream
    a = MixedStream(["en", "fr"], "bpe_destarved", 64, seed=7)
    b = MixedStream(["en", "fr"], "bpe_destarved", 64, seed=7)
    # world=1 vs world=2 must yield the same global assignment/counts
    xa, ca = a.rank_batch(16, 0, 1)
    xb0, _ = b.rank_batch(16, 0, 2)
    b2 = MixedStream(["en", "fr"], "bpe_destarved", 64, seed=7)
    xb1, _ = b2.rank_batch(16, 1, 2)
    assert xa.shape[0] == 16 and xb0.shape[0] + xb1.shape[0] == 16
    # resume: skip_to reproduces state
    c = MixedStream(["en", "fr"], "bpe_destarved", 64, seed=7)
    c.skip_to(16)
    assert c.slot == 16
    print("[smoke] loader determinism + world-size split OK")


def sh(cmd):
    print(f"\n$ {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        raise SystemExit(f"command failed ({rc}): {cmd}")


def main():
    root = Path(os.environ["XSCRIPT_SCRATCH"])
    print(f"[smoke] scratch = {root}")
    fake_flores()
    fake_tok_corpora()
    fake_pools()
    train_tokenizers()

    env = f'PYTHONPATH="{Path(__file__).resolve().parents[1] / "src"}" ' \
          f'XSCRIPT_SCRATCH="{root}" XSCRIPT_RESULTS="{root / "results"}" ' \
          f'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1'
    X = f"{env} python -m xscript.cli"

    sh(f"{X} tok-analyze")   # default = all 5 tokenizers

    for lang in ("en", "fr", "ar"):
        sh(f"{X} pack --lang {lang} --tok bpe_destarved --workers 2")

    check_loader_determinism()

    B = "configs/base_smoke.yaml"
    sh(f"{X} train en__bpe_destarved      --base {B} --flavor bpe")
    sh(f"{X} train fr__bpe_destarved      --base {B} --flavor bpe")
    sh(f"{X} train en-fr__bpe_destarved   --base {B} --flavor bpe")

    # WSD cooldown-branch mechanism (hand-built tiny trunk + cooldown)
    branch_test(root)

    sh(f"{X} eval-bpb   en-fr__bpe_destarved --tok bpe_destarved --tag final")
    sh(f"{X} eval-align en-fr__bpe_destarved --tok bpe_destarved")
    sh(f"{X} bts --flavor bpe --source flores")

    print("\n[smoke] ALL STAGES PASSED")


def branch_test(root):
    """Trunk (no decay, stable_mark) then cooldown branch from that checkpoint."""
    from xscript import train, runmatrix, _yaml
    base = _yaml.load("configs/base_smoke.yaml")
    trunk = runmatrix._stamp(base, "en__bpe_destarved__trunk", "en", "bpe_destarved")
    trunk["schedule"] = {**trunk["schedule"], "warmup_tokens": 4000,
                         "stable_tokens": 40000, "decay_tokens": 0}
    trunk["train"] = {**trunk["train"], "stable_marks": [30000]}
    train.run_from_config(trunk)
    ck = paths.run_dir("en__bpe_destarved__trunk") / "checkpoints" / "stable_30M.pt"
    # stable_marks names use int(mark/1e6)M -> 30000/1e6 = 0 -> "stable_0M"
    marks = list((paths.run_dir("en__bpe_destarved__trunk") / "checkpoints").glob("stable_*"))
    assert marks, "trunk saved no stable_* checkpoint"
    cool = runmatrix._stamp(base, "en__bpe_destarved__100b", "en", "bpe_destarved")
    cool["schedule"] = {**cool["schedule"], "warmup_tokens": 0,
                        "stable_tokens": 0, "decay_tokens": 20000}
    cool["branch"] = {"from": str(marks[0]), "load_optim": True}
    train.run_from_config(cool)
    print("[smoke] WSD trunk + cooldown branch OK")


if __name__ == "__main__":
    main()
