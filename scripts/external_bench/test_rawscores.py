#!/usr/bin/env python
"""Offline correctness test for the raw-loglikelihood capture and the
estimators derived from it. No checkpoint, no accelerator.

It runs lm-eval over the real task configs with a STUB scorer that returns
deterministic pseudo-random loglikelihoods, then asserts:

  1. `extract_raw` + `score_variants` reproduce lm-eval's own `acc`,
     `acc_norm` and `acc_mutual_info` hit lists EXACTLY. If this fails, no
     derived estimator is trustworthy and nothing else here matters.
  2. `acc_cal` is invariant to a constant per-choice offset -- the property
     that makes it immune to the label-length/prior artifact. Concretely:
     add an arbitrary constant to every document's score for choice c, and
     `acc` / `acc_norm` change while `acc_cal` does not.
  3. The degeneracy report fires on a deliberately collapsed scorer.

    python scripts/external_bench/test_rawscores.py
"""
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from xscript.eval.rawscores import (degeneracy, extract_raw,  # noqa: E402
                                    prediction_profile, score_variants)


def _stub_ll(text: str, salt: str = "") -> float:
    """Deterministic pseudo-random loglikelihood, monotone in nothing."""
    h = hashlib.sha256((salt + text).encode("utf-8")).digest()
    return -int.from_bytes(h[:4], "big") / 2**32 * 20.0


class StubLM:
    """Minimal lm-eval LM: scores every request with `_stub_ll`."""

    def __init__(self, offsets=None, collapse_to=None):
        self.offsets = offsets or {}
        self.collapse_to = collapse_to

    def _score(self, context: str, continuation: str) -> float:
        if self.collapse_to is not None:
            # Rank one fixed continuation first for every document.
            return 0.0 if continuation.strip() == self.collapse_to else -50.0
        return _stub_ll(context + "\x00" + continuation) + \
            self.offsets.get(continuation.strip(), 0.0)


def _bind(stub):
    from lm_eval.api.model import TemplateLM

    class _Bound(TemplateLM):
        def __init__(self):
            super().__init__()
            self._rank, self._world_size = 0, 1

        @property
        def eot_token_id(self):
            return 0

        @property
        def prefix_token_id(self):
            return 0

        def tok_encode(self, string, **kw):
            return [1] * max(1, len(string) // 4)

        def _encode_pair(self, context, continuation):
            return self.tok_encode(context), self.tok_encode(continuation)

        def loglikelihood(self, requests, disable_tqdm=False):
            return [(stub._score(*r.args), False) for r in requests]

        def _loglikelihood_tokens(self, requests, disable_tqdm=False):
            raise NotImplementedError

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            raise NotImplementedError

        def generate_until(self, requests, disable_tqdm=False):
            raise NotImplementedError

    return _Bound()


def run_task(task: str, stub, limit: int):
    import lm_eval
    from lm_eval.tasks import TaskManager

    tm = TaskManager(include_path=str(REPO / "src" / "xscript" / "eval" / "c5_tasks"))
    res = lm_eval.simple_evaluate(model=_bind(stub), tasks=[task], num_fewshot=0,
                                  batch_size=1, limit=limit, log_samples=True,
                                  confirm_run_unsafe_code=True, task_manager=tm,
                                  verbosity="ERROR")
    rec = res["results"][task]
    samples = res["samples"][task]
    reported = {k.split(",")[0] for k in rec
                if k.split(",")[0] in ("acc", "acc_norm", "acc_mutual_info")}
    lm_hits = {m: [int(round(s[m])) for s in samples if m in s] for m in reported}
    raw = extract_raw(samples, reported, tok=None)
    return rec, raw, lm_hits


def main() -> int:
    TASK, LIMIT = "sib200_eng_Latn", 120
    failures = []

    print(f"[test] 1. re-derivation matches lm-eval  ({TASK}, n={LIMIT})")
    rec, raw, lm_hits = run_task(TASK, StubLM(), LIMIT)
    ours = score_variants(raw)
    alias = {"acc": "acc", "acc_norm": "acc_norm", "acc_mutual_info": "acc_pmi"}
    for lm_name, our_name in alias.items():
        if lm_name not in lm_hits:
            continue
        n_bad = sum(1 for a, b in zip(lm_hits[lm_name], ours[our_name]) if a != b)
        lm_acc = sum(lm_hits[lm_name]) / len(lm_hits[lm_name])
        our_acc = sum(ours[our_name]) / len(ours[our_name])
        ok = n_bad == 0
        print(f"       {lm_name:16s} lm-eval={lm_acc:.4f}  ours({our_name})={our_acc:.4f}"
              f"  mismatches={n_bad}  {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{lm_name}: {n_bad} mismatched hits")

    print("\n[test] 2. acc_cal is invariant to a constant per-choice offset")
    # Give the majority/longest label a large constant handicap, exactly the
    # shape of the real artifact (unnormalized loglik penalizes the longest
    # label on every single document).
    off = {"science and technology": -8.0, "geography": +4.0}
    _, raw2, _ = run_task(TASK, StubLM(offsets=off), LIMIT)
    a, b = score_variants(raw), score_variants(raw2)
    for v in ("acc", "acc_norm", "acc_pmi", "acc_cal"):
        if v not in a:
            continue
        changed = sum(1 for x, y in zip(a[v], b[v]) if x != y)
        acc_a = sum(a[v]) / len(a[v])
        acc_b = sum(b[v]) / len(b[v])
        print(f"       {v:14s} {acc_a:.4f} -> {acc_b:.4f}   docs changed={changed}")
        if v == "acc_cal" and changed != 0:
            failures.append(f"acc_cal moved under a per-choice offset ({changed} docs)")
        if v == "acc" and changed == 0:
            failures.append("acc did NOT move under a per-choice offset -- "
                            "the test's offset never reached the scorer")

    print("\n[test] 3. degeneracy fires on a collapsed scorer")
    _, raw3, _ = run_task(TASK, StubLM(collapse_to="travel"), LIMIT)
    hits3 = score_variants(raw3)["acc"]
    deg = degeneracy(raw3, hits3)
    prof = prediction_profile(raw3, "acc")
    print(f"       constant={deg['constant']}  n_recalled={deg['n_recalled']}  "
          f"pred_entropy={prof['pred_entropy']:.3f}  acc={prof['acc']:.4f}")
    if not deg["constant"] or prof["pred_entropy"] != 0.0:
        failures.append("degeneracy check did not flag a constant predictor")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
