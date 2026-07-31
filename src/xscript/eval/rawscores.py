"""Persist raw per-choice loglikelihoods, and derive every scoring rule offline.

WHY THIS EXISTS
---------------
`run_extra_bench.py` and `run_appendix_c5.py` used to keep only lm-eval's 0/1
hit lists. That threw away the one thing needed to audit a scoring rule after
the fact, and CLAUDE.md §6/§6b/§6d document four separate occasions where an
uncontrolled scoring choice was mistaken for a training result. A fifth was
sitting in the SIB-200 numbers:

    SIB-200 per-gold-class recall, `acc`, en-fair (30B):
        science/technology 0.43   travel 0.91  politics 0.94  sports 0.92
        health 0.81  entertainment 0.30  geography 0.45
    ... and under `acc_norm`, on the SAME scores:
        science/technology 1.00   everything else 0.00-0.16

`acc` sums an *unnormalized* loglikelihood over the answer string, so the
longest of the seven shared labels ("science and technology", which is also
the 25.1%-majority class) is almost never ranked first. `acc_norm` divides by
the label's length (characters -- lm-eval's `completion_len`, not bytes as
CLAUDE.md §6d states), which makes that same longest label win *always*. Both
numbers are dominated by a per-choice offset that is identical for every document, so
neither measures classification: the argmax is decided before the document is
read. Because that offset drifts between checkpoints, accuracy trajectories
come out non-monotonic (en-starved English SIB-200: .609 @12B, .537 @15B,
.581 @30B) and single cells swing by 0.10+.

The same mechanism drives lm-eval's XNLI for en/de/fr, where the three
candidates differ only by a connective — CLAUDE.md §6 already diagnosed and
fixed exactly this for ar/zh and left the other three languages on raw
loglikelihood.

THE FIX
-------
Two parts, and the first matters more than the second:

1. **Store the raw loglikelihoods** (conditional and, where the task requests
   PMI, unconditional), with each candidate's token and byte length and the
   gold index. Every estimator below is then a pure-CPU re-derivation, and a
   new scoring rule never costs another accelerator pass. Same lesson as
   §6b's cached embeddings.

2. **Score with a calibrated estimator.** For a task whose choice set is
   shared across documents (SIB-200, Taxi-1500, and — at the connective —
   XNLI), subtract each choice's *mean loglikelihood over the evaluation set*:

       s(d, c) = ll(c | x_d) − (1/N) Σ_d' ll(c | x_d')

   This is contextual calibration (Zhao et al. 2021, arXiv:2102.09690) with
   the empirical input distribution standing in for a content-free probe, and
   it is the diagonal-free case of Holtzman et al. 2021's surface-form
   competition argument. It removes the per-choice offset exactly — whatever
   its cause, length or prior or tokenizer fertility — and is invariant to
   how the label happens to be worded. It needs no extra forward passes.

`acc_cal` is the estimator to quote for a shared-choice-set task. `acc_pmi`
(lm-eval's `acc_mutual_info`) is the per-document analogue and is kept because
§6 established it for XNLI-zh; it is noisier here because the unconditional
score is estimated from one short string rather than from N documents.

DEGENERACY
----------
`degeneracy()` reports what an accuracy alone cannot: how concentrated the
predictions are. A cell that ranks one label first for every document scored
that label's frequency while learning nothing, and 16 such cells were already
found in the SIB-200 sweep (§6d). Any estimator can produce them, so the check
travels with the estimator rather than being bolted onto one analysis script.
"""
from __future__ import annotations

import math

# Metric names produced by `score_variants`, in the order they should be
# reported. `acc` / `acc_norm` / `acc_pmi` reproduce lm-eval's `acc` /
# `acc_norm` / `acc_mutual_info` exactly (verified by `check_reproduces`).
VARIANTS = ("acc", "acc_norm", "acc_tokennorm", "acc_pmi", "acc_cal",
            "acc_cal_loo", "acc_cal_pmi")

# Tasks whose choice INDEX means the same thing in every document, which is the
# precondition for `acc_cal` to be the label-prior correction it is advertised
# as. Matched by name prefix.
#
#   sib200/taxi1500  index c is always topic c -- the 7 (or 6) label strings are
#                    literally the same for every document.
#   global_mmlu      lm-eval's registered task scores the LETTER, so the four
#                    candidates are literally "A"/"B"/"C"/"D" for every document.
#   xnli             the three candidates differ per document (they embed the
#                    premise and hypothesis), but index c is always connective c,
#                    and the connective is the shared component whose prior
#                    decides the argmax. Calibrating removes exactly that.
#
# NOT listed, deliberately: hellaswag, arc, belebele_cloze, xstorycloze,
# xwinograd. Their candidates are document-specific text, so "choice 0" of one
# document has nothing to do with "choice 0" of the next. `_calibrate` would
# still RUN on a fixed-arity task like HellaSwag (always 4 endings) and return a
# number -- but that number is a position/order-bias correction, a different and
# far smaller effect, and reading it as the label-prior fix would be precisely
# the kind of unaudited scoring choice CLAUDE.md keeps having to retract. So it
# is withheld rather than left to be misread.
SHARED_CHOICE_TASKS = ("sib200", "taxi1500", "xnli", "global_mmlu",
                       "gmmlu_letter",   # gmmlu_cloze deliberately absent
                       "mub_mnli", "mub_snli")  # label words, shared set


def has_shared_choices(task: str) -> bool:
    """Does choice index c mean the same thing in every document of `task`?"""
    return task.startswith(SHARED_CHOICE_TASKS)


def extract_raw(samples: list[dict], reported_metrics, tok=None,
                target_delimiter: str = " ") -> dict:
    """Pull raw per-choice scores out of lm-eval's `log_samples` records.

    `samples` is `results["samples"][task]`; `reported_metrics` is the set of
    metric names the task reported (used only to decide whether unconditional
    requests were issued). `tok` is our `Tok`, used to count continuation
    tokens; pass None to skip `ntok` (then `acc_tokennorm` is unavailable).

    Returns a dict of parallel lists, one entry per document:

        n_choices  int
        gold       [int]                     index of the correct choice
        ll         [[float] * n_choices]     log P(choice | context)
        ll_uncond  [[float] * n_choices]     log P(choice), or None
        nchars     [[int] * n_choices]       character length of each choice
        nbytes     [[int] * n_choices]       UTF-8 length of each choice
        ntok       [[int] * n_choices]       token count, or None

    Two conventions have to match lm-eval exactly or `acc_norm` will not
    reproduce (it silently differed by one document in 120 during
    development, from a near-tie flipping):

    * lm-eval normalizes by `len(choice)` -- **characters, not bytes** -- see
      `completion_len` in `ConfigurableTask.process_results`. `nbytes` is
      recorded too but is not what `acc_norm` divides by.
    * the length is of the raw choice, while `arguments` carries the
      continuation *with* `target_delimiter` prepended, so the delimiter is
      stripped back off here.

    lm-eval appends the unconditional requests *after* the conditional ones
    when `acc_mutual_info` is in the task's metric list (see
    `construct_requests`), so the split is at the halfway point -- not at the
    first empty context, which would misfire on a task whose own
    `doc_to_text` is empty (lm-eval's XNLI is exactly that).
    """
    has_mi = "acc_mutual_info" in set(reported_metrics)
    out = {"gold": [], "ll": [], "ll_uncond": [] if has_mi else None,
           "nchars": [], "nbytes": [], "ntok": [] if tok is not None else None,
           "n_choices": None, "doc_id": []}
    for s in samples:
        resps = s["filtered_resps"]
        args = s["arguments"]
        k = len(resps) // 2 if has_mi else len(resps)
        if k == 0 or (has_mi and len(resps) != 2 * k):
            raise ValueError(f"unexpected response count {len(resps)} "
                             f"(acc_mutual_info={has_mi})")
        if out["n_choices"] is None:
            out["n_choices"] = k
        elif out["n_choices"] != k:
            # Ragged choice sets (some ARC items have 3 or 5 options) are fine
            # for acc/acc_norm/acc_pmi but make `acc_cal` ill-defined, since
            # there is no stable "choice c" to average over. Recorded so the
            # analysis can refuse to calibrate rather than silently mis-index.
            out["n_choices"] = -1
        out["gold"].append(_gold_index(s))
        out["doc_id"].append(int(s["doc_id"]))
        out["ll"].append([float(r[0]) for r in resps[:k]])
        if has_mi:
            out["ll_uncond"].append([float(r[0]) for r in resps[k:]])
        conts = [a[1] for a in args[:k]]
        choices = [c[len(target_delimiter):]
                   if target_delimiter and c.startswith(target_delimiter) else c
                   for c in conts]
        out["nchars"].append([max(1, len(c)) for c in choices])
        out["nbytes"].append([max(1, len(c.encode("utf-8"))) for c in choices])
        if tok is not None:
            out["ntok"].append([max(1, len(tok.encode(c, bos=False, eos=False)))
                                for c in conts])
    return out


def _gold_index(sample: dict) -> int:
    """Gold choice index from an lm-eval sample record.

    `doc_to_target` normally yields the integer index (SIB-200, XNLI, ARC via
    its `label`), sometimes as a numpy scalar or a decimal string (HellaSwag's
    `label` is "0".."3"). A handful of tasks yield the answer text instead, so
    fall back to matching it against the scored continuations.
    """
    target = sample["target"]
    try:
        return int(target)
    except (TypeError, ValueError):
        pass
    conts = [a[1] for a in sample["arguments"]]
    for i, c in enumerate(conts):
        if c.strip() == str(target).strip():
            return i
    raise ValueError(f"cannot resolve gold index for target {target!r}")


def _argmax_hits(scores: list[list[float]], gold: list[int]) -> list[int]:
    return [int(max(range(len(row)), key=row.__getitem__) == g)
            for row, g in zip(scores, gold)]


def _calibrate(ll: list[list[float]], loo: bool = False) -> list[list[float]]:
    """Subtract each choice's mean loglikelihood over the evaluation set.

    With `loo=True` the mean for document d excludes d itself, so the score of
    a document is computed from the *other* documents only. The correction is
    label-free either way -- no gold is touched -- but the plain version is
    transductive (it reads every input before scoring any of them), and the
    leave-one-out version is not. At n~1000 the two differ by O(1/n) and
    `acc_cal` / `acc_cal_loo` agree on essentially every document; the point of
    keeping both is that the objection can be checked rather than argued.
    """
    n, k = len(ll), len(ll[0])
    totals = [sum(row[c] for row in ll) for c in range(k)]
    if not loo:
        return [[row[c] - totals[c] / n for c in range(k)] for row in ll]
    if n < 2:
        return [list(row) for row in ll]
    return [[row[c] - (totals[c] - row[c]) / (n - 1) for c in range(k)]
            for row in ll]


def score_variants(raw: dict, shared_choices: bool | None = None) -> dict[str, list[int]]:
    """Every estimator, as 0/1 hit lists over the stored documents.

    The `acc_cal*` family is emitted only when the choice INDEX is stable
    across documents. That is taken from `shared_choices` if given, else from
    `raw["shared_choices"]` if the runner recorded it, else assumed True for
    backward compatibility with sidecars written before the flag existed --
    those only ever covered SIB-200 / Taxi-1500 / XNLI, which all qualify.
    A ragged choice set (`n_choices == -1`, e.g. ARC's 3-to-5 options)
    disqualifies regardless, since there is no stable index at all.
    """
    if shared_choices is None:
        shared_choices = raw.get("shared_choices", True)
    gold, ll = raw["gold"], raw["ll"]
    out: dict[str, list[int]] = {}
    out["acc"] = _argmax_hits(ll, gold)
    # Characters, matching lm-eval's `completion_len` -- see extract_raw.
    lens = raw.get("nchars") or raw["nbytes"]
    out["acc_norm"] = _argmax_hits(
        [[x / n for x, n in zip(row, ln)] for row, ln in zip(ll, lens)], gold)
    if raw.get("ntok"):
        out["acc_tokennorm"] = _argmax_hits(
            [[x / n for x, n in zip(row, nt)]
             for row, nt in zip(ll, raw["ntok"])], gold)
    if raw.get("ll_uncond"):
        pmi = [[x - u for x, u in zip(row, urow)]
               for row, urow in zip(ll, raw["ll_uncond"])]
        out["acc_pmi"] = _argmax_hits(pmi, gold)
    if raw.get("n_choices", -1) > 0 and shared_choices:
        out["acc_cal"] = _argmax_hits(_calibrate(ll), gold)
        out["acc_cal_loo"] = _argmax_hits(_calibrate(ll, loo=True), gold)
        if raw.get("ll_uncond"):
            # Redundant whenever the choice set is shared across documents:
            # the unconditional score of choice c is then the same constant for
            # every document, and calibration already removes any per-choice
            # constant, so acc_cal_pmi == acc_cal exactly. Kept because it is
            # NOT redundant for a task whose choices vary per document.
            out["acc_cal_pmi"] = _argmax_hits(_calibrate(pmi), gold)
    return out


def degeneracy(raw: dict, hits: list[int], n_classes: int | None = None) -> dict:
    """How concentrated the predictions are, independent of accuracy.

    Returns:
        pred_frac   [float] fraction of documents assigned to each choice
                    index (only meaningful for a shared choice set)
        pred_entropy   normalized entropy of `pred_frac`, 1.0 = uniform,
                       0.0 = one label for every document
        recall      [float] per-gold-class recall, the diagnostic that made
                    the SIB-200 length artifact visible: `acc` gives the
                    longest label ~0 recall, `acc_norm` gives it 1.0
        constant    True if the hit vector is exactly `gold == c` for some c,
                    i.e. the cell scored that class's frequency having learned
                    nothing (§6d's check, kept)
    """
    gold = raw["gold"]
    k = n_classes or raw.get("n_choices") or (max(gold) + 1)
    rec = {}
    for c in range(k):
        idx = [i for i, g in enumerate(gold) if g == c]
        rec[c] = (sum(hits[i] for i in idx) / len(idx)) if idx else None
    constant = any(
        all(hits[i] == int(g == c) for i, g in enumerate(gold))
        for c in range(k)
    )
    return {"recall": rec, "constant": constant,
            "n_recalled": sum(1 for v in rec.values() if v and v > 0.1)}


def prediction_profile(raw: dict, variant: str = "acc_cal") -> dict:
    """Predicted-choice distribution + its normalized entropy for one variant.

    Separate from `degeneracy` because it needs the argmax itself, not just
    the hit vector, so it is only available where `raw` was stored.
    """
    gold, ll = raw["gold"], raw["ll"]
    if variant == "acc":
        scores = ll
    elif variant == "acc_norm":
        lens = raw.get("nchars") or raw["nbytes"]
        scores = [[x / n for x, n in zip(r, ln)] for r, ln in zip(ll, lens)]
    elif variant == "acc_pmi":
        scores = [[x - u for x, u in zip(r, ur)]
                  for r, ur in zip(ll, raw["ll_uncond"])]
    elif variant == "acc_cal":
        scores = _calibrate(ll)
    elif variant == "acc_cal_loo":
        scores = _calibrate(ll, loo=True)
    elif variant == "acc_cal_pmi":
        scores = _calibrate([[x - u for x, u in zip(r, ur)]
                             for r, ur in zip(ll, raw["ll_uncond"])])
    else:
        raise ValueError(f"unknown variant {variant}")
    # Choice counts can VARY per document (ArabicMMLU has 2-5 options), so the
    # argmax must use each row's own arity and `counts` must be sized to the
    # widest row. Assuming a uniform k here raised IndexError on ArabicMMLU.
    kmax = max(len(row) for row in scores)
    ragged = any(len(row) != kmax for row in scores)
    counts = [0] * kmax
    for row in scores:
        counts[max(range(len(row)), key=row.__getitem__)] += 1
    n = len(scores)
    frac = [c / n for c in counts]
    ent = abs(-sum(p * math.log(p) for p in frac if p > 0) / math.log(kmax)) if kmax > 1 else 0.0
    # Empirical null: the accuracy this same prediction DISTRIBUTION would get
    # if its predictions were independent of the gold labels, i.e. sum_c
    # P(predict c) * P(gold c). This is the right chance level to compare
    # against -- not 1/k, and not the majority rate. A predictor that always
    # answers "travel" scores exactly the travel frequency, and `acc - null`
    # is then 0: it demonstrably used nothing from the document. Closed form,
    # so it costs nothing and needs no permutation sampling.
    if ragged:
        # With varying arity a per-index null is not meaningful (index 4 only
        # exists on some documents). Use the uniform-random baseline instead:
        # mean over documents of 1/k_d. For ArabicMMLU that is 0.293, NOT the
        # 0.25 a fixed-arity benchmark would have -- quoting 25% there would be
        # wrong by four points.
        null = sum(1.0 / len(row) for row in scores) / n
    else:
        gold_frac = [sum(1 for g in gold if g == c) / n for c in range(kmax)]
        null = sum(frac[c] * gold_frac[c] for c in range(kmax))
    acc = sum(int(max(range(len(row)), key=row.__getitem__) == g)
              for row, g in zip(scores, gold)) / n
    return {"pred_frac": frac, "pred_entropy": ent, "acc": acc,
            "null": null, "acc_over_null": acc - null, "ragged": ragged}


def check_reproduces(raw: dict, lm_eval_hits: dict[str, list[int]],
                     tol: int = 0) -> dict[str, int]:
    """Regression check: our re-derived hits vs lm-eval's own, per metric.

    Returns {metric: n_mismatched}. Anything above `tol` means the stored raw
    scores do not reconstruct what lm-eval reported, and no derived metric
    from this file should be trusted. Ties broken differently by `max` vs
    numpy's `argmax` are the only expected source of a small non-zero count.
    """
    ours = score_variants(raw)
    alias = {"acc": "acc", "acc_norm": "acc_norm", "acc_mutual_info": "acc_pmi"}
    out = {}
    for lm_name, our_name in alias.items():
        if lm_name in lm_eval_hits and our_name in ours:
            a, b = lm_eval_hits[lm_name], ours[our_name]
            out[lm_name] = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    return out
