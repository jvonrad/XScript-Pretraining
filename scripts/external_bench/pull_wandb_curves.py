#!/usr/bin/env python
"""Pull dense BPB-vs-tokens curves from W&B into results/wandb_curves/.

This is the puller `results/wandb_curves/README.md` says to use; it was never
actually committed (`bts_from_wandb.py` only *reads* the `histories.json` this
produces), so the cached CSV had no reproducible provenance. It does now.

    export WANDB_API_KEY=...        # or `wandb login`
    python pull_wandb_curves.py [--project jonathan-von-rad/XScript-Pretraining]

Writes, into `results/wandb_curves/`:
    histories.json      {curve_id: {name, tokens_per_step, sources, history}}
                        <- read by bts_from_wandb.py
    bpb_curves.csv      long: run, name, wandb_id, step, tokens, metric, value
    runs_meta.json      per W&B run: state, tokens_per_step, final tokens_b,
                        eval-point counts, roster, seam/merge bookkeeping
    eval_final_bpb.json per run: the end-of-training `eval_final/*_bpb` block,
                        with the `_step` and token offset it was logged at
    histories_other.json / bpb_curves_other.csv
                        everything that is NOT the thesis roster, same schema

THE ROSTER SPLIT
================
The project also holds later, unrelated experiments -- 56 `__100b` attempts as
of 2026-09-01, plus `probe_*`, `ctrl-scratch__*`, `*_scratch` and the
`train.max_tokens` unit tests. They are pulled (a puller that quietly drops
runs is what this file exists to stop being) but written to `*_other.*`, so
the roster artifacts stay the roster and a curve from a different experiment
can never be picked up by an analysis that globs the main CSV.

`is_roster()` is the rule, not a list of bad names, so a new `__100b-v14`
sorts itself. It is deliberately the SAME rule `bts_from_wandb.py`'s `load()`
already applies (`{mix}__unigram_{starved,destarved}`, exactly two `__`
fields); those runs were being filtered there anyway, which is why the
downstream numbers never moved. Splitting at the source means an analysis that
does not happen to reimplement that filter is safe too.

FOUR THINGS THIS HANDLES THAT A NAIVE PULL GETS WRONG
=====================================================

1. **tokens_b and eval/*_bpb are logged in SEPARATE `wandb.log()` calls**, so
   they land on different steps. Requiring both on one row silently drops most
   eval points -- CLAUDE.md 6 records this costing the interaction once
   (`en-fr__unigram_destarved` looked like a 2-point run when it has 29).
   Eval rows therefore stand on their own and tokens are reconstructed as
   `step x tokens_per_step` (+ an offset, see 4), with the ratio taken as the
   median over rows that *do* carry both. The relation is exactly linear.

2. **THE RUN SUMMARY IS NOT AN INDEX OF THE HISTORY.** The eval keys used to be
   read from `r.summary`, and any run whose summary happened to carry no
   `eval/*_bpb` key was skipped entirely -- the inner loop never ran, and the
   run landed in `runs_meta.json` as a silent `n_eval=0`. Three runs were lost
   that way (the two `zh__*__neuron` resumes and `de__unigram_starved__neuron`).
   Keys are now discovered from the HISTORY ROWS THEMSELVES: one unfiltered
   `scan_history()` per run, from which the `eval/`, `eval_final/` and
   `tokens_b` rows are separated locally. An unfiltered scan is what makes this
   possible -- a `keys=` filter can only ask for names you already know.

   (For the three runs above the answer turned out to be that they log NO eval
   metrics under any key at all -- see UNUSABLE_RUNS -- but that is a fact
   about those runs, established by looking, not something the summary could
   have told us.)

3. **SPLICED RUNS.** `de__unigram_starved` is one W&B id containing TWO
   different training runs. The original diverged at the warmup/peak-LR seam
   (CLAUDE.md 6h) and ran to step 7361; the 2026-08-03 retrain resumed the
   same W&B id, so its early points collided with existing steps and were
   dropped, and only its history past the original's last step survived. The
   result is a single page whose BPB falls 1.3012 -> 1.0803 in one eval
   interval -- a 0.221 drop that is not learning, it is the model changing
   identity. Read as one curve it mixes a diverged run with a healthy one.

   `RUN_MIN_STEP` cuts each spliced run at its seam. Verified for this one:
   step x 917,504 reproduces 6h's documented retrain budgets on all three
   checkpoints (7.754/11.754/14.754 B against 7.753/11.754/14.755), so the
   retrain's step counter is its own and NO token offset is needed.

4. **RESUMED RUNS UNDER A NEW W&B ID, SHARING A DISPLAY NAME.** The opposite
   hazard to 3: one training split across TWO W&B ids that report the same
   `name`. `histories`/`meta` are keyed by id but the CSV used to be keyed by
   name, so such a pair would have merged into one curve *implicitly*, through
   a name collision, with no record of it. `RUN_MERGE` makes the merge
   explicit and `curve_id` (not the name) is what the CSV keys on, so every
   other same-name group -- the `__100b` v1..v13 attempts, which are separate
   trainings and must NOT merge -- stays separate.

   A resume also breaks `tokens = step x tokens_per_step`: `_step` continues
   from the parent but `tokens_per_step` can change with the world size
   (917,504 on 2 nodes vs 983,040 on 1). Token offsets are therefore FITTED
   from the resume's own logged `(step, tokens)` rows rather than hard-coded,
   and the fit, the seam continuity and the absence of a step overlap are all
   asserted at pull time.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "wandb_curves"

EVAL = "eval/"
EVAL_FINAL = "eval_final/"
BPB = "_bpb"

# curve id -> first step that belongs to the run we actually want.
# See the module docstring, point 3. Keep the reason with the number.
RUN_MIN_STEP = {
    # Steps <=7361 are the DIVERGED original (BPB 1.55->1.28->1.74->1.30).
    # Steps >=8451 are the 2026-08-03 retrain (1.0803 -> 1.0472, monotone).
    "de__unigram_starved": 8451,
}

# resume W&B id -> parent W&B id. See the module docstring, point 4.
#
# The two `zh__*__neuron` runs are the 2nd leg of the Chinese monolinguals:
# both resume at _step 12820 (the parent's last eval mark is step 12811 =
# 11.754B) and run to _step 16100 = 14.987B. `_step` CONTINUES across the
# resume -- it does not restart -- but tokens/step changes 917,504 -> 983,040
# (2 nodes -> 1), so `step x tps` alone puts the resume 0.84B too far right.
# The offset is fitted per-run in `_fit_tokens()` and asserted at the seam,
# never hard-coded.
#
# NOTE both legs are merged for completeness of the token axis, but NEITHER
# resume logged a single eval row (UNUSABLE_RUNS), so today the merge adds 0
# BPB points and the zh curves still end at the parents' 11.75B / 12.75B.
# The merge is here so that stays true *by construction* rather than by the
# accident of a name collision.
RUN_MERGE = {
    "zh__unigram_starved__neuron": "zh__unigram_starved",
    "zh__unigram_destarved__neuron": "zh__unigram_destarved",
}

# Runs that carry NO eval metric under any key, with why. Recorded here rather
# than left to show up as an unexplained `n_eval: 0`, which is exactly how the
# summary-keyed bug hid for as long as it did. Verified by an unfiltered
# scan_history(): the only keys any of them log are
# _runtime/_step/_timestamp/loss/lr/mix.*/step/tok_per_s/tokens/tokens_b.
UNUSABLE_RUNS = {
    "zh__unigram_starved__neuron":
        "zh 12B->15B resume, 165 rows, _step 12820-16100 (11.763-14.987B), "
        "finished -- train-loss only, in-loop eval never logged. Merged into "
        "zh__unigram_starved for the token axis; contributes 0 BPB points.",
    "zh__unigram_destarved__neuron":
        "zh 12B->15B resume, 165 rows, _step 12820-16100 (11.763-14.987B), "
        "finished -- train-loss only, in-loop eval never logged. Note it "
        "restarted from the 11.75B checkpoint while its parent had already "
        "reached 12.75B, so steps 12820-13900 OVERLAP the parent; harmless "
        "while it has no eval rows, but do not interleave the two blindly.",
    "de__unigram_starved__neuron":
        "UNUSABLE. Crashed 1-node attempt (983,040 tok/step), 26 rows, _step "
        "20-520 = 0.020-0.511B, train-loss only, no eval under any key. It is "
        "NOT the 2026-08-03 retrain -- that resumed the `de__unigram_starved` "
        "id itself (RUN_MIN_STEP above) and ran to 16.09B. It reports the same "
        "display NAME as that run, which is precisely the collision point 4 "
        "guards against: keyed by name it would have poured a 0.5B crashed "
        "attempt into the middle of the retrain's curve.",
}


ROSTER_TOKENIZERS = ("unigram_starved", "unigram_destarved")


def is_roster(name):
    """Is this display name one of the thesis runs?

    `{mix}__unigram_{starved,destarved}` and nothing else -- exactly the shape
    `bts_from_wandb.py`'s `load()` accepts, so the two agree by construction.
    A third `__` field means a later experiment (`__100b`, `__100b_scratch`,
    `__capped`, `__uncapped`) and a prefix means a probe
    (`probe_lr2e3__...`, `ctrl-scratch__...`).
    """
    parts = name.split("__")
    return len(parts) == 2 and parts[1] in ROSTER_TOKENIZERS


def _fit_tokens(pairs):
    """Fit tokens(step) from rows carrying BOTH `_step` and `tokens_b`.

    Returns (tps_ratio, tps_delta, offset, max_err_b).

    `tps_ratio` -- median of tokens/step -- is the historical estimator and
    stays the one used for ordinary runs, so this change does not move any
    token value that was already cached. Some runs mixed world sizes mid-run
    (CLAUDE.md 6h), which puts their step->token map off the grid; the median
    ratio absorbs that, a delta-based slope would not.

    `tps_delta` (median consecutive slope) + `offset` is the affine fit, which
    is what a RESUMED run needs: its step counter continues the parent's while
    its slope does not, so the ratio estimator is meaningless there.
    """
    if not pairs:
        return None, None, 0.0, None
    tps_ratio = statistics.median(t / s for s, t in pairs)
    slopes = [(t2 - t1) / (s2 - s1)
              for (s1, t1), (s2, t2) in zip(pairs, pairs[1:]) if s2 != s1]
    tps_delta = statistics.median(slopes) if slopes else tps_ratio
    offset = statistics.median(t - s * tps_delta for s, t in pairs)
    max_err = max(abs(t - (s * tps_delta + offset)) for s, t in pairs) / 1e9
    return tps_ratio, tps_delta, offset, max_err


def pull_run(r):
    """One unfiltered history scan -> everything this script needs from `r`.

    Unfiltered because the eval key names are what we are trying to discover;
    see the module docstring, point 2.
    """
    rows = list(r.scan_history(page_size=10000))
    tok_pairs, eval_rows, final_rows = [], [], []
    for h in rows:
        step, tb = h.get("_step"), h.get("tokens_b")
        if step and tb:
            tok_pairs.append((step, tb * 1e9))
        ev = {k: v for k, v in h.items()
              if k.startswith(EVAL) and k.endswith(BPB) and v is not None}
        if ev and step is not None:
            eval_rows.append((step, ev))
        fv = {k: v for k, v in h.items()
              if k.startswith(EVAL_FINAL) and k.endswith(BPB) and v is not None}
        if fv and step is not None:
            final_rows.append((step, fv))
    tok_pairs.sort()
    eval_rows.sort()
    tps_ratio, tps_delta, offset, max_err = _fit_tokens(tok_pairs)
    return {
        "id": r.id, "name": r.name, "state": r.state,
        "n_history_rows": len(rows),
        "tok_first_step": tok_pairs[0][0] if tok_pairs else None,
        "tok_last_step": tok_pairs[-1][0] if tok_pairs else None,
        "tokens_per_step": tps_ratio,
        "tokens_per_step_delta": tps_delta,
        "affine_offset_tokens": offset,
        "affine_fit_max_err_b": max_err,
        "final_step": max((h["_step"] for h in rows if h.get("_step") is not None),
                          default=None),
        "eval_rows": eval_rows,
        "eval_final_rows": final_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="jonathan-von-rad/XScript-Pretraining")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    import wandb
    api = wandb.Api()
    runs = list(api.runs(args.project))
    print(f"[wandb] {len(runs)} runs in {args.project}")

    pulled = {}
    for r in runs:
        pulled[r.id] = pull_run(r)
        p = pulled[r.id]
        print(f"  scanned {r.name:44s} {r.state:9s} rows={p['n_history_rows']:5d}"
              f" eval={len(p['eval_rows']):3d} eval_final={len(p['eval_final_rows'])}")

    # --- resolve merges ---------------------------------------------------
    # A resume placed on the token axis with `step x tps` alone lands ~0.84B
    # too far right (its slope is its own, its step counter is the parent's),
    # so a resume uses its FITTED affine map. Everything else keeps the
    # historical `step x median(tokens/step)`, unchanged.
    for res_id, par_id in RUN_MERGE.items():
        assert res_id in pulled and par_id in pulled, f"RUN_MERGE: {res_id}/{par_id} not in project"
        res, par = pulled[res_id], pulled[par_id]
        assert res["name"] == par["name"], \
            f"RUN_MERGE {res_id}: display name {res['name']!r} != parent {par['name']!r}"
        assert abs(res["affine_fit_max_err_b"]) < 1e-6, \
            f"{res_id}: (step, tokens) is not affine (max err {res['affine_fit_max_err_b']}B)"
        # The parent must be on the plain grid, or `step x tps` (which is what
        # the parent's own points still use) would not meet the resume's fit.
        assert abs(par["tokens_per_step"] - par["tokens_per_step_delta"]) < 1e-6, \
            f"{par_id}: mixed step->token grid; merging needs an explicit anchor"
        # No step-axis overlap between EVAL points across the seam, i.e. no
        # eval point can be duplicated or reordered by the merge.
        p_eval = [s for s, _ in par["eval_rows"]]
        r_eval = [s for s, _ in res["eval_rows"]]
        if p_eval and r_eval:
            assert min(r_eval) > max(p_eval), \
                f"{res_id}: eval steps overlap the parent's ({min(r_eval)} <= {max(p_eval)})"
        # Seam continuity on the TOKEN axis. The resume must start from a
        # CHECKPOINT THE PARENT ACTUALLY WROTE -- so it is anchored to the
        # parent's nearest eval mark at or below its own start, NOT to the
        # parent's last mark. The two differ: `zh__unigram_destarved` ran on
        # to 12.754B while its resume restarted from the 11.754B checkpoint,
        # so the parent OVERRUNS the resume by ~0.99B and steps 12820-13900
        # hold two different trainings at once. That overlap is recorded
        # (`parent_overlap_b`) rather than smoothed over; it is safe today
        # only because the resume has no eval rows to interleave.
        assert p_eval and res["tok_first_step"] is not None, \
            f"RUN_MERGE {res_id}: nothing to anchor the seam against"
        seam_res = (res["tok_first_step"] * res["tokens_per_step_delta"]
                    + res["affine_offset_tokens"])
        below = [s_ * par["tokens_per_step"] for s_ in p_eval
                 if s_ * par["tokens_per_step"] <= seam_res]
        assert below, \
            f"{res_id}: starts at {seam_res/1e9:.4f}B, before any {par_id} eval mark"
        gap_b = (seam_res - max(below)) / 1e9
        assert 0.0 <= gap_b < 0.30, \
            f"{res_id}: seam discontinuity {gap_b:+.4f}B against parent {par_id}"
        res["seam_gap_b"] = gap_b
        res["parent_overlap_b"] = (max(p_eval) * par["tokens_per_step"] - seam_res) / 1e9
        res["merged_into"] = par_id
        par.setdefault("merged_sources", []).append(res_id)

    # --- assemble curves --------------------------------------------------
    histories, meta, csv_rows = {}, {}, []
    other_histories, other_csv_rows = {}, []
    for rid, p in pulled.items():
        cid = RUN_MERGE.get(rid, rid)
        roster = is_roster(p["name"])
        is_resume = rid in RUN_MERGE
        tps = p["tokens_per_step_delta"] if is_resume else p["tokens_per_step"]
        off = p["affine_offset_tokens"] if is_resume else 0.0
        cut = RUN_MIN_STEP.get(cid, 0)

        hist, kept, dropped = [], 0, 0
        if tps:
            for step, vals in p["eval_rows"]:
                if step < cut:
                    dropped += 1     # spliced-run seam; see RUN_MIN_STEP
                    continue
                tokens = step * tps + off
                hist.append({"step": step, "tokens_b": tokens / 1e9, **vals})
                for k, v in vals.items():
                    (csv_rows if roster else other_csv_rows).append(
                        (cid, p["name"], rid, step, tokens, k, v))
                kept += 1

        cur = (histories if roster else other_histories).setdefault(
            cid, {"name": pulled[cid]["name"],
                  "tokens_per_step": pulled[cid]["tokens_per_step"],
                  "sources": [], "history": []})
        cur["sources"].append(rid)
        cur["history"].extend(hist)

        meta[rid] = {
            "name": p["name"], "state": p["state"], "curve_id": cid,
            "roster": roster,
            "tokens_per_step": p["tokens_per_step"],
            "tokens_per_step_delta": p["tokens_per_step_delta"],
            "affine_offset_tokens": p["affine_offset_tokens"],
            "affine_fit_max_err_b": p["affine_fit_max_err_b"],
            "final_step": p["final_step"],
            "final_tokens_b": hist[-1]["tokens_b"] if hist else None,
            "n_eval_points": kept, "n_cut_at_seam": dropped,
            "n_history_rows": p["n_history_rows"],
        }
        if is_resume:
            meta[rid]["merged_into"] = p["merged_into"]
            meta[rid]["seam_gap_b"] = p.get("seam_gap_b")
            meta[rid]["parent_overlap_b"] = p.get("parent_overlap_b")
        if p.get("merged_sources"):
            meta[rid]["merged_sources"] = p["merged_sources"]
        if rid in UNUSABLE_RUNS:
            meta[rid]["unusable"] = UNUSABLE_RUNS[rid]
        flag = ""
        if dropped:
            flag += f"  CUT {dropped} pre-seam rows"
        if is_resume:
            flag += f"  MERGED into {p['merged_into']} (+{kept} pts)"
        if rid in UNUSABLE_RUNS and not kept:
            flag += "  NO EVAL (see UNUSABLE_RUNS)"
        print(f"  {p['name']:32s} {p['state']:9s} eval={kept:4d}"
              f" tps={(f'{tps:,.0f}' if tps else '-'):>9s}{flag}")

    # A merged curve is one training; keep it ordered on the token axis.
    for cid, cur in list(histories.items()) + list(other_histories.items()):
        cur["history"].sort(key=lambda h: h["step"])
        steps = [h["step"] for h in cur["history"]]
        assert len(steps) == len(set(steps)), f"{cid}: duplicate eval step after merge"

    # --- eval_final -------------------------------------------------------
    # `eval_final/*_bpb` is the end-of-training eval, logged once at the run's
    # last step. `results/wandb_eval_final_bpb.json` held these with no puller
    # behind it; this reproduces them with provenance, and adds the token
    # placement (`final_step` alone cannot be put on a token axis).
    finals = {}
    for rid, p in pulled.items():
        if not p["eval_final_rows"]:
            continue
        step, vals = p["eval_final_rows"][-1]
        tps = p["tokens_per_step_delta"] if rid in RUN_MERGE else p["tokens_per_step"]
        off = p["affine_offset_tokens"] if rid in RUN_MERGE else 0.0
        finals[rid] = {
            "name": p["name"], "state": p["state"],
            "curve_id": RUN_MERGE.get(rid, rid),
            "roster": is_roster(p["name"]),
            "steps": step,
            "tokens_b": (step * tps + off) / 1e9 if tps else None,
            **{k: v for k, v in sorted(vals.items())},
        }

    def write_csv(path, rows):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "name", "wandb_id", "step", "tokens", "metric", "value"])
            for row in sorted(rows):
                w.writerow(row)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "histories.json").write_text(json.dumps(histories, indent=1))
    (args.out / "histories_other.json").write_text(json.dumps(other_histories, indent=1))
    (args.out / "runs_meta.json").write_text(json.dumps(meta, indent=1))
    (args.out / "eval_final_bpb.json").write_text(json.dumps(finals, indent=1))
    write_csv(args.out / "bpb_curves.csv", csv_rows)
    write_csv(args.out / "bpb_curves_other.csv", other_csv_rows)
    print(f"\nroster : {len(csv_rows):5d} points / {len(histories)} curves"
          f"  -> bpb_curves.csv, histories.json")
    print(f"other  : {len(other_csv_rows):5d} points / {len(other_histories)} curves"
          f"  -> bpb_curves_other.csv, histories_other.json")
    print(f"eval_final blocks: {len(finals)}   ({args.out})")


if __name__ == "__main__":
    raise SystemExit(main())
