#!/usr/bin/env python
"""Consistency and parametric knowledge sharing OVER TRAINING for the 8
bilinguals (2b/5b/10b/15b/23b mid-stable + cooled 30B final), both
tokenizer conditions -- the trajectory view CLAUDE.md 6k argues for instead
of single matched points.

Per checkpoint (one record each, stored in results/trajectories/):
  budget, LR state, FLORES bpb (en, partner)
  consistency:  RankC on PolyFact (4 QID-aligned cands, summed loglik);
                RankC on MuBench-BMLAMA (sumWhole and cand); P@1 per language;
                error consistency (same top-1 among facts wrong in both langs)
  sharing:      knowledge-neuron dKS at K=50/100/200 (same-fact minus
                mismatched-fact top-K Jaccard; analyze_knowneurons' statistic
                on ALL selected facts), pooled ablation transfer rate
                (transfer/specificity), own-fact damage & flip rate, n facts
Inputs: polyfact_traj/<run>.json, rankc/<run>_bmlamamub.json,
knowneurons/<run>_{selection,kn.npz,ablation} under the workdir (the 8
finals' 6j results are symlinked in from results/knowneurons/).

    python analyze_trajectories.py [--workdir /mnt/scratch/xscript_rankc]
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from analyze_knowneurons import Run, jaccard_matrix, jaccard_mismatched, CONDS  # noqa: E402
from analyze_polyfact_traj import load as load_pf  # noqa: E402
from analyze_rankc import load as load_mub  # noqa: E402

PARTNERS = ["de", "fr", "ar", "zh"]
BUDGETS = ["2b", "5b", "10b", "15b", "23b", ""]
KS = (50, 100, 200)


def kn_stats(res: Path, name: str):
    try:
        r = Run(res, name)
    except FileNotFoundError:
        return None
    out = {"kn_n_facts": int(len(r.fact_ids))}
    a, b = r.topk[:, 0, :], r.topk[:, 1, :]
    for k in KS:
        same = jaccard_matrix(a, b, k); mism = jaccard_mismatched(a, b, k)
        out[f"dks_k{k}"] = float((same - mism).mean())
        out[f"sameJ_k{k}"] = float(same.mean())
    trans, spec, own, flips = [], [], [], []
    for ti, tlang in enumerate(r.langs):
        dmg = {c: r.damage(tlang, c) for c in CONDS[1:]}
        spec.append(dmg["own_fact"] - dmg["own_other"]); trans.append(dmg["cross_fact"] - dmg["cross_other"])
        own.append(dmg["own_fact"]); flips.append(r.flips(tlang, "own_fact"))
    trans, spec = np.concatenate(trans), np.concatenate(spec)
    out["transfer_rate"] = float(trans.mean() / spec.mean()) if spec.mean() > 0 else float("nan")
    out["transfer_abs"] = float(trans.mean()); out["specificity"] = float(spec.mean())
    out["own_damage"] = float(np.concatenate(own).mean()); out["own_flip"] = float(np.concatenate(flips).mean())
    return out


def pf_stats(work: Path, name: str):
    p = work / "results" / "polyfact_traj" / f"{name}.json"
    if not p.exists():
        return None
    r = load_pf(p)
    d = json.load(open(p)); l1, l2 = d["langs"]
    ll1, ll2 = np.array(d["ll"][l1]), np.array(d["ll"][l2]); g1, g2 = np.array(d["gold"][l1]), np.array(d["gold"][l2])
    ids1, ids2 = d["option_ids"][l1], d["option_ids"][l2]
    perm = np.array([[ids2[i].index(q) for q in ids1[i]] for i in range(len(ids1))]); ll2 = np.take_along_axis(ll2, perm, 1)
    g2 = np.array([ids1[i].index(ids2[i][g2[i]]) for i in range(len(ids1))])
    t1, t2 = ll1.argmax(1), ll2.argmax(1); wb = (t1 != g1) & (t2 != g2)
    return {"bpb_en": r["bpb"].get("en"), "bpb_partner": r["bpb"].get(l2), "cooled": r["cooled"], "budget_b": r["budget"],
            "pf_rankc": float(r["rc"].mean()), "pf_acc_en": r["acc1"], "pf_acc_partner": r["acc2"],
            "pf_same_wrong": float((t1 == t2)[wb].mean()), "pf_n_wrong_both": int(wb.sum()),
            "pf_consistent": float((t1 == t2).mean())}


def mub_stats(work: Path, name: str):
    p = work / "results" / "rankc" / f"{name}_bmlamamub.json"
    if not p.exists():
        return None
    r = load_mub(p)
    out = {}
    for v in ("sumWhole", "cand"):
        out[f"mub_rankc_{v}"] = float(r["rc"][v].mean())
        out[f"mub_acc_en_{v}"] = float(r["acc1"][v].mean()); out[f"mub_acc_partner_{v}"] = float(r["acc2"][v].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/mnt/scratch/xscript_rankc")
    args = ap.parse_args()
    work = Path(args.workdir); kn_dir = work / "results" / "knowneurons"
    recs = []
    for p in PARTNERS:
        for cond in ("fair", "starved"):
            for b in BUDGETS:
                name = f"en-{p}-{cond}" + (f"-{b}" if b else "")
                rec = {"run": name, "partner": p, "cond": cond, "budget": b or "30b-cooled"}
                for f in (pf_stats(work, name), mub_stats(work, name), kn_stats(kn_dir, name)):
                    if f:
                        rec.update(f)
                recs.append(rec)
    out = ROOT / "results" / "trajectories"; out.mkdir(exist_ok=True)
    json.dump(recs, open(out / "trajectories.json", "w"), indent=1)
    keys = sorted({k for r in recs for k in r})
    with open(out / "trajectories.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(recs)
    # markdown: per partner, budgets x metrics, fair / starved side by side
    M = [("bpb_partner", "bpb X", ".4f"), ("pf_acc_partner", "P@1 X (PolyFact)", ".3f"), ("pf_rankc", "RankC PolyFact", ".3f"),
         ("pf_same_wrong", "same-wrong-answer", ".3f"), ("mub_rankc_sumWhole", "RankC MuBench", ".3f"),
         ("dks_k100", "dKS K=100", ".4f"), ("transfer_rate", "transfer rate", ".3f"), ("own_flip", "own-fact flip", ".3f"), ("kn_n_facts", "n facts", "d")]
    lines = ["# Consistency and parametric sharing over training (8 bilinguals)", "",
             "Each cell: fair / starved. `—` = not measured. Budgets 2b–23b are mid-stable at peak LR; 30b is the cooled final.", ""]
    for p in PARTNERS:
        lines += [f"## en-{p}", "", "| metric | " + " | ".join(b or "30b (cooled)" for b in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
        for key, label, fmt in M:
            row = []
            for b in BUDGETS:
                cells = []
                for cond in ("fair", "starved"):
                    r = next(x for x in recs if x["partner"] == p and x["cond"] == cond and x["budget"] == (b or "30b-cooled"))
                    v = r.get(key); cells.append("—" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt))
                row.append(" / ".join(cells))
            lines.append(f"| {label} | " + " | ".join(row) + " |")
        lines.append("")
    lines += ["## fair − starved by budget", "", "| partner | metric | " + " | ".join(b or "30b" for b in BUDGETS) + " |", "|---|---|" + "---|" * len(BUDGETS)]
    for p in PARTNERS:
        for key, label, fmt in M[:-2]:
            row = []
            for b in BUDGETS:
                F = next(x for x in recs if x["partner"] == p and x["cond"] == "fair" and x["budget"] == (b or "30b-cooled"))
                S = next(x for x in recs if x["partner"] == p and x["cond"] == "starved" and x["budget"] == (b or "30b-cooled"))
                if F.get(key) is None or S.get(key) is None:
                    row.append("—")
                else:
                    row.append(format(F[key] - S[key], "+" + fmt))
            lines.append(f"| {p} | {label} | " + " | ".join(row) + " |")
    (out / "trajectories.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
