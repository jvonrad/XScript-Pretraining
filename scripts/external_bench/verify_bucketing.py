#!/usr/bin/env python
"""Verify the length-bucketed XLA batching reproduces the unbucketed path.

`bench.XScriptLM._loglikelihood_tokens` sorts requests by length and pads each
batch to its own (laddered) width instead of to one task-wide width. That is a
~10x speedup on skewed tasks but it changes which requests share a batch, so it
must be shown not to change any score.

This re-scores a task with the CURRENT code and compares against the raw
loglikelihoods stored by an earlier sweep that used the task-wide width.

    python verify_bucketing.py --repo jvonrad/xscript-eval --device xla \
      --workdir $WORK --run en-fair --lang en --task sib200_eng_Latn
"""
import argparse, json, sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--device", default="xla")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=float, default=200)
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()

    work = Path(args.workdir)
    ref_path = work / "results" / "extra_bench" / "raw" / f"{args.run}_raw.json"
    if not ref_path.exists():
        sys.exit(f"no reference raw scores at {ref_path}")
    ref = json.loads(ref_path.read_text())["raw"][args.lang][args.task]

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    import os
    scratch = work / "xscript"
    os.environ["XSCRIPT_SCRATCH"] = str(scratch)
    import torch, lm_eval
    from lm_eval.tasks import TaskManager
    from xscript.model import ModelConfig, Transformer
    from xscript.tok.wrapper import Tok
    from xscript.paths import tokenizer_dir
    from xscript.eval.bench import _make_lm
    from xscript.eval.rawscores import extract_raw

    models = json.loads((work / "_repo" / "models.json").read_text())
    tok_name = models[args.run]["tok"]
    ckpt = work / "_assembled" / f"runs/{args.run}/checkpoints" / "final.pt"
    if not ckpt.exists():
        sys.exit(f"checkpoint not staged at {ckpt}; run with --keep-checkpoints first")

    device = (__import__("torch_xla.core.xla_model", fromlist=["xla_device"]).xla_device()
              if args.device == "xla" else torch.device(args.device))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**ck["cfg"]["model"])).to(device).eval()
    model.load_state_dict(ck["model"])
    adapter = _make_lm(model, Tok(tokenizer_dir(tok_name)), device, model.cfg.max_seq_len)
    adapter.batch_size = args.batch_size

    tm = TaskManager(include_path=str(repo_root / "src/xscript/eval/c5_tasks"))
    res = lm_eval.simple_evaluate(model=adapter, tasks=[args.task], num_fewshot=0,
                                  batch_size=1, limit=args.limit, log_samples=True,
                                  confirm_run_unsafe_code=True, task_manager=tm)
    rec = res["results"][args.task]
    reported = {k.split(",")[0] for k in rec
                if k.split(",")[0] in ("acc", "acc_norm", "acc_mutual_info")}
    new = extract_raw(res["samples"][args.task], reported, tok=None)

    n = len(new["gold"])
    worst = 0.0
    bad = 0
    for d in range(n):
        for c in range(len(new["ll"][d])):
            delta = abs(new["ll"][d][c] - ref["ll"][d][c])
            worst = max(worst, delta)
            bad += delta > args.tol
    print(f"[verify] {args.run}/{args.lang}/{args.task}: compared {n} docs x "
          f"{new['n_choices']} choices")
    print(f"[verify] max |bucketed - taskwide| loglik delta = {worst:.3e}  "
          f"(tolerance {args.tol})")
    print(f"[verify] values over tolerance: {bad}")
    if bad:
        sys.exit("MISMATCH -- length bucketing changed scores; do not use it")
    print("[verify] OK: bucketing is score-preserving")


if __name__ == "__main__":
    main()
