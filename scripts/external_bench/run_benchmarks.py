#!/usr/bin/env python
"""Standalone downstream-benchmark runner for XScript checkpoints on any GPU.

Isambard-AI is walled off by a CPU-minutes quota, so we export the trained
checkpoints to a (private) HF repo and evaluate them elsewhere. This script
pulls the checkpoints + tokenizers + bundled `xscript` source from that repo,
lays them out exactly as `xscript.paths` expects, and runs the lm-eval-harness
benchmarks (Global-MMLU / Belebele / XNLI) via the same `eval-bench` harness
used on-cluster -- so scores are directly comparable.

Quick validation pass over everything (recommended first):
    pip install -r requirements.txt
    # install a torch build matching your CUDA first, e.g.:
    #   pip install torch --index-url https://download.pytorch.org/whl/cu121
    export HF_TOKEN=hf_...            # needed while the repo is private
    python run_benchmarks.py --repo jvonrad/xscript-eval --limit 200

Full suite (all examples):
    python run_benchmarks.py --repo jvonrad/xscript-eval

Results land in ./xscript_bench/results/bench/<run>_final.json (one per run),
plus a combined summary.json. Send those JSONs back for analysis.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="HF repo id holding the export")
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--workdir", default="./xscript_bench")
    ap.add_argument("--limit", type=float, default=None,
                    help="examples per task (omit for full suite)")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="subset of friendly model names (default: all in models.json)")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="override task list (default: the run's training languages)")
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None,
                    help="cuda / cpu / xla (Neuron). Default: auto (cuda else cpu). "
                         "Use 'xla' on Trainium for the fixed-shape scoring path.")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="keep each 4GB checkpoint after eval (default: delete to save disk)")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    def fetch_checkpoint(rel_dir: str) -> Path:
        """Download final.pt, transparently reassembling `final.pt.partNNN`
        chunks if the checkpoint was uploaded split (see upload_chunked.py)."""
        whole = f"{rel_dir}/final.pt"
        if whole in repo_files:
            return Path(hf_hub_download(filename=whole, **dl))
        parts = sorted(f for f in repo_files
                       if f.startswith(f"{rel_dir}/final.pt.part"))
        if not parts:
            sys.exit(f"no checkpoint found under {rel_dir} (neither final.pt nor parts)")
        n_parts_f = f"{rel_dir}/n_parts.txt"
        if n_parts_f in repo_files:
            expected = int(Path(hf_hub_download(filename=n_parts_f, **dl)).read_text())
            if len(parts) != expected:
                sys.exit(f"{rel_dir}: expected {expected} parts, found {len(parts)} "
                        f"(upload still in progress?)")
        assembled_dir = work / "_assembled" / rel_dir
        assembled_dir.mkdir(parents=True, exist_ok=True)
        out = assembled_dir / "final.pt"
        if not out.exists():
            tmp = out.with_suffix(".tmp")
            with open(tmp, "wb") as w:
                for p in parts:
                    local = hf_hub_download(filename=p, **dl)
                    with open(local, "rb") as r:
                        while chunk := r.read(64 * 1024 * 1024):
                            w.write(chunk)
                    Path(local).unlink(missing_ok=True)  # raw shard now folded into `out`
            tmp.rename(out)
        return out

    work = Path(args.workdir).resolve()
    scratch = work / "xscript"
    (scratch / "runs").mkdir(parents=True, exist_ok=True)
    (scratch / "tokenizers").mkdir(parents=True, exist_ok=True)
    os.environ["XSCRIPT_SCRATCH"] = str(scratch)
    os.environ["XSCRIPT_RESULTS"] = str(work / "results")

    dl = dict(repo_id=args.repo, repo_type=args.repo_type, local_dir=str(scratch.parent / "_repo"))
    repo_files = list_repo_files(args.repo, repo_type=args.repo_type)

    # 1) bundled xscript source -> importable
    src_root = scratch.parent / "_repo"
    for f in repo_files:
        if f.startswith("src/xscript/"):
            hf_hub_download(filename=f, **dl)
    sys.path.insert(0, str(src_root / "src"))
    # When run from inside the XScript-Pretraining repo, prefer its src so local
    # patches (e.g. the fixed-shape XLA/Neuron scoring path in eval/bench.py)
    # take precedence over the bundled export. No-op for standalone use.
    _local_src = Path(__file__).resolve().parents[2] / "src"
    if (_local_src / "xscript" / "eval" / "bench.py").exists():
        sys.path.insert(0, str(_local_src))

    # 2) tokenizers (small)
    for f in repo_files:
        if f.startswith("tokenizers/"):
            local = hf_hub_download(filename=f, **dl)
            dest = scratch / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(local)

    # 3) model manifest (friendly name -> real tokenizer)
    models = json.loads(Path(hf_hub_download(filename="models.json", **dl)).read_text())
    runs = args.runs or sorted(models)
    missing = [r for r in runs if r not in models]
    if missing:
        sys.exit(f"models not in repo: {missing}\navailable: {sorted(models)}")
    print(f"[bench] {len(runs)} model(s) to evaluate: {runs}")

    import torch
    from xscript.eval import bench
    if args.device == "xla":
        print("[bench] using XLA/Neuron (fixed-shape scoring).")
    elif not torch.cuda.is_available() and args.device != "cpu":
        print("[bench] WARNING: no CUDA device -- this will be very slow on CPU.")

    summary = {}
    for i, run in enumerate(runs, 1):
        tok = models[run]["tok"]
        print(f"\n===== [{i}/{len(runs)}] {run} (tok={tok}, limit={args.limit}) =====")
        ckpt_rel = f"runs/{run}/checkpoints/final.pt"
        local_ckpt = fetch_checkpoint(f"runs/{run}/checkpoints")
        dest = scratch / ckpt_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.symlink_to(local_ckpt)
        try:
            scores = bench.run(run, tok, tag="final", tasks=args.tasks,
                               num_fewshot=args.num_fewshot, limit=args.limit,
                               log_wandb=False, batch_size=args.batch_size,
                               device=args.device)
            summary[run] = scores
        except Exception as exc:
            print(f"[bench] {run} FAILED: {type(exc).__name__}: {exc}")
            summary[run] = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            if not args.keep_checkpoints:
                # free the ~4GB blob (both the symlink target in HF cache and our link)
                try:
                    real = Path(local_ckpt).resolve()
                    dest.unlink(missing_ok=True)
                    real.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"[bench] cleanup warning for {run}: {exc}")

    out = work / "results" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"limit": args.limit, "num_fewshot": args.num_fewshot,
                               "scores": summary}, indent=2))
    print(f"\n[bench] wrote {out}")
    print(f"[bench] per-run JSON in {work / 'results' / 'bench'}")


if __name__ == "__main__":
    main()
