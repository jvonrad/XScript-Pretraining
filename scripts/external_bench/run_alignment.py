#!/usr/bin/env python
"""Cross-lingual representation alignment (MEXA-style) for XScript checkpoints,
on Neuron/XLA (``--device xla``), CUDA, or CPU. The login node can't hold a live
~1B model at all, so like benchmarking this has to run on an accelerator box.

**Every model is evaluated on every language pair by default**, not just the
pairs it was trained on. A bilingual EN-AR model's EN-AR retrieval score means
nothing on its own -- FLORES sentences share digits, Latin-script named
entities, punctuation and length, so even a model that never saw Arabic scores
well above 1/n. The monolingual models scored on the *same* pair are the
control that turns the number into evidence, and they also give the
zero-shot readout ("does EN-DE training buy you EN-AR alignment?"). Restrict
with ``--langs`` if you only want the trained cells.

    export HF_TOKEN=hf_...
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; source ~/neuron_venv/bin/activate
    python run_alignment.py --repo jvonrad/xscript-eval --device xla --workdir $WORK

Writes ``$WORK/results/alignment/<model>.json`` (+ ``.md``). Like
run_appendix_c5.py it deliberately writes **no shared summary.json**, so a
fan-out over logical core-pairs can't clobber itself (CLAUDE.md section 5);
aggregate with analyze_alignment.py.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ALL_LANGS = ["en", "de", "fr", "ar", "zh"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--workdir", default="./xscript_bench")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="subset of friendly model names (default: all in models.json)")
    ap.add_argument("--langs", nargs="*", default=ALL_LANGS,
                    help=f"languages to embed (default: {' '.join(ALL_LANGS)}). "
                         "All unordered pairs among these are scored.")
    ap.add_argument("--split", default="both", choices=["dev", "devtest", "both"],
                    help="'both' = dev+devtest (2009 sentences). Default, because "
                         "top-1 retrieval over dev alone (997) saturates above "
                         "0.95 even for monolingual controls.")
    ap.add_argument("--device", default=None, choices=["xla", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="truncate sentences to this many tokens; also caps the "
                         "fixed graph width on XLA")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N parallel sentences (smoke tests)")
    ap.add_argument("--keep-checkpoints", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    work = Path(args.workdir).resolve()
    scratch = work / "xscript"
    (scratch / "runs").mkdir(parents=True, exist_ok=True)
    (scratch / "tokenizers").mkdir(parents=True, exist_ok=True)
    os.environ["XSCRIPT_SCRATCH"] = str(scratch)
    os.environ["XSCRIPT_RESULTS"] = str(work / "results")

    dl = dict(repo_id=args.repo, repo_type=args.repo_type,
              local_dir=str(scratch.parent / "_repo"))
    repo_files = list_repo_files(args.repo, repo_type=args.repo_type)

    def fetch_checkpoint(rel_dir: str) -> Path:
        whole = f"{rel_dir}/final.pt"
        if whole in repo_files:
            return Path(hf_hub_download(filename=whole, **dl))
        parts = sorted(f for f in repo_files if f.startswith(f"{rel_dir}/final.pt.part"))
        if not parts:
            sys.exit(f"no checkpoint found under {rel_dir}")
        n_parts_rel = f"{rel_dir}/n_parts.txt"
        if n_parts_rel in repo_files:
            want = int(Path(hf_hub_download(filename=n_parts_rel, **dl)).read_text().strip())
            if want != len(parts):
                sys.exit(f"{rel_dir}: expected {want} parts, found {len(parts)}")
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
            tmp.rename(out)
        return out

    # Prefer this repo's local src/ over the bundled HF export so local patches
    # to alignment.py take effect (same convention as run_benchmarks.py).
    local_src = Path(__file__).resolve().parents[2] / "src"
    src_root = scratch.parent / "_repo"
    for f in repo_files:
        if f.startswith("src/xscript/"):
            hf_hub_download(filename=f, **dl)
    sys.path.insert(0, str(src_root / "src"))
    if (local_src / "xscript" / "eval" / "alignment.py").exists():
        sys.path.insert(0, str(local_src))
        print(f"[align] using local src: {local_src}")
    for f in repo_files:
        if f.startswith("tokenizers/"):
            local = hf_hub_download(filename=f, **dl)
            dest = scratch / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(local)

    models = json.loads(Path(hf_hub_download(filename="models.json", **dl)).read_text())
    runs = args.runs or sorted(models)
    missing = [r for r in runs if r not in models]
    if missing:
        sys.exit(f"models not in repo: {missing}\navailable: {sorted(models)}")
    bad = [l for l in args.langs if l not in ALL_LANGS]
    if bad:
        sys.exit(f"unknown languages {bad}; choose from {ALL_LANGS}")
    print(f"[align] {len(runs)} model(s) x {len(args.langs)} langs "
          f"({args.langs}): {runs}")

    import torch
    from xscript import flores
    from xscript.eval import alignment

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[align] --device not given, using {device}"
              + ("" if device == "cuda" else " (slow; pass --device xla on Neuron)"))
    if device == "xla":
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
    else:
        dev = torch.device(device)

    splits = ("dev", "devtest") if args.split == "both" else (args.split,)
    flores.download(args.langs, splits=splits, token=os.environ.get("HF_TOKEN"))

    out_dir = work / "results" / "alignment"
    for i, run in enumerate(runs, 1):
        tok = models[run]["tok"]
        print(f"\n===== [{i}/{len(runs)}] {run} (tok={tok}, device={device}) =====",
              flush=True)
        ckpt_rel = f"runs/{run}/checkpoints/final.pt"
        local_ckpt = fetch_checkpoint(f"runs/{run}/checkpoints")
        dest = scratch / ckpt_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.symlink_to(local_ckpt)
        try:
            alignment.run(run, tok, split=args.split, out_dir=out_dir,
                          device=dev, langs=args.langs, batch=args.batch_size,
                          max_tokens=args.max_tokens, limit=args.limit)
        except Exception as exc:
            import traceback
            print(f"[align] {run} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            if not args.keep_checkpoints:
                try:
                    real = Path(local_ckpt).resolve()
                    dest.unlink(missing_ok=True)
                    real.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"[align] cleanup warning for {run}: {exc}")

    print(f"\n[align] per-run JSON/MD in {out_dir}")


if __name__ == "__main__":
    main()
