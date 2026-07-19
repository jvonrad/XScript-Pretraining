"""`xscript` command-line entry point.

Pipeline order (see README):
  flores-download -> byte-premium
  tok-corpus -> tok-train -> tok-analyze            (the tokenizer gate)
  pool -> pack                                       (per language x chosen tok)
  train                                              (one run of the matrix)
  eval-bpb / eval-align -> bts                       (headline analysis)

Heavy steps are meant to run inside Slurm jobs (see slurm/); the CLI is the
single interface those jobs call, so behaviour is identical locally and on the
compute nodes.
"""
import argparse

from .langs import (LANGS, TOK_FLAVORS, TOK_CONDITIONS, MODEL_FLAVORS,
                    tok_name, tok_conditions)


def _add(sub, name, help):
    p = sub.add_parser(name, help=help)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(prog="xscript", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- data prep ----
    p = _add(sub, "flores-download", "download FLORES+ dev/devtest (needs HF_TOKEN)")
    p.add_argument("--langs", nargs="*", default=list(LANGS))

    _add(sub, "byte-premium", "compute FLORES+ byte premiums (+ compare Arnett)")

    p = _add(sub, "tok-corpus", "build raw FineWeb/FineWeb2 tokenizer-training corpora")
    p.add_argument("condition", choices=TOK_CONDITIONS + ["both"])
    p.add_argument("--gb", type=float, default=4.0, help="target corpus size (GB)")

    p = _add(sub, "tok-train", "train tokenizer(s): unigram/bpe/pa")
    p.add_argument("--flavor", choices=TOK_FLAVORS + ["all"], default="all")
    p.add_argument("--condition", choices=TOK_CONDITIONS + ["both"], default="both")

    p = _add(sub, "tok-analyze", "fertility / allocation gate on FLORES+")
    p.add_argument("--toks", nargs="*", default=None)

    # ---- model-data prep ----
    p = _add(sub, "pool", "build FineWeb(-2)-HQ text pool for a language")
    p.add_argument("--lang", required=True, choices=list(LANGS))
    p.add_argument("--gb", type=float, default=None, help="override byte budget (GB)")

    p = _add(sub, "pack", "tokenize a pool into uint16 shards")
    p.add_argument("--lang", required=True, choices=list(LANGS))
    p.add_argument("--tok", required=True)
    p.add_argument("--workers", type=int, default=8)

    _add(sub, "plan", "print per-language pool budgets and the run matrix")

    p = _add(sub, "runs", "list generated run names")
    p.add_argument("--base", default="configs/base_main.yaml")
    p.add_argument("--flavor", default="unigram", choices=MODEL_FLAVORS)
    p.add_argument("--only-30b", action="store_true",
                   help="list 18 independent 30B runs (no extension trunks)")

    # ---- training ----
    p = _add(sub, "train", "train one run of the matrix")
    p.add_argument("name", help="run name (see `xscript runs`)")
    p.add_argument("--base", default="configs/base_main.yaml")
    p.add_argument("--flavor", default="unigram", choices=MODEL_FLAVORS)
    p.add_argument("--only-30b", action="store_true",
                   help="use a self-contained 30B WSD config, never a trunk branch")
    p.add_argument("--output-name", default=None,
                   help="store an independent diagnostic replicate under this run name")
    p.add_argument("--seed", type=int, default=None,
                   help="override model/optimizer RNG seed for a diagnostic replicate")
    p.add_argument("--data-seed", type=int, default=None,
                   help="override packed-stream order seed for a diagnostic replicate")
    p.add_argument("--wandb-id", default=None,
                   help="override the stable W&B run ID (useful for a clean replacement run)")

    # ---- eval ----
    p = _add(sub, "eval-bpb", "re-evaluate a checkpoint's BPB")
    p.add_argument("name"); p.add_argument("--tok", required=True)
    p.add_argument("--tag", default="final")

    p = _add(sub, "eval-align", "MEXA alignment for a run")
    p.add_argument("name"); p.add_argument("--tok", required=True)
    p.add_argument("--split", default="dev")

    p = _add(sub, "eval-bench", "downstream benchmarks (Global-MMLU/Belebele/XNLI) via lm-eval-harness")
    p.add_argument("name"); p.add_argument("--tok", required=True)
    p.add_argument("--tag", default="final")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="override tasks; default is all three benchmarks for the run's languages")
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--limit", type=float, default=None,
                   help="cap examples/task (for quick smoke checks)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="likelihood requests per GPU batch")
    p.add_argument("--no-wandb", action="store_true")

    p = _add(sub, "bts", "compute BTS + interaction across runs")
    p.add_argument("--flavor", default="unigram", choices=MODEL_FLAVORS)
    p.add_argument("--source", default="flores", choices=["flores", "holdout"])

    args = ap.parse_args(argv)
    return _dispatch(args)


def _dispatch(args):
    cmd = args.cmd
    if cmd == "flores-download":
        from . import flores
        flores.download(args.langs)

    elif cmd == "byte-premium":
        from . import byte_premium
        byte_premium.run()

    elif cmd == "tok-corpus":
        from .data import tokcorpus
        conds = TOK_CONDITIONS if args.condition == "both" else [args.condition]
        for c in conds:
            if c == "starved":
                tokcorpus.build_starved(total_bytes=args.gb * 1e9)
            else:
                tokcorpus.build_destarved(total_bytes=args.gb * 1e9)

    elif cmd == "tok-train":
        from .tok import train as toktrain
        flavors = TOK_FLAVORS if args.flavor == "all" else [args.flavor]
        want = TOK_CONDITIONS if args.condition == "both" else [args.condition]
        for f in flavors:
            for c in want:
                if c not in tok_conditions(f):
                    continue  # pa has no starved condition
                print(f"[tok-train] {tok_name(f, c)}")
                toktrain.train(f, c)

    elif cmd == "tok-analyze":
        from .tok import analyze
        analyze.run(args.toks)

    elif cmd == "pool":
        from .data import fineweb
        budget = (args.gb * 1e9) if args.gb else fineweb.plan_budgets()[args.lang]
        fineweb.build_pool(args.lang, budget)

    elif cmd == "pack":
        from .data import pack
        pack.pack(args.lang, args.tok, workers=args.workers)

    elif cmd == "plan":
        _plan()

    elif cmd == "runs":
        from . import runmatrix
        for n in runmatrix.list_runs(args.base, args.flavor, args.only_30b):
            print(n)

    elif cmd == "train":
        from . import runmatrix, train
        cfg = runmatrix.get_run(args.base, args.flavor, args.name, args.only_30b)
        if args.output_name is not None:
            cfg["name"] = args.output_name
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.data_seed is not None:
            cfg["data_seed"] = args.data_seed
        if args.wandb_id is not None:
            cfg["wandb_id"] = args.wandb_id
        train.run_from_config(cfg)

    elif cmd == "eval-bpb":
        from .eval import bpb
        bpb.run(args.name, args.tok, args.tag)

    elif cmd == "eval-align":
        from .eval import alignment
        alignment.run(args.name, args.tok, args.split)

    elif cmd == "eval-bench":
        from .eval import bench
        bench.run(args.name, args.tok, args.tag, tasks=args.tasks,
                  num_fewshot=args.num_fewshot, limit=args.limit,
                  log_wandb=not args.no_wandb, batch_size=args.batch_size)

    elif cmd == "bts":
        from .eval import bts
        bts.run(args.flavor, args.source)


def _plan():
    from .data.fineweb import plan_budgets
    from . import runmatrix
    b = plan_budgets()
    print("Per-language pool byte budgets (worst-case, destarved tokenizer):")
    for l, v in b.items():
        print(f"  {l}: {v/1e9:.1f} GB")
    print("\nRun matrix (flavor=unigram):")
    from . import _yaml
    base = _yaml.load("configs/base_main.yaml")
    for n in sorted(runmatrix.all_runs(base, "unigram")):
        print(f"  {n}")


if __name__ == "__main__":
    main()
