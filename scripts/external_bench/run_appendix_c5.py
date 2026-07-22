#!/usr/bin/env python
"""Replicates the multilingual benchmark suite from Appendix C.5 of Messmer,
Sabolcec & Jaggi 2025, "Enhancing Multilingual LLM Pretraining with
Model-Based Data Selection" (https://arxiv.org/abs/2502.10361), across our
five languages (en/de/fr/ar/zh) and all XScript checkpoints.

Appendix C.5's per-language tables (17-20, plus Sec. 4.2.6 for French) draw
from a larger pool of benchmarks (Table 21) that includes single-language
knowledge exams (CMMLU, C-Eval, ArabicMMLU, AGIEval, EXAMS, ...) and
F1-scored extractive QA (MLQA, CMRC2018, ARCD, FQuAD, TyDiQA, Mintaka).
Those need either a language we don't train on or a generative span-
extraction + F1 pipeline this harness doesn't have (our XScriptLM only
implements loglikelihood scoring, plus a bare greedy generate_until with no
answer-extraction/F1 wiring). This script instead replicates the subset that
is (a) loglikelihood-scorable with the existing Neuron-safe scoring path and
(b) covers en/de/fr/ar/zh, using the paper's own stated methodology
(Appendix D): 0-shot, cloze multiple-choice (answer text as the target, not
A/B/C/D letters), normalized accuracy.

    XNLI          en/de/fr/ar/zh  (ar/zh routed through the debiased path,
                                    see xscript.eval.bench.XNLI_DEBIAS_METHOD)
    Belebele      en/de/fr/ar/zh  (custom CLOZE task -- lm-eval's registered
                                    `belebele` task uses A/B/C/D letters,
                                    which is NOT what the paper does; see
                                    c5_tasks/belebele_cloze/)
    ARC           en/de/fr/ar/zh  (native arc_easy for en, `okapi` M-ARC
                                    translations for de/fr/ar/zh)
    HellaSwag     en/de/fr/ar     (native hellaswag for en, `okapi` for
                                    de/fr/ar -- no Chinese translation exists
                                    in this lm-eval build, so zh is skipped)
    XStoryCloze   en/ar/zh        (dataset doesn't cover de/fr)
    XWinograd     en/fr/zh        (dataset doesn't cover de/ar)

Every model is evaluated against every language's task set, not just its own
training languages, so this also reads off zero-shot cross-lingual transfer.

Usage:
    export HF_TOKEN=hf_...
    python run_appendix_c5.py --repo jvonrad/xscript-eval --device xla \
      --limit 200 --batch-size 8 --workdir $WORK        # quick pass
    python run_appendix_c5.py --repo jvonrad/xscript-eval --device xla \
      --batch-size 8 --workdir $WORK                    # full suite

Results: `$WORK/results/appendix_c5/<run>_final.json`, one per model, with
scores nested as `{lang: {task: accuracy}}`. No shared summary.json -- see
NEURON.md 5's note on concurrent writers clobbering a shared results file when
fanned out across cores; aggregate from the per-run files instead.
"""
import argparse
import json
import os
import sys
from pathlib import Path

C5_TASKS = {
    "en": ["xnli_en", "belebele_cloze_eng_Latn", "arc_easy", "hellaswag",
           "xstorycloze_en", "xwinograd_en"],
    "de": ["xnli_de", "belebele_cloze_deu_Latn", "arc_de", "hellaswag_de"],
    "fr": ["xnli_fr", "belebele_cloze_fra_Latn", "arc_fr", "hellaswag_fr",
           "xwinograd_fr"],
    "ar": ["xnli_ar", "belebele_cloze_arb_Arab", "arc_ar", "hellaswag_ar",
           "xstorycloze_ar"],
    "zh": ["xnli_zh", "belebele_cloze_zho_Hans", "arc_zh",
           "xstorycloze_zh", "xwinograd_zh"],
}


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
    ap.add_argument("--langs", nargs="*", default=None, choices=list(C5_TASKS),
                    help="subset of languages to evaluate every model on "
                         "(default: all 5, regardless of the model's own "
                         "training languages)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="batch size for Belebele (long passages -- this is the "
                         "task that sets the HBM ceiling on --device xla)")
    ap.add_argument("--batch-size-short", type=int, default=32,
                    help="batch size for every other task (XNLI/ARC/HellaSwag/"
                         "StoryCloze/Winograd have much shorter fixed_width than "
                         "Belebele, so on --device xla they clear the same "
                         "per-graph HBM ceiling at a much larger batch size)")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu / xla (Neuron). Default: auto (cuda else cpu).")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="keep each 4GB checkpoint after eval (default: delete to save disk)")
    args = ap.parse_args()
    langs = args.langs or list(C5_TASKS)

    from huggingface_hub import hf_hub_download, list_repo_files

    def _dl_retry(filename: str, tries: int = 8) -> Path:
        """hf_hub_download with backoff -- the hub 429s ('maximum queue size
        reached') when the fan-out starts N processes at once, and EVERY startup
        below touches the network. Without this a 24-way fan-out loses most of
        its jobs in the first seconds, before any Neuron work begins."""
        import random
        import time
        for attempt in range(tries):
            try:
                return Path(hf_hub_download(filename=filename, **dl))
            except Exception as exc:
                if attempt == tries - 1:
                    raise
                wait = min(60, 2 ** attempt) + random.uniform(0, 3)
                print(f"[c5] {type(exc).__name__} on {filename}; "
                      f"retry {attempt + 1}/{tries - 1} in {wait:.0f}s", flush=True)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def fetch_checkpoint(rel_dir: str) -> Path:
        """Download final.pt, transparently reassembling `final.pt.partNNN`
        chunks if the checkpoint was uploaded split (see upload_chunked.py)."""
        parts = sorted(f for f in repo_files
                       if f.startswith(f"{rel_dir}/final.pt.part"))
        # Already assembled (e.g. by prefetch_checkpoints.py)? Then do NOT touch
        # the network -- not even for n_parts.txt. Under fan-out that lookup is
        # pure 429 bait for a file we do not need, and under HF_HUB_OFFLINE it
        # raises outright. Validate size ONLY when the manifest gives one:
        #   want > 0  -> known size; return on match, refetch on mismatch
        #   want == 0 -> size unknown (offline repo_info yields size=None), so
        #                trust the assembly -- the prefetcher already verified it
        #                byte-exact, and a network refetch is impossible anyway.
        out = work / "_assembled" / rel_dir / "final.pt"
        if parts and out.exists():
            want = sum(sizes.get(p, 0) for p in parts)
            if not want or out.stat().st_size == want:
                return out
            print(f"[c5] {rel_dir}: assembled size {out.stat().st_size} != "
                  f"{want}, refetching", flush=True)
            out.unlink(missing_ok=True)

        whole = f"{rel_dir}/final.pt"
        if whole in repo_files:
            return _dl_retry(whole)
        if not parts:
            sys.exit(f"no checkpoint found under {rel_dir} (neither final.pt nor parts)")
        n_parts_f = f"{rel_dir}/n_parts.txt"
        if n_parts_f in repo_files:
            expected = int(_dl_retry(n_parts_f).read_text())
            if len(parts) != expected:
                sys.exit(f"{rel_dir}: expected {expected} parts, found {len(parts)} "
                        f"(upload still in progress?)")
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            # unique tmp + atomic replace: a shared "final.tmp" races if two
            # processes ever assemble the same checkpoint.
            tmp = out.with_suffix(f".tmp.{os.getpid()}")
            try:
                with open(tmp, "wb") as w:
                    for p in parts:
                        local = _dl_retry(p)
                        with open(local, "rb") as r:
                            while chunk := r.read(64 * 1024 * 1024):
                                w.write(chunk)
                        local.unlink(missing_ok=True)  # shard folded into `out`
                os.replace(tmp, out)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        return out

    work = Path(args.workdir).resolve()
    scratch = work / "xscript"
    (scratch / "runs").mkdir(parents=True, exist_ok=True)
    (scratch / "tokenizers").mkdir(parents=True, exist_ok=True)
    os.environ["XSCRIPT_SCRATCH"] = str(scratch)
    os.environ["XSCRIPT_RESULTS"] = str(work / "results")

    dl = dict(repo_id=args.repo, repo_type=args.repo_type, local_dir=str(scratch.parent / "_repo"))

    # Repo file listing + sizes, cached on disk. N parallel `list_repo_files`
    # calls 429 before a single checkpoint transfers (same reason run_bpb.py
    # caches it); the sizes let fetch_checkpoint validate an already-assembled
    # checkpoint without any network call. Delete _repo_files.json after new
    # uploads to refresh.
    _listing = work / "_repo_files.json"
    sizes = {}
    if _listing.exists():
        try:
            sizes = json.loads(_listing.read_text())
        except json.JSONDecodeError:
            sizes = {}
    if not sizes:
        from huggingface_hub import HfApi
        _info = HfApi().repo_info(args.repo, repo_type=args.repo_type,
                                  files_metadata=True)
        sizes = {s.rfilename: (s.size or 0) for s in _info.siblings}
        tmp_l = _listing.with_suffix(f".tmp.{os.getpid()}")
        tmp_l.write_text(json.dumps(sizes))
        os.replace(tmp_l, _listing)     # atomic: many processes race to write it
    repo_files = list(sizes)

    # 1) bundled xscript source -> importable
    src_root = scratch.parent / "_repo"
    for f in repo_files:
        if f.startswith("src/xscript/"):
            _dl_retry(f)
    sys.path.insert(0, str(src_root / "src"))
    # Prefer this repo's local src (this script's own C.5 task suite, the
    # debiased-XNLI routing, and the Neuron scoring fixes all live there).
    _local_src = Path(__file__).resolve().parents[2] / "src"
    if (_local_src / "xscript" / "eval" / "bench.py").exists():
        sys.path.insert(0, str(_local_src))
    c5_tasks_dir = _local_src / "xscript" / "eval" / "c5_tasks"
    if not c5_tasks_dir.exists():
        sys.exit(f"custom C.5 task configs not found at {c5_tasks_dir} -- "
                 "this script only runs from inside the XScript-Pretraining repo.")

    # 2) tokenizers (small)
    for f in repo_files:
        if f.startswith("tokenizers/"):
            local = _dl_retry(f)
            dest = scratch / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(local)

    # 3) model manifest (friendly name -> real tokenizer)
    models = json.loads(_dl_retry("models.json").read_text())
    runs = args.runs or sorted(models)
    missing = [r for r in runs if r not in models]
    if missing:
        sys.exit(f"models not in repo: {missing}\navailable: {sorted(models)}")
    print(f"[c5] {len(runs)} model(s) x {len(langs)} language(s) "
         f"({', '.join(langs)}), limit={args.limit}")

    import torch
    import lm_eval
    from lm_eval.tasks import TaskManager
    from xscript.model import ModelConfig, Transformer
    from xscript.tok.wrapper import Tok
    from xscript.paths import tokenizer_dir, ensure
    from xscript.eval.bench import _make_lm, XNLI_DEBIAS_METHOD, _xnli_debiased

    task_manager = TaskManager(include_path=str(c5_tasks_dir))

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "xla":
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print("[c5] using XLA/Neuron (fixed-shape scoring).")
    else:
        device = torch.device(args.device)

    def _accuracy(rec: dict):
        # "Normalized accuracy" per the paper's Appendix D; falls back to
        # plain acc for tasks with no length-bias correction (XNLI, XStory-
        # Cloze, XWinograd only report acc).
        return rec.get("acc_norm,none", rec.get("acc,none",
               rec.get("acc_norm", rec.get("acc"))))

    def _per_example_hits(samples: list[dict]) -> list[int]:
        # Same acc_norm-over-acc preference as _accuracy, but per doc -- the
        # 0/1 hit list a bootstrap CI needs. lm_eval's log_samples records
        # carry both metrics as direct float fields on each sample dict.
        return [int(round(s.get("acc_norm", s.get("acc")))) for s in samples]

    out_dir = ensure(Path(work / "results" / "appendix_c5"))
    for i, run in enumerate(runs, 1):
        tok_name = models[run]["tok"]
        print(f"\n===== [{i}/{len(runs)}] {run} (tok={tok_name}) =====")
        ckpt_rel = f"runs/{run}/checkpoints/final.pt"
        local_ckpt = fetch_checkpoint(f"runs/{run}/checkpoints")
        dest = scratch / ckpt_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.symlink_to(local_ckpt)

        try:
            ck = torch.load(dest, map_location="cpu", weights_only=False)
            model = Transformer(ModelConfig(**ck["cfg"]["model"])).to(device).eval()
            model.load_state_dict(ck["model"])
            tok = Tok(tokenizer_dir(tok_name))
            adapter = _make_lm(model, tok, device, model.cfg.max_seq_len)

            debiased_langs = [lang for lang in langs
                              if lang in XNLI_DEBIAS_METHOD
                              and f"xnli_{lang}" in C5_TASKS[lang]]
            debiased_tasks = {f"xnli_{lang}" for lang in debiased_langs}
            all_tasks = [t for lang in langs for t in C5_TASKS[lang]]
            harness_tasks = [t for t in all_tasks if t not in debiased_tasks]
            # Belebele's long passages set the per-graph HBM ceiling on
            # --device xla (see NEURON.md sec 4/5); every other task's
            # fixed_width is much shorter and clears the same ceiling at a
            # larger batch size, so score them in a separate, wider-batched
            # simple_evaluate() call rather than paying Belebele's cap
            # everywhere.
            belebele_tasks = [t for t in harness_tasks if t.startswith("belebele_cloze")]
            short_tasks = [t for t in harness_tasks if t not in belebele_tasks]

            # log_samples=True so per-example correctness is available to
            # bootstrap confidence intervals on transfer deltas -- see
            # CLAUDE.md's "Same-script vs. cross-script transfer" open item.
            groups, subtasks, samples = {}, {}, {}
            for task_group, bs in [(belebele_tasks, args.batch_size),
                                    (short_tasks, args.batch_size_short)]:
                if not task_group:
                    continue
                adapter.batch_size = bs
                results = lm_eval.simple_evaluate(
                    model=adapter, tasks=task_group, num_fewshot=0, batch_size=1,
                    limit=args.limit, log_samples=True, confirm_run_unsafe_code=True,
                    task_manager=task_manager,
                )
                groups.update(results.get("groups", {}))
                subtasks.update(results.get("results", {}))
                samples.update(results.get("samples", {}))

            scores = {}
            correct = {}
            for lang in langs:
                lang_scores = {}
                lang_correct = {}
                for t in C5_TASKS[lang]:
                    if lang in debiased_langs and t == f"xnli_{lang}":
                        acc, hits = _xnli_debiased(adapter, lang, args.limit, return_correct=True)
                        lang_scores[t] = acc
                        lang_correct[t] = hits
                    else:
                        rec = groups.get(t, subtasks.get(t, {}))
                        lang_scores[t] = _accuracy(rec)
                        if t in samples:
                            lang_correct[t] = _per_example_hits(samples[t])
                scores[lang] = lang_scores
                correct[lang] = lang_correct
                print(f"[c5] {run} / {lang}: " +
                     ", ".join(f"{k}={v:.4f}" for k, v in lang_scores.items() if v is not None))

            payload = {
                "run": run, "tokenizer": tok_name, "limit": args.limit,
                "langs": langs, "tasks": {l: C5_TASKS[l] for l in langs},
                "xnli_debiased": {l: XNLI_DEBIAS_METHOD[l] for l in debiased_langs},
                "scores": scores, "correct": correct,
            }
            (out_dir / f"{run}_final.json").write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            print(f"[c5] {run} FAILED: {type(exc).__name__}: {exc}")
            (out_dir / f"{run}_final.json").write_text(
                json.dumps({"run": run, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        finally:
            if not args.keep_checkpoints:
                try:
                    real = Path(local_ckpt).resolve()
                    dest.unlink(missing_ok=True)
                    real.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"[c5] cleanup warning for {run}: {exc}")

    print(f"\n[c5] per-model JSON in {out_dir}")


if __name__ == "__main__":
    main()
