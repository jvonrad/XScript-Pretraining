#!/usr/bin/env python
"""Benchmarks that cover all five languages, added on top of the Appendix-C.5
suite (`run_appendix_c5.py`) rather than replacing it.

Motivation: of the six C.5 benchmarks, only XNLI carries signal at this scale,
and four of the six do not even *cover* all five languages -- HellaSwag has no
zh column, XStoryCloze no de/fr, XWinograd no de/ar (CLAUDE.md 6). That makes
cross-language comparison rest on XNLI plus a Belebele that sits near chance.
This script adds:

    hellaswag_zh   the missing Chinese HellaSwag column. okapi's
                   `alexandrainst/m_hellaswag` DOES ship zh; lm-eval 0.4.12
                   has no task for it because 4 malformed `endings` break the
                   split's schema inference. See
                   c5_tasks/hellaswag_zh/utils.build_dataset.

    sib200_*       SIB-200 topic classification (Adelani et al. 2024,
                   arXiv:2309.07445): 7 topics over FLORES-200 sentences,
                   all 5 languages, n=1004/language, chance 1/7 = 0.143.
                   Lexical rather than reasoning-based, so it is a task an
                   undertrained model can plausibly do -- and it is scored on
                   the SAME FLORES sentences as 6's BPB and 6b's alignment,
                   so downstream capability and representation alignment are
                   measured on identical text.

    sib200_enlab_* the same texts with ENGLISH prompt and label words. SIB-200
                   ships English labels for every language, so the primary
                   tasks localize them; this control measures how much of the
                   score is "can read the English label words" rather than
                   "can classify the text".

    taxi1500_*     Taxi-1500 (Ma et al. 2023, arXiv:2305.08487): 6 topics over
                   Parallel Bible Corpus verses, all 5 languages. The backup
                   to SIB-200: scripture is a much worse domain match for
                   FineWeb2-trained checkpoints, and the only full-coverage
                   open Arabic edition is from 1865, so its register is uneven
                   ACROSS languages -- read it as corroboration, not as an
                   independent measurement. Its corpus is assembled on first
                   use (786 MB, once); pre-warm before a fan-out with
                   `python src/xscript/eval/c5_tasks/taxi1500/utils.py`.

Like run_appendix_c5.py, EVERY model is scored on EVERY language, not just its
own training languages, so this doubles as a zero-shot cross-lingual transfer
readout.

Output mirrors run_appendix_c5.py's shape -- `scores` and per-example `correct`,
nested `{lang: {task: ...}}` -- with one deliberate difference: `correct` has an
extra metric level, `{lang: {task: {metric: [0/1, ...]}}}`, because SIB-200
reports acc / acc_norm / acc_mutual_info and they do NOT agree here (see
`_all_metrics`). Picking one inside the runner would bake in exactly the kind of
unaudited scoring choice that CLAUDE.md 6 keeps having to retract, so all three
hit lists are kept and `analyze_extra_bench.py` selects. That extra level means
`bootstrap_transfer.py` cannot read these files directly; it is the same paired
estimator, applied by the analysis script.

Usage:
    export HF_TOKEN=hf_...
    python run_extra_bench.py --repo jvonrad/xscript-eval --device xla \
      --workdir $WORK --runs zh-fair-15b --families hellaswag_zh
    python run_extra_bench.py --repo jvonrad/xscript-eval --device xla \
      --workdir $WORK                       # everything, all models

Results: `$WORK/results/extra_bench/<run>_final.json`, one per model. No shared
summary.json -- see NEURON.md 5 on concurrent writers clobbering it under
fan-out; aggregate from the per-run files.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Task families, keyed by the `--families` flag value. Each maps a language to
# the tasks that exist for it, so a family that covers only one language (like
# hellaswag_zh) simply has one entry.
FAMILIES = {
    "hellaswag_zh": {"zh": ["hellaswag_zh"]},
    "sib200": {
        "en": ["sib200_eng_Latn"],
        "de": ["sib200_deu_Latn"],
        "fr": ["sib200_fra_Latn"],
        "ar": ["sib200_arb_Arab"],
        "zh": ["sib200_zho_Hans"],
    },
    # English-label control. `en`'s control would be byte-identical to
    # `sib200_eng_Latn` (English text, English labels), so there is no
    # `sib200_enlab_eng_Latn` -- reuse the primary English number.
    "sib200_enlab": {
        "de": ["sib200_enlab_deu_Latn"],
        "fr": ["sib200_enlab_fra_Latn"],
        "ar": ["sib200_enlab_arb_Arab"],
        "zh": ["sib200_enlab_zho_Hans"],
    },
    # Global-MMLU, FULL size (n=14042 for en), in both answer formats -- see
    # c5_tasks/gmmlu_probe/utils.py. CLAUDE.md 6's "genuinely at chance"
    # verdict was measured at --limit 200, where the smallest 2-sigma
    # detectable effect is +6.1 points; this is the powered version.
    #   gmmlu_letter_*  A/B/C/D -- shared choice set, so acc_cal removes the
    #                   model's SELECTION bias (a fixed preference for one
    #                   option id). It cannot manufacture knowledge.
    #   gmmlu_cloze_*   the answer TEXT -- per-document choices, so
    #                   calibration does not apply and acc_norm is correct.
    #                   This is the format that tests world knowledge.
    # Split into two families so the formats can run on separate core-pairs:
    # fixed_width here is the task-wide MAX prompt (1088 tokens for en, against
    # a median of 77), so these graphs are ~11x wider than SIB-200's and must
    # use a smaller batch -- batch 16 OOMs at NCC_EOOM002 (28.45GB > 24GB).
    "gmmlu_letter": {lang: [f"gmmlu_letter_{lang}"]
                     for lang in ("en", "de", "fr", "ar", "zh")},
    "gmmlu_cloze": {lang: [f"gmmlu_cloze_{lang}"]
                    for lang in ("en", "de", "fr", "ar", "zh")},
    # MMMLU (openai/MMMLU): the same 14,042 MMLU items rendered by PROFESSIONAL
    # TRANSLATORS, which is what Messmer et al. 2025 evaluate on. Global-MMLU is
    # largely machine-translated, so gmmlu_cloze vs mmmlu_cloze isolates
    # translation quality -- the one prompt-side explanation for Arabic sitting
    # at chance that the cue test did not rule out.
    "mmmlu_cloze": {lang: [f"mmmlu_cloze_{lang}"]
                    for lang in ("en", "de", "fr", "ar", "zh")},
    # ARC-Challenge, like-for-like across all five languages, WITH PMI.
    # Fixes the ARC-Easy(en) vs ARC-Challenge(others) pool mismatch behind
    # 6d's "English-only" ARC claim, and adds the one estimator never tried on
    # ARC. See c5_tasks/arc_pmi/.
    "arc_pmi": {lang: [f"arc_pmi_{lang}"] for lang in ("en","de","fr","ar","zh")},
    # MuBench: 12 benchmarks x 61 aligned languages, so every language is a
    # rendering of ONE item set -- the structural fix for the pool mismatch
    # ARC exposed. Cloze-converted; see c5_tasks/mubench/ and verified by
    # scripts/external_bench/verify_mubench.py.
    "mub_hellaswag": {l: [f"mub_hellaswag_{l}"] for l in ("en","de","fr","ar","zh")},
    "mub_mnli": {l: [f"mub_mnli_{l}"] for l in ("en","de","fr","ar","zh")},
    "mub_snli": {l: [f"mub_snli_{l}"] for l in ("en","de","fr","ar","zh")},
    "mub_bmlama": {l: [f"mub_bmlama_{l}"] for l in ("en","de","fr","ar","zh")},
    "mub_storycloze": {l: [f"mub_storycloze_{l}"] for l in ("en","de","fr","ar","zh")},
    "mub_winogrande": {l: [f"mub_winogrande_{l}"] for l in ("en","de","fr","ar","zh")},
    "mub_mmlu": {l: [f"mub_mmlu_{l}"] for l in ("en","de","fr","ar","zh")},
    "mub_arceasy": {l: [f"mub_arceasy_{l}"] for l in ("en","de","fr","ar","zh")},
    # ARC-EASY aligned across all five languages (MuBench), cloze + PMI. This
    # is the like-for-like ARC our C.5 column never had. See mubench_arc/.
    "mubench_arc": {l: [f"mubench_arceasy_{l}"] for l in ("en","de","fr","ar","zh")},
    # NATIVE-language knowledge exams (ArabicMMLU / CMMLU), localized cloze.
    # Translated MMLU asks whether a model knows Anglocentric facts in another
    # language; this asks whether it knows facts its own web text contains.
    # Messmer et al. 2025 evaluate non-English with FineTasks, which is native
    # per language -- see c5_tasks/native_mmlu/.
    "native_mmlu": {"ar": ["nativemmlu_cloze_ar"], "zh": ["nativemmlu_cloze_zh"]},
    # English-cue control: identical documents, the answer cue left in English
    # ("Answer:" after an Arabic question). Same primary/control split as
    # sib200_* vs sib200_enlab_*; no `en` entry because it would be identical
    # to the primary.
    "gmmlu_cloze_encue": {lang: [f"gmmlu_cloze_encue_{lang}"]
                          for lang in ("de", "fr", "ar", "zh")},
    "taxi1500": {
        "en": ["taxi1500_eng_Latn"],
        "de": ["taxi1500_deu_Latn"],
        "fr": ["taxi1500_fra_Latn"],
        "ar": ["taxi1500_arb_Arab"],
        "zh": ["taxi1500_zho_Hans"],
    },
}
ALL_LANGS = ["en", "de", "fr", "ar", "zh"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="HF repo id holding the export")
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--workdir", default="./xscript_bench")
    ap.add_argument("--limit", type=float, default=None,
                    help="examples per task (omit for the full split)")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="subset of friendly model names (default: all in models.json)")
    ap.add_argument("--langs", nargs="*", default=None, choices=ALL_LANGS,
                    help="subset of languages to evaluate every model on "
                         "(default: all 5, regardless of training languages)")
    ap.add_argument("--families", nargs="*", default=None, choices=list(FAMILIES),
                    help=f"benchmark families to run (default: all of {list(FAMILIES)})")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="batch size on --device xla. SIB-200/HellaSwag are far "
                         "shorter than Belebele, so this can exceed "
                         "run_appendix_c5.py's Belebele-bound value.")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu / xla (Neuron). Default: auto (cuda else cpu).")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="keep each 4GB checkpoint after eval (default: delete)")
    ap.add_argument("--own-langs", action="store_true",
                    help="score each model only on ITS OWN training languages "
                         "(models.json `langs`), instead of all five. Cuts ~74%% "
                         "of the work. The bilingual-vs-monolingual comparisons "
                         "are unaffected -- every transfer cell needs only "
                         "trained-language scores (partner-lang delta: bi(X) vs "
                         "mono-X(X); English delta: bi(en) vs en-mono(en)) -- and "
                         "calibration is computed per (model, lang, task) so it "
                         "does not weaken either. What is given up is the "
                         "zero-shot cross-lingual readout: the out-of-domain "
                         "cells, the untrained arm of the SIB-200 label-language "
                         "control, and the constant-prediction findings, which "
                         "were all in languages the model never trained on.")
    ap.add_argument("--no-require-raw", dest="require_raw", action="store_false",
                    help="treat a task as done even if its raw per-choice "
                         "loglikelihoods are missing. Default is to re-score "
                         "such tasks, so a resumed sweep backfills the raw "
                         "sidecar every offline estimator now reads.")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-score tasks that are already in a model's result "
                         "JSON (default: skip only the models whose requested "
                         "tasks are ALL already present, so the sweep is "
                         "resumable and two runs with different --families "
                         "merge into one file instead of clobbering)")
    args = ap.parse_args()
    langs = args.langs or ALL_LANGS
    families = args.families or list(FAMILIES)

    # lang -> tasks, restricted to the requested families and languages.
    tasks_by_lang = {
        lang: [t for fam in families for t in FAMILIES[fam].get(lang, [])]
        for lang in langs
    }
    tasks_by_lang = {k: v for k, v in tasks_by_lang.items() if v}
    if not tasks_by_lang:
        sys.exit(f"no tasks for languages={langs} families={families}")

    from huggingface_hub import hf_hub_download

    def _dl_retry(filename: str, tries: int = 8) -> Path:
        """hf_hub_download with backoff -- the hub 429s ('maximum queue size
        reached') when a fan-out starts N processes at once, and every startup
        below touches the network before any Neuron work begins."""
        import random
        import time
        for attempt in range(tries):
            try:
                return Path(hf_hub_download(filename=filename, **dl))
            except Exception as exc:
                if attempt == tries - 1:
                    raise
                wait = min(60, 2 ** attempt) + random.uniform(0, 3)
                print(f"[extra] {type(exc).__name__} on {filename}; "
                      f"retry {attempt + 1}/{tries - 1} in {wait:.0f}s", flush=True)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def fetch_checkpoint(rel_dir: str) -> Path:
        """Download final.pt, transparently reassembling `final.pt.partNNN`
        chunks if the checkpoint was uploaded split (see upload_chunked.py).
        Identical to run_appendix_c5.py's; see its comments for the
        already-assembled / offline cases."""
        parts = sorted(f for f in repo_files
                       if f.startswith(f"{rel_dir}/final.pt.part"))
        out = work / "_assembled" / rel_dir / "final.pt"
        if parts and out.exists():
            want = sum(sizes.get(p, 0) for p in parts)
            if not want or out.stat().st_size == want:
                return out
            print(f"[extra] {rel_dir}: assembled size {out.stat().st_size} != "
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

    dl = dict(repo_id=args.repo, repo_type=args.repo_type,
              local_dir=str(scratch.parent / "_repo"))

    # Repo file listing + sizes, cached on disk (N parallel `list_repo_files`
    # calls 429 before a single checkpoint transfers). Delete _repo_files.json
    # after new uploads to refresh.
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
    # Prefer this repo's local src: the custom task suite and the Neuron
    # scoring fixes both live there.
    _local_src = Path(__file__).resolve().parents[2] / "src"
    if (_local_src / "xscript" / "eval" / "bench.py").exists():
        sys.path.insert(0, str(_local_src))
    c5_tasks_dir = _local_src / "xscript" / "eval" / "c5_tasks"
    if not c5_tasks_dir.exists():
        sys.exit(f"custom task configs not found at {c5_tasks_dir} -- this "
                 "script only runs from inside the XScript-Pretraining repo.")

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

    import torch
    import lm_eval
    from lm_eval.tasks import TaskManager
    from xscript.model import ModelConfig, Transformer
    from xscript.tok.wrapper import Tok
    from xscript.paths import tokenizer_dir, ensure
    from xscript.eval.bench import _make_lm
    from xscript.eval.rawscores import (extract_raw, check_reproduces,
                                        has_shared_choices)

    task_manager = TaskManager(include_path=str(c5_tasks_dir))
    # Fail loudly and early on a task the include_path does not define -- e.g.
    # taxi1500_* before its corpus has been staged (see c5_tasks/taxi1500/).
    wanted = sorted({t for ts in tasks_by_lang.values() for t in ts})
    unknown = [t for t in wanted if t not in task_manager.all_tasks]
    if unknown:
        sys.exit(f"tasks not registered in {c5_tasks_dir}: {unknown}")
    print(f"[extra] {len(runs)} model(s); tasks: "
          + "; ".join(f"{k}={v}" for k, v in tasks_by_lang.items()), flush=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "xla":
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print("[extra] using XLA/Neuron (fixed-shape scoring).")
    else:
        device = torch.device(args.device)

    def _accuracy(rec: dict):
        return rec.get("acc_norm,none", rec.get("acc,none",
                       rec.get("acc_norm", rec.get("acc"))))

    def _all_metrics(rec: dict) -> dict:
        """Every headline metric the task reported, `acc`/`acc_norm`/
        `acc_mutual_info` (PMI). SIB-200 asks for all three because they
        disagree in exactly the regime we are in -- PMI is what moved XNLI-zh
        off chance (CLAUDE.md 6) -- and the choice must be auditable after
        the fact, not baked in here."""
        return {k.split(",")[0]: v for k, v in rec.items()
                if k.split(",")[0] in ("acc", "acc_norm", "acc_mutual_info")
                and isinstance(v, float)}

    def _per_example_hits(samples: list[dict], metric: str) -> list[int]:
        # lm_eval's log_samples records carry each metric as a direct float
        # field per sample -- the 0/1 hit list a paired bootstrap needs.
        return [int(round(s[metric])) for s in samples if metric in s]

    def _raw_block(samples: list[dict], metric_names, tok, task: str) -> dict | None:
        """Raw per-choice loglikelihoods for one task, or None if lm-eval's
        records don't carry them.

        This is the part that makes the scoring rule auditable after the run:
        acc / acc_norm / PMI / calibrated accuracy are all re-derivable from
        it on CPU, so changing the estimator never costs another Neuron pass
        (see eval/rawscores.py for why the shipped acc and acc_norm are both
        length-degenerate on SIB-200)."""
        try:
            block = extract_raw(samples, metric_names, tok=tok)
            block["shared_choices"] = has_shared_choices(task)
            return block
        except (KeyError, ValueError) as exc:
            print(f"[extra] raw capture skipped: {type(exc).__name__}: {exc}",
                  flush=True)
            return None

    def read_existing(path: Path) -> dict:
        """Previous result for this model, or {}. Results are MERGED into it
        rather than replacing it: one model's tasks are routinely produced by
        more than one invocation (different --families on different cores, or
        a task added later), and a plain overwrite would silently drop whatever
        the other run had already measured."""
        if not path.exists():
            return {}
        try:
            prev = json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
        return {} if "error" in prev else prev

    def merge(prev: dict, key: str, new: dict) -> dict:
        """Two-level merge of {lang: {task: value}}, new values winning."""
        out = {lang: dict(ts) for lang, ts in prev.get(key, {}).items()}
        for lang, ts in new.items():
            out.setdefault(lang, {}).update(ts)
        return out

    out_dir = ensure(Path(work / "results" / "extra_bench"))
    raw_dir = ensure(Path(work / "results" / "extra_bench" / "raw"))
    for i, run in enumerate(runs, 1):
        out_path = out_dir / f"{run}_final.json"
        prev = read_existing(out_path)
        done = {(lang, t) for lang, ts in prev.get("metrics", {}).items() for t in ts}
        if args.require_raw:
            # A task scored before the raw sidecar existed is NOT done: its
            # estimator can no longer be audited or changed without a rerun.
            raw_path = raw_dir / f"{run}_raw.json"
            have_raw = set()
            if raw_path.exists():
                try:
                    have_raw = {(l, t) for l, ts in
                                json.loads(raw_path.read_text()).get("raw", {}).items()
                                for t in ts}
                except json.JSONDecodeError:
                    have_raw = set()
            done &= have_raw
        wanted = ({lang: ts for lang, ts in tasks_by_lang.items()
                   if lang in set(models[run]["langs"])}
                  if args.own_langs else tasks_by_lang)
        if not wanted:
            print(f"[extra] [{i}/{len(runs)}] {run}: no requested task covers its "
                  f"training languages {models[run]['langs']}, skipping", flush=True)
            continue
        todo = {lang: [t for t in ts if args.overwrite or (lang, t) not in done]
                for lang, ts in wanted.items()}
        todo = {k: v for k, v in todo.items() if v}
        if not todo:
            print(f"[extra] [{i}/{len(runs)}] {run}: all requested tasks already "
                  "scored, skipping (--overwrite to redo)", flush=True)
            continue
        run_tasks = sorted({t for ts in todo.values() for t in ts})
        tok_name = models[run]["tok"]
        print(f"\n===== [{i}/{len(runs)}] {run} (tok={tok_name}) =====", flush=True)
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
            adapter.batch_size = args.batch_size

            results = lm_eval.simple_evaluate(
                model=adapter, tasks=run_tasks, num_fewshot=0, batch_size=1,
                limit=args.limit, log_samples=True, confirm_run_unsafe_code=True,
                task_manager=task_manager,
            )
            subtasks = results.get("results", {})
            samples = results.get("samples", {})

            scores, metrics, correct, raw = {}, {}, {}, {}
            for lang, lang_tasks in todo.items():
                scores[lang], metrics[lang], correct[lang] = {}, {}, {}
                raw[lang] = {}
                for t in lang_tasks:
                    rec = subtasks.get(t, {})
                    if not _all_metrics(rec):
                        # Record nothing rather than an empty dict: an empty
                        # entry would still count as "done" on the next run and
                        # the task would never be retried.
                        print(f"[extra] {run} / {lang} / {t}: no metrics "
                              "returned, leaving it unscored", flush=True)
                        continue
                    scores[lang][t] = _accuracy(rec)
                    metrics[lang][t] = _all_metrics(rec)
                    if t in samples:
                        # One hit list per reported metric: which one is the
                        # right estimator is decided downstream, not here.
                        correct[lang][t] = {
                            m: _per_example_hits(samples[t], m)
                            for m in metrics[lang][t]
                        }
                        block = _raw_block(samples[t], metrics[lang][t], tok, t)
                        if block is not None:
                            raw[lang][t] = block
                            # Fail loudly if the stored raw scores do not
                            # reconstruct lm-eval's own hit lists -- every
                            # derived estimator rests on that identity.
                            mism = check_reproduces(block, correct[lang][t])
                            bad = {k: v for k, v in mism.items() if v}
                            if bad:
                                print(f"[extra] WARNING {run}/{lang}/{t}: raw "
                                      f"scores disagree with lm-eval hits {bad}",
                                      flush=True)
                print(f"[extra] {run} / {lang}: " + ", ".join(
                    f"{k}=" + "/".join(f"{mk}:{mv:.4f}" for mk, mv in
                                       sorted(metrics[lang][k].items()))
                    for k in lang_tasks if k in metrics[lang]), flush=True)

            payload = {
                "run": run, "tokenizer": tok_name, "limit": args.limit,
                "families": sorted(set(prev.get("families", [])) | set(families)),
                "scores": merge(prev, "scores", scores),    # acc_norm where available
                "metrics": merge(prev, "metrics", metrics),  # every metric reported
                "correct": merge(prev, "correct", correct),  # {lang:{task:{metric:[0/1]}}}
            }
            payload["tasks"] = {lang: sorted(ts)
                                for lang, ts in payload["metrics"].items()}
            payload["langs"] = sorted(payload["tasks"])
            tmp = out_path.with_suffix(f".tmp.{os.getpid()}")
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, out_path)   # never leave a half-written result behind

            # Raw loglikelihoods go to a SIDECAR, not into the result JSON:
            # they are ~10x its size, and every existing reader
            # (analyze_extra_bench.py, bootstrap_transfer.py) keeps working
            # untouched. Merged across invocations exactly like `metrics`.
            raw = {l: ts for l, ts in raw.items() if ts}
            if raw:
                raw_path = raw_dir / f"{run}_raw.json"
                prev_raw = {}
                if raw_path.exists():
                    try:
                        prev_raw = json.loads(raw_path.read_text())
                    except json.JSONDecodeError:
                        prev_raw = {}
                merged = {l: dict(ts) for l, ts in prev_raw.get("raw", {}).items()}
                for l, ts in raw.items():
                    merged.setdefault(l, {}).update(ts)
                tmp = raw_path.with_suffix(f".tmp.{os.getpid()}")
                tmp.write_text(json.dumps(
                    {"run": run, "tokenizer": tok_name, "raw": merged},
                    separators=(",", ":")))
                os.replace(tmp, raw_path)
        except Exception as exc:
            print(f"[extra] {run} FAILED: {type(exc).__name__}: {exc}", flush=True)
            if prev:
                # Do not turn an earlier run's good results into an error stub.
                print(f"[extra] {run}: keeping the {len(done)} task(s) already "
                      "in its result file", flush=True)
            else:
                out_path.write_text(json.dumps(
                    {"run": run, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        finally:
            if not args.keep_checkpoints:
                try:
                    real = Path(local_ckpt).resolve()
                    dest.unlink(missing_ok=True)
                    real.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"[extra] cleanup warning for {run}: {exc}")

    print(f"\n[extra] per-model JSON in {out_dir}")


if __name__ == "__main__":
    main()
