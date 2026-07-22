"""Downstream multilingual benchmarks via lm-evaluation-harness.

Complements the intrinsic BPB eval (eval/bpb.py) and the bespoke
cross-lingual-transfer/representation analyses (eval/bts.py, eval/alignment.py)
with real task accuracy: Global-MMLU (knowledge), Belebele (reading
comprehension), XNLI (natural language inference) -- all covering en/de/fr/ar/zh.

By default, evaluation is restricted to the languages in the checkpoint's
training mixture: three tasks for a monolingual run and six for a bilingual
run. ``--tasks`` remains an explicit override for targeted/OOD analyses.

`XScriptLM` wraps our own Transformer/Tok (not HF-standard) in lm_eval's `LM`
interface. Scoring follows the exact shifted-LM convention already used in
eval/bpb.py's score_texts: model(x, y) with x=seq[:-1], y=seq[1:] returns
logits where logits[:, j] predicts y[j].
"""
import importlib.metadata
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ..tok.wrapper import BOS_ID, EOS_ID, PAD_ID

DEFAULT_TASKS = {
    "global_mmlu": ["global_mmlu_en", "global_mmlu_de", "global_mmlu_fr",
                    "global_mmlu_ar", "global_mmlu_zh"],
    "belebele": ["belebele_eng_Latn", "belebele_deu_Latn", "belebele_fra_Latn",
                 "belebele_arb_Arab", "belebele_zho_Hans"],
    "xnli": ["xnli_en", "xnli_de", "xnli_fr", "xnli_ar", "xnli_zh"],
}

LANG_ORDER = ("en", "de", "fr", "ar", "zh")
TASKS_BY_LANG = {
    lang: [DEFAULT_TASKS[family][i] for family in DEFAULT_TASKS]
    for i, lang in enumerate(LANG_ORDER)
}

# lm-eval's stock XNLI is a raw-loglikelihood cloze over three connectives and
# is not usable as-is for ar/zh at this scale (see CLAUDE.md §6): Arabic's
# connectives are mistranslated, and Chinese collapses to surface-form
# competition (Holtzman et al. 2021) even with correct ones. `xnli_ar` /
# `xnli_zh` are therefore always scored by `_xnli_debiased` below instead of
# through lm_eval's task registry -- corrected connectives + standard scoring
# for ar, PMI (prior-normalized) scoring for zh.
# fr/de/en entries below are NOT used for scoring -- XNLI_DEBIAS_METHOD has no
# "fr"/"de"/"en" key, so those languages are never routed through
# _xnli_debiased() and always score via lm_eval's own registered xnli_{lang}
# task instead (see run()'s harness_tasks split below). Kept here only because
# they happen to match lm-eval's own connective words (fr: "Oui"/"Aussi"/"Non",
# copied from lm_eval/tasks/xnli/utils.py) -- do not assume changing them here
# changes fr/de/en scoring; it only would if those languages were added to
# XNLI_DEBIAS_METHOD.
#
# NOTE on a confirmed confound in lm-eval's OWN xnli_fr (not this dict, but the
# same words): "Oui"/"Aussi" cost 1 more token than "Non" under unigram_starved
# in the full "{premise}, correct? {c}, {hypothesis}" template (0 extra under
# unigram_destarved -- verified in-template, not just standalone). lm-eval's
# xnli_common_yaml scores via unnormalized `acc` (raw summed loglikelihood, no
# length normalization), so this token-count asymmetry is a real, tokenizer-
# dependent scoring bias toward "Non" specifically under starved -- plausibly
# contributing to fr's significant fair-vs-starved delta on XNLI (CLAUDE.md
# section 6, "does the tokenizer change the transfer delta"). Checked and ruled
# out for ar (0 marginal tokens per connective under both tokenizers, in-
# template) and for zh (2/1/2 marginal tokens, but IDENTICAL under both
# tokenizers, so it can't produce a fair-vs-starved difference there).
XNLI_CONNECTIVES = {
    # lang: (question_word, entailment, neutral, contradiction)
    "en": ("right",   "Yes",  "Also",  "No"),
    "de": ("richtig", "Ja",   "Auch",  "Nein"),
    "fr": ("correct", "Oui",  "Aussi", "Non"),
    "ar": ("صحيح",    "نعم",  "أيضا",  "لا"),      # corrected: was لذا/رقم
    "zh": ("正确",     "是的", "所以",   "不是的"),
}
XNLI_DEBIAS_METHOD = {"ar": "standard", "zh": "pmi"}


def _xnli_debiased(lm, lang: str, limit: int | float | None,
                   disable_tqdm: bool = False, return_correct: bool = False):
    """Debiased XNLI accuracy for `lang` (must be a key of XNLI_DEBIAS_METHOD).

    Reuses `lm._loglikelihood_tokens` -- the verified fixed-shape/Neuron
    scoring path -- by framing each candidate string as a bare continuation
    (empty context), so the returned logprob is the string's total loglik
    exactly as CLAUDE.md's debiasing recipe requires.

    With `return_correct=True`, returns `(accuracy, per_example_hits)`
    instead of just `accuracy` -- the 0/1 hit list needed to bootstrap a
    confidence interval, e.g. for transfer-delta significance testing.
    """
    import datasets

    qw, ent, neu, con = XNLI_CONNECTIVES[lang]
    conns = [ent, neu, con]
    ds = datasets.load_dataset("xnli", lang, split="validation")
    n = len(ds) if limit is None else min(int(limit), len(ds))
    full_strs, null_strs, golds = [], [], []
    for i in range(n):
        d = ds[i]
        golds.append(d["label"])
        for c in conns:
            full_strs.append(f"{d['premise']}, {qw}? {c}, {d['hypothesis']}")
            null_strs.append(f"{qw}? {c}, {d['hypothesis']}")

    requests = [(None, [], lm.tok_encode(s)) for s in full_strs + null_strs]
    scored = lm._loglikelihood_tokens(requests, disable_tqdm=disable_tqdm)
    lls = [lp for lp, _ in scored]
    full_ll, null_ll = lls[:len(full_strs)], lls[len(full_strs):]

    method = XNLI_DEBIAS_METHOD[lang]
    hits = []
    for j, g in enumerate(golds):
        f3 = full_ll[3 * j:3 * j + 3]
        if method == "pmi":
            n3 = null_ll[3 * j:3 * j + 3]
            scores3 = [f3[k] - n3[k] for k in range(3)]
        else:
            scores3 = f3
        hits.append(int(max(range(3), key=lambda k: scores3[k]) == g))
    accuracy = sum(hits) / n
    return (accuracy, hits) if return_correct else accuracy


def tasks_for_langs(langs: list[str]) -> list[str]:
    """Harness task names for exactly the languages in a run's mixture."""
    unknown = [lang for lang in langs if lang not in TASKS_BY_LANG]
    if unknown:
        raise ValueError(f"no downstream task mapping for languages: {unknown}")
    # Keep benchmark families together in the output, then run language order.
    return [TASKS_BY_LANG[lang][family_i]
            for family_i in range(len(DEFAULT_TASKS)) for lang in langs]


class XScriptLM:
    """lm_eval.api.model.TemplateLM subclass wrapping our Transformer + Tok.

    Inherits from TemplateLM lazily (import-time, so lm_eval/torch stay
    optional deps of the base package) via _make_lm() below.
    """

    def __init__(self, model, tok, device, max_seq_len: int, batch_size: int = 4):
        super().__init__()
        self.model = model.eval()
        self.tok = tok
        # lm_eval.api.model.LM exposes `device` as a read-only property backed
        # by `_device`; assigning self.device raises AttributeError.
        self._device = torch.device(device)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.tokenizer = None  # no chat template support needed for these tasks

    @property
    def eot_token_id(self) -> int:
        return EOS_ID

    @property
    def prefix_token_id(self) -> int:
        # our documents are always BOS-prefixed, not EOS-prefixed
        return BOS_ID

    def tok_encode(self, string: str, add_special_tokens=None, **kwargs) -> list[int]:
        return self.tok.encode(string, bos=False, eos=False)

    def _prepare(self, context_enc: list[int], continuation_enc: list[int]) -> list[int]:
        """Return a model-ready sequence with one BOS and an intact target.

        TemplateLM supplies ``[prefix_token_id]`` for an empty string context,
        whereas non-empty contexts contain no special token.  Normalize both
        cases here so BOS is added exactly once.  Context is left-truncated;
        benchmark answer continuations are never silently truncated.
        """
        if not continuation_enc:
            return []
        has_bos = bool(context_enc) and context_enc[0] == BOS_ID
        context = context_enc[1:] if has_bos else context_enc
        if len(continuation_enc) > self.max_seq_len:
            raise ValueError(
                f"continuation has {len(continuation_enc)} tokens, exceeding "
                f"max_seq_len={self.max_seq_len}"
            )
        budget = self.max_seq_len - len(continuation_enc)
        context = context[-budget:] if budget < len(context) else context
        return [BOS_ID] + context + continuation_enc

    @torch.no_grad()
    def _score_batch(self, batch, fixed_width: int | None = None) -> list[tuple[float, bool]]:
        """Score variable-length requests with right padding.

        Padding is strictly after each real sequence, so causal attention
        cannot let it affect any scored position.  Passing targets asks our
        Transformer for all-position logits; the returned scalar loss is
        intentionally ignored.

        On XLA/Neuron (``fixed_width`` set) every forward is padded to one
        constant ``[batch_size, fixed_width]`` shape so the whole task hits a
        single compiled graph; the graph is weight-independent, so it compiles
        once on the first model and is reused for every checkpoint after.
        """
        prepared = [(self._prepare(list(c), list(k)), len(k)) for c, k in batch]
        out: list[tuple[float, bool] | None] = [None] * len(prepared)
        active = [(i, seq, n) for i, (seq, n) in enumerate(prepared) if n]
        for i, (_, n) in enumerate(prepared):
            if not n:
                out[i] = (0.0, True)
        if not active:
            return out  # type: ignore[return-value]

        if fixed_width is not None:
            self._score_active_xla(active, out, fixed_width)
            return out  # type: ignore[return-value]

        width = max(len(seq) - 1 for _, seq, _ in active)
        x = torch.full((len(active), width), PAD_ID, dtype=torch.long,
                       device=self.device)
        y = torch.full((len(active), width), -100, dtype=torch.long,
                       device=self.device)
        lengths = []
        for row, (_, seq, _) in enumerate(active):
            m = len(seq) - 1
            lengths.append(m)
            x[row, :m] = torch.tensor(seq[:-1], device=self.device)
            y[row, :m] = torch.tensor(seq[1:], device=self.device)

        amp = (torch.autocast("cuda", dtype=torch.bfloat16)
               if self.device.type == "cuda" else _null())
        with amp:
            logits, _ = self.model(x, y)
        for row, (out_i, _, n) in enumerate(active):
            m = lengths[row]
            cont_logits = logits[row, m - n:m, :].float()
            target = y[row, m - n:m]
            logprobs = F.log_softmax(cont_logits, dim=-1)
            token_lp = logprobs.gather(1, target.unsqueeze(1)).squeeze(1)
            greedy = bool((cont_logits.argmax(-1) == target).all().item())
            out[out_i] = (float(token_lp.sum().item()), greedy)
        return out  # type: ignore[return-value]

    @torch.no_grad()
    def _score_active_xla(self, active, out, fixed_width: int) -> None:
        """Fixed-shape scoring for XLA/Neuron.

        Pads to a constant ``[batch_size, fixed_width]`` graph (rows past the
        real requests are dummy PAD rows, computed then ignored) and reduces to
        per-token logprobs / greedy-match ON DEVICE, moving only the small
        ``[batch_size, fixed_width]`` results back to host -- never the
        ``[batch, width, vocab]`` logits, which stay on the Neuron core.
        """
        import torch_xla.core.xla_model as xm
        R = self.batch_size
        # Build the fixed-shape batch on the HOST, then move once. Assembling it
        # with per-row in-place updates on the XLA tensor emits on-device scatter
        # ops that trip NRT_EXEC_OOB on Neuron.
        x = torch.full((R, fixed_width), PAD_ID, dtype=torch.long)
        y = torch.full((R, fixed_width), -100, dtype=torch.long)
        lengths = []
        for row, (_, seq, _) in enumerate(active):
            m = len(seq) - 1
            if m > fixed_width:
                raise ValueError(f"sequence len {m} exceeds fixed_width {fixed_width}")
            lengths.append(m)
            x[row, :m] = torch.tensor(seq[:-1])
            y[row, :m] = torch.tensor(seq[1:])
        # Clamp the pad target (-100) to a valid id ON THE HOST: an on-device
        # clamp_min does not materialize before one_hot's scatter, which then
        # sees -100 and trips NRT_EXEC_OOB.
        y_idx = y.clamp_min(0)
        x = x.to(self.device)
        y = y.to(self.device)
        y_idx = y_idx.to(self.device)
        logits, _ = self.model(x, y)
        logits = logits.float()
        # NOTE: torch.gather along the vocab dim silently returns zeros on this
        # Neuron/torch-xla build. Select the target logit with a one-hot
        # multiply-sum and score as logit - logsumexp instead (both verified
        # fp32-accurate vs CPU on Neuron; gather is the only broken op here).
        onehot = F.one_hot(y_idx, logits.size(-1)).to(logits.dtype)
        target_logit = (logits * onehot).sum(-1)                   # [R, W]
        token_lp = target_logit - torch.logsumexp(logits, dim=-1)  # [R, W]
        greedy = (logits.argmax(-1) == y)                          # [R, W]
        xm.mark_step()
        token_lp = token_lp.cpu()
        greedy = greedy.cpu()
        for row, (out_i, _, n) in enumerate(active):
            m = lengths[row]
            lp = float(token_lp[row, m - n:m].sum().item())
            ok = bool(greedy[row, m - n:m].all().item())
            out[out_i] = (lp, ok)

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        from tqdm import tqdm
        out = []
        fixed_width = None
        if self.device.type == "xla":
            # One constant graph shape for the entire task: pad every batch to
            # the longest prepared sequence in this task (bounded by the model's
            # max_seq_len via _prepare's left-truncation of context).
            fixed_width = max(
                (len(self._prepare(list(c), list(k))) - 1 for _, c, k in requests),
                default=1,
            )
            fixed_width = max(fixed_width, 1)
            # NCC-5266: neuronx-cc's matmul transpose lowering requires an
            # even step for non-FP32 dst dims -- an odd fixed_width reliably
            # fails compilation (confirmed: zh debiased-XNLI fixed_width=85
            # crashes every time, solo or concurrent; ar's fixed_width=88
            # never does). Round up to the next even width; the extra column
            # is inert padding, identical to any other padded position.
            if fixed_width % 2:
                fixed_width += 1
        batches = range(0, len(requests), self.batch_size)
        for st in tqdm(batches, disable=disable_tqdm, desc="[bench] scoring"):
            chunk = requests[st:st + self.batch_size]
            out.extend(self._score_batch([(c, k) for _, c, k in chunk],
                                         fixed_width=fixed_width))
        return out

    @torch.no_grad()
    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        from lm_eval import utils
        from tqdm import tqdm
        out = []
        for req in tqdm(requests, disable=disable_tqdm, desc="[bench] rolling"):
            (text,) = req.args
            ids = self.tok_encode(text)
            windows = list(utils.get_rolling_token_windows(
                token_list=ids, prefix_token=BOS_ID,
                max_seq_len=self.max_seq_len, context_len=1,
            ))
            # The utility's contexts are already complete windows (the first
            # starts with BOS), so score them without _prepare adding BOS.
            total = 0.0
            for context, target in windows:
                x = torch.tensor(context, device=self.device).unsqueeze(0)
                # Input and prediction windows are aligned by the harness;
                # only the final len(target) logits are part of this window.
                y_ids = [-100] * (len(context) - len(target)) + target
                y = torch.tensor(y_ids, device=self.device).unsqueeze(0)
                logits, _ = self.model(x, y)
                n = len(target)
                lp = F.log_softmax(logits[0, -n:, :].float(), -1)
                total += float(lp.gather(1, y[0, -n:].unsqueeze(1)).sum().item())
            out.append(total)
        return out

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm: bool = False):
        from tqdm import tqdm
        out = []
        for req in tqdm(requests, disable=disable_tqdm, desc="[bench] generating"):
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
            max_gen = (gen_kwargs.get("max_gen_toks", 256)
                      if isinstance(gen_kwargs, dict) else 256)
            ids = [BOS_ID] + self.tok_encode(context)[-(self.max_seq_len - 1):]
            gen = []
            text_so_far = ""
            for _ in range(max_gen):
                x = torch.tensor(ids[-self.max_seq_len:], device=self.device).unsqueeze(0)
                logits = self.model(x)                     # (1, 1, vocab) -- no targets
                next_id = int(logits[0, -1].argmax(-1).item())
                if next_id == EOS_ID:
                    break
                gen.append(next_id)
                ids.append(next_id)
                text_so_far = self.tok.decode(gen)
                if until and any(u in text_so_far for u in until):
                    for u in until:
                        idx = text_so_far.find(u)
                        if idx != -1:
                            text_so_far = text_so_far[:idx]
                    break
            out.append(text_so_far)
        return out


def _make_lm(model, tok, device, max_seq_len):
    """Bind XScriptLM to lm_eval.api.model.TemplateLM at call time (keeps
    lm_eval/torch optional for anything that only imports xscript.eval.bench
    for DEFAULT_TASKS)."""
    from lm_eval.api.model import TemplateLM

    class _Bound(XScriptLM, TemplateLM):
        def __init__(self):
            XScriptLM.__init__(self, model, tok, device, max_seq_len)

    return _Bound()


def run(run_name: str, tok_name: str, tag: str = "final", tasks: list[str] | None = None,
        num_fewshot: int = 0, limit: int | float | None = None,
        out_dir: Path | None = None, log_wandb: bool = True,
        batch_size: int = 4, device: str | None = None) -> dict:
    """Evaluate a checkpoint on its training languages by default.

    ``device`` may be ``cuda`` / ``cpu`` / ``xla`` (Neuron). ``None`` auto-picks
    CUDA if present else CPU (Neuron must be requested explicitly since it needs
    the fixed-shape scoring path).
    """
    import lm_eval
    from ..model import ModelConfig, Transformer
    from ..tok.wrapper import Tok
    from ..paths import RUNS, RESULTS, tokenizer_dir, ensure

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "xla":
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device(device)
    ck = torch.load(RUNS / run_name / "checkpoints" / f"{tag}.pt",
                    map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**ck["cfg"]["model"])).to(device).eval()
    model.load_state_dict(ck["model"])
    tok = Tok(tokenizer_dir(tok_name))
    if tok_name != ck["cfg"]["tok_name"]:
        raise ValueError(f"checkpoint uses {ck['cfg']['tok_name']}, not {tok_name}")

    task_list = tasks if tasks is not None else tasks_for_langs(ck["cfg"]["langs"])
    adapter = _make_lm(model, tok, device, model.cfg.max_seq_len)
    adapter.batch_size = batch_size

    # xnli_ar / xnli_zh are scored by the debiased path (see XNLI_DEBIAS_METHOD
    # above), never by lm_eval's task registry -- pull them out of the harness
    # task list before calling simple_evaluate.
    debiased_langs = {lang: name for lang in XNLI_DEBIAS_METHOD
                      if (name := f"xnli_{lang}") in task_list}
    harness_tasks = [t for t in task_list if t not in debiased_langs.values()]

    results = lm_eval.simple_evaluate(
        model=adapter, tasks=harness_tasks, num_fewshot=num_fewshot,
        batch_size=1, limit=limit, log_samples=False, confirm_run_unsafe_code=True,
    ) if harness_tasks else {"results": {}, "groups": {}}

    def _accuracy(rec):
        # Use ordinary accuracy consistently across all three benchmark
        # families. Belebele additionally reports length-normalized accuracy,
        # which remains available in the preserved raw harness output.
        return rec.get("acc,none", rec.get("acc"))

    scores = {}
    groups = results.get("groups", {})
    subtasks = results.get("results", {})
    for name in task_list:
        if name in debiased_langs.values():
            lang = name[len("xnli_"):]
            scores[name] = _xnli_debiased(adapter, lang, limit)
        else:
            rec = groups.get(name, subtasks.get(name, {}))
            scores[name] = _accuracy(rec)

    out_dir = ensure(Path(out_dir) if out_dir else RESULTS / "bench")
    payload = {
        "run": run_name, "checkpoint": tag, "tokenizer": tok_name,
        "lm_eval_version": importlib.metadata.version("lm_eval"),
        "num_fewshot": num_fewshot, "limit": limit, "tasks": task_list,
        "xnli_debiased": {lang: XNLI_DEBIAS_METHOD[lang] for lang in debiased_langs},
        "scores": scores, "results": results.get("results", {}),
        "groups": groups, "versions": results.get("versions", {}),
        "n-shot": results.get("n-shot", {}),
    }
    (out_dir / f"{run_name}_{tag}.json").write_text(
        json.dumps(payload, indent=2, default=_json_default)
    )
    print(f"[bench] {run_name} ({tag}): " +
          ", ".join(f"{k}={v:.4f}" for k, v in scores.items() if v is not None))

    if log_wandb:
        try:
            import wandb
            wb = wandb.init(project="XScript-Pretraining", id=run_name, resume="allow")
            wb.log({f"bench/{k}": v for k, v in scores.items() if v is not None})
            wb.finish()
        except Exception as exc:
            print(f"[bench] wandb logging skipped ({exc})")

    return scores


class _null:
    def __enter__(self): return self
    def __exit__(self, *args): return False


def _json_default(value):
    """Serialize NumPy scalars and other scalar-like harness values."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
